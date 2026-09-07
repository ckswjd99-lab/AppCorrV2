"""Backbone FLOP accounting, split into critical and overlappable work.

The question this exists to answer: **how much of a request's backbone compute can only start
once the server holds the whole image?** That is the part progressive transmission cannot hide
behind the network, so it is what sets the floor on end-to-end latency.

The rule is one line: **an operation is critical if it runs after the final arrival of that
request.** Everything else overlapped with transmission and is free in the latency sense, however
many FLOPs it consumed.

Read against the arms this repo measures, the rule reproduces the expected answers without any
special cases:

    floor (L2 approx-only)   the degraded image IS the whole transmission, so its single pass runs
                             after the last byte -> 100% critical
    ceiling (L0 exact)       same, at full resolution -> 100% critical
    one-shot correction      the approximate pass runs on the base image, which arrives first, so
                             it is NOT critical; the correction waits for the last detail -> the
                             correction alone is critical
    interleaved g=4          rounds 0..2 run as their groups land; only round 3 and whatever
                             approximate frontier follows it are critical -> about 1/g

Arrivals are numbered, not named, and `critical` resolves at report time as "everything tagged with
the highest arrival index seen". That is why floor and ceiling need no special handling: they never
open an arrival, so every op carries index 0, which is also the maximum, and the whole request comes
out critical.

**Scope is the backbone.** Heads are excluded by construction rather than by name: hooks are
installed on the module subtree the caller designates, so anything outside it is invisible. For a
VFM that subtree ends where features are produced; for a VLM and a VLA it is the vision encoder plus
the LLM. Decode is excluded -- it always follows the whole image, so counting it would inflate every
arm's critical share by an amount that has nothing to do with the approximation being studied.

**What is counted.** Multiply-accumulates in the matmuls, at 2 FLOPs each:

    Linear      2 * numel(output) * in_features
    Conv        2 * numel(output) * (in_channels / groups) * prod(kernel)
    attention   2 * B * H * Sq * Sk * D  for QK^T, and again for the AV product

Elementwise work -- norms, activations, softmax, residual adds -- is not counted. It is under 1% of
these models at these shapes and it is not what any of the arms here change. Bias adds are likewise
skipped, per the usual convention.

**Off means off.** When accounting is disabled nothing is installed: no hooks, no patched
`scaled_dot_product_attention`, no per-call branch. The disabled cost is not "a cheap check", it is
the absence of a call site.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class Bucket:
    """FLOPs accumulated under one (arrival index, stage) pair."""

    linear: int = 0
    conv: int = 0
    attention: int = 0

    @property
    def total(self) -> int:
        return self.linear + self.conv + self.attention

    def __iadd__(self, other: "Bucket") -> "Bucket":
        self.linear += other.linear
        self.conv += other.conv
        self.attention += other.attention
        return self


@dataclass
class RequestFlops:
    """One request's backbone FLOPs, keyed by (arrival index, stage).

    `stage` is free-form and only for reporting -- "approx", "correct", "prefill", "vision". The
    critical/overlappable split never looks at it; that split is decided by the arrival index alone.

    One stage IS excluded from `total`/`critical`, not just reported differently: `PREPARE_TOKENS`
    (the offload pipeline's `OpType.PREPARE_TOKENS.name`, used verbatim as the stage label for every
    model -- see `worker.py`'s `fl.stage(instr.op_type.name)`). For ADE20K's m2f segmentor this stage
    re-runs the ViT-Adapter's SPM from scratch on EVERY interleaved round (`ADE20KWindowInterleavedPolicy`
    prepends `PREPARE_TOKENS` to every group's task list), measured at 6.75x the model's entire
    ceiling forward and identical to the last digit between keep=0.25 and keep=0.50 -- i.e. it scales
    with round count, not with what is actually being corrected. That is real waste worth fixing
    before a latency pass (cache the embedding across a request's rounds instead of recomputing it),
    but it is not backbone compute, which is the only thing this counter exists to report. Cheap
    executors (plain ViT patch-embed) barely notice excluding it; only ADE20K's number was ever
    dominated by it. TODO: fix the re-embed-per-round waste itself instead of just not counting it.

    `PSCORE` is excluded for a different and more deliberate reason. The patch-importance score is
    residual energy times the attention each token RECEIVES, and those attention weights are already
    computed by the layer's own forward -- a faithful implementation would read them out of it. The
    forks recompute them instead because pulling the value out of a fused attention call is fiddly,
    so that recomputation is an artefact of how this was written, not a cost AppCorr would pay in a
    real deployment. Charging the technique for it overstates what it spends.

    Most forks avoid the issue by accident rather than design: OpenCLIP and Gemma 3 score with a bare
    `torch.softmax(q @ k.T)` and DINOv3 with a Triton kernel, none of which is an `nn.Linear` or
    `F.scaled_dot_product_attention`, so the hooks never see them. Gemma 3 is the exception that made
    this scope necessary -- its `_incoming_attention` re-runs the q/k PROJECTIONS, which ARE
    `nn.Linear` and were being counted twice. If a future fork routes its score through SDPA it will
    start being charged silently; wrap it in this stage.
    """

    EXCLUDED_STAGES = frozenset({"PREPARE_TOKENS", "PSCORE"})

    request_id: object = None
    buckets: Dict[Tuple[int, str], Bucket] = field(default_factory=dict)
    meta: Dict[str, object] = field(default_factory=dict)

    def add(self, arrival: int, stage: str, *, linear: int = 0, conv: int = 0,
            attention: int = 0) -> None:
        b = self.buckets.get((arrival, stage))
        if b is None:
            b = self.buckets[(arrival, stage)] = Bucket()
        b.linear += linear
        b.conv += conv
        b.attention += attention

    # --- the split ------------------------------------------------------------------------------ #

    @property
    def final_arrival(self) -> int:
        """The last arrival this request saw. 0 when nothing ever arrived progressively."""
        return max((a for a, _ in self.buckets), default=0)

    @property
    def total(self) -> int:
        return sum(b.total for (_, s), b in self.buckets.items() if s not in self.EXCLUDED_STAGES)

    @property
    def critical(self) -> int:
        """FLOPs that could not start until the whole image was in hand."""
        last = self.final_arrival
        return sum(b.total for (a, s), b in self.buckets.items()
                   if a == last and s not in self.EXCLUDED_STAGES)

    @property
    def overlappable(self) -> int:
        return self.total - self.critical

    @property
    def critical_fraction(self) -> float:
        t = self.total
        return (self.critical / t) if t else 0.0

    def by_stage(self) -> Dict[str, int]:
        """Every stage, INCLUDING `EXCLUDED_STAGES` -- this is the debugging view that should
        catch the next `PREPARE_TOKENS`-shaped surprise, so it stays complete even though
        `total`/`critical` do not."""
        out: Dict[str, int] = {}
        for (_, stage), b in self.buckets.items():
            out[stage] = out.get(stage, 0) + b.total
        return out

    def by_arrival(self) -> Dict[int, int]:
        out: Dict[int, int] = {}
        for (a, _), b in self.buckets.items():
            out[a] = out.get(a, 0) + b.total
        return out

    def summary(self) -> Dict[str, object]:
        return {
            "total_flops": self.total,
            "critical_flops": self.critical,
            "overlappable_flops": self.overlappable,
            "critical_fraction": self.critical_fraction,
            "final_arrival": self.final_arrival,
            "by_arrival": self.by_arrival(),
            "by_stage": self.by_stage(),
            **({"meta": dict(self.meta)} if self.meta else {}),
        }


class FlopCounter:
    """Accumulates backbone FLOPs for one process, request by request.

    Thread-local arrival/stage state, because the offload server runs its executors on worker
    threads and a global "current round" would be attributed to whichever request happened to be
    executing.
    """

    def __init__(self) -> None:
        self.requests: List[RequestFlops] = []
        self._by_id: Dict[object, RequestFlops] = {}
        self._local = threading.local()

    # --- the state the hooks read ---------------------------------------------------------------- #

    def _ctx(self) -> Tuple[Optional[RequestFlops], int, str]:
        req = getattr(self._local, "request", None)
        return req, getattr(self._local, "arrival", 0), getattr(self._local, "stage", "backbone")

    def record(self, *, linear: int = 0, conv: int = 0, attention: int = 0) -> None:
        """Called by the hooks. A record outside any `request()` scope is dropped on purpose --
        model loading, warmup and head inference all run outside one."""
        req, arrival, stage = self._ctx()
        if req is None:
            return
        req.add(arrival, stage, linear=linear, conv=conv, attention=attention)

    # --- scopes ---------------------------------------------------------------------------------- #

    @contextmanager
    def request(self, request_id: object = None, **meta):
        """One instruction / one served request. FLOPs recorded outside this are not attributed."""
        prev = getattr(self._local, "request", None)
        prev_a = getattr(self._local, "arrival", 0)
        prev_s = getattr(self._local, "stage", "backbone")
        req = self._by_id.get(request_id) if request_id is not None else None
        if req is None:
            req = RequestFlops(request_id=request_id)
            self.requests.append(req)
            if request_id is not None:
                self._by_id[request_id] = req
        req.meta.update(meta)
        self._local.request, self._local.arrival, self._local.stage = req, 0, "backbone"
        try:
            yield req
        finally:
            self._local.request, self._local.arrival, self._local.stage = prev, prev_a, prev_s

    @contextmanager
    def arrival(self, index: int):
        """Work that became possible once arrival `index` landed.

        Index the arrivals in the order they occur. Whichever index is highest at report time is
        the one the critical split reads, so an arm with no progressive transmission can simply
        never open one of these and still come out 100% critical.
        """
        prev = getattr(self._local, "arrival", 0)
        self._local.arrival = int(index)
        try:
            yield
        finally:
            self._local.arrival = prev

    @contextmanager
    def stage(self, name: str):
        """A reporting label only -- it never affects the critical split."""
        prev = getattr(self._local, "stage", "backbone")
        self._local.stage = str(name)
        try:
            yield
        finally:
            self._local.stage = prev

    # --- reporting ------------------------------------------------------------------------------- #

    def reset(self) -> None:
        self.requests.clear()
        self._by_id.clear()

    def aggregate(self) -> Dict[str, object]:
        """Means over requests, which is the per-instruction number, plus the totals behind it."""
        n = len(self.requests)
        if not n:
            return {"requests": 0}
        tot = sum(r.total for r in self.requests)
        crit = sum(r.critical for r in self.requests)
        stages: Dict[str, int] = {}
        for r in self.requests:
            for k, v in r.by_stage().items():
                stages[k] = stages.get(k, 0) + v
        return {
            "requests": n,
            "mean_total_gflops": tot / n / 1e9,
            "mean_critical_gflops": crit / n / 1e9,
            "mean_overlappable_gflops": (tot - crit) / n / 1e9,
            # Fraction of the AGGREGATE, not the mean of per-request fractions: requests differ in
            # size by an order of magnitude here (256 to 2975 image tokens), so a mean of ratios
            # would weight a thumbnail the same as a full-page infographic.
            "critical_fraction": (crit / tot) if tot else 0.0,
            "mean_stage_gflops": {k: v / n / 1e9 for k, v in sorted(stages.items())},
        }
