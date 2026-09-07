"""Backbone FLOP accounting with a critical / overlappable split. Off by default.

    from appcorr import flops

    with flops.session(model.vision_tower, model.language_model) as fl:   # backbone subtree
        for sample in data:
            with fl.request(sample.id):
                with fl.arrival(0), fl.stage("approx"):
                    ...                      # runs on the base image -> overlaps transmission
                for r in range(1, g):
                    with fl.arrival(r), fl.stage("correct"):
                        ...                  # only the last of these is critical
    print(fl.aggregate())

`session` is the on/off switch and it is the whole mechanism. Disabled, it installs no hooks and
does not wrap `scaled_dot_product_attention`, so the model runs exactly the code it ran before --
the cost of "off" is the absence of a call site, not a cheap branch. Enable it with
`session(..., enabled=True)` or by exporting `APPCORR_FLOPS=1`.

What is measured, and what is deliberately not:

  * **Backbone only.** The hooks go on the subtree you pass. A VFM's trunk up to its features, a
    VLM's vision encoder plus LLM, a VLA's the same. Task heads sit outside the subtree and are
    therefore never counted.
  * **Prefill only.** Decode is excluded. It always follows the whole image, so including it would
    raise every arm's critical share by an amount unrelated to the approximation being studied.
  * **Matmuls only**, at 2 FLOPs per multiply-accumulate. Norms, activations, softmax and residual
    adds are not counted; they are under 1% at these shapes and no arm here changes them.

See `counter.py` for the critical rule and why floor and ceiling come out 100% critical with no
special-casing.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator

from .counter import Bucket, FlopCounter, RequestFlops
from .hooks import attention_capture_ratio, install, patch_attention, record_attention, remove

__all__ = [
    "Bucket", "FlopCounter", "RequestFlops",
    "session", "enabled_by_default", "attention_capture_ratio", "record_attention",
]


def enabled_by_default() -> bool:
    """`APPCORR_FLOPS=1` turns accounting on without touching a driver's arguments."""
    return os.environ.get("APPCORR_FLOPS", "").strip().lower() in ("1", "true", "yes", "on")


@contextmanager
def session(*roots, enabled: bool | None = None) -> Iterator[FlopCounter]:
    """Account for backbone FLOPs under `roots` for the duration.

    Yields a `FlopCounter` either way, so a caller's `with fl.request(...)` / `fl.arrival(...)`
    blocks are written once and cost nothing when disabled: with no hooks installed and attention
    unwrapped, nothing ever calls `record`, and the scopes are a couple of thread-local
    assignments per request rather than per operation.
    """
    counter = FlopCounter()
    if enabled is None:
        enabled = enabled_by_default()
    if not enabled:
        yield counter
        return
    handles = install(counter, roots)
    try:
        with patch_attention(counter):
            yield counter
    finally:
        remove(handles)
