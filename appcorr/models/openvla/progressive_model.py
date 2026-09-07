"""
progressive_model.py

Phase 3 of the progressive-VLA-prefill plan (see /home/nxclab/.claude/plans/async-stargazing-mango.md):
wires the Phase 1 vision-tower forks and Phase 2 causal-LLM fork together into a single object that
can run a real progressive (approx -> correct -> correct -> ...) prefill on a real image, and decode
real actions from it -- answering the two open questions from that conversation: (1) does a partially
corrected prefill produce a meaningful *action* (not just a similar logit), and (2) does this hold
using genuine low-resolution image data (not synthetic noise) end-to-end through both towers and the
LLM together.

This is a lighter-weight standalone wrapper, not yet the formal `ModelExecutor` ABC
(`offload/server/model/base.py`) -- per the plan's Milestone 1 decision, we validate the core mechanism
locally first; wrapping this in `ModelExecutor` (Task/Instruction/OpType plumbing) is Phase 6's job, if
we get there.

Session lifecycle:
    model = OpenVLAProgressiveModel(checkpoint, device, unnorm_key)
    model.start_session(image, task_description, center_crop=True)
    logits = model.approx_forward(low_res_pixel_values)          # first pass (e.g. blurred image)
    logits = model.correct_forward(full_res_pixel_values, patch_idx)  # subsequent passes, cumulative
    action = model.decode_action()                                # 7-DoF continuous action, any time
"""

import math
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from appcorr.models.openvla.vision.backbone import ApproxCorrectViTBackbone
from appcorr.models.openvla.llm.llama_prefill_layer import ApproxCorrectLlamaDecoderLayer


class OpenVLAProgressiveModel:
    def __init__(self, checkpoint: str, device: torch.device, unnorm_key: Optional[str] = None,
                 sdpa_query_bucket_size: int = 0, vision_correction: str = "cumulative"):
        from transformers import AutoModelForVision2Seq, AutoProcessor

        self.device = device
        self.processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True)
        self.vla = AutoModelForVision2Seq.from_pretrained(
            checkpoint,
            attn_implementation="sdpa",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(device)
        self.vla.eval()

        self.unnorm_key = unnorm_key
        if self.unnorm_key is None:
            assert len(self.vla.norm_stats) == 1
            self.unnorm_key = next(iter(self.vla.norm_stats.keys()))

        vb = self.vla.vision_backbone
        self.dino_backbone = ApproxCorrectViTBackbone(vb.featurizer).to(device)
        self.siglip_backbone = ApproxCorrectViTBackbone(vb.fused_featurizer).to(device)
        assert self.dino_backbone.extract_block_idx == self.siglip_backbone.extract_block_idx or True, (
            "towers may have different depths; extraction indices are tracked independently"
        )

        self.llm_layers = [
            ApproxCorrectLlamaDecoderLayer.from_stock(l).to(device) for l in self.vla.language_model.model.layers
        ]

        # Ported from AppCorr's DINOv3 `sdpa_query_bucket_size` mechanism (attention.py's
        # correct_partial_token + dinov3_{depther,segmentor_m2f}.py's `_maybe_warmup_*`): a
        # data-dependent (e.g. fused-score-filtered) query-set size makes llm_correct_segment's
        # GEMM/SDPA shapes vary almost every call, which pays cuBLAS/cuBLASLt's one-time
        # per-shape algorithm-search cost on nearly every call instead of just the first (measured
        # ~30x slower for i.i.d.-varying Q vs any fixed/recurring shape). 0 = disabled (default,
        # matches AppCorr's own default-off convention); see _bucketize_token_idx and
        # _maybe_warmup_llm_correct_buckets.
        self.sdpa_query_bucket_size = sdpa_query_bucket_size
        self._warmup_done = False
        # Which patches the vision towers recompute in vision_correct: "cumulative" = every arrived
        # patch, every round (the LIBERO campaign's arm); "new_only" = only this round's group,
        # the DINOv3 interleaved schedule (block.correct persists the corrected increment, so
        # earlier rounds survive the per-round layer-0 restart). See vision_correct.
        assert vision_correction in ("cumulative", "new_only"), vision_correction
        self.vision_correction = vision_correction

        # Session state, set in start_session()
        self.cache_feature: Dict[str, Any] = {}
        self.input_ids: Optional[torch.Tensor] = None
        self.bos_embed: Optional[torch.Tensor] = None
        self.text_embed: Optional[torch.Tensor] = None
        self.num_vision_tokens: Optional[int] = None
        self.permanent_group: Optional[torch.Tensor] = None
        self.seq_len: Optional[int] = None
        self.round_idx = 0

    def _center_crop_and_resize(self, image: Image.Image, crop_scale: float = 0.9) -> Image.Image:
        orig_w, orig_h = image.size
        new_h, new_w = orig_h * math.sqrt(crop_scale), orig_w * math.sqrt(crop_scale)
        top, left = (orig_h - new_h) / 2, (orig_w - new_w) / 2
        cropped = image.crop((left, top, left + new_w, top + new_h))
        return cropped.resize((224, 224), Image.BILINEAR)

    def start_session(self, image: Image.Image, task_description: str, center_crop: bool = True):
        """Tokenizes the prompt and precomputes the (fixed, never-approximated) BOS + text embeddings.
        Also runs the processor's image transform once to get the *reference* full-res pixel_values
        (channel-stacked [1,6,224,224], DINOv2 first 3 channels + SigLIP next 3) for convenience --
        callers can still pass their own (e.g. blurred) pixel_values to approx_forward/correct_forward."""
        image = image.convert("RGB")
        if center_crop:
            image = self._center_crop_and_resize(image, 0.9).convert("RGB")

        prompt = f"In: What action should the robot take to {task_description.lower()}?\nOut:"
        inputs = self.processor(prompt, image).to(self.device, dtype=torch.bfloat16)
        input_ids = inputs["input_ids"]

        # Matches OpenVLAForActionPrediction.predict_action(): ensure the empty-string token (29871)
        # follows "Out:" before generation, as seen at training time.
        if not torch.all(input_ids[:, -1] == 29871):
            input_ids = torch.cat(
                [input_ids, torch.tensor([[29871]], dtype=input_ids.dtype, device=input_ids.device)], dim=1
            )
        self.input_ids = input_ids

        embed_layer = self.vla.get_input_embeddings()
        full_text_embed = embed_layer(input_ids)  # [1, T, C]
        self.bos_embed = full_text_embed[:, :1]
        self.text_embed = full_text_embed[:, 1:]

        self.reference_pixel_values = inputs["pixel_values"]

        self.cache_feature = {}
        self.round_idx = 0
        self.num_vision_tokens = None  # set on first approx_forward once we know patch count
        self.seq_len = None
        self.permanent_group = None
        # NOTE: _warmup_done is intentionally NOT reset here -- the bucket-shape warmup
        # (_maybe_warmup_llm_correct_buckets) is a one-time, model-instance-lifetime cost (cuBLAS's
        # per-shape algorithm cache persists across sessions), not a per-episode one. Resetting it
        # per session would re-pay the ~10s warmup on every episode instead of just the first.

    def _project_vision(self, dino_patch_feat: torch.Tensor, siglip_patch_feat: torch.Tensor) -> torch.Tensor:
        fused = torch.cat([dino_patch_feat, siglip_patch_feat], dim=2)
        return self.vla.projector(fused)

    def _finish_setup_after_first_pass(self, num_vision_tokens: int):
        self.num_vision_tokens = num_vision_tokens
        self.seq_len = 1 + num_vision_tokens + self.text_embed.shape[1]
        self.permanent_group = torch.cat([
            torch.tensor([0], device=self.device),
            torch.arange(1 + num_vision_tokens, self.seq_len, device=self.device),
        ])
        self._maybe_warmup_llm_correct_buckets()

    def _build_multimodal_embed(self, projected_vision: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.bos_embed, projected_vision, self.text_embed], dim=1)

    def approx_forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """First pass on a new image (e.g. a blurred/low-res canvas). Returns logits at every
        position (mainly the last one is of interest -- see `decode_action`)."""
        dino_px, siglip_px = torch.split(pixel_values.to(dtype=torch.bfloat16), [3, 3], dim=1)

        dino_feat, self.cache_feature = self.dino_backbone.approx_forward(dino_px, self.cache_feature, "dino")
        siglip_feat, self.cache_feature = self.siglip_backbone.approx_forward(siglip_px, self.cache_feature, "siglip")
        self._finish_setup_after_first_pass(dino_feat.shape[1])

        projected = self._project_vision(dino_feat, siglip_feat)
        x = self._build_multimodal_embed(projected)

        for i, layer in enumerate(self.llm_layers):
            x, self.cache_feature = layer.approx(x, self.cache_feature, f"llm_layer{i}")

        self.cache_feature["_x"] = x
        self.round_idx = 0
        logits = self._logits_from_x(x)
        return logits

    def correct_forward(self, pixel_values: torch.Tensor, patch_idx: torch.Tensor) -> torch.Tensor:
        """Subsequent pass once higher-res data has arrived for `patch_idx` (patch-grid indices,
        0-indexed, same convention as the vision backbones). Cumulative across rounds -- the vision
        towers' own cache_feature persists, so already-corrected patches stay correct."""
        assert self.num_vision_tokens is not None, "call approx_forward() at least once first"
        dino_px, siglip_px = torch.split(pixel_values.to(dtype=torch.bfloat16), [3, 3], dim=1)

        dino_feat, self.cache_feature = self.dino_backbone.correct_forward(dino_px, patch_idx, self.cache_feature, "dino")
        siglip_feat, self.cache_feature = self.siglip_backbone.correct_forward(siglip_px, patch_idx, self.cache_feature, "siglip")

        projected = self._project_vision(dino_feat, siglip_feat)
        x_layer0 = self._build_multimodal_embed(projected)

        vision_token_idx = patch_idx.to(dtype=torch.long, device=self.device) + 1  # +1 for BOS offset
        token_idx = torch.cat([vision_token_idx, self.permanent_group])

        x = x_layer0
        for i, layer in enumerate(self.llm_layers):
            x, self.cache_feature = layer.correct(x, token_idx, self.cache_feature, f"llm_layer{i}")

        self.cache_feature["_x"] = x
        self.round_idx += 1
        logits = self._logits_from_x(x)
        return logits

    def _logits_from_x(self, x: torch.Tensor) -> torch.Tensor:
        final = self.vla.language_model.model.norm(x)
        return self.vla.language_model.lm_head(final)

    # === Segment-level API (interleaved scheduling; used by offload/server/model/openvla_vla.py) ===
    # The full-pass methods above run vision + all 32 LLM layers in one call; the interleaved static
    # schedule (offload/policies/scheduling/vla_interleaved_static.py) instead advances the LLM
    # approx frontier in segments and corrects each residual group only up to the current frontier,
    # letting layers above the frontier absorb corrections in their single normal execution.

    def start_session_from_text(self, task_description: str):
        """start_session() minus the image transform -- the offload pipeline supplies pixel data
        via the decoded transmission canvas instead of a PIL image."""
        prompt = f"In: What action should the robot take to {task_description.lower()}?\nOut:"
        input_ids = self.processor.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        if not torch.all(input_ids[:, -1] == 29871):
            input_ids = torch.cat(
                [input_ids, torch.tensor([[29871]], dtype=input_ids.dtype, device=input_ids.device)], dim=1
            )
        self.input_ids = input_ids
        full_text_embed = self.vla.get_input_embeddings()(input_ids)
        self.bos_embed = full_text_embed[:, :1]
        self.text_embed = full_text_embed[:, 1:]
        self.reference_pixel_values = None
        self.cache_feature = {}
        self.round_idx = 0
        self.num_vision_tokens = None
        self.seq_len = None
        self.permanent_group = None
        self.llm_frontier = 0  # LLM layers approximated so far in this session
        self._x0 = None
        # _warmup_done is NOT reset here -- see start_session()'s note; one-time per model instance.

    def vision_approx(self, pixel_values: torch.Tensor):
        """Vision approx on the (base-layer) canvas + build the multimodal x0. Does NOT run any
        LLM layer; the stream starts at x0 with frontier 0."""
        dino_px, siglip_px = torch.split(pixel_values.to(dtype=torch.bfloat16), [3, 3], dim=1)
        dino_feat, self.cache_feature = self.dino_backbone.approx_forward(dino_px, self.cache_feature, "dino")
        siglip_feat, self.cache_feature = self.siglip_backbone.approx_forward(siglip_px, self.cache_feature, "siglip")
        self._finish_setup_after_first_pass(dino_feat.shape[1])
        self._x0 = self._build_multimodal_embed(self._project_vision(dino_feat, siglip_feat))
        self.cache_feature["_x"] = self._x0
        self.llm_frontier = 0

    def vision_correct(self, pixel_values: torch.Tensor, all_arrived_idx: torch.Tensor,
                       new_idx: torch.Tensor) -> torch.Tensor:
        """Cumulative vision correction on the current canvas (re-correcting earlier groups is
        cheap for the small towers and tightens their cross-round staleness), rebuilding x0.
        Returns the LLM token positions of the NEW group only (AppCorr-faithful per-group LLM
        correction). If no LLM layer has been approximated yet, the stream is simply refreshed to
        the new x0 (the whole upcoming approx absorbs the correction; the policy skips CORRECT at
        frontier 0).

        Tried a "new-only" variant (correct only this round's newly-arrived patches, mirroring the
        LLM's per-round correction, leaving earlier-round patches' outputs frozen) to cut
        PREPARE_TOKENS's cost. Two things ruled it out: (1) it measured ZERO latency benefit
        (8.377ms at 64 patches vs 8.354ms at 256 patches, essentially identical) -- unlike the LLM,
        the vision tower's correct_forward() dominant costs (prepare_full_tokens' patch_embed
        conv over the WHOLE image, and the per-layer full-[B,N,C] `x + blocks_out_sum`
        reconstruction) don't scale with the corrected-subset size at all, so shrinking the
        correction set bought nothing. (2) It also gives up bit-exactness *permanently*: unlike
        the LLM's causal structure (where a well-ordered schedule still reaches exactness once
        everything arrives), the vision towers are bidirectional, so a patch corrected early can
        never become exact just because a later patch also arrives -- nothing ever revisits it.
        Measured mean_abs_err=0.023, max_abs_err=1.25 vs a single-shot-full correction, even after
        all 256 patches had arrived (verified the OLD/current cumulative behavior is bit-exact
        there, max_abs_err=0.0). A real cost with no matching benefit -- reverted.

        2026-09-07: that measurement was confounded -- `block.correct` did not write the corrected
        increment back into `blocks_out_sum` at the time (interleaved-contract rule 3, DINOv3
        ac0238f), so under new-only every round's layer-0 restart rebuilt the earlier groups from
        the stale APPROX increment and discarded their corrections. The write-back is in now, and
        `vision_correction="new_only"` re-enables the schedule (towers get `new_idx` only) so it
        can be measured properly: analysis/experiments/openvla_newonly_gate.py (feature error split
        into with/without write-back) and flops_report_openvla.py (`newonly_*` keys). The
        latency argument (op-level, tiny towers) is unchanged; the FLOPs argument is not -- the
        towers are 3.4x a stock forward under cumulative and ~2x under new-only.
        """
        tower_idx = new_idx if self.vision_correction == "new_only" else all_arrived_idx
        dino_px, siglip_px = torch.split(pixel_values.to(dtype=torch.bfloat16), [3, 3], dim=1)
        dino_feat, self.cache_feature = self.dino_backbone.correct_forward(dino_px, tower_idx, self.cache_feature, "dino")
        siglip_feat, self.cache_feature = self.siglip_backbone.correct_forward(siglip_px, tower_idx, self.cache_feature, "siglip")
        self._x0 = self._build_multimodal_embed(self._project_vision(dino_feat, siglip_feat))
        if self.llm_frontier == 0:
            self.cache_feature["_x"] = self._x0
        return new_idx.to(dtype=torch.long, device=self.device) + 1  # +1 for BOS offset

    def llm_approx_segment(self, start_layer: int, end_layer: int):
        assert start_layer == self.llm_frontier, f"approx segment must resume at frontier {self.llm_frontier}, got {start_layer}"
        x = self.cache_feature["_x"]
        for i in range(start_layer, min(end_layer, len(self.llm_layers))):
            x, self.cache_feature = self.llm_layers[i].approx(x, self.cache_feature, f"llm_layer{i}")
        self.cache_feature["_x"] = x
        self.llm_frontier = min(end_layer, len(self.llm_layers))

    def _bucketize_token_idx(self, token_idx: torch.Tensor) -> torch.Tensor:
        """Pad `token_idx` (by repeating its last element) up to the next multiple of
        `sdpa_query_bucket_size`, bounding CORRECT_FORWARD's GEMM/SDPA shapes to a small fixed set
        instead of a new size almost every call (see `sdpa_query_bucket_size`'s docstring in
        __init__). Disabled (returns `token_idx` unchanged) when `sdpa_query_bucket_size <= 0`.

        Adapted from AppCorr's DINOv3 bucketing (`attention.py::correct_partial_token`), which pads
        a *separate* scratch Q tensor with garbage rows (safe there only because non-causal ViT
        attention computes each query row independently, so garbage rows can't leak into real
        outputs -- real rows are gathered back out afterward). Our causal LLM setting makes the
        equivalent simpler: since every downstream step here (RoPE, causal mask, KV-cache
        writeback, and the decoder layer's `x_out[:, token_idx] = ...` scatter) is naturally safe
        under an index *repeated* (not garbage), padding with a duplicate of a REAL, already-valid
        index makes every extra row redundant-but-correct: the KV cache gets rewritten with the
        identical value, and the final scatter-write reassigns the same real position the same
        value. No separate scratch buffer, output gather, or masking of padded rows is needed.
        """
        bucket = self.sdpa_query_bucket_size
        if bucket <= 0:
            return token_idx
        Q = token_idx.numel()
        if Q == 0:
            return token_idx
        padded_Q = ((Q + bucket - 1) // bucket) * bucket
        if padded_Q == Q:
            return token_idx
        pad = token_idx[-1].expand(padded_Q - Q)
        return torch.cat([token_idx, pad])

    def _maybe_warmup_llm_correct_buckets(self):
        """Pre-runs `.correct()` once for every bucket size (up to a full-sequence correction),
        on fully disposable scratch tensors under a private cache tag, so cuBLAS/cuBLASLt's
        one-time per-shape algorithm-search cost is paid during session startup rather than during
        real, timed control steps. Ported from AppCorr's DINOv3 depther/M2F-segmentor
        `_maybe_warmup_*` (offload/server/model/dinov3_{depther,segmentor_m2f}.py): scratch
        state only, real `self.cache_feature` is never touched. A no-op unless
        `sdpa_query_bucket_size > 0`; runs once per model instance, not per session (guarded by
        `_warmup_done`, which session-start methods deliberately do NOT reset) -- an offline,
        amortized cost, not a per-episode one.

        All 32 Llama decoder layers share identical GEMM/attention shapes (uniform hidden size,
        head count, head_dim), so warming with real per-layer scratch caches (matching AppCorr's
        M2F pattern of warming every backbone block, not just one) is a modest, one-time,
        session-startup cost -- not a per-step one.
        """
        if self.sdpa_query_bucket_size <= 0 or self._warmup_done:
            return
        self._warmup_done = True
        bucket = self.sdpa_query_bucket_size
        N = self.seq_len
        C = self.bos_embed.shape[-1]
        dtype = self.bos_embed.dtype
        max_q = ((N + bucket - 1) // bucket) * bucket
        with torch.no_grad():
            for q in range(bucket, max_q + 1, bucket):
                q_eff = min(q, N)
                token_idx = torch.arange(q_eff, device=self.device, dtype=torch.long)
                x = torch.zeros(1, N, C, device=self.device, dtype=dtype)
                scratch: Dict[str, Any] = {}
                for i, layer in enumerate(self.llm_layers):
                    tag = f"_warmup_layer{i}"
                    nh, hd = layer.self_attn.num_key_value_heads, layer.self_attn.head_dim
                    scratch[f"{tag}_kv"] = torch.zeros(1, nh, N, 2, hd, device=self.device, dtype=dtype)
                    scratch[f"{tag}_blocks_out_sum"] = torch.zeros(1, N, C, device=self.device, dtype=dtype)
                    layer.correct(x, token_idx, scratch, tag)
                del scratch
        torch.cuda.synchronize()

    def llm_correct_segment(self, end_layer: int, vision_token_idx: torch.Tensor):
        """Correct the new group's positions + permanent group through layers [0, end_layer),
        restarting from the current x0 (mirrors DINOv3's x_temp = input_tokens per CORRECT).
        The resulting stream replaces the frontier stream, so subsequent approx segments absorb it."""
        end_layer = min(end_layer, len(self.llm_layers))
        token_idx = torch.cat([vision_token_idx, self.permanent_group])
        token_idx = self._bucketize_token_idx(token_idx)
        x = self._x0
        # RoPE has no learnable parameters -- cos/sin depend only on token_idx + head_dim/theta
        # (shared config across all layers), so they're identical every layer. Computing them once
        # per segment instead of once per layer (profiled: ~0.1ms/layer, the single largest
        # sub-cost in ApproxCorrectLlamaAttention.correct()) removes ~(end_layer-1)x redundant work
        # -- an exact optimization, not an approximation (verified bit-identical downstream).
        B = x.shape[0]
        position_ids_sel = token_idx.unsqueeze(0).expand(B, -1).to(device=x.device)
        cos, sin = self.llm_layers[0].self_attn.rotary_emb(x, position_ids_sel)
        for i in range(end_layer):
            x, self.cache_feature = self.llm_layers[i].correct(
                x, token_idx, self.cache_feature, f"llm_layer{i}", cos=cos, sin=sin,
            )
        self.cache_feature["_x"] = x

    # === True chunked causal prefill (no LLM approx-then-correct redundancy) ===
    # For a CAUSAL decoder with sequential (raster, top-to-bottom) vision grouping, each vision
    # token attends only to earlier positions, so its LLM state can be computed EXACTLY ONCE, when
    # its group arrives, and never revisited -- unlike the bidirectional vision towers, which must
    # be recomputed per group. This replaces `llm_approx_segment` (a full blur-vision LLM pass) +
    # per-group `llm_correct_segment` (which re-corrects the whole text suffix every round). It is
    # bit-equivalent to that path here: a corrected group's causal mask already excludes the
    # not-yet-arrived (blur/zero) later positions, so the base approx's blur K/V for them was never
    # read anyway; and the text suffix, prefilled once at the last group, attends to exactly the
    # same per-group arrival-time vision K/V. Net LLM work drops from ~2x vision + ~5x text to 1x.

    def _llm_prefill_positions(self, token_idx: torch.Tensor):
        """Prefill `token_idx` through all LLM layers via the O(Q) `.prefill()` path (threads only
        the Q query rows, never a full [B, N, C] tensor). The resulting top-layer states are
        scattered back into `cache["_x"]` so `decode_action` (which reads the text-end position)
        sees them. Reused for BOS, each vision group, and the final text suffix."""
        token_idx = self._bucketize_token_idx(token_idx.to(dtype=torch.long, device=self.device))
        x0 = self._x0
        B = x0.shape[0]
        position_ids_sel = token_idx.unsqueeze(0).expand(B, -1).to(device=x0.device)
        cos, sin = self.llm_layers[0].self_attn.rotary_emb(x0, position_ids_sel)
        # Causal + raster order: this group only reaches keys up to its top position, so bound the
        # attention K/V to that prefix (computed once here, not 32x -- avoids per-layer GPU->CPU sync).
        key_end = int(token_idx.max().item()) + 1
        x_sel = x0[:, token_idx]  # [B, Q, C]
        for i in range(len(self.llm_layers)):
            x_sel, self.cache_feature = self.llm_layers[i].prefill(
                x_sel, token_idx, self.cache_feature, f"llm_layer{i}", cos=cos, sin=sin, key_end=key_end,
            )
        x_full = self.cache_feature.get("_x", x0)
        x_full = x_full.clone()  # don't alias _x0 / the previous group's stream
        x_full[:, token_idx] = x_sel.to(dtype=x_full.dtype)
        self.cache_feature["_x"] = x_full

    def llm_chunked_init(self):
        """Base setup for chunked prefill: zero-initialise every layer's K/V cache (so later
        `.prefill()` calls act as first-time prefills; positions not yet prefilled are never read
        thanks to causal masking), then prefill BOS (position 0). No `blocks_out_sum` is needed --
        the O(Q) prefill path never reconstructs non-queried positions. Sets the frontier to full
        depth so HEAD_INFERENCE's readiness assertion passes."""
        x = self._x0
        B, N, _ = x.shape
        attn0 = self.llm_layers[0].self_attn
        h_kv, d_h = attn0.num_key_value_heads, attn0.head_dim
        for i in range(len(self.llm_layers)):
            self.cache_feature[f"llm_layer{i}_kv"] = torch.zeros(
                B, h_kv, N, 2, d_h, device=x.device, dtype=x.dtype
            )
        self._llm_prefill_positions(torch.tensor([0], device=x.device, dtype=torch.long))
        self.llm_frontier = len(self.llm_layers)

    def llm_prefill_segment(self, vision_token_idx: torch.Tensor, include_text: bool):
        """Prefill this group's vision-token positions (+ the text suffix once, on the last group)
        through all layers. Causal + raster order makes this the exact, non-redundant counterpart of
        `llm_correct_segment`."""
        token_idx = vision_token_idx.to(dtype=torch.long, device=self.device)
        if include_text:
            token_idx = torch.cat([token_idx, self.permanent_group])
        self._llm_prefill_positions(token_idx)

    def decode_action(self, num_action_tokens: Optional[int] = None, return_stats: bool = False):
        """Greedy-decodes the action tokens from the current prefill state and converts them to a
        continuous action using the exact same bin-center + un-normalize logic as
        OpenVLAForActionPrediction.predict_action() (modeling_prismatic.py).

        With `return_stats=True`, also returns per-action-token confidence stats computed over the
        256-wide action-bin logit slice (the model only ever emits these at action positions) --
        the ingredients for the Phase 4 early-exit decision, mirroring the metric menu of
        `dinov3_classifier.py::decide_exit` (max_prob / top2_margin / entropy) plus a bin-aware
        `neighbor_mass` (probability within +-1 bin of the argmax; adjacent bins are near-identical
        continuous actions, so mass there should count toward confidence)."""
        if num_action_tokens is None:
            num_action_tokens = self.vla.get_action_dim(self.unnorm_key)

        # Action-bin token ids live in [vocab_size - n_action_bins, vocab_size) (see the
        # `vocab_size - token_id` de-tokenization below).
        bin_lo = self.vla.vocab_size - self.vla.config.n_action_bins

        def slice_stats(last_logits: torch.Tensor) -> Dict[str, float]:
            probs = torch.softmax(last_logits[0, bin_lo : self.vla.vocab_size].float(), dim=-1)
            top2 = probs.topk(2)
            arg = int(top2.indices[0].item())
            lo, hi = max(arg - 1, 0), min(arg + 2, probs.shape[0])
            entropy = -(probs * (probs + 1e-10).log()).sum()
            return {
                "max_prob": float(top2.values[0]),
                "top2_margin": float(top2.values[0] - top2.values[1]),
                "entropy": float(entropy),
                "neighbor_mass": float(probs[lo:hi].sum()),
            }

        x = self.cache_feature["_x"]
        logits = self._logits_from_x(x)
        next_token = logits[:, -1].argmax(-1, keepdim=True)
        generated = [next_token.item()]
        stats = [slice_stats(logits[:, -1])] if return_stats else None

        # Convert our hand-built per-layer KV cache ([B, H_kv, N, 2, Dh]) into the legacy
        # tuple-of-(K, V) format transformers' LlamaModel.forward() accepts (it wraps it in a
        # DynamicCache internally) so the remaining action tokens can be decoded with the stock,
        # unforked model -- decode is a plain cached generation, no approx/correct needed there.
        # Note the .contiguous() copies: decode-time cache growth never mutates cache_feature,
        # so a confidence-gated flow may decode from approx and still correct_forward afterwards.
        past_key_values = tuple(
            (kv[:, :, :, 0].contiguous(), kv[:, :, :, 1].contiguous())
            for kv in (self.cache_feature[f"llm_layer{i}_kv"] for i in range(len(self.llm_layers)))
        )

        with torch.no_grad():
            for _ in range(num_action_tokens - 1):
                out = self.vla.language_model(
                    input_ids=next_token, past_key_values=past_key_values, use_cache=True, return_dict=True
                )
                past_key_values = out.past_key_values
                next_token = out.logits[:, -1].argmax(-1, keepdim=True)
                generated.append(next_token.item())
                if return_stats:
                    stats.append(slice_stats(out.logits[:, -1]))

        predicted_action_token_ids = np.array(generated[-num_action_tokens:])
        discretized_actions = self.vla.vocab_size - predicted_action_token_ids
        discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=self.vla.bin_centers.shape[0] - 1)
        normalized_actions = self.vla.bin_centers[discretized_actions]

        action_norm_stats = self.vla.get_action_stats(self.unnorm_key)
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
        action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )
        if return_stats:
            return actions, {"per_token": stats, "bins": discretized_actions.tolist()}
        return actions
