"""
openvla_vla.py

ModelExecutor for OpenVLA progressive prefill, wiring the appcorr.models.openvla library (Phase 1-3
forks, validated against the stock model) into the offload Task/Instruction pipeline so the existing
worker monitoring applies (CUDA-event server_events, InferenceResult telemetry, nsys ranges).

Op mapping (driven by offload/policies/scheduling/vla_interleaved_static.py):
  LOAD_INPUT       -> preprocess: normalized per-tower tensors from the decoded canvas; track newly
                      arrived patch indices, their mobile pscore hints (Patch.pscore_hint), and the
                      instruction text (Patch.text on group 0).
  PREPARE_TOKENS   -> first call: text session + vision approx + x0. Later calls: cumulative vision
                      correction on the updated canvas + x0 rebuild. The vision towers always
                      correct every arrived patch; which of the newly-arrived patches also get
                      their LLM position marked for CORRECT_FORWARD is optionally gated by
                      `scheduler_kwargs['server_pscore_threshold']` (see _filter_by_server_pscore,
                      AppCorr's _combine_patch_scores x _select_patch_keep_mask, faithfully ported).
                      Unset (default) = no filtering, every arrived patch gets LLM-corrected.
  APPROX_FORWARD   -> LLM approx segment {layers: (start, end)} advancing the frontier.
  CORRECT_FORWARD  -> LLM correction of the last-prepared group through {layers: (0, frontier)}.
  FULL_INFERENCE   -> stock predict_action() on the current canvas (baseline, same monitoring path).
  HEAD_INFERENCE   -> greedy 7-token action decode from the current prefill state.
  DECIDE_EXIT      -> no-op ({}): approx-pass confidence was shown uninformative under blur
                      (see analysis/experiments/ exit calibration); interface kept for parity.

Note: session state (KV caches, streams) lives on the wrapped OpenVLAProgressiveModel, which
supports ONE in-flight request at a time -- fine for the sequential control-loop driver
(batch_size=1, one frame at a time), asserted in preprocess.
"""

from typing import Any, Dict

import numpy as np
import torch

from offload.common import Task

from .base import ModelExecutor


class OpenVLAExecutor(ModelExecutor):
    def __init__(self, device: torch.device):
        super().__init__(device)
        self.pm = None  # OpenVLAProgressiveModel
        self._norms = None  # per-tower (mean, std) tensors

    def load_model(self, model_name: str, config: Any):
        from appcorr.models.openvla.progressive_model import OpenVLAProgressiveModel

        checkpoint = config.dataset_kwargs.get("checkpoint", "openvla/openvla-7b-finetuned-libero-spatial")
        unnorm_key = config.dataset_kwargs.get("unnorm_key")
        bucket_size = int(config.scheduler_kwargs.get("sdpa_query_bucket_size", 0))
        vision_correction = config.scheduler_kwargs.get("vision_correction", "cumulative")
        print(f"[Executor] Loading OpenVLA progressive model: {checkpoint} "
              f"(sdpa_query_bucket_size={bucket_size}, vision_correction={vision_correction})")
        self.pm = OpenVLAProgressiveModel(
            checkpoint, self.device, unnorm_key=unnorm_key, sdpa_query_bucket_size=bucket_size,
            vision_correction=vision_correction,
        )
        self.model = self.pm.vla

        # Per-tower normalization (dino: ImageNet stats, siglip: 0.5) straight from the HF processor.
        ip = self.pm.processor.image_processor
        self._norms = [
            (
                torch.tensor(ip.means[i]).view(1, 3, 1, 1).to(self.device),
                torch.tensor(ip.stds[i]).view(1, 3, 1, 1).to(self.device),
            )
            for i in range(len(ip.means))
        ]

    def _canvas_to_pixel_values(self, canvas_np: np.ndarray) -> torch.Tensor:
        """[1, H, W, 3] uint8 canvas -> channel-stacked [1, 6, H, W] bf16 (dino norm | siglip norm)."""
        t = torch.from_numpy(np.ascontiguousarray(canvas_np)).to(self.device).permute(0, 3, 1, 2).float() / 255.0
        towers = [(t - mean) / std for mean, std in self._norms]
        return torch.cat(towers, dim=1).to(torch.bfloat16)

    def preprocess(self, batch_data: Any, task: Task, context: Dict[str, Any], config: Any):
        assert isinstance(batch_data, np.ndarray) and batch_data.ndim == 4 and batch_data.shape[0] == 1, (
            "OpenVLAExecutor supports batch_size 1 (one control step at a time)"
        )
        context["pixel_values"] = self._canvas_to_pixel_values(batch_data)

        new_idx = sorted({p.spatial_idx for p in task.payload if p.group_id > 0 and p.spatial_idx >= 0})
        context.setdefault("arrived_idx", [])
        context.setdefault("pending_idx", [])
        context.setdefault("pscore_hint", {})
        context["pending_idx"].extend(i for i in new_idx if i not in context["arrived_idx"])
        context["arrived_idx"].extend(new_idx)

        for p in task.payload:
            if p.group_id > 0 and p.spatial_idx >= 0:
                context["pscore_hint"][int(p.spatial_idx)] = float(p.pscore_hint)
            if p.group_id == 0 and p.text:
                context["text"] = p.text
        return {"num_new_patches": len(new_idx)}

    def _filter_by_server_pscore(self, pending: list, context: Dict[str, Any], config: Any) -> list:
        """AppCorr-faithful fused-score gate (dinov3/layers/block.py: _combine_patch_scores with
        pscore_fusion='multiply', both weights=1.0, then _select_patch_keep_mask's hard
        token_keep_thres cutoff -- no min-keep floor, matching the original). Server pscore is
        DINOv2's CLS->patch attention, layer-averaged once at approx (`dino_cls_attn_layermean`,
        AppCorr's cls_attn_prob_layermean); mobile pscore is each patch's residual-RMS hint
        (`Patch.pscore_hint`, computed mobile-side by vla_patch_canvas.py). Filters which of this
        round's newly-arrived patches get their LLM position marked for CORRECT_FORWARD -- the
        vision towers still correct every arrived patch regardless (vision_correct's
        `all_arrived_idx` argument is untouched), so a filtered-out patch's embedding stays
        accurate while only its LLM hidden state remains stale for this round.

        Unit note: `Patch.pscore_hint` is a residual RMS over *raw* 0-255 canvas pixels (see
        vla_patch_canvas.py's encode()), while progressive_vla_libero_eval.py's validated
        `attnthresh_<N>` threshold was calibrated against a residual RMS over DINO-tower
        *normalized* pixels ((x/255 - mean)/std) -- ~255*std (~57x for ImageNet std) smaller.
        Rescale mobile_score by that factor before fusing so the same threshold value is
        comparable across both drivers; skipping this made an earlier version of this filter
        pass ~100% of patches at threshold=200e-6 (empirically confirmed via
        openvla_offload_libero_eval.py) since the un-rescaled raw-pixel residual dwarfed the
        threshold.
        """
        thresh = config.scheduler_kwargs.get("server_pscore_threshold")
        if thresh is None:
            return pending
        attn = self.pm.cache_feature.get("dino_cls_attn_layermean")
        if attn is None:
            return pending
        idx_t = torch.tensor(pending, dtype=torch.long, device=attn.device)
        server_score = attn[0, idx_t]
        mobile_score = torch.tensor(
            [context["pscore_hint"].get(i, 0.0) for i in pending],
            dtype=server_score.dtype, device=attn.device,
        )
        dino_std_mean = self._norms[0][1].mean().to(dtype=mobile_score.dtype, device=mobile_score.device)
        mobile_score = mobile_score / (255.0 * dino_std_mean)
        fused = server_score * mobile_score
        keep = (fused >= float(thresh)).tolist()
        return [i for i, k in zip(pending, keep) if k]

    def prepare_tokens(self, task: Task, context: Dict[str, Any], config: Any):
        px = context["pixel_values"]
        if not context.get("session_started"):
            self.pm.start_session_from_text(context.get("text", ""))
            self.pm.vision_approx(px)
            if config.scheduler_kwargs.get("schedule") == "chunked":
                # No blur-vision LLM approx pass; zero-init the K/V cache + prefill BOS so each
                # arriving group's LLM positions are prefilled exactly once (chunked causal prefill).
                self.pm.llm_chunked_init()
            context["session_started"] = True
            context["pending_idx"] = []
            return {"phase": "vision_approx"}

        pending = context.get("pending_idx", [])
        if not pending:
            return {"phase": "vision_noop"}
        all_idx = torch.tensor(context["arrived_idx"], dtype=torch.long, device=self.device)
        filtered = self._filter_by_server_pscore(pending, context, config)
        new_idx = torch.tensor(filtered, dtype=torch.long, device=self.device)
        context["last_vision_token_idx"] = self.pm.vision_correct(px, all_idx, new_idx)
        context["pending_idx"] = []
        return {
            "phase": "vision_correct",
            "num_arrived": len(pending),
            "num_corrected": int(new_idx.numel()),
        }

    def approx_forward(self, params: Dict[str, Any], context: Dict[str, Any], config: Any):
        start_l, end_l = params.get("layers", (0, len(self.pm.llm_layers)))
        self.pm.llm_approx_segment(start_l, end_l)

    def correct_forward(self, params: Dict[str, Any], context: Dict[str, Any], config: Any):
        _, end_l = params.get("layers", (0, self.pm.llm_frontier))
        token_idx = context.get("last_vision_token_idx")
        if token_idx is None or token_idx.numel() == 0:
            return {"skipped": "no vision tokens to correct"}
        if config.scheduler_kwargs.get("schedule") == "chunked":
            # Chunked causal prefill: prefill this group's vision positions once (+ text suffix on
            # the last group), never re-correcting earlier groups or the text.
            self.pm.llm_prefill_segment(token_idx, include_text=bool(params.get("include_text", False)))
        else:
            self.pm.llm_correct_segment(end_l, token_idx)

    def full_inference(self, task: Task, context: Dict[str, Any], config: Any):
        """Stock baseline: unmodified predict_action() on the current (fully assembled) canvas."""
        if not context.get("session_started"):
            self.pm.start_session_from_text(context.get("text", ""))
            context["session_started"] = True
        action = self.pm.vla.predict_action(
            input_ids=self.pm.input_ids,
            pixel_values=context["pixel_values"],
            unnorm_key=self.pm.unnorm_key,
            do_sample=False,
        )
        context["final_action"] = np.asarray(action)

    def head_inference(self, task: Task, context: Dict[str, Any], config: Any) -> Dict[str, Any]:
        assert self.pm.llm_frontier == len(self.pm.llm_layers), (
            f"HEAD_INFERENCE before the approx frontier reached the top ({self.pm.llm_frontier})"
        )
        action, stats = self.pm.decode_action(return_stats=True)
        context["final_action"] = np.asarray(action)
        worst = min(t["neighbor_mass"] for t in stats["per_token"])
        return {"bins": stats["bins"], "worst_neighbor_mass": float(worst)}

    def decide_exit(self, task: Task, context: Dict[str, Any], config: Any) -> Dict[str, Any]:
        # Approx-pass confidence is uninformative under blur (calibration: safe-exit precision at
        # base rate); kept as a no-op for interface parity with the DINOv3 executors.
        return {}

    def get_final_results(self, task: Task, context: Dict[str, Any], config: Any) -> Dict[int, Any]:
        if "final_action" not in context:
            return {}
        return {0: context["final_action"].tolist()}
