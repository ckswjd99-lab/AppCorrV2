"""
qwen25vl_executor.py

ModelExecutor for Qwen2.5-VL (32B/72B), driving the forked vision tower
(appcorr/models/qwen25vl/vision/) and forked causal LLM decoder
(appcorr/models/qwen25vl/llm/decoder_layer.py) through the existing GroupTriggerPolicy scheduling
contract. `total_layers` for GroupTrigger's chunking refers to the LLM's layer count (64) -- the
vision tower is corrected to FULL depth (all 32 layers) once per group arrival (not chunked against
the LLM's schedule), matching OpenVLA's own design (DINOv2/SigLIP correction also always runs each
tower to completion per round; only the LLM's causal correction is what GroupTriggerPolicy chunks).

Each image's patches are transmitted at MERGE-GROUP granularity (`transmission_kwargs.patch_size`
should be set to `(28,28)`, matching `spatial_merge_size(2) * patch_size(14)` -- see Phase 1's
fork), so `Patch.spatial_idx` directly indexes the same `num_merge_groups = seq_len // 4` space the
vision tower fork's `correct_forward(group_idx=...)` expects, with no extra mapping needed.

Per-request question text travels via `Patch.text_payload` (set by the driver on the base-layer
patches), read once in `preprocess()` and cached in `context`.

batch_size is forced to 1 (same convention as every prior driver this session).
"""

import os
from typing import Any, Dict

import torch

from offload.common import Task
from .base import ModelExecutor

MODEL_ID_32B = "Qwen/Qwen2.5-VL-32B-Instruct"


def _trace_save(name: str, tensor: torch.Tensor):
    """Bisection probe (temporary): if APPCORR_QWEN_TRACE_DIR is set, dump a tensor at a named
    checkpoint so two arms' runs can be diffed offline. Gated, no-op otherwise."""
    trace_dir = os.environ.get("APPCORR_QWEN_TRACE_DIR")
    if not trace_dir:
        return
    label = os.environ.get("APPCORR_QWEN_TRACE_LABEL", "run")
    os.makedirs(trace_dir, exist_ok=True)
    torch.save(tensor.detach().cpu(), os.path.join(trace_dir, f"{label}_{name}.pt"))


class Qwen25VLExecutor(ModelExecutor):
    _position_ids_verified = False  # one-shot latch for the mm_token_type_ids independence assertion

    def __init__(self, device: torch.device):
        super().__init__(device)
        self.processor = None
        self.vision_tower = None
        self.llm_layers = None
        self.num_llm_layers = 0
        self.image_token_id = None

    def backbone_modules(self):
        """A VLM's backbone is both halves: the vision tower and the language model.

        `lm_head` is excluded -- it is the header, and it is also the only part whose cost scales
        with the vocabulary rather than with the image.
        """
        inner = getattr(self.model, "model", None)
        return [getattr(inner, "visual", None), getattr(inner, "language_model", None)]

    def load_model(self, model_name: str, config: Any):
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        from appcorr.models.qwen25vl.vision.backbone import ApproxCorrectQwen25VLVisionTower
        from appcorr.models.qwen25vl.llm.decoder_layer import ApproxCorrectQwen25VLDecoderLayer

        model_path = config.dataset_kwargs.get("model_path", MODEL_ID_32B)
        # dataset_kwargs.model_dtype: "bfloat16" (default, every eval) or "float32" (numerical
        # gates only -- e.g. analysis/experiments/qwen25vl_text_correct_gate.py).
        self.dtype = getattr(torch, str(config.dataset_kwargs.get("model_dtype", "bfloat16")))
        print(f"[Executor] Loading Model: {model_path} ({self.dtype}) ...")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, dtype=self.dtype, attn_implementation="sdpa"
        ).to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.image_token_id = self.model.config.image_token_id

        self.vision_tower = ApproxCorrectQwen25VLVisionTower(self.model.model.visual).to(self.device).eval()
        text_model = self.model.model.language_model
        self.llm_layers = [
            ApproxCorrectQwen25VLDecoderLayer.from_stock(layer, text_model.rotary_emb) for layer in text_model.layers
        ]
        self.num_llm_layers = len(self.llm_layers)
        print(f"[Executor] Loaded. vision_layers={len(self.vision_tower.blocks)} llm_layers={self.num_llm_layers}")

    def preprocess(self, reconstructed_canvas: Any, task: Task, context: Dict[str, Any], config: Any):
        """`reconstructed_canvas` MUST be the image reconstructed from whatever patches the
        transmission layer has decoded so far (blurred base + any arrived corrections) -- i.e.
        exactly what `WorkerModule` builds via `policy.decode(patch_buffer, config, canvas=prev)`
        (worker.py:177-188), NOT a raw/original-resolution image. This method has no way to detect
        the difference (it will happily process a full-resolution image and produce plausible
        `pixel_values`), which is exactly what let an earlier bug (a caller passing the raw image
        for every keep_rate/approx group) go undetected until an approx-only accuracy sanity check
        caught it -- see `refcoco_gqa_batched_eval.py`'s `build_first_token_context` docstring."""
        if isinstance(reconstructed_canvas, torch.Tensor):
            canvas = reconstructed_canvas[0].cpu().numpy() if reconstructed_canvas.ndim == 4 else reconstructed_canvas.cpu().numpy()
        else:
            canvas = reconstructed_canvas[0] if reconstructed_canvas.ndim == 4 else reconstructed_canvas
        canvas = canvas.astype("uint8")

        proc_out = self.processor.image_processor(images=[canvas], return_tensors="pt")
        context["pixel_values"] = proc_out["pixel_values"].to(device=self.device, dtype=self.dtype)
        context["image_grid_thw"] = proc_out["image_grid_thw"].to(self.device)
        _trace_save("pixel_values", context["pixel_values"])
        _trace_save("image_grid_thw", context["image_grid_thw"])

        num_groups = context["pixel_values"].shape[0] // self.vision_tower.spatial_merge_unit
        if "group_map" not in context:
            context["group_map"] = torch.full((1, num_groups), -1, device=self.device, dtype=torch.long)
        if "pscore_map" not in context:
            # Per-merge-group importance hint (Patch.pscore_hint -- residual energy, already
            # computed client-side by every transmission policy that sets `mobile_pscore`; see
            # progressive.py's `_compute_patch_pscore_hint`). Used by `_prune_patch_idx` for the
            # vision-side keep rate. 0.0 for groups whose patches haven't arrived yet, matching
            # `group_map`'s -1 sentinel convention.
            context["pscore_map"] = torch.zeros((1, num_groups), device=self.device, dtype=torch.float32)
        if "question" not in context:
            for p in task.payload:
                if getattr(p, "text_payload", ""):
                    context["question"] = p.text_payload
                    break
        group_map = context["group_map"]
        pscore_map = context["pscore_map"]
        for p in task.payload:
            if 0 <= p.spatial_idx < num_groups:
                group_map[0, p.spatial_idx] = p.group_id
                pscore_map[0, p.spatial_idx] = float(p.pscore_hint)

    def _build_prompt(self, context: Dict[str, Any]):
        """Builds the text prompt + multimodal input scaffolding ONCE per request. Always exact
        (tokenization/text embedding has no notion of corrected-vs-stale), matching every prior
        fork's philosophy for cheap non-block prep steps -- but unlike DINOv3/CLIP, this also needs
        the image_grid_thw (to know how many image-token placeholders to insert), which is why it
        runs inside prepare_tokens (after preprocess has set pixel_values/image_grid_thw) rather
        than being independent of the image."""
        question = context["question"]
        grid_thw = context["image_grid_thw"]
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        merge_unit = self.vision_tower.spatial_merge_unit
        num_image_tokens = int((grid_thw.prod(dim=-1) // merge_unit).sum().item())
        image_pad = "<|image_pad|>" * num_image_tokens
        text = text.replace("<|vision_start|><|image_pad|><|vision_end|>", f"<|vision_start|>{image_pad}<|vision_end|>")

        tok_out = self.processor.tokenizer(text, return_tensors="pt")
        input_ids = tok_out["input_ids"].to(self.device)
        attention_mask = tok_out["attention_mask"].to(self.device)
        image_mask_1d = (input_ids[0] == self.image_token_id)
        mm_token_type_ids = image_mask_1d.long().unsqueeze(0)

        position_ids, _ = self.model.model.get_rope_index(
            input_ids, mm_token_type_ids, image_grid_thw=grid_thw, attention_mask=attention_mask
        )
        _trace_save("position_ids", position_ids)
        context["input_ids"] = input_ids
        context["attention_mask"] = attention_mask
        context["image_mask_1d"] = image_mask_1d
        context["position_ids"] = position_ids
        context["permanent_group_idx"] = (~image_mask_1d).nonzero(as_tuple=True)[0]
        image_token_positions = image_mask_1d.nonzero(as_tuple=True)[0]  # merge-group g -> this[g]
        # Structural precondition for chunked prefill (docs/memo/qwen25vl_baseline_mrope_bug.md's
        # neighbor design work): merge-group native order (spatial_idx) must equal LLM sequence
        # order with no gaps, so a `sequential`-grouping band is a contiguous prefill chunk with no
        # row/token mapping layer needed. Verified on 3 real single-image RefCOCO prompts at 3
        # resolutions -- but it is a claim about PROMPT LAYOUT, not about resolution, and the
        # obvious way it breaks is a prompt with more than one image (two image blocks separated by
        # text): image_token_positions stops being one contiguous run, chunk boundaries stop being
        # contiguous, and a chunked prefill would silently read holes. Assert here rather than only
        # in the one script that checked it once -- an unmet structural assumption in this kind of
        # code should crash, not produce a plausible number from a discontiguous range.
        if image_token_positions.numel() > 0:
            gaps = image_token_positions[1:] - image_token_positions[:-1]
            assert bool((gaps == 1).all()), (
                f"image_token_positions is not one contiguous run ({image_token_positions.numel()} "
                f"tokens, first={int(image_token_positions[0])}, last={int(image_token_positions[-1])}) "
                f"-- chunked prefill assumes a single-image prompt with image tokens forming one "
                f"unbroken block; a multi-image prompt would violate this."
            )
        context["image_token_positions"] = image_token_positions

    def prepare_tokens(self, task: Task, context: Dict[str, Any], config: Any):
        if "pixel_values" not in context:
            return
        vision_ctx = self.vision_tower.prepare_full_tokens(context["pixel_values"], context["image_grid_thw"])
        context["vision_ctx"] = vision_ctx
        context["vision_current_feature"] = vision_ctx["hidden_states"]

        if "input_ids" not in context:
            self._build_prompt(context)
            input_embeds = self.model.model.language_model.embed_tokens(context["input_ids"])
            context["llm_input_embeds_template"] = input_embeds  # text rows exact; image rows placeholder

    def _splice_image_embeds(self, context: Dict[str, Any], image_embeds_merged: torch.Tensor) -> torch.Tensor:
        """Scatters the current (possibly partially-corrected) merged image embeddings into the
        cached text-embedding template at the real image-token positions, matching stock's own
        `masked_scatter` -- always exact given whatever `image_embeds_merged` currently is."""
        template = context["llm_input_embeds_template"].clone()
        image_mask = context["image_mask_1d"].unsqueeze(0).unsqueeze(-1).expand_as(template)
        return template.masked_scatter(image_mask, image_embeds_merged.to(template.dtype))

    def approx_forward(self, params: Dict[str, Any], context: Dict[str, Any], config: Any):
        start_l, end_l = params.get("layers", (0, self.num_llm_layers))

        if start_l == 0:
            # First call of the request: approx the vision tower to FULL depth once (matching
            # OpenVLA's vision towers, always run to completion per round, not chunked), then
            # build the initial multimodal embedding sequence from the (approx-only) image.
            vctx = context["vision_ctx"]
            x_v, cache_v = self.vision_tower.approx_forward(
                context["vision_current_feature"], 0, len(self.vision_tower.blocks), vctx, {}, tag_prefix="vision",
                collect_attn=True,
            )
            context["vision_current_feature"] = x_v
            context["vision_cache"] = cache_v
            attn_layermean = cache_v.get("vision_patch_attn_layermean")
            if attn_layermean is not None:
                # Server-side pscore (received attention, this project's standing residual-energy
                # x attention pattern -- see attention.py's `incoming_attention`). Pooled from raw
                # patch rows to merge-group granularity (mean over each group's `spatial_merge_unit`
                # rows, matching how `correct_forward` addresses merge-groups), then un-permuted
                # from window-permuted order back to ORIGINAL merge-group order (`spatial_idx`) via
                # `inv_window_index` -- the same gather `get_merged_output` uses for the same
                # reason: everything downstream (`pscore_map`, `group_map`) is indexed by the
                # ORIGINAL merge-group index, never the permuted one.
                unit = self.vision_tower.spatial_merge_unit
                permuted_pooled = attn_layermean.reshape(-1, unit).mean(dim=-1)
                context["server_pscore_map"] = permuted_pooled[vctx["inv_window_index"]].unsqueeze(0)
            merged = self.vision_tower.get_merged_output(x_v, vctx)
            context["llm_input_embeds"] = self._splice_image_embeds(context, merged)
            x_feature = context["llm_input_embeds"]
        else:
            x_feature = context["llm_current_feature"]

        # STREAMING skips the LLM approx pass entirely. In that schedule the approx LLM values
        # are semantically dead: every position is prefilled by exactly one chunk, the causal
        # mask means no chunk ever attends an approx-only position (keys beyond a chunk's end are
        # unread; keys before it were written by earlier chunks), and decode reads the final
        # chunk. Measured before removal: keeping it put the streaming arm's total at 200.4% of
        # full vs the streaming category's ~1.25x identity (full + one vision pass) -- an entire
        # dead full-LLM forward per request. The cost of removal is the round-0 first-token
        # preview, which the streaming category does not have anyway (its first token arrives
        # after the last chunk by construction). The caches correct() reads are zero-initialized
        # with the exact shapes approx() would have produced; zeros are correct because (above)
        # nothing ever consumes an unwritten row. APPCORR_STREAMING_KEEP_APPROX=1 restores the
        # old path -- GATE USE ONLY (in-process bitwise A/B of stripped-vs-kept), never an arm.
        raw_appcorr = getattr(config, "appcorr_kwargs", None) or {}
        if (str(raw_appcorr.get("llm_schedule", "interleaved")) == "streaming"
                and not os.environ.get("APPCORR_STREAMING_KEEP_APPROX")):
            B, N, D = x_feature.shape
            cache = context.get("llm_cache", {})
            for i in range(start_l, end_l):
                attn = self.llm_layers[i].self_attn
                cache[f"llm_layer{i}_kv"] = torch.zeros(
                    B, attn.num_key_value_heads, N, 2, attn.head_dim,
                    device=x_feature.device, dtype=x_feature.dtype)
                cache[f"llm_layer{i}_blocks_out_sum"] = torch.zeros(
                    B, N, D, device=x_feature.device, dtype=x_feature.dtype)
            context["llm_current_feature"] = x_feature
            context["llm_cache"] = cache
            return

        cache = context.get("llm_cache", {})
        for i in range(start_l, end_l):
            layer = self.llm_layers[i]
            x_feature, cache = layer.approx(x_feature, context["position_ids"], cache, tag=f"llm_layer{i}")
        context["llm_current_feature"] = x_feature
        context["llm_cache"] = cache

    def _prune_patch_idx(self, group_idx: torch.Tensor, context: Dict[str, Any], config: Any) -> torch.Tensor:
        """Vision-side keep rate: within the merge-groups that just arrived (`group_idx`), keep
        only the top-`token_keep_ratio` fraction by this project's standing importance score
        (residual energy x received attention -- see `attention.py`'s `incoming_attention` and
        `docs/memo/qwen25vl_baseline_mrope_bug.md`'s note on why Qwen originally shipped with
        energy alone: no CLS token was never the actual blocker, the vision fork just had no
        attention-collection point yet) and correct only those; the rest stay at whatever the
        current approx state already has them at (never separately re-corrected -- same
        "static per-round selection" semantics as OpenCLIP's `_prune_patch_idx`,
        `openclip_executor.py:147-211`, this is adapted from, including its
        `combined = server_score * mobile_score` combination).

        Reads `token_keep_ratio` from the RAW `config.appcorr_kwargs`, NOT
        `normalize_appcorr_kwargs`'s output -- that function defaults the ratio to 0.2, so reading
        the normalized value would silently turn every existing Qwen config that never mentions
        this knob into a 20%-keep run. No-op (returns group_idx unchanged) unless the config
        explicitly sets a ratio < 1.0.
        """
        raw_appcorr = getattr(config, "appcorr_kwargs", None) or {}
        token_keep_ratio = (float(raw_appcorr["token_keep_ratio"])
                            if "token_keep_ratio" in raw_appcorr else None)
        if token_keep_ratio is None or token_keep_ratio >= 1.0:
            return group_idx

        pscore_map = context.get("pscore_map")
        if pscore_map is None:
            return group_idx
        mobile_score = pscore_map[0, group_idx].float()
        if bool((mobile_score == 0).all()):
            # No real residual-energy hint for this group (e.g. mobile_pscore not configured on
            # the transmission policy) -- pruning would be meaningless, so skip it rather than
            # keep an arbitrary subset.
            return group_idx

        server_pscore_map = context.get("server_pscore_map")
        if server_pscore_map is not None:
            scores = mobile_score * server_pscore_map[0, group_idx].float()
        else:
            # Round 0's approx pass never ran collect_attn (should not happen given
            # approx_forward always requests it), or this group's attention statistic could not be
            # computed -- fall back to energy alone rather than block pruning entirely.
            scores = mobile_score

        n = int(group_idx.numel())
        k = max(1, min(int(round(n * token_keep_ratio)), n))
        keep_mask = torch.zeros(n, dtype=torch.bool, device=group_idx.device)
        keep_mask.scatter_(0, scores.topk(k).indices, True)
        return group_idx[keep_mask]

    def correct_forward(self, params: Dict[str, Any], context: Dict[str, Any], config: Any):
        start_l, end_l = params.get("layers", (0, self.num_llm_layers))
        group_id = params.get("group_id", 1)
        group_map = context["group_map"]
        group_idx = torch.where(group_map[0] == group_id)[0]
        if group_idx.numel() == 0:
            return
        _n_before_prune = group_idx.numel()
        group_idx_raw = group_idx  # pre-prune positions -- streaming chunk boundaries need these
        group_idx = self._prune_patch_idx(group_idx, context, config)
        if os.environ.get("APPCORR_QWEN_TRACE_DIR"):
            print(f"[TRACE-PRUNE] group_id={group_id} before={_n_before_prune} after={group_idx.numel()}", flush=True)

        # Vision tower: correct the newly-arrived merge-groups to FULL depth (all 32 layers), from
        # the CURRENT canvas (already reconstructed by the transmission layer's decode, matching
        # every prior fork's "restart x from prepare_full_tokens's output each round" invariant).
        vctx = self.vision_tower.prepare_full_tokens(context["pixel_values"], context["image_grid_thw"])
        x_v, vcache = self.vision_tower.correct_forward(
            vctx["hidden_states"], group_idx, 0, len(self.vision_tower.blocks), vctx, context["vision_cache"], tag_prefix="vision"
        )
        context["vision_cache"] = vcache
        merged = self.vision_tower.get_merged_output(x_v, vctx)
        _trace_save("vision_merged", merged)
        if os.environ.get("APPCORR_QWEN_TRACE_DIR"):
            total_merge_groups = group_map.shape[1]
            print(f"[TRACE-VISION] group_idx selected {group_idx.numel()}/{total_merge_groups} merge-groups "
                  f"(group_id={group_id})", flush=True)
        context["llm_input_embeds"] = self._splice_image_embeds(context, merged)
        _trace_save("inputs_embeds", context["llm_input_embeds"])

        # LLM: correct THIS round's image tokens plus text, with text split BY POSITION relative
        # to the (contiguous, asserted in `_build_prompt`) image block:
        #
        #   * PRE-image text (system preamble): NEVER corrected. On a causal LLM these rows read
        #     only rows before themselves -- none of which is an image row -- so the approx pass
        #     (a plain causal forward over all N rows, `decoder_layer.approx`) already computed
        #     them exactly; re-correcting them returns the same values. An earlier version
        #     corrected them every round on the mistaken premise that their K/V had been computed
        #     "against the degraded image state" -- pure waste, and for small images (MMVP 224px:
        #     64 image tokens, N=116) it was 60 of the 113 correction queries, pushing total
        #     compute to 195% of the ceiling. Gated before removal in fp32 on the 7B model (12
        #     runs, 3 datasets x k in {1.0, 0.25}, analysis/experiments/qwen25vl_text_correct_gate.py):
        #     final hidden state within arithmetic noise (max|d| <= 5.7e-2 of ~1e3), first-token
        #     logits max|d| <= 7.2e-5, argmax and top-5 identical everywhere.
        #   * POST-image text (question + generation prompt): corrected on the FINAL round ONLY.
        #     Nothing sits after them in the sequence, so no consumer reads their intermediate
        #     corrections before decode -- and every round restarts from the fresh
        #     `llm_input_embeds` with only the final round's state feeding `decode_first_token`,
        #     so intermediate-round corrections of these rows were pure waste.
        #
        # Measured before this split (FLOPs audit, refcoco n=6): re-correcting the full text every
        # round was 0.56x of a whole ceiling forward -- 56% of the correct-stage LLM cost for
        # 14.5% of the rows -- entirely keep-independent, and the whole reason Qwen's total sat at
        # 177-199% of full while every other fixed model converged to 114-143%. This is the same
        # class as Gemma3's text-last-round schedule, made positionally safer for a causal LLM
        # (Gemma3's vision tokens are bidirectional; here only the pre-image half is ever read by
        # image rows, so only that half needs freshening).
        image_token_positions = context["image_token_positions"][group_idx.to(context["image_token_positions"].device)]
        all_img = context["image_token_positions"]
        perm = context["permanent_group_idx"]
        num_groups = max(int(getattr(config, "transmission_kwargs", {}).get("num_groups", 1)), 1)
        is_final_round = group_id >= num_groups

        # LLM schedule. Read from the RAW appcorr_kwargs (the `normalize_appcorr_kwargs` trap:
        # never let a normalizer default silently pick an arm).
        raw_appcorr = getattr(config, "appcorr_kwargs", None) or {}
        llm_schedule = str(raw_appcorr.get("llm_schedule", "interleaved"))

        if llm_schedule == "streaming":
            # STREAMING (exact chunked prefill -- the causal-LLM arm the OV2/Qwen3.5 rows use):
            # each round prefill the CONTIGUOUS band [frontier, end-of-this-round's-image-band)
            # exactly once, over the vision state as of this round; earlier chunks' K/V stay
            # locked (never revisited), later positions' stale round-0 keys are unread thanks to
            # the causal mask. Round 1 folds the leading text (frontier starts at 0); the final
            # round extends through the trailing text to N. Correctness of this exact pattern was
            # gated in fp32 chunked-vs-reference at 4.77e-05 max-abs logit diff (arithmetic-noise
            # band) before it was promoted here from the diagnostic scripts.
            #
            # Chunk boundaries come from the group's RAW (pre-prune) positions -- the vision-side
            # keep rate selects which VISION tokens get recomputed, but the LLM prefills every
            # arrived position exactly once over whatever embedding the vision state currently
            # holds; a selection-shaped LLM chunk would leave unprefilled holes that a later
            # chunk's causal attention WOULD read.
            if group_idx_raw.numel() > 0:
                band = context["image_token_positions"][group_idx_raw.to(context["image_token_positions"].device)]
                gaps = band[1:] - band[:-1]
                assert bool((gaps == 1).all()), (
                    "streaming llm_schedule requires contiguous per-round image bands -- use "
                    "grouping_strategy='sequential' (grid/top_energy bands interleave positions)."
                )
                band_end = int(band.max()) + 1
            else:
                band_end = int(context.get("stream_frontier", 0))
            N_seq = context["position_ids"].shape[-1]
            start = int(context.get("stream_frontier", 0))
            end = N_seq if is_final_round else band_end
            context["stream_frontier"] = end
            token_idx = torch.arange(start, end, device=all_img.device)
            if token_idx.numel() == 0:
                return
        else:
            if all_img.numel() == 0:
                text_idx = perm
            elif is_final_round:
                text_idx = perm[perm > all_img[-1]]
            else:
                text_idx = perm[:0]
            token_idx = torch.cat([text_idx, image_token_positions]).unique()

        x_feature = context["llm_input_embeds"]

        # RoPE hoist: compute the mrope-interleaved cos_sel/sin_sel ONCE per correction round
        # instead of once per decoder layer (up to 64x redundant otherwise) -- valid because
        # rotary_emb and mrope_section are shared across every layer (see
        # `llm/decoder_layer.py`'s module docstring). `Qwen2_5_VLRotaryEmbedding.forward` only
        # reads `x.dtype`/`x.device` from its first argument, so `x_feature` (not yet
        # per-head-projected) is a valid stand-in for the per-layer `v_new` tensor it was
        # previously called with.
        first_attn = self.llm_layers[0].self_attn
        position_ids_sel = context["position_ids"][:, :, token_idx]
        cos_full, sin_full = first_attn.rotary_emb(x_feature, position_ids_sel)
        mrope_section2 = first_attn.mrope_section * 2
        cos_sel = torch.cat([m[i % 3] for i, m in enumerate(cos_full.split(mrope_section2, dim=-1))], dim=-1).unsqueeze(1)
        sin_sel = torch.cat([m[i % 3] for i, m in enumerate(sin_full.split(mrope_section2, dim=-1))], dim=-1).unsqueeze(1)

        # Causal-mask hoist: `is_full_causal`/`attn_mask` are pure functions of `token_idx` and the
        # total sequence length N (invariant across every layer in this round), so decide/build them
        # once here instead of once per decoder layer -- avoids a `bool(torch.equal(...))` GPU->CPU
        # sync and an `[Q,N]` mask allocation up to 64x per round (see decoder_layer.py's docstring
        # for why the is_causal=True fast path matters for kernel-path parity with stock).
        N = context["position_ids"].shape[-1]
        Q = token_idx.shape[0]
        is_full_causal = Q == N and bool(torch.equal(token_idx, torch.arange(N, device=token_idx.device)))
        attn_mask = None
        if not is_full_causal:
            key_positions = torch.arange(N, device=token_idx.device).view(1, N)
            allowed = key_positions <= token_idx.view(Q, 1)
            attn_mask = torch.zeros((Q, N), device=token_idx.device, dtype=x_feature.dtype)
            attn_mask.masked_fill_(~allowed, torch.finfo(x_feature.dtype).min)
            attn_mask = attn_mask.view(1, 1, Q, N)

        cache = context.get("llm_cache", {})
        for i in range(start_l, end_l):
            layer = self.llm_layers[i]
            x_feature, cache = layer.correct(
                x_feature, token_idx, cache, tag=f"llm_layer{i}", position_ids=context["position_ids"],
                cos_sel=cos_sel, sin_sel=sin_sel, is_full_causal=is_full_causal, attn_mask=attn_mask,
            )
            _trace_save(f"layer{i}_out", x_feature)
        context["llm_current_feature"] = x_feature
        context["llm_cache"] = cache

    def decode_first_token(self, x_full: torch.Tensor) -> torch.Tensor:
        """norm -> lm_head -> argmax on the final sequence position of a (possibly partially
        corrected) LLM hidden state -- the single first-generated-token decode used by both
        `head_inference` below and `analysis/experiments/refcoco_gqa_batched_eval.py`'s
        `build_first_token_context`. Pulled into one place after an earlier bug where the two
        copies silently diverged (one fed a raw canvas, the other a reconstructed one) went
        undetected because the duplicated logic itself looked identical."""
        hidden = self.model.model.language_model.norm(x_full)
        logits_last = self.model.lm_head(hidden[:, -1, :].to(self.model.lm_head.weight.dtype))
        return logits_last.argmax(dim=-1)

    def head_inference(self, task: Task, context: Dict[str, Any], config: Any) -> Dict[str, Any]:
        """The FIRST generated token is decoded from the actual corrected/approx prefill state
        (`context["llm_current_feature"]`'s final position) -- this is the token whose correctness
        this whole mechanism is meant to validate (RealWorldQA answers are frequently a single
        letter/word, so this token alone often IS the whole answer). If more tokens are needed,
        remaining generation falls back to a stock `model.generate()` call on
        `[input_ids, first_token]` -- this recomputes the full sequence via stock forward for the
        (typically short) continuation, a deliberate, documented simplification: building a real
        incremental KV-cache-append decode loop on top of the hand-rolled correction cache was out
        of scope for validating the core approx/correct mechanism, and RealWorldQA answers are
        short enough (1 letter to a few words) that this fallback's cost is bounded and does not
        change what the FIRST token (already fixed before the fallback runs) was."""
        prefill_hidden = context.get("llm_current_feature")
        _trace_save("prefill_hidden", prefill_hidden)
        first_token = self.decode_first_token(prefill_hidden)
        with torch.no_grad():
            _hidden = self.model.model.language_model.norm(prefill_hidden)
            _logits_last = self.model.lm_head(_hidden[:, -1, :].to(self.model.lm_head.weight.dtype))
        _trace_save("first_token_logits", _logits_last)
        _trace_save("first_token_id", first_token)

        if first_token.item() == self.processor.tokenizer.eos_token_id:
            answer_text = ""
        else:
            with torch.no_grad():
                extended_ids = torch.cat([context["input_ids"], first_token.unsqueeze(0)], dim=1)
                extended_mask = torch.cat(
                    [context["attention_mask"], torch.ones_like(first_token.unsqueeze(0))], dim=1
                )
                mm_token_type_ids = context["image_mask_1d"].long().unsqueeze(0)
                extended_mm_token_type_ids = torch.cat(
                    [mm_token_type_ids, torch.zeros_like(first_token.unsqueeze(0))], dim=1
                )
                gen_ids = self.model.generate(
                    input_ids=extended_ids,
                    attention_mask=extended_mask,
                    pixel_values=context["pixel_values"],
                    image_grid_thw=context["image_grid_thw"],
                    mm_token_type_ids=extended_mm_token_type_ids,
                    max_new_tokens=63,
                    do_sample=False,
                )
            gen_trimmed = gen_ids[:, context["input_ids"].shape[1]:]
            answer_text = self.processor.tokenizer.decode(gen_trimmed[0], skip_special_tokens=True)

        context["output"] = answer_text
        return {"answer_text": answer_text}

    def full_inference(self, task: Task, context: Dict[str, Any], config: Any):
        """Mechanism-matched baseline: mirrors `head_inference`'s two-stage decode (argmax the
        first token from a forward pass's final-position logits, then fall back to a SEPARATE
        stock `model.generate()` call for the rest) instead of one continuous `generate()` call.
        This used to differ from every corrected condition's generation mechanism -- confirmed via
        `analysis/experiments/refcoco_matched_decode_diagnostic.py` (nr=400, 100% stock computation
        both ways) that the two mechanisms produce different generated text on 30-35% of RefCOCO
        samples even with zero correction involved (32B: +2.25pp, 65% text agreement; 72B: -1.00pp,
        70% agreement) -- a real, un-negligible confound, not just noise. Baseline now uses the
        IDENTICAL mechanism as every keep_rate/corrected condition, so gaps reported against this
        baseline isolate correction quality rather than mixing it with this mechanism artifact."""
        if "pixel_values" not in context:
            return
        if "input_ids" not in context:
            self._build_prompt(context)

        # `mm_token_type_ids` (NOT `position_ids` directly) is the fix: without it,
        # `compute_3d_position_ids` cannot compute real M-RoPE and falls through to plain
        # sequential 1D positions replicated across all 3 mrope axes, discarding every image
        # token's real (temporal, height, width) grid position. Passing `mm_token_type_ids` lets
        # the model derive positions itself via its own `get_rope_index` call, so this stays a
        # genuine independent stock reference rather than sharing a tensor with the fork by
        # construction -- see the assertion below, which proves the two derivations agree rather
        # than assuming it. Found 2026-08-25: this is why `full_inference` disagreed with a
        # bit-exact-correct g=1 fork correction -- see docs/memo/qwen25vl_baseline_mrope_bug.md.
        mm_token_type_ids = context["image_mask_1d"].long().unsqueeze(0)

        if os.environ.get("APPCORR_QWEN_TRACE_DIR"):
            # Side computation for tracing only -- does not feed into `outputs` below, matches
            # exactly how the LLM-fork unit test's `build_inputs_embeds` derives the stock image
            # embedding (`model.model.visual(...).pooler_output`), so it is comparable to the
            # fork's post-merger `merged` tensor at the same stage (post-merge, pre-splice).
            with torch.no_grad():
                _stock_vision_out = self.model.model.visual(
                    context["pixel_values"], grid_thw=context["image_grid_thw"]
                )
            _trace_save("vision_merged", _stock_vision_out.pooler_output)
            # Same masked_scatter the stock forward does internally, replicated here purely to
            # capture the post-splice/pre-layer-0 `inputs_embeds` -- comparable to the fork's
            # `context["llm_input_embeds"]` at the identical stage.
            _embed_tokens = self.model.model.language_model.embed_tokens(context["input_ids"])
            _image_mask = context["image_mask_1d"].unsqueeze(0).unsqueeze(-1).expand_as(_embed_tokens)
            _inputs_embeds = _embed_tokens.masked_scatter(_image_mask, _stock_vision_out.pooler_output.to(_embed_tokens.dtype))
            _trace_save("inputs_embeds", _inputs_embeds)

        _layer_hooks = []
        _orig_apply_rope = None
        if os.environ.get("APPCORR_QWEN_TRACE_DIR"):
            def _make_hook(idx):
                def _hook(module, inp, out):
                    hs = out[0] if isinstance(out, tuple) else out
                    _trace_save(f"layer{idx}_out", hs)
                return _hook
            layer0 = self.model.model.language_model.layers[0]
            _layer_hooks.append(layer0.register_forward_hook(_make_hook(0)))
            _layer_hooks.append(layer0.input_layernorm.register_forward_hook(
                lambda m, i, o: _trace_save("layer0_input_layernorm_out", o)))
            _layer_hooks.append(layer0.self_attn.register_forward_hook(
                lambda m, i, o: _trace_save("layer0_attn_out", o[0] if isinstance(o, tuple) else o)))
            _layer_hooks.append(layer0.mlp.register_forward_hook(
                lambda m, i, o: _trace_save("layer0_mlp_out", o)))

            def _v_proj_hook(module, inp, out):
                B, T, _ = out.shape
                h_kv = layer0.self_attn.num_key_value_heads
                head_dim = layer0.self_attn.head_dim
                _trace_save("layer0_v", out.view(B, T, h_kv, head_dim).transpose(1, 2))
            _layer_hooks.append(layer0.self_attn.v_proj.register_forward_hook(_v_proj_hook))

            # apply_multimodal_rotary_pos_emb is a plain function call inside Qwen2_5_VLAttention
            # .forward, not a module -- monkeypatch it for the duration of this one forward call to
            # capture layer 0's post-RoPE q/k (layers run strictly in order, so the first call is
            # always layer 0). Restored unconditionally after, even on exception.
            import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl as _qwen_mod
            _orig_apply_rope = _qwen_mod.apply_multimodal_rotary_pos_emb
            _rope_call_count = [0]

            def _traced_apply_rope(q, k, cos, sin, mrope_section, unsqueeze_dim=1):
                q_out, k_out = _orig_apply_rope(q, k, cos, sin, mrope_section, unsqueeze_dim=unsqueeze_dim)
                if _rope_call_count[0] == 0:
                    _trace_save("layer0_q_postrope", q_out)
                    _trace_save("layer0_k_postrope", k_out)
                _rope_call_count[0] += 1
                return q_out, k_out
            _qwen_mod.apply_multimodal_rotary_pos_emb = _traced_apply_rope

        # Independence assertion (unconditional -- one [3,1,N] int-tensor compare, negligible
        # cost): confirm HF's own internally-derived M-RoPE position_ids (now that
        # `mm_token_type_ids` reaches the model) match the fork's `context["position_ids"]` from
        # `_build_prompt`'s explicit `get_rope_index` call. This is what makes the fix a genuine
        # independent reference rather than "the baseline agrees with the fork because we handed
        # it the fork's own tensor" -- if this ever fails, the fork and the model's own derivation
        # disagree about positions for a NEW reason, and nothing downstream should be trusted
        # until that is understood.
        _orig_compute_3d = self.model.model.compute_3d_position_ids
        _captured_position_ids = [None]
        def _capturing_compute_3d(*a, **kw):
            pos = _orig_compute_3d(*a, **kw)
            _captured_position_ids[0] = pos
            return pos
        self.model.model.compute_3d_position_ids = _capturing_compute_3d

        with torch.no_grad():
            outputs = self.model(
                input_ids=context["input_ids"],
                attention_mask=context["attention_mask"],
                pixel_values=context["pixel_values"],
                image_grid_thw=context["image_grid_thw"],
                mm_token_type_ids=mm_token_type_ids,
                use_cache=False,
                output_hidden_states=bool(os.environ.get("APPCORR_QWEN_TRACE_DIR")),
            )
            for _h in _layer_hooks:
                _h.remove()
            if _orig_apply_rope is not None:
                import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl as _qwen_mod
                _qwen_mod.apply_multimodal_rotary_pos_emb = _orig_apply_rope
            self.model.model.compute_3d_position_ids = _orig_compute_3d

            _actual_pos = _captured_position_ids[0]
            if _actual_pos is None or not torch.equal(_actual_pos, context["position_ids"]):
                raise AssertionError(
                    "full_inference: HF's internally-derived position_ids (via mm_token_type_ids) "
                    "no longer match context['position_ids'] (the fork's get_rope_index call) -- "
                    "the two derivations have diverged for a reason beyond the known "
                    "can_compute_mrope fall-through. Do not trust any number past this point."
                )
            if not Qwen25VLExecutor._position_ids_verified:
                Qwen25VLExecutor._position_ids_verified = True
                print("[full_inference] verified: HF-internal position_ids == fork's "
                      "context['position_ids']", flush=True)

            if os.environ.get("APPCORR_QWEN_TRACE_DIR"):
                # hidden_states[-1] is the last decoder layer's raw output, pre-final-norm --
                # the same point `context["llm_current_feature"]` captures on the fork side.
                _trace_save("prefill_hidden", outputs.hidden_states[-1])
            logits_last = outputs.logits[:, -1, :]
            first_token = logits_last.argmax(dim=-1)
            _trace_save("first_token_logits", logits_last)
            _trace_save("first_token_id", first_token)

            if first_token.item() == self.processor.tokenizer.eos_token_id:
                answer_text = ""
            else:
                extended_ids = torch.cat([context["input_ids"], first_token.unsqueeze(0)], dim=1)
                extended_mask = torch.cat(
                    [context["attention_mask"], torch.ones_like(first_token.unsqueeze(0))], dim=1
                )
                extended_mm_token_type_ids = torch.cat(
                    [mm_token_type_ids, torch.zeros_like(first_token.unsqueeze(0))], dim=1
                )
                gen_ids = self.model.generate(
                    input_ids=extended_ids,
                    attention_mask=extended_mask,
                    pixel_values=context["pixel_values"],
                    image_grid_thw=context["image_grid_thw"],
                    mm_token_type_ids=extended_mm_token_type_ids,
                    max_new_tokens=63,
                    do_sample=False,
                )
                gen_trimmed = gen_ids[:, context["input_ids"].shape[1]:]
                answer_text = self.processor.tokenizer.decode(gen_trimmed[0], skip_special_tokens=True)
        context["output"] = answer_text

    def get_final_results(self, task: Task, context: Dict[str, Any], config: Any) -> Dict[int, Any]:
        if "output" not in context:
            return {}
        return {0: context["output"]}

    def decide_exit(self, task: Task, context: Dict[str, Any], config: Any) -> Dict[str, Any]:
        return {}
