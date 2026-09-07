"""Gate/measure: DINOv3-style NEW-ONLY vision correction (each round recomputes only its own
group's patches in both towers) vs the campaign's CUMULATIVE arm (every arrived patch, every
round), both under the chunked causal LLM prefill, on real LIBERO frames.

Three arms per frame, identical canvases by construction (openvla_chunked_gate.py's emulation:
8x-blur base, 64 real 14x14 cells pasted per raster group):

  cumulative       towers corrected with all arrived patches   (reference: last round == a
                   single-shot full correction, bit-exact to a stock tower forward)
  new_only         towers corrected with this round's 64 only; block.correct persists the
                   corrected increment (interleaved-contract rule 3, DINOv3 ac0238f)
  new_only_nowb    same schedule with the pre-2026-09-07 block.correct (no write-back) -- the
                   condition under which the July `1eaded2` try/revert measured mean 0.023 /
                   max 1.25 feature error; reproduced here so the bug's share of that number is
                   separable from the bidirectional-tower staleness that remains.

Reported per frame and worst-case: projected vision embedding error (the July metric, on
`pm._x0`'s 256 vision rows), top-layer LLM state error on the vision and text rows, last-position
logits, decoded action bins, and the stock forward's last-position logits as an outside anchor.

Run (GPU0, openvla env):
  CUDA_VISIBLE_DEVICES=0 <openvla-env>/python analysis/experiments/openvla_newonly_gate.py
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from appcorr.models.openvla.progressive_model import OpenVLAProgressiveModel  # noqa: E402
from appcorr.models.openvla.vision.block import ApproxCorrectBlock  # noqa: E402
from analysis.experiments.openvla_chunked_gate import CKPT, GRID, TASK, canvases, frames_from_demo  # noqa: E402


def _correct_no_writeback(self, x, token_idx, cache_feature, tag):
    """`ApproxCorrectBlock.correct` as of 4872003 (before the persist write) -- probe only."""
    token_idx = token_idx.to(x.device)
    x_active = x[:, token_idx]
    x_norm_sel = self.norm1(x_active)
    x_attn_sel, cache_feature = self.attn.correct(x_norm_sel, token_idx, cache_feature, tag)
    x_attn_active = x_active + self.ls1(x_attn_sel)
    mlp_out_new = self.ls2(self.mlp(self.norm2(x_attn_active)))
    x_out = (x + cache_feature[f"{tag}_blocks_out_sum"].to(dtype=x.dtype)).clone()
    x_out[:, token_idx] = (x_attn_active + mlp_out_new).to(dtype=x_out.dtype)
    return x_out, cache_feature


_CORRECT_WB = ApproxCorrectBlock.correct


@torch.no_grad()
def run(pm, arm: str, cvs, groups, text):
    pm.vision_correction = "cumulative" if arm == "cumulative" else "new_only"
    ApproxCorrectBlock.correct = _correct_no_writeback if arm == "new_only_nowb" else _CORRECT_WB
    try:
        pm.start_session_from_text(text)
        pm.vision_approx(cvs[0])
        pm.llm_chunked_init()
        arrived = []
        for g, new in enumerate(groups, start=1):
            arrived += new
            all_idx = torch.tensor(arrived, dtype=torch.long, device=pm.device)
            new_idx = torch.tensor(new, dtype=torch.long, device=pm.device)
            tok = pm.vision_correct(cvs[g], all_idx, new_idx)
            pm.llm_prefill_segment(tok, include_text=(g == len(groups)))
        x0 = pm._x0[:, 1:1 + GRID * GRID].float().clone()
        x = pm.cache_feature["_x"].float().clone()
        logits = pm._logits_from_x(pm.cache_feature["_x"])[:, -1].float()
        actions, st = pm.decode_action(return_stats=True)
        return x0, x, logits, np.asarray(actions), st["bins"]
    finally:
        ApproxCorrectBlock.correct = _CORRECT_WB
        pm.vision_correction = "cumulative"


@torch.no_grad()
def stock(pm, px, text):
    """Stock forward on the final (fully arrived) canvas: last-position logits and the greedy
    7-token action bins (what `predict_action` does), the outside anchor for every arm."""
    pm.start_session_from_text(text)
    out = pm.vla(input_ids=pm.input_ids, pixel_values=px, use_cache=False, return_dict=True)
    n = pm.vla.get_action_dim(pm.unnorm_key)
    gen = pm.vla.generate(input_ids=pm.input_ids, pixel_values=px, max_new_tokens=n, do_sample=False)
    tok = gen[0, -n:].cpu().numpy()
    bins = np.clip(pm.vla.vocab_size - tok - 1, 0, pm.vla.bin_centers.shape[0] - 1)
    return out.logits[:, -1].float(), bins.tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--groups", type=int, default=4)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    pm = OpenVLAProgressiveModel(CKPT, torch.device(a.device))
    bounds = np.linspace(0, GRID * GRID, a.groups + 1).round().astype(int)
    groups = [list(range(bounds[g], bounds[g + 1])) for g in range(a.groups)]
    arms = ["new_only", "new_only_nowb"]
    worst = {k: {"x0_max": 0.0, "x0_mean": 0.0, "xv_max": 0.0, "xt_max": 0.0, "logit": 0.0,
                 "bin_mismatch": 0, "bins_off": 0, "stock_logit": 0.0, "stock_bins_off": 0}
             for k in ["cumulative"] + arms}
    for fi, fr in enumerate(frames_from_demo(a.frames)):
        cvs = canvases(pm, fr, groups)
        ref = run(pm, "cumulative", cvs, groups, TASK)
        lg_stock, bins_stock = stock(pm, cvs[-1], TASK)
        n_text = int(pm.input_ids.shape[1]) - 1
        d_stock = float((ref[2] - lg_stock).abs().max())
        off_stock = int(np.abs(np.asarray(ref[4]) - np.asarray(bins_stock)).sum())
        worst["cumulative"]["stock_logit"] = max(worst["cumulative"]["stock_logit"], d_stock)
        worst["cumulative"]["stock_bins_off"] += off_stock
        print(f"[frame {fi}] stock bins={bins_stock}; cumulative vs stock: last-pos logits max|d|={d_stock:.3e} "
              f"(max|logit|={float(lg_stock.abs().max()):.1f}) bins {ref[4]} sum|dbin|={off_stock}", flush=True)
        for arm in arms:
            x0, x, lg, act, bins = run(pm, arm, cvs, groups, TASK)
            d0 = (x0 - ref[0]).abs()
            dx = (x - ref[1]).abs()
            w = worst[arm]
            m = {"x0_max": float(d0.max()), "x0_mean": float(d0.mean()),
                 "xv_max": float(dx[:, 1:1 + GRID * GRID].max()), "xt_max": float(dx[:, -n_text:].max()),
                 "logit": float((lg - ref[2]).abs().max())}
            for k, v in m.items():
                w[k] = max(w[k], v)
            off = int(np.abs(np.asarray(bins) - np.asarray(ref[4])).sum())
            w["bin_mismatch"] += 0 if bins == ref[4] else 1
            w["bins_off"] += off
            w["stock_logit"] = max(w["stock_logit"], float((lg - lg_stock).abs().max()))
            w["stock_bins_off"] += int(np.abs(np.asarray(bins) - np.asarray(bins_stock)).sum())
            print(f"    {arm:14s} vision-embed |d| mean={m['x0_mean']:.3e} max={m['x0_max']:.3e} "
                  f"(ref |x0| mean={float(ref[0].abs().mean()):.3e}) | top-layer max|d| vision={m['xv_max']:.3e} "
                  f"text={m['xt_max']:.3e} | logits max|d|={m['logit']:.3e} | bins "
                  f"{'same' if off == 0 else f'DIFF sum|dbin|={off}'} {bins} | max|d action|={np.abs(act - ref[3]).max():.3e}",
                  flush=True)
    w = worst["cumulative"]
    print(f"GATE: {a.frames} frames; cumulative vs stock: worst logit max|d|={w['stock_logit']:.3e} "
          f"sum|dbin|={w['stock_bins_off']}")
    for arm in arms:
        w = worst[arm]
        print(f"  {arm:14s} vs cumulative worst: vision-embed mean={w['x0_mean']:.3e} max={w['x0_max']:.3e}  "
              f"top-layer vision={w['xv_max']:.3e} text={w['xt_max']:.3e}  logits={w['logit']:.3e}  "
              f"bin-mismatch frames={w['bin_mismatch']}/{a.frames} sum|dbin|={w['bins_off']}  ||  "
              f"vs stock: logits={w['stock_logit']:.3e} sum|dbin|={w['stock_bins_off']}", flush=True)


if __name__ == "__main__":
    main()
