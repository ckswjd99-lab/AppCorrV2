"""Gate: `chunked` (true chunked causal prefill) vs the campaign's `interleaved` schedule
(frontiers 32x4, sequential grouping, k=1.00) must agree on the final LLM state and the decoded
action. Both schedules share the vision path exactly (vision_approx on the base canvas, cumulative
vision_correct per group); only the LLM work differs (approx pass + per-group correct incl. text
vs one prefill per position), and `progressive_model.py` claims the two are equivalent for a
causal decoder with raster grouping. This checks that claim on real LIBERO frames.

Canvas emulation of the transmission pipeline: base = 8x box-downsampled/upsampled frame (the
blur base layer); each residual group pastes its 64 real 14x14 patch cells (raster order) --
identical inputs to both schedules by construction.

Run (GPU0, openvla env):
  CUDA_VISIBLE_DEVICES=0 <openvla-env>/python analysis/experiments/openvla_chunked_gate.py
"""
from __future__ import annotations

import argparse
import os
import sys

import h5py
import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from appcorr.models.openvla.progressive_model import OpenVLAProgressiveModel  # noqa: E402

DEMO = ("/NHNHOME/share/cjpark/openvla_deps/LIBERO/datasets/libero_spatial/"
        "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo.hdf5")
TASK = "pick up the black bowl between the plate and the ramekin and place it on the plate"
CKPT = "openvla/openvla-7b-finetuned-libero-spatial"
IMG, PATCH, GRID = 224, 14, 16


def frames_from_demo(n: int):
    with h5py.File(DEMO, "r") as f:
        demos = sorted(f["data"].keys())[:n]
        out = []
        for d in demos:
            v = f["data"][d]["obs"]["agentview_rgb"]
            fr = v[len(v) // 2]  # mid-episode frame; LIBERO stores frames flipped vertically
            out.append(np.ascontiguousarray(fr[::-1]))
    return out


def to_px(pm, frame_u8: np.ndarray) -> torch.Tensor:
    t = torch.from_numpy(frame_u8).to(pm.device).permute(2, 0, 1)[None].float()
    t = F.interpolate(t, size=(IMG, IMG), mode="bilinear", align_corners=False) / 255.0
    ip = pm.processor.image_processor
    towers = []
    for i in range(len(ip.means)):
        mean = torch.tensor(ip.means[i]).view(1, 3, 1, 1).to(pm.device)
        std = torch.tensor(ip.stds[i]).view(1, 3, 1, 1).to(pm.device)
        towers.append((t - mean) / std)
    return torch.cat(towers, dim=1).to(torch.bfloat16)


def canvases(pm, frame_u8: np.ndarray, groups):
    """[base canvas, canvas after group 1, ..., after group G] as pixel tensors."""
    full = torch.from_numpy(frame_u8).permute(2, 0, 1)[None].float()
    full = F.interpolate(full, size=(IMG, IMG), mode="bilinear", align_corners=False)
    base = F.interpolate(F.avg_pool2d(full, 8), size=(IMG, IMG), mode="bilinear", align_corners=False)
    cur = base.clone()
    out = [to_px(pm, cur[0].permute(1, 2, 0).round().clamp(0, 255).byte().numpy())]
    for g in groups:
        for i in g:
            r, c = divmod(i, GRID)
            cur[:, :, r * PATCH:(r + 1) * PATCH, c * PATCH:(c + 1) * PATCH] = \
                full[:, :, r * PATCH:(r + 1) * PATCH, c * PATCH:(c + 1) * PATCH]
        out.append(to_px(pm, cur[0].permute(1, 2, 0).round().clamp(0, 255).byte().numpy()))
    return out


@torch.no_grad()
def run(pm, sched: str, cvs, groups, text):
    pm.start_session_from_text(text)
    pm.vision_approx(cvs[0])
    if sched == "chunked":
        pm.llm_chunked_init()
    else:
        pm.llm_approx_segment(0, len(pm.llm_layers))
    arrived = []
    for g, new in enumerate(groups, start=1):
        arrived += new
        all_idx = torch.tensor(arrived, dtype=torch.long, device=pm.device)
        new_idx = torch.tensor(new, dtype=torch.long, device=pm.device)
        tok = pm.vision_correct(cvs[g], all_idx, new_idx)
        if sched == "chunked":
            pm.llm_prefill_segment(tok, include_text=(g == len(groups)))
        else:
            pm.llm_correct_segment(len(pm.llm_layers), tok)
    x = pm.cache_feature["_x"].clone()
    logits = pm._logits_from_x(x)[:, -1].float()
    actions, st = pm.decode_action(return_stats=True)
    return x.float(), logits, np.asarray(actions), st["bins"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--groups", type=int, default=4)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    pm = OpenVLAProgressiveModel(CKPT, torch.device(a.device))
    bounds = np.linspace(0, GRID * GRID, a.groups + 1).round().astype(int)
    groups = [list(range(bounds[g], bounds[g + 1])) for g in range(a.groups)]
    n_text = None
    worst_logit = worst_x_text = 0.0
    bin_mismatch = 0
    for fi, fr in enumerate(frames_from_demo(a.frames)):
        cvs = canvases(pm, fr, groups)
        x_i, lg_i, act_i, bins_i = run(pm, "interleaved", cvs, groups, TASK)
        x_c, lg_c, act_c, bins_c = run(pm, "chunked", cvs, groups, TASK)
        n_text = int(pm.input_ids.shape[1]) - 1
        dx = (x_i - x_c).abs()
        d_text = float(dx[:, -n_text:].max())
        d_vis = float(dx[:, 1:1 + GRID * GRID].max())
        dl = float((lg_i - lg_c).abs().max())
        same = bins_i == bins_c
        worst_logit = max(worst_logit, dl); worst_x_text = max(worst_x_text, d_text)
        bin_mismatch += 0 if same else 1
        print(f"[frame {fi}] top-layer state max|d|: vision={d_vis:.3e} text={d_text:.3e}  "
              f"last-pos logits max|d|={dl:.3e}  action bins {'same' if same else 'DIFF'} "
              f"{bins_i} vs {bins_c}  max|d action|={np.abs(act_i - act_c).max():.3e}", flush=True)
    print(f"GATE: {a.frames} frames, worst logit max|d|={worst_logit:.3e}, worst text-state max|d|="
          f"{worst_x_text:.3e}, action-bin mismatches={bin_mismatch}", flush=True)


if __name__ == "__main__":
    main()
