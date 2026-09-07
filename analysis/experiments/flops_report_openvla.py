"""Backbone FLOPs for the OpenVLA LIBERO arms: full (ceiling), approx (floor), chunked stream,
and the campaign's interleaved schedule.

Mirrors AppCorr-qwen35-eval's flops_report_*.py reporters (same `appcorr.flops` session API,
ported verbatim into this worktree) so the numbers drop into `inprocess_flops.json` under
`["openvla"][<suite>]` with the usual keys: `full`, `k1.00` (critical), `total_k1.00`.

Arms reproduce exactly what `offload/server/model/openvla_vla.py` + `vla_interleaved_static.py`
run in the LIBERO campaign (`--frontiers 32,32,32,32 --grouping sequential`, num_groups=4):

  full         predict_action's prefill: one stock forward over [BOS, 256 vision, text]
  approx       vision_approx (both towers on the upsampled base canvas) + llm_approx (0, 32)
  interleaved  approx, then per residual group g=1..4:
                 vision_correct(all arrived patches, this group's 64 new)   -- cumulative towers
                 llm_correct_segment(32, new_idx + 1)                        -- 32 layers, 64+text q
  chunked      vision_approx + llm_chunked_init (BOS only), then per group g=1..4:
                 vision_correct(...) as above; llm_prefill_segment(new_idx + 1, include_text=g==4)
               -- the exact chunked causal prefill (`schedule=chunked` in vla_interleaved_static.py):
               every LLM position is computed exactly once, no LLM approx pass, text once.
               Gated equivalent to `interleaved` on the final state / decoded action
               (analysis/experiments/openvla_chunked_gate.py), so the accuracy numbers carry over
               and this is the stream arm's compute (`k1.00`/`total_k1.00`); the interleaved
               schedule's cost is kept under `interleaved_k1.00`/`interleaved_total_k1.00`.

Backbone = both vision towers + projector + the 32 Llama layers. `lm_head` and `decode_action`'s
7-token generation are outside (decode excluded, as everywhere else). FLOPs are shape-determined
here -- 224x224 canvas, 256 patches, a fixed instruction, no server pscore filter in the campaign
-- so a synthetic canvas gives the same count as a LIBERO frame; a few instructions of different
lengths are averaged so the text length matches the suite's real prompts. No MuJoCo needed.

Run (GPU0, `openvla` env):
  CUDA_VISIBLE_DEVICES=0 /NHNHOME/storage/users/cjpark/shk/conda_envs/openvla/bin/python \
      analysis/experiments/flops_report_openvla.py --out-json analysis/results/flops/openvla_libero_flops.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from appcorr import flops  # noqa: E402
from appcorr.models.openvla.progressive_model import OpenVLAProgressiveModel  # noqa: E402

# Real task strings from each suite's bddl files (LIBERO), so the text-token count is the campaign's.
SUITES = {
    "libero_spatial": ("openvla/openvla-7b-finetuned-libero-spatial", [
        "pick up the black bowl between the plate and the ramekin and place it on the plate",
        "pick up the black bowl next to the ramekin and place it on the plate",
        "pick up the black bowl on the stove and place it on the plate",
    ]),
    "libero_object": ("openvla/openvla-7b-finetuned-libero-object", [
        "pick up the alphabet soup and place it in the basket",
        "pick up the cream cheese and place it in the basket",
        "pick up the salad dressing and place it in the basket",
    ]),
    "libero_goal": ("openvla/openvla-7b-finetuned-libero-goal", [
        "open the middle drawer of the cabinet",
        "put the bowl on the stove",
        "put the wine bottle on top of the cabinet",
    ]),
    "libero_10": ("openvla/openvla-7b-finetuned-libero-10", [
        "put both the alphabet soup and the tomato sauce in the basket",
        "turn on the stove and put the moka pot on it",
        "put the black bowl in the bottom drawer of the cabinet and close it",
    ]),
}
IMG = 224
PATCH = 14
NUM_PATCHES = (IMG // PATCH) ** 2  # 256


def make_canvas(pm: OpenVLAProgressiveModel, seed: int) -> torch.Tensor:
    """[1, 6, 224, 224] bf16 (dino norm | siglip norm), the executor's _canvas_to_pixel_values on
    a synthetic uint8 frame. Values are irrelevant to the count."""
    rng = np.random.default_rng(seed)
    frame = rng.integers(0, 256, size=(1, IMG, IMG, 3), dtype=np.uint8)
    t = torch.from_numpy(frame).to(pm.device).permute(0, 3, 1, 2).float() / 255.0
    ip = pm.processor.image_processor
    towers = []
    for i in range(len(ip.means)):
        mean = torch.tensor(ip.means[i]).view(1, 3, 1, 1).to(pm.device)
        std = torch.tensor(ip.stds[i]).view(1, 3, 1, 1).to(pm.device)
        towers.append((t - mean) / std)
    return torch.cat(towers, dim=1).to(torch.bfloat16)


def sequential_groups(num_groups: int):
    bounds = np.linspace(0, NUM_PATCHES, num_groups + 1).round().astype(int)
    return [list(range(bounds[g], bounds[g + 1])) for g in range(num_groups)]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suites", nargs="+", default=list(SUITES))
    ap.add_argument("--groups", type=int, default=4)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out-json", default=None)
    a = ap.parse_args()
    dev = torch.device(a.device)
    out = {"_meta": {"groups": a.groups, "unit": "GFLOPs/instruction",
                     "note": "backbone prefill only (both towers + projector + 32 Llama layers); "
                             "decode_action's 7-token generation excluded; 4 sequential groups, no "
                             "server pscore filter. k1.00/total_k1.00 = chunked causal prefill (the "
                             "stream arm's compute); interleaved_* = the campaign's frontiers-32x4 "
                             "approx-then-correct schedule (gated equivalent, redundant)"},
           "openvla": {"_samples": None}}

    for suite in a.suites:
        ckpt, tasks = SUITES[suite]
        t0 = time.time()
        pm = OpenVLAProgressiveModel(ckpt, dev)
        print(f"[{suite}] loaded {ckpt} in {time.time()-t0:.0f}s", flush=True)
        roots = [pm.vla.vision_backbone, pm.vla.projector, pm.vla.language_model.model,
                 pm.dino_backbone, pm.siglip_backbone, *pm.llm_layers]
        groups = sequential_groups(a.groups)

        def run(fn):
            with flops.session(*roots, enabled=True) as fl:
                for i, text in enumerate(tasks):
                    px = make_canvas(pm, seed=i)
                    with fl.request(i, text=text):
                        fn(fl, px, text)
            return fl.aggregate()

        def full(fl, px, text):
            pm.start_session_from_text(text)
            # predict_action's prefill: stock forward over [BOS, 256 vision, text]. lm_head is
            # outside the roots; generation is not run.
            pm.vla(input_ids=pm.input_ids, pixel_values=px)

        def approx(fl, px, text):
            pm.start_session_from_text(text)
            with fl.stage("approx"):
                pm.vision_approx(px)
                pm.llm_approx_segment(0, len(pm.llm_layers))

        def interleaved(fl, px, text):
            pm.start_session_from_text(text)
            with fl.arrival(0), fl.stage("approx"):
                pm.vision_approx(px)
                pm.llm_approx_segment(0, len(pm.llm_layers))
            arrived = []
            for g, new in enumerate(groups, start=1):
                arrived += new
                all_idx = torch.tensor(arrived, dtype=torch.long, device=dev)
                new_idx = torch.tensor(new, dtype=torch.long, device=dev)
                with fl.arrival(g), fl.stage("correct"):
                    tok = pm.vision_correct(px, all_idx, new_idx)
                    pm.llm_correct_segment(len(pm.llm_layers), tok)

        def chunked(fl, px, text):
            pm.start_session_from_text(text)
            with fl.arrival(0), fl.stage("approx"):
                pm.vision_approx(px)
                pm.llm_chunked_init()
            arrived = []
            for g, new in enumerate(groups, start=1):
                arrived += new
                all_idx = torch.tensor(arrived, dtype=torch.long, device=dev)
                new_idx = torch.tensor(new, dtype=torch.long, device=dev)
                with fl.arrival(g), fl.stage("correct"):
                    tok = pm.vision_correct(px, all_idx, new_idx)
                    pm.llm_prefill_segment(tok, include_text=(g == len(groups)))

        full_g = run(full)["mean_total_gflops"]
        floor_g = run(approx)["mean_total_gflops"]
        agg_i = run(interleaved)
        agg_c = run(chunked)
        crit_i, tot_i = agg_i["mean_critical_gflops"], agg_i["mean_total_gflops"]
        crit_c, tot_c = agg_c["mean_critical_gflops"], agg_c["mean_total_gflops"]
        n_text = int(pm.input_ids.shape[1])
        print(f"\n══ {suite}  (n={len(tasks)}, seq = 1 + {NUM_PATCHES} + {n_text - 1} text) ══")
        print(f"  full inference (ceiling prefill)   {full_g:10.1f} GFLOPs/instruction")
        print(f"  approx only (floor)                {floor_g:10.1f} GFLOPs   floor/full = {100*floor_g/full_g:5.1f}%")
        print(f"  chunked g={a.groups} (stream)        critical {crit_c:9.1f}  total {tot_c:9.1f} GFLOPs   "
              f"critical/full = {100*crit_c/full_g:5.1f}%   total/full = {100*tot_c/full_g:5.1f}%")
        print(f"  interleaved g={a.groups} k=1.00      critical {crit_i:9.1f}  total {tot_i:9.1f} GFLOPs   "
              f"critical/full = {100*crit_i/full_g:5.1f}%   total/full = {100*tot_i/full_g:5.1f}%")
        out["openvla"][suite] = {"full": round(full_g, 1), "floor": round(floor_g, 1),
                                 "k1.00": round(crit_c, 1), "total_k1.00": round(tot_c, 1),
                                 "interleaved_k1.00": round(crit_i, 1),
                                 "interleaved_total_k1.00": round(tot_i, 1)}
        out["openvla"]["_samples"] = len(tasks)
        del pm, roots
        torch.cuda.empty_cache()

    if a.out_json:
        os.makedirs(os.path.dirname(a.out_json) or ".", exist_ok=True)
        with open(a.out_json, "w") as f:
            json.dump(out, f, indent=1)
        print(f"wrote {a.out_json}")


if __name__ == "__main__":
    main()
