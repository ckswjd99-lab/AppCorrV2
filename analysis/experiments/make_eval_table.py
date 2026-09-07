"""Emit the evaluation table as LaTeX from whatever results exist on disk.

Run it any time. Cells with no measurement print `--`, so the table is always current and never
carries a number that was typed by hand -- which is the point: transcribing these by eye already
put a wrong MMMU row into one progress report.

    python analysis/experiments/make_eval_table.py                  # LaTeX, keeps 25/50
    python analysis/experiments/make_eval_table.py --keeps 0.30 0.50
    python analysis/experiments/make_eval_table.py --format md      # readable while working
    python analysis/experiments/make_eval_table.py --status         # what is still missing

Where the numbers come from:

  accuracy   analysis/results/{model}_{dataset}/{arm}.json -> summary.accuracy, written by the
             oracle drivers. `ceiling`, `floor`, `interleaved_g{g}_k{keep}`.
  FLOPs      analysis/results/flops/*.json for the offload-driven models, written by the worker;
             analysis/results/flops/inprocess_flops.json for the ones driven in process.

Two conventions the table depends on, both enforced here rather than left to the reader:

  * Crit.\\ Comp. is per INSTRUCTION. Offload arms differ in batch size within a single model --
    NYU's ceiling runs at 1 and its interleaved arms at 8 -- so every offload value is divided by
    its recorded batch size. Skipping that made NYU's critical exceed its own ceiling.
  * A VFM runs one backbone regardless of task, so its rows repeat the same FLOPs by construction.
    They are emitted repeatedly because the table has one row per task, not because they were
    measured per task.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Dict, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(ROOT, "analysis", "results")
FLOPS_DIR = os.path.join(RESULTS, "flops")

# Ours accuracy cells at or above this preservation get a light-gray background (user request
# 2026-08-31). Needs \usepackage[table]{xcolor} in the paper preamble; emit_md strips the macro.
SHADE_PRES = 98.0
SHADE_MACRO = r"\cellcolor{gray!15}"

# Qwen3.5 pyr-filter campaign (2026-08-31, B200 box): jsonl per arm, appended incrementally with
# resume. A cell is emitted only when the arm is COMPLETE (every split index present), so a table
# generated mid-run never shows a partial number; oom-skips count toward completeness but are
# excluded from the mean. RefCOCO reports mean(ok) as Acc@0.5 and mean(val) as mIoU; TextVQA's
# headline is mean(val) -- the VQA soft score -- NOT the driver's running % (which prints the
# ok-rate, sc>=0.5).
QWEN35_PYR_DIR = os.path.join(RESULTS, "qwen35_accuracy_pyr")
QWEN35_PYR_EXPECTED = {"refcoco": 8811, "textvqa": 5000, "vstar": 191,
                       "visdrone_det": 448, "visdrone_count": 2350}
QWEN35_PYR_ROWS = {"RefCOCO val (Acc.@0.5)": ("refcoco", "ok"),
                   "RefCOCO val (mIoU)":     ("refcoco", "val"),
                   "TextVQA (VQA Acc.)":     ("textvqa", "val"),
                   "V*Bench (Acc.)":         ("vstar", "ok"),
                   "VisDrone Count (Exact Acc.)": ("visdrone_count", "ok"),
                   "VisDrone Count (Soft)":       ("visdrone_count", "val"),
                   "VisDrone Det (Acc.@0.5)":     ("visdrone_det", "ok"),
                   "VisDrone Det (mIoU)":         ("visdrone_det", "val")}

# Muse Glimmer campaign jsonls (same schema, same driver family).
MG_PYR_DIR = os.path.join(RESULTS, "museglimmer_accuracy_pyr")
MG_PYR_EXPECTED = {"vstar": 191, "textvqa": 5000, "refcoco": 8811,
                   "visdrone_det": 448, "visdrone_count": 2350}
MG_PYR_ROWS = {"V*Bench (Acc.)":              ("vstar", "ok"),
               "TextVQA (VQA Acc.)":          ("textvqa", "val"),
               "RefCOCO val (Acc.@0.5)":      ("refcoco", "ok"),
               "RefCOCO val (mIoU)":          ("refcoco", "val"),
               "VisDrone Count (Exact Acc.)": ("visdrone_count", "ok"),
               "VisDrone Count (Soft)":       ("visdrone_count", "val"),
               "VisDrone Det (Acc.@0.5)":     ("visdrone_det", "ok"),
               "VisDrone Det (mIoU)":         ("visdrone_det", "val")}


# 122B probe (2026-09-01, NHN box): same jsonl schema, REDUCED SCALE (n=240 per arm, user-directed)
# under the Triton finegrained-fp8 fallback -- see the pilcrow footnote. Completeness is checked
# against the probe's own n, not the dataset size.
QWEN122B_DIR = os.path.join(RESULTS, "qwen35_122b_probe")
QWEN122B_EXPECTED = {"refcoco": 240, "textvqa": 240}
QWEN122B_SLUG = "_qwen3.5-122b-a10b-fp8"


def qwen35_pyr_lit(dataset: str, metric: str, slug: str = "",
                   dir_: str = None, expected: Dict[str, int] = None) -> Dict[str, float]:
    """LITERALS-shaped dict computed from the pyr campaign's jsonls; complete arms only."""
    exp = (expected or QWEN35_PYR_EXPECTED).get(dataset)
    out: Dict[str, float] = {}
    if exp is None:
        return out
    QWEN35_PYR_DIR_ = dir_ or QWEN35_PYR_DIR
    for key, tag in (("floor", "floor"), ("ceiling", "ceiling"),
                     ("k0.25", "streaming_g4_k0.25"), ("k0.50", "streaming_g4_k0.50"),
                     ("stream", "streaming_g4")):
        p = os.path.join(QWEN35_PYR_DIR_, f"{dataset}{slug}_{tag}.jsonl")
        if not os.path.exists(p):
            continue
        rows = [json.loads(l) for l in open(p) if l.strip()]
        if len({r["i"] for r in rows}) < exp:
            continue
        sc = [r for r in rows if "skip" not in r]
        if not sc:
            continue
        out[key] = 100.0 * sum(float(r[metric]) for r in sc) / len(sc)
    return out

# (model label, [(dataset label, accuracy key, flops key)]). `accuracy key` is (dir_prefix, dataset)
# or None when no accuracy arm exists; `flops key` selects the FLOPs source.
SPEC = [
    ("Gemma 3 (4.3B)$^\\ddagger$", [
        ("ChartQA (Relaxed Acc.)", ("gemma3", "chartqa"),     ("inproc", "gemma3", "chartqa")),
        ("InfoVQA (ANLS)",         ("gemma3", "infovqa"),     ("inproc", "gemma3", "infovqa")),
        ("TextVQA (VQA Acc.)",     ("gemma3", "textvqa"),     ("inproc", "gemma3", "textvqa")),
        ("POPE (Acc.)",            ("gemma3", "pope"),        ("inproc", "gemma3", "pope")),
        ("RealWorldQA (Acc.)",     ("gemma3", "realworldqa"), ("inproc", "gemma3", "realworldqa")),
        ("DocVQA (ANLS)",          ("gemma3", "docvqa"),      ("inproc", "gemma3", "docvqa")),
        ("GQA testdev (Exact Match)", ("gemma3", "gqa"),      ("inproc", "gemma3", "gqa")),
        ("MMMU val (Acc.)",        ("gemma3", "mmmu"),        ("inproc", "gemma3", "mmmu")),
        ("VSR zeroshot (Acc.)",    ("gemma3", "vsr"),         ("inproc", "gemma3", "vsr")),
    ]),
    # Qwen3.5's arm is STREAMING (vision approx/correct + chunked LLM prefill), not interleaved
    # correction: it progressively recomputes 100% of tokens and has no keep-rate knob -- the
    # operating point is the round count (g=4 measured). Its FLOPs sit under the k0.50 keys purely
    # so the existing column machinery renders them; the dagger footnote in the caption says so.
    # Accuracy cells stay empty until the dataset driver runs.
    ("LLaVA-OV2 (8.5B)$^\\ddagger$", [
        ("ChartQA (Relaxed Acc.)", ("ov2", "chartqa"),     ("inproc", "ov2", "chartqa")),
        ("InfoVQA (ANLS)",         ("ov2", "infovqa"),     ("inproc", "ov2", "infovqa")),
        ("TextVQA (VQA Acc.)",     ("ov2", "textvqa"),     ("inproc", "ov2", "textvqa")),
        ("DocVQA (ANLS)",          ("ov2", "docvqa"),      ("inproc", "ov2", "docvqa")),
        ("RealWorldQA (Acc.)",     ("ov2", "realworldqa"), ("inproc", "ov2", "realworldqa")),
        ("POPE (Acc.)",            ("ov2", "pope"),        ("inproc", "ov2", "pope")),
        ("GQA testdev (Exact Match)", ("ov2", "gqa"),      ("inproc", "ov2", "gqa")),
        ("MMMU val (Acc.)",        ("ov2", "mmmu"),        ("inproc", "ov2", "mmmu")),
        ("RefCOCO val (Acc.@0.5)", ("ov2", "refcoco"),     ("inproc", "ov2", "refcoco")),
        ("VSR zeroshot (Acc.)",    ("ov2", "vsr"),         ("inproc", "ov2", "vsr")),
        ("V*Bench (Acc.)",         None, ("inproc", "ov2", "vstar")),
    ]),
    ("Qwen2.5-VL (33.5B)$^\\S$", [
        ("RefCOCO val (Acc.@0.5)",    None, ("inproc", "qwen25vl_32b", "refcoco")),
        ("RefCOCO val (mIoU)",        None, ("inproc", "qwen25vl_32b", "refcoco")),
        ("GQA testdev (Exact Match)", None, ("inproc", "qwen25vl_32b", "gqa")),
        ("RealWorldQA (Acc.)",        None, ("inproc", "qwen25vl_32b", "realworldqa")),
        ("MMVP (Acc.)",               None, ("inproc", "qwen25vl_32b", "mmvp")),
        ("CV-Bench (Acc.)",           None, ("inproc", "qwen25vl_32b", "cvbench")),
        # VisDrone: neither accuracy nor FLOPs exist for this model. The spec hands the encoder
        # the NATIVE frame (270x480 etc.), which ProgressiveLPyramidPolicy rejects (not a multiple
        # of the 28px merge patch) -- the fix is a resize policy that would also define the
        # accuracy arms, so it is not a fill (2026-09-07).
        ("VisDrone Count (Exact Acc.)", None, None),
        ("VisDrone Det (Acc.@0.5)",     None, None),
        ("V*Bench (Acc.)",              None, ("inproc", "qwen25vl_32b", "vstar")),
    ]),
    ("Qwen3.5-MoE (35B-A3B)$^\\dagger$", [
        ("ChartQA (Relaxed Acc.)", None, ("inproc", "qwen35_moe", "chartqa")),
        ("RealWorldQA (Acc.)",     None, ("inproc", "qwen35_moe", "realworldqa")),
        ("VSR zeroshot (Acc.)",    None, ("inproc", "qwen35_moe", "vsr")),
        ("MMVP (Acc.)",            None, ("inproc", "qwen35_moe", "mmvp")),
        ("CV-Bench (Acc.)",        None, ("inproc", "qwen35_moe", "cvbench")),
        # Resolution-sensitive track (2026-08-31, B200 box): accuracy is measured under the
        # pyr filter (Option B) by qwen35_accuracy.py into qwen35_accuracy_pyr/ -- jsonl, not
        # the {arm}.json layout, so cells stay "--" until a loader is wired. FLOPs n=12.
        ("RefCOCO val (Acc.@0.5)", None, ("inproc", "qwen35_moe", "refcoco")),
        ("RefCOCO val (mIoU)",     None, ("inproc", "qwen35_moe", "refcoco")),
        ("TextVQA (VQA Acc.)",     None, ("inproc", "qwen35_moe", "textvqa")),
        ("VisDrone Count (Exact Acc.)", None, ("inproc", "qwen35_moe", "visdrone_count")),
        ("VisDrone Count (Soft)",       None, ("inproc", "qwen35_moe", "visdrone_count")),
        ("VisDrone Det (Acc.@0.5)",     None, ("inproc", "qwen35_moe", "visdrone_det")),
        ("VisDrone Det (mIoU)",         None, ("inproc", "qwen35_moe", "visdrone_det")),
        ("V*Bench (Acc.)",              None, ("inproc", "qwen35_moe", "vstar")),
    ]),
    # Gemma 4 31B: Ours = INTERLEAVED g=4 since 2026-08-31 (port-plan step 4 landed; walk
    # gate bitwise, identity gate in ceiling's flicker set). Accuracy cells prefer
    # interleaved_g4_k*.json and fall back to corrected_k*.json while reruns land; FLOPs in
    # inprocess_flops.json are the interleaved accounting (crit 17-35%, total 154-182% --
    # the one-shot structure's 90%+ crit / 103-106% total is archived in
    # analysis/results/flops/gemma4_flops*.json).
    ("Gemma 4 (31B)", [
        ("MMVP (Acc.)",            ("gemma4", "mmvp"),    ("inproc", "gemma4", "mmvp")),
        ("CV-Bench (Acc.)",        ("gemma4", "cvbench"), ("inproc", "gemma4", "cvbench")),
        # Resolution-sensitive track (GH200 campaign, post-Mistral). NOTE the port has bounds +
        # one-shot corrected only (interleaved/streaming are port-plan steps 4-6, not yet built),
        # so the Streaming column stays "--" for now by construction.
        ("RefCOCO val (Acc.@0.5)", ("gemma4", "refcoco"), ("inproc", "gemma4", "refcoco")),
        ("RefCOCO val (mIoU)",     ("gemma4", "refcoco"), ("inproc", "gemma4", "refcoco")),
        ("TextVQA (VQA Acc.)",     ("gemma4", "textvqa"), ("inproc", "gemma4", "textvqa")),
        ("VisDrone Count (Exact Acc.)", ("gemma4", "visdrone_count"), ("inproc", "gemma4", "visdrone_count")),
        ("VisDrone Count (Soft)",       ("gemma4", "visdrone_count"), ("inproc", "gemma4", "visdrone_count")),
        ("VisDrone Det (Acc.@0.5)",     ("gemma4", "visdrone_det"),   ("inproc", "gemma4", "visdrone_det")),
        ("VisDrone Det (mIoU)",         ("gemma4", "visdrone_det"),   ("inproc", "gemma4", "visdrone_det")),
    ]),
    # New 30B-class models (2026-08-28 sweep): bounds via the generic oracle; ours arms pending
    # their axis ports. WildVision is judge-only (prediction dumps) and has no accuracy row.
    ("Mistral Small 3.1 (24B)", [
        ("MMVP (Acc.)",            ("mistral24b", "mmvp"),    ("inproc", "mistral24b", "mmvp")),
        ("CV-Bench (Acc.)",        ("mistral24b", "cvbench"), ("inproc", "mistral24b", "cvbench")),
        # Resolution-sensitive track (2026-08-28 priority shift): grounding + OCR + drone-scale
        # tiny objects. File-based: cells fill as the GH200 campaign's arms land.
        ("RefCOCO val (Acc.@0.5)", ("mistral24b", "refcoco"), ("inproc", "mistral24b", "refcoco")),
        ("RefCOCO val (mIoU)",     ("mistral24b", "refcoco"), ("inproc", "mistral24b", "refcoco")),
        ("TextVQA (VQA Acc.)",     ("mistral24b", "textvqa"), ("inproc", "mistral24b", "textvqa")),
        ("VisDrone Count (Exact Acc.)", ("mistral24b", "visdrone_count"), ("inproc", "mistral24b", "visdrone_count")),
        ("VisDrone Count (Soft)",       ("mistral24b", "visdrone_count"), ("inproc", "mistral24b", "visdrone_count")),
        ("VisDrone Det (Acc.@0.5)",     ("mistral24b", "visdrone_det"),   ("inproc", "mistral24b", "visdrone_det")),
        ("VisDrone Det (mIoU)",         ("mistral24b", "visdrone_det"),   ("inproc", "mistral24b", "visdrone_det")),
        ("V*Bench (Acc.)",              None, ("inproc", "mistral24b", "vstar")),
    ]),
    ("Muse Glimmer (29.6B)", [
        ("MMVP (Acc.)",            ("museglimmer30b", "mmvp"),    ("inproc", "museglimmer30b", "mmvp")),
        ("CV-Bench (Acc.)",        ("museglimmer30b", "cvbench"), ("inproc", "museglimmer30b", "cvbench")),
        ("V*Bench (Acc.)",         None, ("inproc", "museglimmer30b", "vstar")),
        # 2026-09-01 MG campaign (user scope: RefCOCO/VisDrone/TextVQA, arms floor/ceiling/
        # streaming k1.0). visdrone_det is a capability-limit row (~1-2% every arm), kept per
        # the Mistral precedent. FLOPs: refcoco/textvqa from museglimmer_arms_flops.json.
        ("RefCOCO val (Acc.@0.5)", None, ("inproc", "museglimmer30b", "refcoco")),
        ("RefCOCO val (mIoU)",     None, ("inproc", "museglimmer30b", "refcoco")),
        ("TextVQA (VQA Acc.)",     None, ("inproc", "museglimmer30b", "textvqa")),
        ("VisDrone Count (Exact Acc.)", None, ("inproc", "museglimmer30b", "visdrone_count")),
        ("VisDrone Count (Soft)",       None, ("inproc", "museglimmer30b", "visdrone_count")),
        ("VisDrone Det (Acc.@0.5)",     None, ("inproc", "museglimmer30b", "visdrone_det")),
        ("VisDrone Det (mIoU)",         None, ("inproc", "museglimmer30b", "visdrone_det")),
    ]),
    # 122B-FP8: DeepGEMM mis-generates on this B200 (sm_100; per-row outputs bit-perfect, end
    # tokens drift -- transformers 5.13 documents it); accuracy IS measurable under the Triton
    # finegrained-fp8 fallback (TRANSFORMERS_DISABLE_DEEPGEMM_LINEAR=1). Cells below are the
    # 2026-09-01 REDUCED-SCALE probe (n=240/arm, user-directed) via the QWEN122B_* loader; the
    # pilcrow footnote states both the fallback and the n. COMPUTE figures are shape-determined
    # and kernel-independent (n=12, attention term included except chartqa -- see _note).
    ("Qwen3.5-MoE (122B-A10B FP8)$^\\dagger$\\textsuperscript{\\P}", [
        ("ChartQA (Relaxed Acc.)", None, ("inproc", "qwen35_122b", "chartqa")),
        ("RealWorldQA (Acc.)",     None, ("inproc", "qwen35_122b", "realworldqa")),
        ("RefCOCO val (Acc.@0.5)", None, ("inproc", "qwen35_122b", "refcoco")),
        ("RefCOCO val (mIoU)",     None, ("inproc", "qwen35_122b", "refcoco")),
        ("TextVQA (VQA Acc.)",     None, ("inproc", "qwen35_122b", "textvqa")),
        ("VisDrone Count (Exact Acc.)", None, ("inproc", "qwen35_122b", "visdrone_count")),
        ("VisDrone Det (Acc.@0.5)",     None, ("inproc", "qwen35_122b", "visdrone_det")),
        ("V*Bench (Acc.)",              None, ("inproc", "qwen35_122b", "vstar")),
    ]),
    # OpenVLA FLOPs (2026-09-07): AppCorr-openvla/analysis/experiments/flops_report_openvla.py,
    # in-process, 4 sequential groups with cumulative vision correction. Backbone = both towers +
    # projector + 32 Llama layers; the 7-token action decode is excluded like every other model's
    # decode. k1.00/total_k1.00 are the CHUNKED causal prefill (no LLM approx pass, each position
    # once, text once; total ~127%). The accuracy campaign ran the interleaved schedule (frontiers
    # 32x4), gated bit-identical to chunked on final state / action bins (openvla_chunked_gate.py),
    # whose redundant cost (~244%) is kept as interleaved_total_k1.00. Only the streaming (k=1.00)
    # arm exists for the VLA, so the one-shot columns stay "--" by construction.
    ("OpenVLA (7B)", [
        ("LIBERO-Spatial (Success Rate)", None, ("inproc", "openvla", "libero_spatial")),
        ("LIBERO-Object (Success Rate)",  None, ("inproc", "openvla", "libero_object")),
        ("LIBERO-Goal (Success Rate)",    None, ("inproc", "openvla", "libero_goal")),
        ("LIBERO-Long (Success Rate)",    None, ("inproc", "openvla", "libero_10")),
    ]),
    ("DINOv3 (7B)", [
        (r"ImageNet-1k (Top-1 $\uparrow$)", None, ("offload", "dinov3_imagenet")),
        (r"COCO Detector (mAP $\uparrow$)", None, ("offload", "dinov3_coco")),
        (r"ADE20K m2f (mIoU $\uparrow$)",   None, ("offload", "dinov3_ade20k")),
        (r"NYUv2 (AbsRel $\downarrow$)",    None, ("offload", "dinov3_nyu")),
        (r"Co3Dv2 (Rot. deg $\downarrow$)", None, None),
    ]),
    ("SAM 3 (0.85B)", [
        ("COCO Tracker (Mask AP)",  None, ("inproc", "sam3", "coco")),
        ("COCO Detector (Mask AP)", None, ("inproc", "sam3", "coco")),
        ("LVIS Detector (Mask AP)", None, ("inproc", "sam3", "coco")),
        ("SA-Co crowded (cgF1)",    None, ("inproc", "sam3", "coco")),
        ("SA-Co sa1b (cgF1)",       None, ("inproc", "sam3", "coco")),
        ("SA-Co attributes (cgF1)", None, ("inproc", "sam3", "coco")),
        ("SA-Co metaclip (cgF1)",    None, ("inproc", "sam3", "coco")),
        ("SA-Co fg-sports (cgF1)",   None, ("inproc", "sam3", "coco")),
        ("SA-Co fg-food (cgF1)",     None, ("inproc", "sam3", "coco")),
        ("SA-Co wiki-common (cgF1)", None, ("inproc", "sam3", "coco")),
    ]),
    ("OpenCLIP (2.5B)", [
        ("ImageNet-1k (Top-1)",        None, ("offload", "openclip_imagenet")),
        ("ImageNet-1k (Top-5)",        None, ("offload", "openclip_imagenet")),
        # Same vision tower and same 224px canvas as the ImageNet rows, so the ceiling comes out
        # bit-identical (967.5 GF/image) -- measured rather than aliased, which is what makes that
        # agreement a cross-check instead of an assumption.
        ("COCO Ret. val2017 (i2t R@1)", None, ("offload", "openclip_cocoret")),
        ("COCO Ret. val2017 (t2i R@1)", None, ("offload", "openclip_cocoret")),
        # One-shot (g=1) at the same keep: the diagnosis decomposition's headline. CLIP is the one
        # model where interleaving costs real accuracy (staleness 5.76pp > selection 3.06pp at
        # keep=0.50); this row shows the trade the g=4 row hides. Measured on the full 5000-image
        # split, 2026-08-26 (docs/memo/openclip_staleness_decomposition.md).
        # FLOPs: the same worker accounting with num_groups=1 (openclip_cocoret_g1_k0.50.json,
        # 2026-09-07); the ceiling file is shared with the g=4 rows because a one-shot ceiling is
        # the same forward on the same canvas -- an identity, not a measurement to repeat.
        ("COCO Ret. one-shot g=1 (i2t R@1)", None, ("offload", "openclip_cocoret", "g1")),
        ("COCO Ret. one-shot g=1 (t2i R@1)", None, ("offload", "openclip_cocoret", "g1")),
    ]),
    ("VGGT-Omega (7B)", [
        (r"Co3Dv2 (Depth AbsRel $\downarrow$)", None, ("offload", "vggt_co3d")),
        (r"Co3Dv2 (Rot. deg $\downarrow$)",     None, ("offload", "vggt_co3d")),
        (r"Co3Dv2 ($\delta < 1.10$ $\uparrow$)", None, ("offload", "vggt_co3d")),
        (r"Co3Dv2 (3D Point Err. $\downarrow$)", None, ("offload", "vggt_co3d")),
        (r"Co3Dv2 (3D Inlier $<10\%$)",         None, ("offload", "vggt_co3d")),
    ]),
]

# Presentation order (user, 2026-08-31): VFMs first, then the VLMs, then the VLA. Sorting here
# instead of moving the literal blocks keeps each block's comments next to its rows.
_ROW_ORDER = ["DINOv3", "SAM 3", "OpenCLIP", "VGGT-Omega", "Gemma 3", "Gemma 4", "LLaVA-OV2",
              "Mistral Small", "Muse Glimmer", "Qwen2.5-VL", "Qwen3.5-MoE (35B",
              "Qwen3.5-MoE (122B", "OpenVLA"]


def _row_order_key(entry):
    name = entry[0]
    for i, prefix in enumerate(_ROW_ORDER):
        if name.startswith(prefix):
            return i
    return len(_ROW_ORDER)  # unknown models sink to the bottom rather than crash


SPEC.sort(key=_row_order_key)

# Values that exist only in prose (other branches, published memos) and have no JSON to read.
# Kept separate from anything measured here so the two are never confused.
# SUSPECT, 2026-08-25: every Qwen2.5-VL value below came from `full_inference` in
# `qwen25vl_executor.py`, which called the stock model without `mm_token_type_ids`. transformers
# then takes `can_compute_mrope = False`, `compute_3d_position_ids` returns None, and the text model
# falls back to plain 1D positions replicated across all three M-RoPE axes -- every image token
# loses its (t, h, w) grid position and is treated as text at its sequence offset. Traced and proven
# on the GH200 box: a correctly-called stock forward matches our interleaved g=1 arm bit-exactly
# (0/8,714,240 elements differing), while the degraded call does not. So the ARM was always right
# and the BASELINE was wrong. These numbers, and every gap or crossing point measured against them,
# have been re-established. RefCOCO and GQA below now carry the corrected values; the RealWorldQA
# ceilings still do not, and neither does anything derived from the old bounds (the keep-rate sweeps
# and the "-1pp crossing at ~58%" conclusion in QWEN25VL_APPCORR_LOG.md), which all need re-deriving
# rather than re-centering -- RefCOCO's gap NARROWED from 10.99pp to 8.51pp because its floor moved
# further than its ceiling (+4.92 against +2.44), so a recovery fraction computed against the old
# bounds is wrong by more than a shift. GQA's bounds barely moved at all (-0.04 / +0.08), which is
# the same insensitivity its churn analysis showed.
#
# Re-measurement in progress on the GH200 box. RefCOCO baseline has landed: 85.75 -> **88.19**
# (mean IoU 0.7620 -> 0.8024), full 8811 split. It moved TOWARD the published Qwen2.5-VL figures,
# which is the independent corroboration that the mechanism is what the trace says it is -- a fix
# that left the number flat, or moved it down, would have meant the story was wrong even though the
# tensors matched. GQA is running. The floor arms came from the same broken function and are being
# re-measured too, so nothing here is updated until BOTH bounds are back: quoting a new ceiling
# against an old floor would invent a gap neither measurement supports.
LITERALS = {
    # V*Bench full split (191), 2026-09-01 campaign, pyr L2, from the oracle Final Summary lines
    # (analysis/results/logs/vstar_mistral_*.log / vstar_ov2_*.log -- --out-json was not passed).
    ("Mistral Small 3.1 (24B)", "V*Bench (Acc.)"): {"floor": 50.26, "ceiling": 52.36,
                                                    "k0.25": 53.40, "k0.50": 54.97,
                                                    "stream": 51.83},
    ("LLaVA-OV2 (8.5B)", "V*Bench (Acc.)"): {"floor": 74.87, "ceiling": 85.86,
                                             "k0.25": 78.01, "k0.50": 83.25,
                                             "stream": 84.29},
    # V*Bench full split via qwen25vl_bench_eval retry (logs vstar_qwen25_*.log, '=== Summary').
    ("Qwen2.5-VL (33.5B)", "V*Bench (Acc.)"): {"floor": 60.73, "ceiling": 79.06,
                                               "k0.25": 70.68, "k0.50": 74.87,
                                               "stream": 78.01},
    # Qwen3.5 accuracy, full RealWorldQA split (765), 2026-08-27, thinking disabled, shared greedy
    # decode across all three arms. Single streaming arm (g=4) sits under the k0.50 columns per the
    # dagger footnote's nominal-placement rule.
    ("Qwen3.5-MoE (35B-A3B)", "RealWorldQA (Acc.)"): {"floor": 74.51, "ceiling": 77.39,
                                                      "k0.25": 77.52, "k0.50": 77.25,
                                                      "stream": 77.25},
    ("Qwen3.5-MoE (35B-A3B)", "ChartQA (Relaxed Acc.)"): {"floor": 60.76, "ceiling": 88.56,
                                                          "k0.25": 83.68, "k0.50": 86.80,
                                                          "stream": 88.32},
    ("Qwen3.5-MoE (35B-A3B)", "VSR zeroshot (Acc.)"): {"floor": 88.46, "ceiling": 89.77,
                                                       "k0.25": 88.63, "k0.50": 88.95},
    # MMVP full 300 (2026-08-28 real-photo sweep): floor / streaming g=4 / ceiling.
    # NOTE: floor/streaming measured with the bicubic-era filter; BOX re-measurement
    # (qwen35_accuracy_box/) supersedes these when it lands.
    ("Qwen3.5-MoE (35B-A3B)", "MMVP (Acc.)"): {"floor": 79.00, "ceiling": 82.00,
                                               "k0.25": 80.67, "k0.50": 80.00,
                                               "stream": 81.67},
    # GH200 campaign (2026-08-28): cvbench ceiling full 2638, zero skips, FINAL.
    # Floor arrives after their chunked-attention fix rerun; do not backfill early values.
    # floor is the completed full-2638 rerun after the chunked-attention fix (the 111
    # recovered 2K-res images scored 62%, harder than average). k0.50/streaming pending.
    # MMVP full 300, five arms, zero skips (GH200). Largest Qwen2.5 gap measured (9.34pp --
    # CLIP-blind discrimination is exactly what a level-2 pyramid destroys). n=300: one sample
    # = 0.33pp, recovery CI ~ +-5pp -- read recovery coarsely, preservation ordering is solid.
    ("Qwen2.5-VL (33.5B)", "MMVP (Acc.)"): {"floor": 65.33, "ceiling": 74.67,
                                            "k0.25": 68.33, "k0.50": 69.00,
                                            "stream": 71.67},
    ("Qwen2.5-VL (33.5B)", "CV-Bench (Acc.)"): {"ceiling": 79.87, "floor": 72.52,
                                                "k0.25": 75.09, "k0.50": 77.52,
                                                "stream": 78.96},
    # full 2638, BOX floor (post-convention-audit), shared greedy decode.
    ("Qwen3.5-MoE (35B-A3B)", "CV-Bench (Acc.)"): {"floor": 84.08, "ceiling": 85.06,
                                                   "k0.25": 84.50, "k0.50": 84.76,
                                                   "stream": 84.87},
    # One-shot rows share the interleaved rows' bounds (same floor/ceiling arms).
    ("OpenCLIP (2.5B)", "COCO Ret. one-shot g=1 (i2t R@1)"): {"floor": 50.14, "ceiling": 67.92},
    ("OpenCLIP (2.5B)", "COCO Ret. one-shot g=1 (t2i R@1)"): {"floor": 40.37, "ceiling": 50.64},
    # Re-measured 2026-08-26 on the M-RoPE-fixed code, full splits, both bounds through the same
    # driver. RefCOCO N=8811, GQA N=12578. These REPLACE the pre-fix values (which were
    # 85.75/74.76, 76.20/65.02, 60.84/55.16) -- see the block comment above.
    # RefCOCO ours: GH200 full-split (8811 images) 2026-08-27, energy x attention, bs=1 vs
    # bs=16 bounds (measured Acc-identical, 0.001 mIoU).
    # k0.50 is the text-split-schedule run at FULL n=8811 coverage (2026-08-28): the original
    # 207-image OOM exclusion turned out to be a driver defect (missing no_grad pinning ~27GB of
    # autograd graph per image -- fixed same day), and the skipped images were completed via the
    # jsonl resume under the identical schedule. Bounds are therefore the plain full-split bounds,
    # no kept-set restriction needed, and the section-mark footnote no longer applies to k0.50.
    # Schedule A/B at full scale: -0.02pp (flips 21:23) vs every-round.
    # k0.25 kept from the every-round run at n=8803 (8 OOM skips from the same driver defect,
    # predating text-split -- resuming them would mix schedules in one file, and the schedule
    # A/B says the number would not move; the footnote still covers this arm).
    # Streaming (k=1.0) full n=8811, 2026-08-28: 89.55/81.50 -- ABOVE ceiling by +1.36pp.
    # Verified real, not mismeasurement: paired flips 278:158 (net +120, ~4sigma) vs ceiling;
    # control (k0.50, same driver/decode structure) nets -78 as expected. Mechanism isolated by
    # elimination: interleaved k=1.0 subset ALSO beats ceiling (+1.9pp) -> not chunked-prefill-
    # specific; the g=1 identity gate is bitwise-exact vs stock -> not the fork decode; what
    # remains is the multi-round vision correction's partial staleness acting as a beneficial
    # perturbation for grounding. Cross-model: Mistral-24B MMVP streaming +1.33pp (no chunked
    # LLM at all) and OV2 GQA/VSR at 100.2% point the same way.
    ("Qwen2.5-VL (33.5B)", "RefCOCO val (Acc.@0.5)"):    {"floor": 79.68, "ceiling": 88.19,
                                                          "k0.25": 86.10, "k0.50": 87.30,
                                                          "stream": 89.55},
    ("Qwen2.5-VL (33.5B)", "RefCOCO val (mIoU)"):        {"floor": 70.36, "ceiling": 80.24,
                                                          "k0.25": 78.22, "k0.50": 79.35,
                                                          "stream": 81.50},
    ("Qwen2.5-VL (33.5B)", "GQA testdev (Exact Match)"): {"floor": 55.24, "ceiling": 60.80},
    # 72B dropped 2026-08-26: not worth the run. It also does not fit -- the GH200 box has ~66 GB
    # free against a ~130 GB pull, so the row could only ever have carried a prose ceiling.
    ("Qwen2.5-VL (33.5B)", "RealWorldQA (Acc.)"):        {"ceiling": 68.89},
    ("SAM 3 (0.85B)", "COCO Tracker (Mask AP)"):  {"floor": 53.74, "ceiling": 60.10},
    ("SAM 3 (0.85B)", "COCO Detector (Mask AP)"): {"floor": 43.32, "ceiling": 50.92},
    ("SAM 3 (0.85B)", "LVIS Detector (Mask AP)"): {"floor": 41.21, "ceiling": 56.38},
    ("SAM 3 (0.85B)", "SA-Co crowded (cgF1)"):    {"floor": 53.15, "ceiling": 58.95},
    ("SAM 3 (0.85B)", "SA-Co sa1b (cgF1)"):       {"floor": 52.78, "ceiling": 53.94},
    ("SAM 3 (0.85B)", "SA-Co attributes (cgF1)"): {"floor": 53.96, "ceiling": 54.21},
    ("DINOv3 (7B)", r"ImageNet-1k (Top-1 $\uparrow$)"): {"floor": 84.50, "ceiling": 88.11},
    ("DINOv3 (7B)", r"COCO Detector (mAP $\uparrow$)"): {"floor": 55.83, "ceiling": 63.14},
    ("DINOv3 (7B)", r"ADE20K m2f (mIoU $\uparrow$)"):   {"floor": 56.01, "ceiling": 62.24},
    # measured here 2026-08-25, full 654-sample split, same driver as the ours arms
    ("DINOv3 (7B)", r"NYUv2 (AbsRel $\downarrow$)"):    {"floor": 0.05302, "ceiling": 0.05013,
                                                        "fmt": "{:.4f}"},
    ("DINOv3 (7B)", r"Co3Dv2 (Rot. deg $\downarrow$)"): {"floor": 5.440, "ceiling": 2.885,
                                                        "fmt": "{:.3f}"},
    # measured here 2026-08-25, full 310 sequences -- these confirmed the literals to 3 decimals,
    # which is what made VGGT's ours-below-floor anomaly real rather than a bad reference
    ("VGGT-Omega (7B)", r"Co3Dv2 (Depth AbsRel $\downarrow$)"): {"floor": 0.04773, "ceiling": 0.04255,
                                                                "fmt": "{:.4f}"},
    ("VGGT-Omega (7B)", r"Co3Dv2 (Rot. deg $\downarrow$)"):     {"floor": 1.552, "ceiling": 1.332,
                                                                "fmt": "{:.3f}"},
    ("VGGT-Omega (7B)", r"Co3Dv2 ($\delta < 1.10$ $\uparrow$)"): {"floor": 92.83, "ceiling": 92.97},
    ("VGGT-Omega (7B)", r"Co3Dv2 (3D Point Err. $\downarrow$)"): {"floor": 0.1464, "ceiling": 0.1572,
                                                                 "fmt": "{:.4f}"},
    ("VGGT-Omega (7B)", r"Co3Dv2 (3D Inlier $<10\%$)"):          {"floor": 66.25, "ceiling": 62.61},
    ("OpenCLIP (2.5B)", "ImageNet-1k (Top-1)"): {"floor": 65.92, "ceiling": 77.14},
    ("OpenCLIP (2.5B)", "ImageNet-1k (Top-5)"): {"floor": 88.20, "ceiling": 94.88},
    # measured here 2026-08-26 through COCOCaptionsLoader, full 5000/25014 -- our own ceiling landed
    # within 0.06 of the literal, which is what validates the new loader against the prior protocol
    ("OpenCLIP (2.5B)", "COCO Ret. val2017 (i2t R@1)"): {"floor": 50.14, "ceiling": 67.92},
    ("OpenCLIP (2.5B)", "COCO Ret. val2017 (t2i R@1)"): {"floor": 40.37, "ceiling": 50.64},
    # LIBERO-Spatial: ceiling re-measured 2026-09-04 on the rebuilt env (numpy 1.26 fix; llvmpipe
    # EGL) with the offload driver's `full` schedule, 500 episodes = 10 tasks x 50 trials, primary
    # evidence AppCorr-openvla/analysis/results/openvla/libero_spatial_{full,approx,interleaved}_t50.jsonl
    # (July's same-harness 82.8/17.2/81.6 were lost with /tmp; paper 84.7 +- 0.9). floor = approx-only
    # schedule 93/500; stream = interleaved schedule, frontiers 32x4, sequential grouping, g=4,
    # 409/500 (2026-09-04) -- ties the 408/500 ceiling. (Gated bit-identical to the chunked causal
    # prefill whose compute the Comp cells report; see the FLOPs comment in SPEC.)
    ("OpenVLA (7B)", "LIBERO-Spatial (Success Rate)"): {"ceiling": 81.60, "floor": 18.60,
                                                        "stream": 81.80},
    # LIBERO-Object/Goal/Long: same campaign, 2026-09-05..07 (offload full/approx/interleaved
    # schedules, 500 episodes each, official max_steps 280/300/520); prior ceiling literals
    # 89/73/54. Object 430/99/424, Goal 375/89/374, Long 259/13/227 (ceiling/floor/stream).
    # Long stream ran on cuda:1 (--device) alongside approx on cuda:0; paired delta vs ceiling
    # -6.4pp, 95% CI [-12.0, -1.2], 85 wins / 117 losses over 500 (task,trial) pairs -- the
    # only suite where stream is below the ceiling beyond noise (520-step horizon).
    ("OpenVLA (7B)", "LIBERO-Object (Success Rate)"):  {"ceiling": 86.00, "floor": 19.80,
                                                        "stream": 84.80},
    ("OpenVLA (7B)", "LIBERO-Goal (Success Rate)"):    {"ceiling": 75.00, "floor": 17.80, "stream": 74.80},
    ("OpenVLA (7B)", "LIBERO-Long (Success Rate)"):    {"ceiling": 51.80, "floor": 2.60, "stream": 45.40},
}


# The VFM "ours" arms, produced by run_vfm_accuracy_campaign.sh / run_vfm_bounds.sh, which write a
# `Final Summary: {...}` line to a log rather than a summary JSON. Keyed by (model label, dataset
# label) -> (log tag prefix, metric key, scale).
#
# Scale is per-row and NOT guessable from the value: ImageNet reports `top1_acc` already in percent,
# COCO reports `mAP` as a fraction, and NYU/VGGT report raw error values that must not be scaled at
# all. Getting one wrong yields a number off by 100x that still looks like a plausible metric for
# some other task, so each is written out rather than inferred.
VFM_OURS = {
    ("DINOv3 (7B)", r"ImageNet-1k (Top-1 $\uparrow$)"): ("dinov3_imagenet", "top1_acc", 1.0),
    ("DINOv3 (7B)", r"COCO Detector (mAP $\uparrow$)"): ("dinov3_coco", "mAP", 100.0),
    ("DINOv3 (7B)", r"ADE20K m2f (mIoU $\uparrow$)"):   ("dinov3_ade20k", "mIoU", 1.0),
    ("DINOv3 (7B)", r"NYUv2 (AbsRel $\downarrow$)"):    ("dinov3_nyu", "abs_rel", 1.0),
    ("VGGT-Omega (7B)", r"Co3Dv2 (Depth AbsRel $\downarrow$)"): ("vggt_co3d", "abs_rel", 1.0),
    ("VGGT-Omega (7B)", r"Co3Dv2 (Rot. deg $\downarrow$)"):     ("vggt_co3d", "rot_deg", 1.0),
    ("VGGT-Omega (7B)", r"Co3Dv2 ($\delta < 1.10$ $\uparrow$)"): ("vggt_co3d", "delta_1.10", 100.0),
    # SAM 3's six rows share one vision encoder -- which is why their Crit. Comp. column repeats the
    # same FLOPs by construction -- but they are different TASKS, so accuracy is measured per task.
    # Metric keys differ by evaluator, but every SAM 3 summary -- mask_AP AND cgF1 -- is a
    # 0..1 FRACTION (verified against the measured logs: crowded k0.25 prints cgF1 0.5594).
    # An earlier version of this block asserted cgF1 was "already in percent" and gave it scale
    # 1.0, which rendered 0.56 in a column of 55-60s and read as a collapsed model. Writing the
    # scale per row stays deliberate for exactly that reason -- a wrong scale still looks like a
    # plausible metric, so each entry is checked against its own log line, not inferred.
    ("SAM 3 (0.85B)", "COCO Tracker (Mask AP)"):  ("sam3_coco", "mask_AP", 100.0),
    ("SAM 3 (0.85B)", "COCO Detector (Mask AP)"): ("sam3_cocodet", "mask_AP", 100.0),
    ("SAM 3 (0.85B)", "LVIS Detector (Mask AP)"): ("sam3_lvis", "mask_AP", 100.0),
    ("SAM 3 (0.85B)", "SA-Co crowded (cgF1)"):    ("sam3_saco_crowded", "cgF1", 100.0),
    ("SAM 3 (0.85B)", "SA-Co sa1b (cgF1)"):       ("sam3_saco_sa1b", "cgF1", 100.0),
    ("SAM 3 (0.85B)", "SA-Co attributes (cgF1)"): ("sam3_saco_attributes", "cgF1", 100.0),
    ("SAM 3 (0.85B)", "SA-Co metaclip (cgF1)"):    ("sam3_saco_metaclip", "cgF1", 100.0),
    ("SAM 3 (0.85B)", "SA-Co fg-sports (cgF1)"):   ("sam3_saco_fg_sports_equipment", "cgF1", 100.0),
    ("SAM 3 (0.85B)", "SA-Co fg-food (cgF1)"):     ("sam3_saco_fg_food", "cgF1", 100.0),
    ("SAM 3 (0.85B)", "SA-Co wiki-common (cgF1)"): ("sam3_saco_wiki_common", "cgF1", 100.0),
    ("OpenCLIP (2.5B)", "ImageNet-1k (Top-1)"): ("openclip_imagenet", "top1_acc", 1.0),
    ("OpenCLIP (2.5B)", "ImageNet-1k (Top-5)"): ("openclip_imagenet", "top5_acc", 1.0),
    ("OpenCLIP (2.5B)", "COCO Ret. val2017 (i2t R@1)"): ("cocoret", "i2t_R@1", 1.0),
    ("OpenCLIP (2.5B)", "COCO Ret. val2017 (t2i R@1)"): ("cocoret", "t2i_R@1", 1.0),
    ("OpenCLIP (2.5B)", "COCO Ret. one-shot g=1 (i2t R@1)"): ("cocoret_g1", "i2t_R@1", 1.0),
    ("OpenCLIP (2.5B)", "COCO Ret. one-shot g=1 (t2i R@1)"): ("cocoret_g1", "t2i_R@1", 1.0),
}

# (base model, dataset label) pairs where the MODEL fails the TASK outright (floor ~= ceiling at
# degenerate accuracy), so the cells say nothing about the streaming axis. Rendered with a
# $\diamond$ after the dataset label per the user's 2026-09-01 directive. MG RefCOCO was cut at
# ceiling 7.26% (6450/8811, decision: skip remaining arms); MG VisDrone Det ceiling is 2.23%.
CAPABILITY_LIMIT = {
    ("Muse Glimmer (29.6B)", "RefCOCO val (Acc.@0.5)"),
    ("Muse Glimmer (29.6B)", "RefCOCO val (mIoU)"),
    ("Muse Glimmer (29.6B)", "VisDrone Det (Acc.@0.5)"),
    ("Muse Glimmer (29.6B)", "VisDrone Det (mIoU)"),
}

VFM_DIR = os.path.join(RESULTS, "vfm_accuracy")

for _sub, _lbl in (("metaclip", "metaclip"), ("fg_sports_equipment", "fg-sports"),
                   ("fg_food", "fg-food"), ("wiki_common", "wiki-common")):
    _fc = {}
    for _arm in ("floor", "ceiling"):
        _p = os.path.join(VFM_DIR, f"sam3_saco_{_sub}_{_arm}.json")
        if os.path.exists(_p):
            try:
                _fc[_arm] = 100.0 * json.load(open(_p))["cgF1"]
            except (KeyError, ValueError):
                pass
    if _fc:
        LITERALS.setdefault(("SAM 3 (0.85B)", f"SA-Co {_lbl} (cgF1)"), {}).update(_fc)


def vfm_accuracy(tag: str, key: str, scale: float) -> Optional[float]:
    """Read one metric out of a `Final Summary: {...}` line.

    A run that died still leaves a log, and several have exited rc=0 while producing nothing
    (a missing dataset loader, an invalid device ordinal). Absence of the summary line is the
    only reliable "this arm did not happen" signal, so it is what this returns None on.

    Both quoting styles appear: the offload drivers print a Python dict (single quotes), the SAM 3
    oracle prints JSON (double quotes).
    """
    # The SAM 3 oracle rewrite (048835e) writes proper result JSONs instead of Final-Summary
    # logs; try {tag}.json first (committed, box-independent), then the legacy {tag}.log
    # (B200-local campaign logs -- dinov3/vggt cells stay "--" on other boxes until those are
    # pushed; they were never committed, which is why this table renders them empty here while
    # B200's local renders showed them).
    jpath = os.path.join(VFM_DIR, f"{tag}.json")
    if os.path.exists(jpath) and os.path.getsize(jpath) > 0:
        try:
            v = json.load(open(jpath)).get(key)
            return None if v is None else v * scale
        except Exception:
            return None
    path = os.path.join(VFM_DIR, f"{tag}.log")
    if not (os.path.exists(path) and os.path.getsize(path) > 0):
        return None
    try:
        text = open(path, errors="ignore").read()
    except Exception:
        return None
    summaries = re.findall(r"Final Summary: (\{.*?\})", text)
    if not summaries:
        return None
    m = re.search(rf"['\"]{re.escape(key)}['\"]\s*:\s*(-?[0-9.eE+]+)", summaries[-1])
    if not m:
        return None
    try:
        return float(m.group(1)) * scale
    except ValueError:
        return None


def load_accuracy(model: str, dataset: str, tag: str, key: str = "accuracy") -> Optional[float]:
    """key="accuracy" is the headline; key="mean_score" is the graded companion the oracle
    writes alongside it (mIoU for bbox specs, soft score for counting/VQA-soft) -- rows whose
    label carries "(mIoU)" or "(Soft" read it instead."""
    p = os.path.join(RESULTS, f"{model}_{dataset}", f"{tag}.json")
    if not (os.path.exists(p) and os.path.getsize(p) > 0):
        return None
    try:
        v = json.load(open(p))["summary"].get(key)
        return None if v is None else v * 100.0
    except Exception:
        return None


_INPROC = None


def inproc_flops(model: str, dataset: str, key: str) -> Optional[float]:
    global _INPROC
    if _INPROC is None:
        p = os.path.join(FLOPS_DIR, "inprocess_flops.json")
        _INPROC = json.load(open(p)) if os.path.exists(p) else {}
    return (_INPROC.get(model, {}) or {}).get(dataset, {}).get(key)


def offload_total(base: str, key: str, groups: str = "g4") -> Optional[float]:
    """Per-instruction TOTAL FLOPs of an arm: approximate pass plus every correction round.

    total/full is the compute OVERHEAD the schedule pays. It is a different question from the
    critical share, and the two move in opposite directions -- deferring less past the last byte
    generally costs more work overall.
    """
    p = os.path.join(FLOPS_DIR, f"{base}_{groups}_{key}.json")
    if not (os.path.exists(p) and os.path.getsize(p) > 0):
        return None
    try:
        j = json.load(open(p))
        return j["mean_total_gflops"] / max(int(j.get("batch_size", 1) or 1), 1)
    except Exception:
        return None


def get_total(spec, key: str) -> Optional[float]:
    if spec is None:
        return None
    if spec[0] == "inproc":
        return inproc_flops(spec[1], spec[2], f"total_{key}")
    return offload_total(spec[1], key, *spec[2:])


def offload_flops(base: str, key: str, groups: str = "g4") -> Optional[float]:
    """Per-INSTRUCTION FLOPs from a worker-written JSON.

    Divides by the recorded batch size. Arms of one model disagree on it -- NYU's ceiling runs at 1
    while its interleaved arms run at 8 -- so per-request means are not comparable and reading them
    directly made NYU's critical exceed its own ceiling.
    """
    # Optional third spec element selects the arm's group-count infix ("g1" for the one-shot rows);
    # the ceiling has no groups and is shared across them.
    tag = f"{base}_ceiling" if key == "full" else f"{base}_{groups}_{key}"
    p = os.path.join(FLOPS_DIR, f"{tag}.json")
    if not (os.path.exists(p) and os.path.getsize(p) > 0):
        return None
    try:
        j = json.load(open(p))
        bs = max(int(j.get("batch_size", 1) or 1), 1)
        field = "mean_total_gflops" if key == "full" else "mean_critical_gflops"
        return j[field] / bs
    except Exception:
        return None


def get_flops(spec, key: str) -> Optional[float]:
    if spec is None:
        return None
    if spec[0] == "inproc":
        return inproc_flops(spec[1], spec[2], key)
    return offload_flops(spec[1], key, *spec[2:])


def fmt_tf(gf: Optional[float], full: Optional[float]) -> str:
    """GFLOPs -> a TF cell, with the share of the full-res critical computation in parentheses."""
    if gf is None:
        return "--"
    tf = gf / 1000.0
    body = f"{tf:.3f}" if tf < 0.1 else f"{tf:.2f}"
    if full:
        return f"{body}\\,TF ({100 * gf / full:.1f}\\%)"
    return f"{body}\\,TF"


def build_rows(keeps, groups: int):
    out = []
    for model, rows in SPEC:
        # Footnote marks ($^\dagger$ etc.) are display-only; every lookup table is keyed by the
        # BASE model name. Keying lookups on the decorated name silently blanks the whole block --
        # adding $^\S$ to Qwen2.5 turned its bounds into "--" before this split existed.
        base_model = model.split("$")[0]
        block = []
        for label, acc_key, fl_key in rows:
            lit = LITERALS.get((base_model, label), {})
            if base_model == "Qwen3.5-MoE (35B-A3B)" and label in QWEN35_PYR_ROWS:
                ds_name, metric = QWEN35_PYR_ROWS[label]
                lit = {**lit, **qwen35_pyr_lit(ds_name, metric)}
            if base_model.startswith("Qwen3.5-MoE (122B") and label in QWEN35_PYR_ROWS:
                ds_name, metric = QWEN35_PYR_ROWS[label]
                if label == "V*Bench (Acc.)":
                    # V*Bench n=191 IS the whole benchmark: a full-split value, rendered
                    # normally. The parenthesized-probe rule covers reduced-n subsets only.
                    lit = {**lit, **qwen35_pyr_lit(ds_name, metric, slug=QWEN122B_SLUG,
                                                   dir_=QWEN35_PYR_DIR,
                                                   expected=QWEN35_PYR_EXPECTED)}
                else:
                    # User directive (2026-09-01): 122B probe numbers render PARENTHESIZED, never
                    # shaded, no preservation % -- they must not read as full-split values.
                    lit = {**lit, **qwen35_pyr_lit(ds_name, metric, slug=QWEN122B_SLUG,
                                                   dir_=QWEN122B_DIR,
                                                   expected=QWEN122B_EXPECTED),
                           "probe": True}
            if base_model == "Muse Glimmer (29.6B)" and label in MG_PYR_ROWS:
                ds_name, metric = MG_PYR_ROWS[label]
                lit = {**lit, **qwen35_pyr_lit(ds_name, metric, dir_=MG_PYR_DIR,
                                               expected=MG_PYR_EXPECTED)}
            f = "{:.2f}"
            if "fmt" in lit:
                f = lit["fmt"]
            lower_better = r"\downarrow" in label

            _metric_key = "mean_score" if ("(mIoU)" in label or "(Soft" in label) else "accuracy"

            def acc_raw(tag, lit_key=None):
                v = load_accuracy(*acc_key, tag, _metric_key) if acc_key else None
                if v is None and lit_key and lit_key in lit:
                    v = lit[lit_key]
                return v

            ceiling_v = acc_raw("ceiling", "ceiling")

            def acc_with_pres(tag, lit_key=None, shade_hi=False):
                """Accuracy, with (preservation vs.\\ ceiling %) in parentheses.

                Preservation is this-value-relative-to-ceiling, not the other way round: for a
                lower-is-better metric that means ceiling/value, so a value further from the
                ceiling in the bad direction still reads as a preservation < 100%. Applied to
                Low-res. and Ours -- Full-res. is the reference point itself, always 100%.

                `shade_hi` (Ours cells only) marks preservation >= SHADE_PRES with a light-gray
                \\cellcolor so near-ceiling cells read at a glance; emit_md strips the macro.
                """
                v = acc_raw(tag, lit_key)
                if v is None:
                    return "--"
                if lit.get("probe"):
                    return f"({f.format(v)})"
                s = f.format(v)
                if ceiling_v:
                    pres = 100.0 * ((ceiling_v / v) if lower_better else (v / ceiling_v))
                    s += f" ({pres:.1f}\\%)"
                    if shade_hi and pres >= SHADE_PRES:
                        s = SHADE_MACRO + s
                return s

            vfm = VFM_OURS.get((base_model, label))

            def ours(k):
                """Ours at keep `k`. VLM rows read a summary JSON; VFM rows read a campaign log."""
                if vfm is not None:
                    tag, key, scale = vfm
                    v = vfm_accuracy(f"{tag}_k{k:.2f}", key, scale)
                    if v is None:
                        return "--"
                    out = f.format(v)
                    if ceiling_v:
                        pres = 100.0 * ((ceiling_v / v) if lower_better else (v / ceiling_v))
                        out += f" ({pres:.1f}\\%)"
                        if pres >= SHADE_PRES:
                            out = SHADE_MACRO + out
                    return out
                # Tag preference: canonical progressive arm where re-measured; the upfront
                # interleaved arm's file otherwise (ddagger caveat); the one-shot corrected
                # arm last (gemma4-class models whose interleaved walk is not ported yet).
                for tag in (f"progressive_g{groups}_k{k:.2f}",
                            f"interleaved_g{groups}_k{k:.2f}",
                            f"corrected_k{k:.2f}"):
                    v = acc_with_pres(tag, lit_key=f"k{k:.2f}", shade_hi=True)
                    if v != "--":
                        return v
                return "--"

            full_gf = get_flops(fl_key, "full")
            cells = [acc_with_pres("floor", "floor")]
            for k in keeps:
                cells.append(ours(k))
                # Comp. = the arm's TOTAL backbone compute (approximate pass + every correction
                # round, overlapped work included), vs Crit. Comp. = only what waits on the last
                # byte. The two move in opposite directions -- deferring less costs more overall --
                # which is why both columns exist side by side.
                cells.append(fmt_tf(get_total(fl_key, f"k{k:.2f}"), full_gf))
                cells.append(fmt_tf(get_flops(fl_key, f"k{k:.2f}"), full_gf))
            # Streaming (keep=1.0) block: the causal-LLM category (LLM prefills exactly once,
            # vision corrects everything progressively). Accuracy comes from a "stream" literal
            # (Qwen3.5's jsonl-driven runs) or a streaming_g4.json beside the row's other arms
            # (OV2's oracle); compute from the k1.00 inproc keys. Non-causal models leave all
            # three cells empty -- Gemma 3's image tokens are bidirectional, and the VFMs have no
            # LLM to stream.
            sv = None
            if "stream" in lit:
                sv = lit["stream"]
            elif acc_key:
                sv = load_accuracy(*acc_key, "streaming_g4", _metric_key)
            if sv is None:
                cells.append("--")
            elif lit.get("probe"):
                cells.append(f"({f.format(sv)})")
            else:
                out_s = f.format(sv)
                if ceiling_v:
                    pres = 100.0 * ((ceiling_v / sv) if lower_better else (sv / ceiling_v))
                    out_s += f" ({pres:.1f}\\%)"
                    if pres >= SHADE_PRES:
                        out_s = SHADE_MACRO + out_s
                cells.append(out_s)
            cells.append(fmt_tf(get_total(fl_key, "k1.00"), full_gf))
            cells.append(fmt_tf(get_flops(fl_key, "k1.00"), full_gf))
            if ceiling_v is None:
                cells.append("--")
            else:
                cells.append(f"({f.format(ceiling_v)})" if lit.get("probe") else f.format(ceiling_v))
            cells.append(fmt_tf(full_gf, None))
            if (base_model, label) in CAPABILITY_LIMIT:
                label = label + r"\,$\diamond$"
            block.append((label, cells))
        out.append((model, block))
    return out


def emit_latex(table, keeps) -> str:
    heads = " & ".join(f"\\multicolumn{{3}}{{c}}{{Ours ({int(k*100)}\\%)}}" for k in keeps)
    heads += " & \\multicolumn{3}{c}{Streaming (k$=$1.0)}"
    cmids, col = [], 4
    for _ in keeps:
        cmids.append(f"\\cmidrule(lr){{{col}-{col+2}}}")
        col += 3
    cmids.append(f"\\cmidrule(lr){{{col}-{col+2}}}")     # streaming block
    col += 3
    cmids.append(f"\\cmidrule(lr){{{col}-{col+1}}}")
    sub = " & ".join(["Acc. (\\%) & Comp. & Crit. Comp."] * (len(keeps) + 1)
                     + ["Acc. (\\%) & Comp."])
    # 2 labels + Low-res. + three per Ours block + three for Streaming + two for Full-res.
    ncol = 2 + 1 + 3 * len(keeps) + 3 + 2
    L = []
    L.append(r"% requires \usepackage[table]{xcolor} in the preamble (for \cellcolor)")
    L.append(r"\begin{table*}[t]")
    L.append(r"\vspace{-0.1in}")
    L.append(r"\caption{Evaluation Results across Different Configurations. Crit.\ Comp.\ is "
             r"backbone prefill FLOPs per instruction that can only begin once the whole image has "
             r"arrived (decode excluded); Comp.\ is the arm's total backbone compute including "
             r"work overlapped with transmission. Parentheses give the ratio to the Full-res.\ "
             r"computation. Ours uses interleaved $g{=}4$; shaded Ours and Streaming accuracy "
             r"cells retain $\geq$98\% of the Full-res.\ accuracy. "
             r"The Streaming (k$=$1.0) block is the causal-LLM category: the LLM prefills exactly "
             r"once in arrival-order chunks while the vision encoder corrects everything "
             r"progressively -- total $\approx$ full + one vision pass, critical $\approx 1/g$. "
             r"Only causal models qualify (Gemma 3's image tokens are bidirectional; VFMs have no "
             r"LLM). $^\dagger$Qwen3.5's Ours columns are keep-limited STREAMING arms (band-wise "
             r"top-$k$ selection), not interleaved correction. "
             r"$^\ddagger$Gemma 3 and LLaVA-OV2 compute figures are from the progressive "
             r"per-round selection arm (2026-08-26); accuracy cells are the progressive arm's "
             r"where re-measured (2026-08-28 sweep) and the earlier upfront arm's otherwise. "
             r"$^\S$Qwen2.5 ours ran at batch size 1 against batch-16 bounds "
             r"(measured equivalent); the 25\% arm excludes 8/8811 images (0.09\%, a since-fixed "
             r"driver defect) with bounds restricted to the same kept set -- the 50\% arm has "
             r"full coverage. \textsuperscript{\P}122B-FP8 accuracy cells are PARENTHESIZED and "
             r"unshaded: they are a reduced-scale probe (n$=$240 per arm) measured under the "
             r"Triton finegrained-fp8 fallback (the DeepGEMM path mis-generates on sm\_100), not "
             r"full-split values, and must not be compared 1:1 against full-split cells; compute "
             r"figures are shape-determined and kernel-independent. $\diamond$: the model is not capable of the task itself (floor $\approx$ ceiling at degenerate accuracy), so these rows carry no signal about the method.}")
    L.append(r"\label{tab:evaluation_results}")
    L.append(r"\begin{center}\begin{small}\begin{sc}")
    L.append(r"\resizebox{\textwidth}{!}{%")
    L.append(r"\begin{tabular}{ll" + "c" * (ncol - 2) + "}")
    L.append(r"\toprule")
    L.append(r"\multirow{2}{*}{Model} & \multirow{2}{*}{Dataset (Metric)} & "
             r"\multicolumn{1}{c}{Low-res.} & " + heads +
             r" & \multicolumn{2}{c}{Full-res.} \\")
    L.append(r"\cmidrule(lr){3-3} " + " ".join(cmids))
    L.append(r"& & Acc. (\%) & " + sub + r" \\")
    L.append(r"\midrule")
    # Section boundaries get a DOUBLE rule (user 2026-09-01): before Gemma 3 (VFM->VLM),
    # before LLaVA-OV2 (Gemma family -> the rest), and before OpenVLA (VLM -> VLA).
    DOUBLE_RULE_BEFORE = ("Gemma 3", "LLaVA-OV2", "OpenVLA")
    for i, (model, rows) in enumerate(table):
        if i:
            L.append(r"\midrule\midrule" if model.startswith(DOUBLE_RULE_BEFORE)
                     else r"\midrule")
        # "Name (size)" wraps to two lines inside the multirow cell -- the size (and any footnote
        # marks after it) drops to the second line, keeping the model column narrow.
        m = re.match(r"^(.*?) (\(.*)$", model)
        cell = f"\\shortstack[l]{{{m.group(1)}\\\\{m.group(2)}}}" if m else model
        L.append(f"\\multirow{{{len(rows)}}}{{*}}{{{cell}}}")
        for label, cells in rows:
            L.append(f"& {label} & " + " & ".join(cells) + r" \\")
    L.append(r"\bottomrule")
    L.append(r"\end{tabular}}")
    L.append(r"\end{sc}\end{small}\end{center}")
    L.append(r"\vspace{-0.22in}")
    L.append(r"\end{table*}")
    return "\n".join(L)


def emit_md(table, keeps) -> str:
    hdr = ["model", "dataset", "low-res"]
    for k in keeps:
        hdr += [f"ours{int(k*100)} acc", f"ours{int(k*100)} comp", f"ours{int(k*100)} crit"]
    hdr += ["full acc", "full comp"]
    L = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for model, rows in table:
        for label, cells in rows:
            L.append("| " + " | ".join([model, label] +
                                       [c.replace(SHADE_MACRO, "").replace("\\,", " ")
                                         .replace("\\%", "%")
                                        for c in cells]) + " |")
    return "\n".join(L)


def emit_status(keeps, groups: int) -> str:
    L, missing, total = [], 0, 0
    for model, rows in SPEC:
        for label, acc_key, fl_key in rows:
            if acc_key is None:
                continue
            for tag in ["floor", "ceiling"] + [f"interleaved_g{groups}_k{k:.2f}" for k in keeps]:
                total += 1
                if load_accuracy(*acc_key, tag) is None:
                    missing += 1
                    L.append(f"  MISSING  {acc_key[0]}/{acc_key[1]}/{tag}")
    L.append(f"\n  accuracy cells: {total - missing}/{total} present")
    return "\n".join(L)


def emit_overhead(keeps) -> str:
    hdr = ["model", "dataset", "full (TF)"]
    for k in keeps:
        hdr += [f"total{int(k*100)} (TF)", f"x full"]
    L = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for model, rows in SPEC:
        for label, _, fl_key in rows:
            full = get_flops(fl_key, "full")
            cells = [model, label, f"{full/1000:.2f}" if full else "--"]
            for k in keeps:
                t = get_total(fl_key, f"k{k:.2f}")
                cells += [f"{t/1000:.2f}" if t else "--",
                          f"{t/full:.2f}x" if (t and full) else "--"]
            L.append("| " + " | ".join(cells) + " |")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keeps", type=float, nargs="+", default=[0.25, 0.50])
    ap.add_argument("--groups", type=int, default=4)
    ap.add_argument("--format", choices=["latex", "md"], default="latex")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--overhead", action="store_true",
                    help="report TOTAL compute (approx + all corrections) against the ceiling, "
                         "i.e. what the schedule costs rather than what it defers")
    a = ap.parse_args()
    if a.status:
        print(emit_status(a.keeps, a.groups))
        return
    if a.overhead:
        print(emit_overhead(a.keeps))
        return
    table = build_rows(a.keeps, a.groups)
    print(emit_latex(table, a.keeps) if a.format == "latex" else emit_md(table, a.keeps))


if __name__ == "__main__":
    main()
