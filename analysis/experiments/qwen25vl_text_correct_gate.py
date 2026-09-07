"""Gate: the text-correction schedule change in qwen25vl_executor.correct_forward (pre-image text
never corrected, post-image text on the final round only) must leave the final prefill state and
first-token logits unchanged vs the previous schedule (pre-image text corrected every round).

Both schedules run on ONE loaded model: the old executor class (from `--old-module`, a copy of
the previous file) is instantiated without __init__ and shares the new executor's __dict__.

Run (GPU0, appcorr env, HF offline):
  python analysis/experiments/qwen25vl_text_correct_gate.py --old-module <path/to/old_executor.py>
"""
import argparse, importlib.util, json, os, sys
import numpy as np, torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT); sys.path.insert(0, os.path.join(REPO_ROOT, "analysis"))
import analysis.experiments.flops_report_qwen25vl_arms as R  # noqa: E402 (BASE_CONFIG, get_spec)


def run(executor, encoder, raw_config, image_np, prompt, keep):
    from offload.common import ExperimentConfig, Task
    cfg = dict(raw_config)
    cfg["image_shape"] = [int(image_np.shape[0]), int(image_np.shape[1]), 3]
    cfg["transmission_kwargs"]["grouping_strategy"] = "sequential"
    cfg["transmission_kwargs"]["num_groups"] = 4
    cfg.setdefault("appcorr_kwargs", {})["token_keep_ratio"] = keep
    config = ExperimentConfig(**cfg)
    context, buf, canvas = {}, [], None
    for group_patches in encoder.encode(image_np[None], config):
        gid = group_patches[0].group_id
        for p in group_patches:
            p.text_payload = prompt
        buf.extend(group_patches)
        canvas = encoder.decode(buf, config, canvas=canvas)
        task = Task(task_id=0, request_id=0, payload=group_patches, instructions=[])
        executor.preprocess(canvas, task, context, config)
        executor.prepare_tokens(task, context, config)
        if gid == 0:
            executor.approx_forward({"layers": (0, executor.num_llm_layers)}, context, config)
        else:
            executor.correct_forward({"layers": (0, executor.num_llm_layers), "group_id": gid}, context, config)
    h = context["llm_current_feature"]
    hidden = executor.model.model.language_model.norm(h)
    logits = executor.model.lm_head(hidden[:, -1, :].to(executor.model.lm_head.weight.dtype)).float()
    # Stock reference on the final canvas (identical inputs: same input_ids / pixel_values).
    # Captured at the last decoder layer's output (pre-final-norm) -- the point
    # `llm_current_feature` holds; `output_hidden_states[-1]` is not that point on every
    # transformers version, so hook it explicitly.
    mm = context["image_mask_1d"].long().unsqueeze(0)
    cap = {}
    hk = executor.model.model.language_model.layers[-1].register_forward_hook(
        lambda m, i, o: cap.__setitem__("h", (o[0] if isinstance(o, tuple) else o).detach()))
    executor.model(input_ids=context["input_ids"], attention_mask=context["attention_mask"],
                   pixel_values=context["pixel_values"], image_grid_thw=context["image_grid_thw"],
                   mm_token_type_ids=mm, use_cache=False)
    hk.remove()
    h_stock = cap["h"].float().cpu()
    img = context["image_token_positions"].cpu()
    return h.float().cpu(), logits.cpu(), (int(img[0]), int(img[-1]) + 1, h_stock,
                                           context["pixel_values"].float().cpu())


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old-module", required=True)
    ap.add_argument("--datasets", nargs="+", default=["mmvp", "realworldqa", "refcoco"])
    ap.add_argument("--keeps", type=float, nargs="+", default=[0.25, 0.50])
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--model-path", default=None, help="e.g. Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    ap.add_argument("--order", default="old,new",
                    help="comma-separated run order per (sample, keep); every run is compared against "
                         "the FIRST run of the same kind (determinism/state leak) and old[0] vs new[0]")
    a = ap.parse_args()

    from datasets import load_dataset
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
    from offload.common import ExperimentConfig
    from offload.policies import get_transmission
    from offload.server.model.qwen25vl_executor import Qwen25VLExecutor

    spec_ = importlib.util.spec_from_file_location("qwen25vl_executor_old", a.old_module)
    old_mod = importlib.util.module_from_spec(spec_); spec_.loader.exec_module(old_mod)

    raw_config = json.load(open(R.BASE_CONFIG)); raw_config["batch_size"] = 1; raw_config["device"] = a.device
    raw_config.setdefault("dataset_kwargs", {})["model_dtype"] = a.dtype
    if a.model_path:
        raw_config["dataset_kwargs"]["model_path"] = a.model_path
    new = Qwen25VLExecutor(torch.device(a.device))
    new.load_model("qwen25vl_realworldqa", ExperimentConfig(**raw_config))
    old = old_mod.Qwen25VLExecutor.__new__(old_mod.Qwen25VLExecutor)
    old.__dict__ = new.__dict__  # one model, two correct_forward schedules
    ip = new.processor.image_processor
    min_px, max_px = ip.size["shortest_edge"], ip.size["longest_edge"]
    factor = ip.patch_size * ip.merge_size * 4
    encoder = get_transmission(raw_config["transmission_policy_name"])

    worst = 0.0; argmax_mismatch = 0; n = 0
    for ds_name in a.datasets:
        spec = R.get_spec(ds_name); ds = spec.load(load_dataset)
        idxs = list(range(0, len(ds), max(1, len(ds) // a.samples)))[:a.samples]
        for i in idxs:
            img, prompt, _ = spec.prepare(ds[i], smart_resize, factor, min_px, max_px)
            image_np = np.array(img, dtype=np.uint8)
            for keep in a.keeps:
                runs = {"old": [], "new": []}
                for kind in a.order.split(","):
                    ex = old if kind == "old" else new
                    h_, lg_, (img0, img1, h_stock, px_) = run(ex, encoder, raw_config, image_np, prompt, keep)
                    runs[kind].append((h_, lg_, px_))
                def region(dh):
                    n_img = img1 - img0; q = n_img // 4
                    parts = [("pre", dh[:, :img0])] + [(f"img{g+1}", dh[:, img0 + g*q: img0 + (g+1)*q]) for g in range(4)] + [("post", dh[:, img1:])]
                    return " ".join(f"{k}={float(v.max()):.2e}" for k, v in parts)
                for kind, rs in runs.items():
                    for j, (h_, lg_, px_) in enumerate(rs[1:], start=1):
                        print(f"    {kind}[{j}] vs {kind}[0]: {region((h_ - rs[0][0]).abs())}  "
                              f"canvas max|d|={float((px_ - rs[0][2]).abs().max()):.2e}", flush=True)
                h_old, lg_old, px_old = runs["old"][0]
                h_new, lg_new, px_new = runs["new"][0]
                print(f"    canvas old vs new max|d|={float((px_old - px_new).abs().max()):.2e}", flush=True)
                print(f"    vs STOCK  old: {region((h_old - h_stock).abs())}", flush=True)
                print(f"    vs STOCK  new: {region((h_new - h_stock).abs())}", flush=True)
                print(f"    old vs new   : {region((h_old - h_new).abs())}", flush=True)
                dh = (h_old - h_new).abs()
                # bf16 ulp of the reference value: differences within 1 ulp are kernel-shape
                # rounding (Q=19 vs Q=4 GEMMs), not a schedule difference. Qwen2.5's top-layer
                # state carries ~1e4-magnitude outlier channels, so absolute max|d| alone misleads.
                ulp = 2.0 ** (torch.floor(torch.log2(h_old.abs().clamp_min(1e-30))) - 7)
                over = (dh > 2 * ulp)
                d_pre = float(dh[:, :img0].max()); d_rest = float(dh[:, img0:].max())
                x_pre = float(h_old[:, :img0].abs().max()); x_rest = float(h_old[:, img0:].abs().max())
                frac_over = float(over.float().mean())
                dl = float((lg_old - lg_new).abs().max()); lmax = float(lg_old.abs().max())
                top5_same = bool(torch.equal(lg_old.topk(5).indices, lg_new.topk(5).indices))
                same = bool(lg_old.argmax(-1).item() == lg_new.argmax(-1).item())
                worst = max(worst, dl); argmax_mismatch += (not same); n += 1
                print(f"[{ds_name} idx={i} keep={keep}] N={h_old.shape[1]} img0={img0} "
                      f"hidden max|d|/max|x| pre-image={d_pre:.3e}/{x_pre:.3e} rest={d_rest:.3e}/{x_rest:.3e} "
                      f"frac(|d|>2ulp)={frac_over:.2e}  logits max|d|={dl:.3e} (max|logit|={lmax:.1f}) "
                      f"argmax {'same' if same else 'DIFF'} top5 {'same' if top5_same else 'DIFF'}",
                      flush=True)
    print(f"GATE: {n} runs, worst logit max|d|={worst:.3e}, argmax mismatches={argmax_mismatch}", flush=True)


if __name__ == "__main__":
    main()
