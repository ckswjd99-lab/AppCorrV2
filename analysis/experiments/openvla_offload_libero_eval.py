"""
openvla_offload_libero_eval.py

Runs the OpenVLA progressive prefill through the REAL offload pipeline (SchedulerModule +
WorkerModule via multiprocessing queues -- everything from offload/server/main.py minus the TCP
transport), stepping actual LIBERO episodes. The driver plays the mobile side: per control step it
center-crops the camera frame, encodes it with VLAPatchCanvasPolicy (base layer + residual-ranked
patch groups), pushes the patches, and consumes the InferenceResult -- which carries the full
server_events CUDA-event timeline, giving per-op (APPROX/CORRECT/HEAD/...) timing through the
existing monitoring machinery.

Schedules (scheduler_kwargs['schedule'] on VLAInterleavedStatic): interleaved | sequential | full.

Run (from repo root, `openvla` conda env, LIBERO software-EGL env vars set):
    MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=2 MUJOCO_EGL_ALLOW_ANY_DEVICE=1 USE_TF=0 USE_TORCH=1 \
    python analysis/experiments/openvla_offload_libero_eval.py --task-ids all --schedules interleaved,sequential,full
"""

import argparse
import copy
import json
import math
import multiprocessing
import queue as queue_mod
import sys
import time
from collections import defaultdict
from pathlib import Path

APPCORR_ROOT = Path(__file__).resolve().parents[2]
OPENVLA_ROOT = Path("/NHNHOME/share/cjpark/openvla")
for p in (APPCORR_ROOT, OPENVLA_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="openvla/openvla-7b-finetuned-libero-spatial")
    parser.add_argument("--task-suite", type=str, default="libero_spatial")
    parser.add_argument("--task-ids", type=str, default="0")
    parser.add_argument("--num-trials", type=int, default=1)
    parser.add_argument("--trial-start", type=int, default=0,
                         help="Index of the first LIBERO initial state to use (trial t -> "
                              "initial_states[trial_start + t]). Lets two processes cover disjoint "
                              "state ranges of the same schedule in parallel across GPUs.")
    parser.add_argument("--max-steps", type=int, default=220)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--num-groups", type=int, default=4)
    parser.add_argument("--grouping", type=str, default="rank", choices=["rank", "sequential", "energy"],
                         help="VLAPatchCanvasPolicy transmission_kwargs['grouping'].")
    parser.add_argument("--coverage", type=float, default=1.0)
    parser.add_argument("--base-factor", type=int, default=4)
    parser.add_argument("--schedules", type=str, default="interleaved,sequential,full")
    parser.add_argument("--frontiers", type=str, default=None,
                         help="Comma-separated per-group LLM correction depths for the 'interleaved' "
                              "schedule, e.g. '32,32,32,32' = each arriving group is prefilled to FULL "
                              "depth (per-group causal chunked prefill). Omit for uniform default "
                              "frontiers (depth-interleaved).")
    parser.add_argument("--result-timeout", type=float, default=180.0)
    parser.add_argument("--server-pscore-threshold", type=float, default=None,
                         help="Gate CORRECT_FORWARD by the fused attn x residual score "
                              "(OpenVLAExecutor._filter_by_server_pscore); unset = no filtering. "
                              "Units match progressive_vla_libero_eval.py's attnthresh_<N> (real "
                              "value, e.g. 200e-6 for the validated attnthresh_200 setting).")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="torch device for the OpenVLA server model (MuJoCo rendering is EGL/llvmpipe, unaffected)")
    parser.add_argument("--out", type=str, default=None,
                         help="Path of a JSONL file that receives one line per finished episode "
                              "(appended as it lands) plus a final 'summary' line per schedule. "
                              "Unset = print only.")
    parser.add_argument("--sdpa-query-bucket-size", type=int, default=0,
                         help="Pad CORRECT_FORWARD's query-set size up to a multiple of this "
                              "(OpenVLAProgressiveModel._bucketize_token_idx); 0 = disabled. "
                              "Fixes the cuBLAS/SDPA per-shape one-time cost triggered by a "
                              "data-dependent query size (e.g. under server_pscore_threshold).")
    return parser.parse_args()


def center_crop_resize(img_np: np.ndarray, crop_scale: float = 0.9) -> np.ndarray:
    image = Image.fromarray(img_np).convert("RGB")
    w, h = image.size
    nh, nw = h * math.sqrt(crop_scale), w * math.sqrt(crop_scale)
    image = image.crop(((w - nw) / 2, (h - nh) / 2, (w + nw) / 2, (h + nh) / 2)).resize((224, 224), Image.BILINEAR)
    return np.asarray(image, dtype=np.uint8)


def make_config(args, schedule: str):
    from offload.common.protocol import ExperimentConfig

    scheduler_kwargs = {"schedule": schedule, "total_layers": 32}
    if args.server_pscore_threshold is not None:
        scheduler_kwargs["server_pscore_threshold"] = args.server_pscore_threshold
    if args.sdpa_query_bucket_size:
        scheduler_kwargs["sdpa_query_bucket_size"] = args.sdpa_query_bucket_size
    if getattr(args, "frontiers", None):
        scheduler_kwargs["frontiers"] = [int(f) for f in args.frontiers.split(",")]

    return ExperimentConfig(
        exp_id=f"vla_{schedule}",
        model_name="openvla",
        device=args.device,
        dataset_name=args.task_suite,
        dataset_kwargs={"checkpoint": args.checkpoint, "unnorm_key": args.task_suite},
        batch_size=1,
        image_shape=(224, 224, 3),
        patch_size=(14, 14),
        scheduler_policy_name="VLAInterleavedStatic",
        transmission_policy_name="VLAPatchCanvas",
        scheduler_kwargs=scheduler_kwargs,
        transmission_kwargs={
            "num_groups": args.num_groups,
            "grouping": args.grouping,
            "coverage": args.coverage,
            "base_factor": args.base_factor,
            "text": "",
        },
    )


def _append_jsonl(path, record):
    """Append one record to the JSONL results file (no-op when --out is unset). Written per
    episode so a crash/reboot keeps everything that already landed."""
    if not path:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")
        f.flush()


def request_action(encoder, config, frame_np, text, sched_q, result_q, timeout):
    """Mobile side of one control step: encode + push all groups, wait for the InferenceResult."""
    config.transmission_kwargs["text"] = text
    now = time.time()
    for group_patches in encoder.encode(frame_np, config):
        for p in group_patches:
            p.arrival_time = now
        for p in group_patches:
            sched_q.put(p)
    result = result_q.get(timeout=timeout)
    action = np.array(result.output[0], dtype=np.float64)
    return action, result.server_events


def run_episode(task_suite_name, task_id, args, encoder, config, sched_q, result_q, op_times, correction_stats,
                 correct_by_group=None, init_state_idx=0):
    from libero.libero import benchmark

    from experiments.robot.libero.libero_utils import get_libero_dummy_action, get_libero_env, get_libero_image
    from experiments.robot.robot_utils import invert_gripper_action, normalize_gripper_action

    task_suite = benchmark.get_benchmark_dict()[task_suite_name]()
    task = task_suite.get_task(task_id)
    initial_states = task_suite.get_task_init_states(task_id)
    env, task_description = get_libero_env(task, "openvla", resolution=256)
    env.reset()
    # Each trial uses a DISTINCT LIBERO initial state so trials are independent samples
    # (a greedy policy replays an identical episode from the same init state, so reusing
    # initial_states[0] would make num_trials>1 pure duplication).
    obs = env.set_init_state(initial_states[init_state_idx % len(initial_states)])

    t, done = 0, False
    while t < args.max_steps + args.num_steps_wait:
        if t < args.num_steps_wait:
            obs, _, done, _ = env.step(get_libero_dummy_action("openvla"))
            t += 1
            continue
        frame = center_crop_resize(get_libero_image(obs, 224))
        action, server_events = request_action(
            encoder, config, frame, task_description, sched_q, result_q, args.result_timeout
        )
        for ev in server_events:
            op_times[ev["type"]].append(ev["end"] - ev["start"])
            meta = ev.get("meta") or {}
            if "num_arrived" in meta and "num_corrected" in meta:
                correction_stats["arrived"] += meta["num_arrived"]
                correction_stats["corrected"] += meta["num_corrected"]
            if correct_by_group is not None and ev["type"] == "CORRECT_FORWARD":
                gid = (ev.get("params") or {}).get("group_id")
                if gid is not None:
                    correct_by_group[gid].append(ev["end"] - ev["start"])
        action = normalize_gripper_action(action, binarize=True)
        action = invert_gripper_action(action)
        obs, _, done, _ = env.step(action.tolist())
        if done:
            break
        t += 1
    env.close()
    return done, t


def main():
    args = parse_args()
    schedules = args.schedules.split(",")

    from offload.policies import get_transmission
    from offload.server.scheduler import SchedulerModule
    from offload.server.worker import WorkerModule

    sched_q = multiprocessing.Queue()
    worker_q = multiprocessing.Queue()
    result_q = multiprocessing.Queue()
    control_q = multiprocessing.Queue()
    feedback_q = multiprocessing.Queue()
    scheduler = SchedulerModule(sched_q, worker_q, control_q, feedback_q)
    worker = WorkerModule(worker_q, result_q, feedback_q)
    scheduler.start()
    worker.start()

    if args.task_ids.strip().lower() == "all":
        from libero.libero import benchmark

        task_ids = list(range(benchmark.get_benchmark_dict()[args.task_suite]().n_tasks))
    else:
        task_ids = [int(x) for x in args.task_ids.split(",")]
    print(f"[driver] Task ids: {task_ids}, schedules: {schedules}")

    encoder = get_transmission("VLAPatchCanvas")
    results = {}
    timing = {}
    correction = {}
    group_timing = {}
    try:
        for schedule in schedules:
            config = make_config(args, schedule)
            control_q.put(("CONFIG", config))
            time.sleep(1.0)  # let CONFIG reach the worker ahead of the first patches

            # Offline warmup: a schedule change reloads the model fresh (worker.py's CONFIG
            # handler always rebuilds the executor), so the bucket-shape warmup
            # (OpenVLAProgressiveModel._maybe_warmup_llm_correct_buckets) and any other
            # first-call cold-start cost (cuDNN/cuBLAS heuristics, CUDA kernel loading) would
            # otherwise land inside the first *timed* episode. Run one short, throwaway episode
            # first and discard its stats entirely -- this is the "offline phase" the real
            # per-op latency breakdown should not include.
            print(f"\n[driver] === Schedule: {schedule}: offline warmup episode (discarded) ===")
            warmup_args = copy.copy(args)
            warmup_args.max_steps = min(args.max_steps, 15)
            t0 = time.time()
            run_episode(
                args.task_suite, task_ids[0], warmup_args, encoder, config, sched_q, result_q,
                defaultdict(list), defaultdict(int),
            )
            print(f"    warmup episode done, wall={time.time() - t0:.1f}s (excluded from all stats below)")

            op_times = defaultdict(list)
            correction_stats = defaultdict(int)
            correct_by_group = defaultdict(list)
            outcomes = []
            print(f"\n[driver] === Schedule: {schedule} ===")
            for task_id in task_ids:
                for trial in range(args.num_trials):
                    t0 = time.time()
                    try:
                        success, steps = run_episode(
                            args.task_suite, task_id, args, encoder, config, sched_q, result_q,
                            op_times, correction_stats, correct_by_group,
                            init_state_idx=args.trial_start + trial,
                        )
                    except queue_mod.Empty:
                        print(f"    task {task_id}: TIMEOUT waiting for InferenceResult -- aborting schedule")
                        raise
                    outcomes.append((task_id, success))
                    print(f"    task {task_id} trial {trial + 1}: success={success} steps={steps} wall={time.time() - t0:.1f}s")
                    sys.stdout.flush()
                    _append_jsonl(args.out, {
                        "kind": "episode", "suite": args.task_suite, "schedule": schedule,
                        "task_id": task_id, "trial": trial, "init_state_idx": args.trial_start + trial,
                        "success": bool(success), "steps": int(steps), "wall_s": round(time.time() - t0, 1),
                        "grouping": args.grouping, "num_groups": args.num_groups,
                        "frontiers": args.frontiers, "max_steps": args.max_steps,
                    })
            results[schedule] = outcomes
            timing[schedule] = {k: (float(np.mean(v)), len(v)) for k, v in op_times.items()}
            _append_jsonl(args.out, {
                "kind": "summary", "suite": args.task_suite, "schedule": schedule,
                "n": len(outcomes), "successes": int(sum(s for _, s in outcomes)),
                "success_rate": float(sum(s for _, s in outcomes)) / max(1, len(outcomes)),
                "op_time_ms": {k: [v[0] * 1000, v[1]] for k, v in timing[schedule].items()},
                "correction": dict(correction_stats),
            })
            correction[schedule] = dict(correction_stats)
            group_timing[schedule] = {g: (float(np.mean(v)), len(v)) for g, v in correct_by_group.items()}
    finally:
        control_q.put(("STOP", None))
        result_q.cancel_join_thread()
        for proc in (scheduler, worker):
            proc.join(timeout=15)
            if proc.is_alive():
                proc.terminate()

    print("\n[driver] === Summary: success rate by schedule ===")
    for schedule, outcomes in results.items():
        succ = [s for _, s in outcomes]
        print(f"    {schedule:12s}  {sum(succ)}/{len(succ)}  ({sum(succ) / len(succ):.0%})")

    print("\n[driver] === Mean per-op GPU time (ms) from server_events ===")
    for schedule, ops in timing.items():
        parts = "  ".join(f"{k}={v[0] * 1000:.1f}({v[1]})" for k, v in sorted(ops.items()))
        print(f"    {schedule:12s}  {parts}")

    print("\n[driver] === APPROX_FORWARD vs CORRECT_FORWARD ===")
    for schedule, ops in timing.items():
        a = ops.get("APPROX_FORWARD")
        c = ops.get("CORRECT_FORWARD")
        if a and c:
            ratio = c[0] / a[0] if a[0] else float("nan")
            print(f"    {schedule:12s}  APPROX={a[0]*1000:.2f}ms({a[1]})  "
                  f"CORRECT={c[0]*1000:.2f}ms({c[1]})  CORRECT/APPROX={ratio:.2f}x")

    print("\n[driver] === Per-task breakdown ===")
    for schedule, outcomes in results.items():
        line = "  ".join(f"t{tid}={'1' if s else '0'}" for tid, s in outcomes)
        print(f"    {schedule:12s}  {line}")

    print("\n[driver] === LLM recompute-token rate (server_pscore_threshold filter) ===")
    for schedule, stats in correction.items():
        arrived, corrected = stats.get("arrived", 0), stats.get("corrected", 0)
        rate = corrected / arrived if arrived else float("nan")
        print(f"    {schedule:12s}  corrected={corrected}/{arrived} patch-rounds  ({rate:.0%})")

    print(f"\n[driver] === CORRECT_FORWARD latency by group_id (num_groups={args.num_groups}) ===")
    for schedule, groups in group_timing.items():
        for gid in sorted(groups):
            mean_s, n = groups[gid]
            print(f"    {schedule:12s}  group {gid}: {mean_s * 1000:.2f}ms  (n={n})")


if __name__ == "__main__":
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
