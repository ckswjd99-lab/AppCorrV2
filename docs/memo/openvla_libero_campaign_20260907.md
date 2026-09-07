# OpenVLA-7B LIBERO campaign (2026-09-04 .. 09-07)

Full table campaign for the OpenVLA row of the MLSys'27 eval table: 4 LIBERO suites x 3 arms,
10 tasks x 50 trials = 500 episodes per arm, official protocol (max_steps spatial 220 / object 280 /
goal 300 / libero_10 520, num_steps_wait 10, center_crop 0.9). Driver:
`analysis/experiments/openvla_offload_libero_eval.py`, one process per (suite, arm), per-episode
outcomes in `analysis/results/openvla/<suite>_<arm>_t50.jsonl` (kind=episode lines, then one
kind=summary line with op timings).

Arms (offload scheduler `VLAInterleavedStatic`):
- **full** = ceiling (FULL_INFERENCE, stock model)
- **approx** = floor (approx-only pass, no correction)
- **interleaved** = stream: `--frontiers 32,32,32,32 --grouping sequential` (per-group chunked causal
  prefill, g=4)

## Results (success rate, %)

| Suite | Floor | Stream | Ceiling | Stream/Ceiling | Paired stream-ceiling (95% CI) | wins/losses | Paper |
|---|---|---|---|---|---|---|---|
| Spatial | 18.6 | 81.8 | 81.6 | 100.2% | +0.2 [-3.8, +4.2] | 56/55 | 84.7 |
| Object | 19.8 | 84.8 | 86.0 | 98.6% | -1.2 [-5.6, +2.8] | 53/59 | 88.4 |
| Goal | 17.8 | 74.8 | 75.0 | 99.7% | -0.2 [-5.0, +4.8] | 73/74 | 79.2 |
| Long (libero_10) | 2.6 | 45.4 | 51.8 | 87.6% | **-6.4 [-11.8, -0.8]** | 85/117 | 53.7 |

Paired = per (task, trial) pair (same init state), 2000-sample bootstrap of the mean delta.

Per-task successes /50 (task 0..9):

| Suite | Ceiling | Floor | Stream |
|---|---|---|---|
| Spatial | 46,43,43,47,29,45,43,42,36,34 | 9,12,2,14,9,0,18,11,11,7 | 48,45,33,43,38,41,50,46,32,33 |
| Object | 40,40,42,30,45,45,44,49,46,49 | 21,5,7,2,44,1,1,6,9,3 | 41,35,46,29,46,37,45,49,46,50 |
| Goal | 28,42,43,33,45,39,34,50,37,24 | 9,7,11,0,27,0,18,12,1,4 | 21,45,46,24,47,37,38,50,33,33 |
| Long | 26,37,30,12,25,38,25,34,10,22 | 1,1,1,1,0,8,0,1,0,0 | 21,31,26,16,21,33,19,30,11,19 |

Per-task n=50 gives roughly +-7pp per cell; only suite totals are comparable.

## Reading

- Spatial / Object / Goal: stream ties the ceiling (all CIs contain 0). Floor is a consistent 23% of
  ceiling, i.e. the approx pass alone is not a usable policy but the per-group correction recovers
  essentially everything.
- Long: the only suite where stream is below the ceiling beyond noise (-6.4pp, CI excludes 0), and
  the floor collapses to 2.6%. Hypothesis (NOT measured): the 520-step horizon accumulates small
  per-step action deviations that the shorter suites absorb. A per-step action-divergence trace
  (stream vs full on identical observations) would confirm or kill this.
- Ceilings sit 3-4pp under the paper on every suite (81.6/86.0/75.0/51.8 vs 84.7/88.4/79.2/53.7).
  All three arms share the environment (llvmpipe EGL rendering, torch 2.7.1+cu128, numpy 1.26.4), so
  the within-row comparison stands; the absolute offset is an environment effect, not a model one.

## Server op timings (ms per call, from the summary lines)

| | FULL_INFERENCE | PREPARE_TOKENS | APPROX_FORWARD | CORRECT_FORWARD (x4/step) | HEAD_INFERENCE |
|---|---|---|---|---|---|
| full | 88.8-96.3 | - | - | - | - |
| approx | - | 12.9-13.9 | 13.3-13.6 | - | 67.4-71.9 |
| interleaved | - | 11.0-11.7 | 12.5-13.5 | 11.3-12.2 | 64.2-71.7 |

HEAD_INFERENCE (the 7-token action decode) is common to approx and interleaved. Use these op
timings, not episode wall time, for any latency column: Long's approx (cuda:0) and interleaved
(cuda:1, `--device cuda:1`) ran concurrently and shared the CPU for llvmpipe rendering, which
inflated wall time (94 -> 112-150 s per 530-step episode) without touching the GPU op timings.

## Environment note

The whole campaign ran on the rebuilt `openvla` conda env after the numpy-2 / mujoco-2.3.0
`MjvOption` bug was fixed (collision geoms were being rendered; ceiling read 42%). Fix =
`numpy==1.26.4` + `opencv-python==4.10.0.84`. Run recipe:
`env -u CUDA_VISIBLE_DEVICES MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=2 MUJOCO_EGL_ALLOW_ANY_DEVICE=1 USE_TF=0 USE_TORCH=1`.
CUDA_VISIBLE_DEVICES must stay unset (robosuite asserts MUJOCO_EGL_DEVICE_ID is in it); pick the
torch GPU with the driver's `--device` flag instead.

Table literals: `AppCorr-qwen35-eval/analysis/experiments/make_eval_table.py` (OpenVLA rows).
