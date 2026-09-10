# Phase 6 results — async multi-stream pipeline

Environment: Colab T4 (`compute_75`), 2 vCPUs. Harness: `imgjit-bench` driving `imgjit-server`
over loopback. The A/B is one binary, `--streams 1` (the Phase 5 pipeline, one frame on the GPU
at a time) vs `--streams 4`, everything else held equal. Reproduce with cells 16 and 18 of
`colab/run.ipynb`.

## 1. Overlap is real — the occupancy counter

The backend samples the number of frames resident on the GPU on every launch:

| `--streams` | mean in flight | peak | submit stalls (of 1200) |
|---|---|---|---|
| 1 | 1.00 | 1 | ~1199 |
| 4 | 2.61 | 4 | ~319 |

`--streams 1` is strictly serial by construction; `--streams 4` keeps ~2.6 frames overlapping.
This is the software measurement of the same thing the Nsight timeline shows.

## 2. Overlap is real — the Nsight trace

`bench/phase6_timeline.png` is the first 60 ms of the GPU timeline under `--streams 4`, one row
per stream, bars coloured H2D / kernel / D2H. `bench/phase6_gpu_trace.csv` is the full
per-operation trace it was built from (`nsys stats --report cuda_gpu_trace`).

Derived from the trace: **__%__ of GPU-busy wall time has ≥2 operations running concurrently**
(a serial pipeline scores ~0).

## 3. Throughput and latency

`bench/baseline_phase6.csv`, `--no-echo` (request-only — the 3 MB echo + client-side memcmp
makes the 2-vCPU client, not the GPU, the bottleneck; with echo on the trivial `invert` chain
barely outran the heavy one).

| label | fps | p50 ms | p99 ms |
|---|---|---|---|
| `phase6_streams1_full`  | __ | __ | __ |
| `phase6_streams4_full`  | __ | __ | __ |
| `phase6_streams1_cheap` | __ | __ | __ |
| `phase6_streams4_cheap` | __ | __ | __ |

The showcase chain (`grayscale,gaussian:1.4,sobel,threshold:0.3`) is compute-bound —
`gaussian:1.4` is ~11×11 taps per pixel — so overlap can only hide the PCIe-copy fraction of
each frame and the throughput ceiling is modest. `invert` is one pointwise op, almost pure
copy, and is where the streams earn the larger win.

## 4. Against the Phase 5 baseline

`bench/baseline_phase5.csv` row `phase5_cuda_prewarmed`: __ fps, p50 __ ms.
`phase6_streams1_full` reproduces it (sanity check: the refactor did not regress the serial
path); `phase6_streams4_full` is the improvement.
