# Phase 6 results — async multi-stream pipeline

Environment: Colab T4 (`compute_75`), 2 vCPUs. Harness: `imgjit-bench` driving `imgjit-server`
over loopback, `--no-echo` (request-only). The A/B is one binary, `--streams 1` (the Phase 5
pipeline — one frame on the GPU at a time) vs `--streams 4`, everything else held equal.
`--streams 1` here *is* the Phase 5 baseline, re-measured under the same harness. Reproduce with
cells 16 and 18 of `colab/run.ipynb`.

## 1. Overlap is real — the occupancy counter

The backend samples the number of frames resident on the GPU on every launch:

| `--streams` | mean in flight | peak | submit stalls (of 1200) |
|---|---|---|---|
| 1 | 1.00 | 1 | ~1199 |
| 4 | 2.61 | 4 | 319 |

`--streams 1` is strictly serial by construction; `--streams 4` keeps ~2.6 frames overlapping.

## 2. Overlap is real — the Nsight trace

`bench/phase6_timeline.png` is the first ~55 ms of the GPU timeline under `--streams 4`, one row
per stream, bars coloured H2D / kernel / D2H. `bench/phase6_gpu_trace.csv` is the full
per-operation trace it was built from (`nsys export --type sqlite`, 4800 ops over a 3.9 s run:
1200 H2D, 2400 kernel — the chain fuses to **two** stages, so two launches per frame — 1200 D2H).

Sweep-line over the trace:

| | ms | of GPU-busy |
|---|---|---|
| GPU busy (≥1 op) | 3278 | — |
| ≥2 ops concurrent | 556 | **17%** |
| GPU idle | 655 (of 3933 wall) | 17% wall |

**The 17% is the whole copy budget, hidden.** H2D + D2H total 1200 × (0.27 + 0.24) ms ≈ 612 ms;
539 ms of that runs concurrently with a kernel on another stream. The copies have been moved off
the critical path almost entirely — which is exactly the +18% throughput in §3.

**Kernels do not overlap kernels** (time at concurrency ≥3 is ~17 ms, ≈ 0). A 1024² image kernel
already occupies every SM on the T4, so two cannot run at once — the compute engine time-slices
them. Streams buy copy/compute overlap here, not compute/compute, and for a compute-bound chain
that ceiling is ~the copy fraction.

**GPU utilisation is 83%**, and the idle is front-loaded: the first four gaps (10.3, 7.3, 5.9,
4.7 ms) are the client filling its window and the reader threads draining the initial burst;
steady-state gaps are ~2 ms every ~24 ms. A faster client (this one is 4 threads on 2 vCPUs)
would close most of the remaining 17%.

## 3. Throughput and latency

`bench/baseline_phase6.csv`:

| label | fps | p50 ms | p99 ms | max ms |
|---|---|---|---|---|
| `phase6_streams1_full`  | 263.4 | 111.5 | 202.1 | 219.3 |
| `phase6_streams4_full`  | 310.6 | 102.5 | 129.3 | 135.3 |
| `phase6_streams1_cheap` | 590.3 |  52.2 |  66.0 |  67.4 |
| `phase6_streams4_cheap` | 551.9 |  56.3 |  77.8 |  92.7 |

**Full chain (`grayscale,gaussian:1.4,sobel,threshold:0.3`): +18% throughput, −36% p99, −38%
max, at `--streams 4`.** This chain is compute-bound (`gaussian:1.4` is ~11×11 taps per pixel),
so the streams overlap the PCIe copies behind the kernel time. That is the Phase 6 win.

**Cheap chain (`invert`): `--streams 4` is ~6% *slower*.** A single pointwise op is ~1.7 ms per
frame end to end, and almost all of that is host work — the worker thread's staging-buffer
`memcpy` out of pinned memory, event polling, completion routing — not GPU time. There is nothing
for the streams to overlap, and the extra per-frame stream bookkeeping (4 events to poll, 4 slots
to sweep) costs ~120 µs. The ~120 µs delta ≈ one worker poll interval, and it disappears into
noise the moment the kernel does real work (the full chain above). `K=4` is tuned for realistic
multi-op chains; `--streams 1` is the right choice for a pure-passthrough workload.

## 4. Correctness

`phase6_gpu_async` (multi-stream residency + per-handle answers) and `phase6_gpu_recovery`
(context recreation fails in-flight work with status 6, flushes the cache, resumes; frame slots
survive) pass on the T4. `phase6_gpu_stress` (800 frames, zero checksum drift under slot
recycling) passes after the oracle tolerance was corrected for `sobel`'s float divergence — see
the comment in `tests/test_gpu_pipeline.cpp`. Portable suite green on macOS in plain / ASan / TSan.
