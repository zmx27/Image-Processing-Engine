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

`bench/phase6_timeline.png` is the first 60 ms of the GPU timeline under `--streams 4`, one row
per stream, bars coloured H2D / kernel / D2H. `bench/phase6_gpu_trace.csv` is the per-operation
trace it was built from (`nsys export --type sqlite`, kernel + memcpy activity).

Derived from the trace by sweep-line: **__%__ of GPU-busy wall time has ≥2 operations running
concurrently** (a serial pipeline scores ~0).

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

All three GPU gates green on the T4: `phase6_gpu_async` (multi-stream residency + per-handle
answers), `phase6_gpu_stress` (800 frames, zero checksum drift under slot recycling),
`phase6_gpu_recovery` (context recreation fails in-flight work with status 6, flushes the cache,
resumes; frame slots survive). Portable suite green on macOS in plain / ASan / TSan.
