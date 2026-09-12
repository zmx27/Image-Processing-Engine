# Benchmarks

`imgjit-bench` is the timing harness introduced in Phase 5 (`docs/PLAN.md`). It drives a running
`imgjit-server` over a socket with `--connections` concurrent clients, each holding `--window`
frames in flight, and reports FPS plus p50/p99 round-trip latency. Phase 8 widens the same binary
over the naive|tiled × sync|async matrix rather than starting a second one, so every number Phases
6 and 7 are gated against was produced by the same instrument.

## Why the numbers here matter

Phases 6 and 7 are both gated on being *measurably faster*. That is only a claim if something
recorded a number first, under a harness that still exists when the comparison is made. This
directory is that record: `baseline_phase5.csv` is the single-stream, synchronous, naive-kernel
starting point, and every later row is compared against it like-for-like.

## Recording the Phase 5 baseline

Needs a real NVIDIA GPU, so this runs on Colab (`colab/run.ipynb` has it as a cell). Two runs,
differing only in whether the chain was prewarmed:

    ./build/tools/imgjit-server --port 9000 --backend cuda \
        --slots 8 --slots-per-conn 2 --queue 16 --max-payload 16777216 &

    ./build/bench/imgjit-bench --port 9000 --connections 4 --frames 200 \
        --width 1024 --height 1024 --channels 3 \
        --ops "grayscale,gaussian:1.4,sobel,threshold:0.3" \
        --label phase5_cuda_cold --csv bench/baseline_phase5.csv

Then restart the server with `--prewarm "grayscale,gaussian:1.4,sobel,threshold:0.3"` and rerun
with `--label phase5_cuda_prewarmed`. The two rows differ in `cold_first_ms` and nowhere else,
which is what turns `docs/ARCHITECTURE.md`'s "NVRTC cold-start is ~50–200 ms, exactly why the cache
exists" from an assertion into a measurement.

A CPU-backend row (`--backend cpu`, no prewarm) is worth recording alongside them. It is not a
fair GPU comparison — it is the scalar oracle — but it is the only row that can be reproduced on a
Mac, which makes it the one that catches a harness regression without a GPU.

## Recording the Phase 6 comparison

Phase 6 made the pipeline asynchronous across K streams. The A/B is **two runs of the same
server binary**, differing only in `--streams`, so nothing about the comparison depends on a build
that no longer exists:

    # the Phase 5 pipeline: one frame on the GPU at a time
    ./build/tools/imgjit-server --port 9000 --backend cuda --streams 1 \
        --slots 32 --slots-per-conn 8 --queue 32 --max-payload 16777216 \
        --prewarm "grayscale,gaussian:1.4,sobel,threshold:0.3" &

    ./build/bench/imgjit-bench --port 9000 --connections 4 --window 8 --frames 300 --no-echo \
        --width 1024 --height 1024 --channels 3 \
        --ops "grayscale,gaussian:1.4,sobel,threshold:0.3" \
        --label phase6_streams1_full --csv bench/baseline_phase6.csv

Then restart with `--streams 4` and rerun with `--label phase6_streams4_full`. Same slots, same
queue, same payload, same chain — only the number of frames the GPU may hold at once changes.
`bench/phase6_results.md` collects the numbers, the occupancy counter and the Nsight artifacts.

**The `--window`, `--no-echo` and slot counts here are not the Phase 5 baseline's, on purpose.**
`imgjit-bench` is a closed loop, and on a 2-vCPU Colab box two things cap it below the GPU:
`--window 2` / `--slots-per-conn 2` lets only 8 frames exist at once (`4 conns * 2 / round-trip ≈
300 fps`), and the 3 MB echo + client-side `memcmp` per frame makes the *client* the bottleneck —
with echo on, the trivial `invert` chain barely outran the heavy one, which is the tell. `--window
8` with 32 slots and `--no-echo` puts the GPU back in the critical path, which is the only regime
where `--streams` changes the answer.

**Record a cheap chain too** (`--ops invert`, same sizes, labels `..._cheap`). The full chain is
compute-bound — `gaussian:1.4` is ~11x11 taps per pixel — so overlap can only hide the ~20% of the
frame that is PCIe copy, and the ceiling is roughly 1.2-1.4x. `invert` is one pointwise op, almost
pure copy, and that is where the streams earn their keep.

**Read the server's shutdown line as well as the CSV.** It reports mean and peak stream occupancy,
and that is the half of the phase gate throughput cannot answer: a run whose `mean in flight` sits
near 1.0 under `--streams 4` did not overlap anything, and whatever made it faster was not the
pipeline. `mean in flight` well above 1 (≈2.6 on a T4 at 4 streams) is the software proxy for the
Nsight timeline below.

    nsys profile -o phase6 --trace=cuda ./build/tools/imgjit-server --port 9000 \
        --backend cuda --streams 4 ...

The timeline must show H2D, kernel and D2H rows genuinely interleaved across streams, not a single
file of segments with gaps between them. If `nsys` is unavailable (some Colab images ship without
it), the occupancy counter carries the claim and the timeline is produced later on any GPU box.

## Recording the Phase 7 comparison

Phase 7 added the tiled stencil variant. The A/B is again two runs of one server binary, differing
only in `--tile`:

    ./build/tools/imgjit-server --port 9000 --backend cuda --streams 1 --tile naive \
        --slots 32 --slots-per-conn 8 --queue 32 --max-payload 16777216 \
        --prewarm "gaussian:1.4" &

    ./build/bench/imgjit-bench --port 9000 --connections 4 --window 8 --frames 300 --no-echo \
        --width 1024 --height 1024 --channels 3 --ops "gaussian:1.4" \
        --label phase7_gaussian_naive_streams1 --csv bench/baseline_phase7.csv

Then `--tile 16` and `--tile 32`, and the same three for `--ops sobel` and the showcase chain.
`--prewarm` warms the variant the server was started with — the tile is part of the kernel key.
Cell 20 of `colab/run.ipynb` runs the whole matrix and writes both files below.

**The number the gate turns on is the server's `mean kernel`, not the CSV's `fps`.** It is read
off the GPU's own clock — a timing event either side of the stages — so it contains the kernels
and nothing else: no PCIe copy, no host work, no NVRTC compile. That distinction is the whole
reason it exists. A 3×3 `sobel` over 1024² is a sliver of a frame whose end-to-end cost is
dominated by copies and the worker's host work (Phase 6 measured ~1.7 ms/frame for a trivial
chain), so a several-percent change in sobel's kernel time — in either direction — is invisible in
`fps` and only readable off this number. The notebook cell collects the kernel times into
`bench/phase7_kernel_ms.csv`, keyed by the same labels. **Measured result** (`bench/phase7_results.md`):
Gaussian and the fused chain are large wins (up to 3.33x kernel time), tiled Sobel is not — a
small, repeatable ~3% regression, not noise, because a radius-1 apron carries too little redundant
memory traffic to pay back the tiled path's fixed overhead.

**`--streams 1` on purpose.** One frame on the GPU at a time is what makes the kernel time a clean
per-frame number: with several streams the GPU time-slices other frames' kernels into this one's
window (Phase 6: kernels do not overlap kernels on a full-size image), and the number includes
the wait. The two extra rows — the showcase chain, naive vs tile 16, at `--streams 4` — are the
end-to-end view: what tiling is worth on top of the Phase 6 pipeline, where the compute engine is
the bottleneck and a faster kernel should show up in `fps` directly.

## Recording the Phase 8 matrix

Phase 8 widens the same harness over the orthogonal matrix. Cell 21 of `colab/run.ipynb` runs it
as **six focused sweeps** — each varying one axis with the others pinned — rather than one
cartesian product, which at six axes would be hundreds of runs and no clearer about any of them:

| Sweep | Axis | Read off |
|---|---|---|
| `resolution` | 512² → 4K | `fps`, `gb_per_s` |
| `constants` | baked vs parameterized | `mean_kernel_ms` |
| `cache` | cold vs warm, four different sigmas | `nvrtc_compiles`, `cold_first_ms` |
| `tile_x_streams` | naive/tiled × 1/4 streams | `mean_kernel_ms`, `fps` |
| `fusion` | four one-op chains vs one fused chain | `mean_kernel_ms` |
| `chain_length` | 1, 2, 4, 6 ops | `mean_kernel_ms`, `fps` |

It writes two files. `baseline_phase8.csv` is what `imgjit-bench` appends, exactly as in every
earlier phase. `phase8_matrix.csv` joins each of those rows to the configuration the server was
started with plus the two server-side instruments (`mean_kernel_ms`, `nvrtc_compiles`), so the
README results table is built from one file with real axis columns instead of by decoding label
strings — which is what the earlier phases did, and what stops scaling at six axes.

**The `constants` axis has a cost side and a benefit side, and only one of them is a speed.**
Parameterized kernels take each op's parameters as launch arguments rather than baked literals, so
the gaussian's tap loop has bounds NVRTC cannot unroll — that shows up as a slower
`mean_kernel_ms`, measured at `--streams 1` for Phase 7's reason. The payoff is the `cache` sweep:
four connections asking for four *different* sigmas compile four kernels when constants are baked
and **one** when they are parameterized, which is `nvrtc_compiles` and `cold_first_ms`. A run that
only measured kernel time would conclude the mode is strictly worse.

**Parameterized is naive-only** (`--constants parameterized` with `--tile` is refused at startup):
a tiled stage sizes its `__shared__` array from the stencil radius at compile time, which a
parameterized kernel does not know until launch. See `docs/PLAN.md` Phase 8.

**The fusion sweep's unfused baseline is an approximation.** Codegen has no "don't fuse" mode —
fusion is structural — so the stand-in is the same four ops sent as four separate one-op chains,
and the comparison is the *sum of their kernel times* against the fused chain's. Four chains are
also four round trips, which is why the summary compares `mean_kernel_ms` and never `fps`.

**Measured result** (`bench/phase8_results.md`, Colab T4): parameterized constants cost
**1.36-1.42x** kernel time and buy **4 compiles → 1** with a 3.1x lower worst first frame on a
cold cache, break-even ≈110 frames per distinct parameter value. **Tile 16 at one stream is the
fastest configuration in the matrix** (1.80x the naive baseline); streams add +11% on naive and
nothing on top of tiling. Fusion is **1.28x slower** in kernel time for the showcase chain, because
a prologue re-runs at every stencil tap. An NVRTC compile measured ≈97 ms.

**Read §0 of the results before any single number.** One configuration appears in three sweeps, so
it was measured three times: the spread is **3.5% on kernel time, 1.6% on FPS**. That is the
harness's repeatability and the bar a difference must clear to be real — which is also why the
Phase 7 tiled-Sobel result is restated there as "not measurably faster" rather than a regression.

## Reading the columns

| Column | Meaning |
|---|---|
| `fps` | total frames ÷ wall time, across all connections |
| `req_mb_per_s` | request payload only; the echo doubles the bytes actually on the wire |
| `p50_ms` / `p99_ms` / `mean_ms` / `max_ms` | round trip, from `send()` entry to matched response |
| `cold_first_ms` | the worst first frame across connections, excluded from the percentiles |
| `failures` | frames answered with a non-zero status; a baseline row must have 0 |

`cold_first_ms` is reported separately rather than folded in because on the CUDA backend that frame
pays the NVRTC compile unless the chain was prewarmed — averaging a compile into a run that
measures execution would put a 100 ms outlier in the p99 of every cold run and hide the effect the
cache exists to produce.

Round trip is measured per `seq_num`, never by arrival order: under pipelining the Nth response is
not the Nth request, which is the entire reason `seq_num` is in the protocol
(`docs/PROTOCOL.md`). The timestamp lives on the client's pending map
(`include/imgjit/net/client.h`) for that reason.
