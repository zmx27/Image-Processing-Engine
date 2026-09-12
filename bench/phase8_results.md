# Phase 8 results — the benchmark matrix

Environment: Colab T4 (`compute_75`), 2 vCPUs. Harness: `imgjit-bench` driving `imgjit-server`
over loopback, `--no-echo` (request-only), `--connections 4 --window 8 --frames 300` at
1024²×3 unless a sweep says otherwise. Reproduce with cell 21 of `colab/run.ipynb`, which writes
both `baseline_phase8.csv` (one row per run, as in every earlier phase) and `phase8_matrix.csv`
(the same rows joined to the server configuration and the two server-side instruments).

Six sweeps, each varying one axis with the rest pinned. A cartesian product of six axes would be
hundreds of runs and no clearer about any one of them.

## 0. Method: the noise floor, and why it is stated first

One configuration — naive, `--streams 1`, 1024², the showcase chain, prewarmed — appears in three
different sweeps, so it was measured three times in one session:

| label | mean kernel | fps |
|---|---|---|
| `phase8_full_naive_streams1` | 2.509 ms | 260.6 |
| `phase8_len4` | 2.546 ms | 256.6 |
| `phase8_fused_full` | 2.598 ms | 260.6 |

**Spread: 3.5% on kernel time, 1.6% on FPS.** That is this harness's repeatability, and it is the
bar every difference below has to clear to mean anything. Two consequences worth stating plainly:

- Differences under ~4% in kernel time are not readable from a single run. **This qualifies Phase 7's
  tiled-Sobel finding** (`bench/phase7_results.md` §1), which called a ~3% regression "consistent,
  not noise" on the strength of two tile sizes landing on the same value. That reasoning still
  holds — two independent configurations agreeing is evidence a single repeat is not — but the
  honest restatement is that tiled Sobel is *not measurably faster*, and the sign of the small
  residual is at the edge of what this harness resolves.
- Cross-session comparison is looser still. The same naive full-chain kernel measured 2.348 ms in
  the Phase 7 session and 2.509–2.598 ms here (+7–10%), on a nominally identical T4. Compare within
  a session; across sessions, compare ratios rather than absolutes.

Every number below is a single run unless stated. They are consistent and the effects that matter
are far larger than the noise floor, but nothing here is an average over repeats.

## 1. Baked vs parameterized constants — this phase's own axis

The axis reserved in the kernel key since Phase 2 and implemented here: op parameters as compiled-in
literals versus kernel arguments passed at launch. **It has a cost side and a benefit side, and only
the cost is a speed** — a run that measured kernel time alone would conclude the mode is strictly
worse and miss the entire point of it.

### 1a. The cost: kernels get 1.36–1.42x slower

`--streams 1`, so `mean_kernel_ms` is a clean per-frame number:

| chain | baked | parameterized | penalty |
|---|---|---|---|
| `gaussian:1.4` | 1.632 ms | 2.310 ms | **1.42x** |
| full chain | 2.430 ms | 3.300 ms | **1.36x** |

End to end that is 354 → 277 fps (−22%) for the Gaussian and 275 → 223 fps (−19%) for the chain.

This is the expected and intended result, and it is what the mode was built to put a number on. A
baked `gaussian:1.4` compiles a loop with literal bounds `-5..5` and 11 constant weights: NVRTC
unrolls all 121 taps and folds the weight products. Parameterized, the bound is an `int radius`
argument and the weights are read from a by-value struct, so the loop stays a loop.

### 1b. The benefit: one compile instead of one per parameter value

The `cache` sweep is the one that earns the mode its keep — four connections asking for four
*different* sigmas (`0.8, 1.4, 2.2, 3`), cold:

| mode | compiles | worst first frame | p99 | fps |
|---|---|---|---|---|
| baked | **4** | 431.6 ms | 468.4 ms | 233.4 |
| parameterized | **1** | **140.1 ms** (−68%) | **198.0 ms** (−58%) | 194.9 |

Four distinct sigmas are four distinct kernels when constants are baked, and one kernel when they
are not. The worst first frame falls 3.1x and p99 falls 2.4x, because on a cold cache the compiles
are in the latency path.

Warm (the same four sigmas prewarmed), the picture inverts, as it should:

| mode | compiles | worst first frame | fps |
|---|---|---|---|
| baked | 4 | **47.8 ms** | 247.7 |
| parameterized | 1 | 77.1 ms | 195.9 |

With every kernel already compiled there is nothing left to amortize, and the slower parameterized
kernel simply loses.

### 1c. The break-even, derived

The cold/warm pair brackets one NVRTC compile: baked pays 431.6 ms for a first frame that costs
47.8 ms warm, over 4 compiles — **≈97 ms per compile**, which lands inside `ARCHITECTURE.md`'s
"~50–200 ms NVRTC cold-start" claim and is the first direct measurement of it. The same arithmetic
on the parameterized rows gives ≈63 ms, a secondary finding that makes sense: with no 121-iteration
unroll to perform, there is less for NVRTC to do.

Against a steady-state penalty of 0.87 ms/frame (full chain) and 0.68 ms/frame (Gaussian):

> **A distinct parameter value pays for its own compile after ~110 frames (full chain) or ~145
> frames (Gaussian alone).** Below that, parameterized constants win; above it, baked wins.

That is the decision rule this axis exists to produce. A server whose clients hold a handful of
fixed presets should bake; one where sigma is a per-request slider — a preview UI, a parameter
sweep — should not. The default stays baked, which is the right default for the showcase chain.

## 2. Resolution, 512² → 4K — PENDING RERUN, numbers below are known wrong

**The first pass of this sweep is invalidated and its numbers are withheld.** It used a fixed
8-frame concurrency budget (`--window 2 --slots 8`, inherited from the Phase 5 baseline and sized
for a hardcoded 64 MiB slot) for every resolution. `imgjit-bench` is a closed loop by construction,
so throughput is mechanically tied to latency by Little's Law at fixed concurrency
(`fps ≈ concurrency / p50`) whether or not the GPU is actually the bottleneck. Checking the
recorded fps against that formula afterwards showed every row within ~15% of the theoretical cap —
meaning a "peak Mpx/s at 1024²" shape drawn from those rows cannot be distinguished from "8 frames
in flight was not enough to saturate the pipeline at every size," which is a documented failure
mode of this exact harness (`bench/README.md`, the Phase 6 window-sizing note).

The corrected methodology (`colab/run.ipynb`, the dedicated resolution-sweep cell) sizes the slot
to each resolution's real frame instead of a fixed 64 MiB, and raises concurrency per resolution
until `CudaBackend::BackendStats::submit_stalls` — incremented only when a frame had to wait for a
free GPU stream — goes nonzero, which is a direct signal of GPU-bound saturation rather than an
inference from an fps ratio. This section will be rewritten from that rerun's `submit_stalls > 0`
rows. `mean_kernel_ms`, read off the GPU's own clock, is unaffected by the closed-loop concern and
was already trustworthy in the withdrawn run; only the fps/Mpx/s/GiB/s figures and the "peak at
1024²" narrative are retracted.

## 3. Tiling × streams — the Phase 6 and Phase 7 axes crossed

Neither earlier phase ran the full 2×2. Showcase chain, 1024²:

| config | fps | vs naive/1 stream | mean kernel |
|---|---|---|---|
| naive, 1 stream | 260.6 | — | 2.509 ms |
| naive, 4 streams | 289.6 | 1.11x | 9.894 ms (time-sliced) |
| **tile 16, 1 stream** | **468.9** | **1.80x** | **0.948 ms** (2.65x faster) |
| tile 16, 4 streams | 451.0 | 1.73x | 2.485 ms (time-sliced) |

**Tiling is worth far more than streams on this chain, and the two do not stack here.** Streams
buy +11% on the naive kernel (Phase 6 measured +18% on its own session, same direction), but on
top of tiling they buy **−4%** — within shouting distance of the 1.6% FPS noise floor, so the
honest reading is "nothing", not "a regression".

That is coherent rather than surprising: multi-stream overlap earns its keep by hiding PCIe copies
behind compute, and tiling cuts the compute it was hiding them behind by 2.65x. Less compute means
less to overlap with, while the per-stream cost (deeper queues, more events, kernel time-slicing)
stays. The best configuration in the whole matrix is the simplest one — **tile 16 at a single
stream**.

## 4. Fusion — the one result that contradicts a documented claim

The showcase chain's four ops fuse into two kernels. The unfused stand-in is the same four ops sent
as four separate one-op chains; codegen has no "don't fuse" mode, since fusion is structural.
Compared by **summed kernel time only** — four chains are also four round trips and four
H2D/D2H pairs, which fusion genuinely does save, and which this comparison deliberately excludes:

| chain | mean kernel |
|---|---|
| `grayscale` | 0.051 ms |
| `gaussian:1.4` | 1.742 ms |
| `sobel` | 0.147 ms |
| `threshold:0.3` | 0.088 ms |
| **sum of the four** | **2.028 ms** |
| **fused chain** | **2.598 ms** |

**The fused chain costs 1.28x the kernel time of the four unfused kernels.** That is 8x the noise
floor, and it is a real effect with a specific cause: `grayscale` sits *before* the Gaussian, so it
is folded into the stage's tap helper and **re-executed at every one of the 121 taps** (radius 5 →
11×11) instead of once per output pixel. The excess over Gaussian alone — 2.598 − 1.742 − 0.147 −
0.088 ≈ 0.62 ms — is almost exactly the cost of that redundancy, and the `chain_length` sweep
corroborates it independently: adding `brightness` (another prologue op) plus `invert` costs
+0.686 ms.

This does not make fusion wrong, and it does not contradict the design — codegen already *prefers*
attaching pointwise runs backwards as epilogues precisely because "a prologue is recomputed at
every stencil tap" (`src/backend/cuda/codegen.h`). What it contradicts is the unconditional framing
in `ARCHITECTURE.md`: fusion is described there as avoiding "4 kernel launches and 4 round-trips
through global memory", with no mention that a *prologue* trades that memory traffic for redundant
compute scaled by the stencil footprint. For a pointwise op in front of an 11×11 stencil, the trade
loses on compute.

Two honest caveats keep this from being a verdict on fusion:
- It measures kernel time only. Fusion still saves three global round trips and three launches per
  frame, plus — in the real server — three network round trips. End-to-end, one fused request beats
  four sequential requests comfortably.
- An epilogue is unaffected: it runs once per output pixel in either arrangement.

The finding suggests an optimization this phase deliberately does **not** implement: a prologue
could be emitted as its own pointwise kernel when the following stencil's radius is large enough
that re-execution outweighs the saved round trip. That is a codegen change with its own correctness
surface, and the op set and fusion plan were settled in Phase 2. Recorded as a finding, not a task.

## 5. Chain length

`--streams 1`, 1024², each chain prewarmed:

| ops | chain | mean kernel | fps |
|---|---|---|---|
| 1 | `invert` | 0.051 ms | 592.8 |
| 2 | `grayscale,sobel` | 0.218 ms | 565.3 |
| 4 | showcase | 2.546 ms | 256.6 |
| 6 | showcase + `brightness` + `invert` | 3.232 ms | 225.4 |

**Chain length is not the cost driver; stencil radius is.** Going 1 → 2 ops adds a 3×3 Sobel for
+0.167 ms. Going 2 → 4 adds an 11×11 Gaussian for +2.328 ms — 14x more, for the same "two more
ops". Going 4 → 6 adds two *pointwise* ops for +0.686 ms, which is large for pointwise work and is
§4's effect again: `brightness` lands in the prologue and runs at all 121 taps.

A useful way to read the whole table: the per-frame cost is roughly (number of stencil taps) ×
(cost of the prologue chain), plus a small constant per pointwise epilogue.

## 6. Architectural claims mapped to numbers

`docs/PLAN.md`'s "Done when" for this phase. Claims from `docs/ARCHITECTURE.md`:

| Claim | Measured | Where |
|---|---|---|
| "NVRTC cold-start is ~50–200 ms" | ≈97 ms baked, ≈63 ms parameterized | §1c |
| Kernel cache makes a repeat free | 4 compiles → 1 for 4 sigmas; cold p99 468 → 198 ms | §1b |
| Constant baking lets NVRTC unroll and constant-fold | 1.36–1.42x faster kernels than parameterized | §1a |
| Baked vs parameterized is "a clean A/B axis" | both directions measured; break-even ≈110 frames | §1 |
| Fusion avoids launches and global round trips | true for epilogues; **a prologue costs 1.28x** | §4 |
| Shared-memory tiling helps large stencils | 2.65x kernel, 1.80x FPS on the showcase chain | §3 |
| Multi-stream overlap raises throughput | +11% naive; **±0 once tiled** | §3 |
| Width/height never enter the kernel key | 3 resolutions, 1 compile, one backend | `phase3_gpu_kernel_cache` |

**The last row is a test result, not a matrix row, and the distinction matters.** Every row in this
matrix does report exactly one compile, including all four resolutions — but each row is a *fresh
server process*, so that is four processes compiling once each, which would be equally true if
resolution were part of the kernel key. What actually proves invariant 4 is the ctest that runs
three resolutions through **one** backend and asserts the compile counter never moves
(`tests/test_cuda_backend.cpp`, "resolution is not part of the kernel identity"). The matrix is
consistent with the claim; it does not demonstrate it.

Two rows above are qualified rather than confirmed, and both are recorded as findings in this file
and in `docs/PLAN.md` rather than smoothed over: fusion's prologue cost, and streams adding nothing
on top of tiling.

## 7. What this run does not cover

- **Error injection** is the remaining Phase 8 checklist item: malformed protocol frames and a
  forced illegal access against a live GPU-backed server, to show the Phase 6 context recreation
  holds under a real fault. Nothing here exercises it; every row in the matrix reports
  `failures = 0` and zero context recreations, which is the healthy baseline it will be measured
  against.
- **Single runs, one session, one T4.** See §0.
- **`--no-echo` throughout**, so no row includes the response payload on the wire. On Colab's 2
  vCPUs the echo makes the client the bottleneck (Phase 6).
- **Parameterized is naive-only**, so the constants axis is not crossed with tiling — a tiled stage
  sizes its `__shared__` array from the stencil radius at compile time, which a parameterized kernel
  does not know until launch (`docs/PLAN.md` Phase 8).
