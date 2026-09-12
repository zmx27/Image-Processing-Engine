# Phase 8 results — the benchmark matrix

Environment: Colab T4 (`compute_75`), 2 vCPUs. Harness: `imgjit-bench` driving `imgjit-server`
over loopback, `--no-echo` (request-only), `--connections 4 --window 8 --frames 300` at
1024²×3 unless a sweep says otherwise. Reproduce with cells 22 and 24 of `colab/run.ipynb`, which
write `baseline_phase8.csv` / `phase8_matrix.csv` (the six-axis sweep) and
`phase8_resolution.csv` / `phase8_resolution_summary.csv` (the resolution sweep) respectively.

Six sweeps, each varying one axis with the rest pinned. A cartesian product of six axes would be
hundreds of runs and no clearer about any one of them.

**This is the second full run of this matrix.** The first run's resolution sweep was invalidated
by a methodology bug (fixed concurrency confounding the closed loop with the GPU — see §2) and
had to be redone; everything else was rerun alongside it for a clean, internally consistent
session rather than splicing two sessions' numbers together. Where the two sessions disagree, both
are reported — one disagreement (§3) reverses a headline finding outright, which is exactly the
kind of thing a single run cannot be trusted to catch. See §0.

## 0. Method: repeatability, within one session and across two

One configuration — naive, `--streams 1`, 1024², the showcase chain, prewarmed — appears in *four*
sweeps this session, so it was measured four times in one run:

| label | mean kernel | fps |
|---|---|---|
| `phase8_full_baked` | 2.305 ms | 267.2 |
| `phase8_full_naive_streams1` | 2.473 ms | 272.8 |
| `phase8_fused_full` | 2.516 ms | 270.5 |
| `phase8_len4` | 2.518 ms | 269.9 |

**Spread: 9.2% on kernel time, 2.1% on FPS.** The first session's spread on the same comparison
was 3.5% / 1.6% (`bench/phase7_results.md`'s tiled-Sobel qualification cites that number). FPS
repeatability held roughly steady between sessions; kernel-time repeatability did not — the
within-session floor itself varies by more than 2x from one full run to the next. Treat 9.2% as
the current, more conservative estimate of what a kernel-time difference must clear to be real.

**A second full session is also what surfaces disagreements a single run cannot.** Comparing this
run against the first for every finding that both sessions measured:

| Finding | Session 1 | Session 2 | Agreement |
|---|---|---|---|
| Parameterized kernel penalty (full chain) | 1.36x | 1.38x | close |
| Parameterized kernel penalty (Gaussian) | 1.42x | 1.36x | close |
| Cold cache: compiles, baked vs parameterized | 4 vs 1 | 4 vs 1 | **identical** (discrete count) |
| Cold cache: worst-first-frame reduction | 68% | 39% | same direction, **large magnitude gap** |
| Tiled kernel speedup (Gaussian+chain, `--streams 1`) | 2.65x | 2.68x | close |
| Streams on top of tiling, full chain | **−3.8%** ("nothing") | **+28.9%** | **sign reversal** |
| Fusion cost (showcase chain) | 1.28x slower | 1.26x slower | close |

Five of seven replicate well. Two do not, and neither failure is subtle:

- **The compile-benefit magnitude for parameterized constants is real but not a fixed number.**
  Both sessions show baked recompiling 4x on four distinct sigmas and parameterized recompiling
  once — that discrete count is exactly reproducible, because it isn't a timing measurement. How
  much *latency* that difference is worth on a cold cache varied from a 68% to a 39% reduction in
  the worst first frame. §1 keeps the count claim unqualified and reports the latency benefit as a
  range instead of a point estimate. The derived "cost per compile in ms" figure from session 1
  (~97 ms baked, ~63 ms parameterized, "parameterized compiles faster") does **not** replicate —
  session 2's same arithmetic gives ~43 ms baked, ~84 ms parameterized, the *opposite* ranking.
  That sub-claim is retracted; see §1.
- **"Streams add nothing on top of tiling" was wrong.** Session 1 measured tile16 `--streams 4` at
  −3.8% versus tile16 `--streams 1` and, combined with a 1.6% fps noise floor at the time, called
  it noise. Session 2 measures **+28.9%** on the identical configuration — `phase8_full_tile16_streams4`
  at 504.2 fps is now the fastest full-chain configuration in the entire matrix, not tile16 at one
  stream. See §3 for the full reversal and what's now known versus still uncertain.

The practical rule this leaves: **a kernel-time claim under roughly 10% is not trustworthy from one
session**, and an interaction between two axes (streams × tiling) apparently needs more than two
sessions to pin down at all. Every ratio in §1–§5 below is this session's number; where session 1's
number differs enough to matter, both are shown rather than one silently overwriting the other.

## 1. Baked vs parameterized constants — this phase's own axis

The axis reserved in the kernel key since Phase 2: op parameters as compiled-in literals versus
kernel arguments passed at launch. It has a cost side and a benefit side, and only the cost is a
speed — a run that measured kernel time alone would conclude the mode is strictly worse.

### 1a. The cost: kernels run 1.36–1.38x slower

`--streams 1`, so `mean_kernel_ms` is a clean per-frame number:

| chain | baked | parameterized | penalty |
|---|---|---|---|
| `gaussian:1.4` | 1.650 ms | 2.247 ms | **1.36x** |
| full chain | 2.305 ms | 3.183 ms | **1.38x** |

Consistent with session 1 (1.42x / 1.36x) to within the noise floor. End to end: 345 → 289 fps
(−16%) for the Gaussian, 267 → 224 fps (−16%) for the chain.

### 1b. The benefit: fewer compiles, real but variable latency payoff

Four connections asking for four *different* sigmas (`0.8, 1.4, 2.2, 3`), cold cache:

| mode | compiles | worst first frame | p99 |
|---|---|---|---|
| baked | **4** | 234.2 ms | 288.8 ms |
| parameterized | **1** | **141.9 ms** | **193.3 ms** |

Four distinct sigmas are four distinct kernels baked, one kernel parameterized — that count is
identical across both sessions and is not a timing measurement, so it is the load-bearing claim
here. The *latency* payoff of that difference is real in both sessions but not a fixed number:
**a 39–68% reduction in worst-first-frame latency and a 33–58% reduction in p99**, depending on
session. Report it as a range, not a point estimate.

Warm (the same four sigmas prewarmed), the picture inverts, consistently across both sessions:
parameterized's steady-state cost outweighs any compile-avoidance benefit once nothing is left to
compile (baked 63.2 ms vs parameterized 58.0 ms worst first frame this session — both cheap, and
close enough that the ranking here is itself within the noise floor).

**Retracted: the "ms per compile" sub-claim.** Session 1 divided the cold/warm gap by the compile
count to estimate a per-compile cost (~97 ms baked, ~63 ms parameterized) and concluded
parameterized compiles faster ("less to unroll"). Repeating that arithmetic this session gives
~43 ms baked, ~84 ms parameterized — the opposite ranking. Whatever this derived quantity is
measuring, it isn't stable enough across sessions to support a directional claim about which mode
compiles faster, and it is dropped rather than reported with a caveat that undersells how wrong it
could be. What survives: an NVRTC compile costs somewhere in the tens-to-low-hundreds of ms, which
is still consistent with `ARCHITECTURE.md`'s "~50–200 ms" claim in both sessions.

### 1c. The decision this axis produces

Despite the magnitude uncertainty, the direction is unambiguous and replicates: a server whose
clients hold a handful of fixed presets should bake; one where a parameter is a per-request slider
should not, because every additional distinct value is a full recompile under baked and free under
parameterized. The default stays baked.

## 2. Resolution, 512² → 4K — corrected methodology, now confirmed GPU-bound

The first pass of this sweep held concurrency fixed at 8 frames in flight for every resolution.
`imgjit-bench` is a closed loop, so its throughput is mechanically tied to latency by Little's Law
at fixed concurrency (`fps ≈ concurrency / p50`) regardless of whether the GPU is the bottleneck —
checking the first pass's numbers against that formula afterward showed every row within ~15% of
the theoretical cap, so its "peak Mpx/s at 1024²" shape could not be told apart from "8 in flight
did not saturate the pipeline at every size." Those numbers were withdrawn rather than kept.

**The fix:** size the slot to each resolution's real frame instead of a fixed 64 MiB (so raising
concurrency for small frames costs little pinned memory), and raise concurrency at each resolution
until `CudaBackend::BackendStats::submit_stalls` — incremented only when a frame had to wait for a
free GPU stream — goes nonzero. That is a direct measurement of GPU-side contention, not an
inference from an fps ratio, and it needs no second data point to interpret on its own.

| resolution | concurrency | fps | GiB/s | p50 | submit stalls |
|---|---|---|---|---|---|
| 512² | 64 | 963.2 | 0.705 | 54.8 ms | 496 |
| 512² | **128** | **1276.0** | **0.935** | 97.0 ms | 621 |
| 1024² | 64 | 309.1 | 0.906 | 208.1 ms | 434 |
| 1024² | **128** | **322.5** | **0.945** | 366.6 ms | 624 |
| 2048² | 64 | 68.9 | 0.807 | 900.3 ms | 151 |
| 2048² | **128** | **69.2** | **0.811** | 1749.5 ms | 177 |
| 4096² | 40 | 12.4 | 0.579 | 3160.8 ms | 79 |

Every row shows `submit_stalls > 0`: at no resolution or concurrency level was the client's window
the limiter — the GPU worker's own 4 stream slots were. That is the thing the first pass could not
establish and this one does.

**Stalls appearing does not mean throughput has plateaued, and the doubled-concurrency rows show
why.** Going from 64 to 128 concurrent frames still raised fps by **32.5% at 512²** and **4.3% at
1024²**, despite both levels already showing contention — more requests queued behind the 4 busy
streams still shortens the gaps between them finishing. By 2048² the same doubling buys only
**0.6%**, which is close enough to flat to call genuinely saturated. The reading: *stalls>0
confirms the number isn't a window-cap artifact; it does not by itself confirm the number is the
ceiling.* 512² and 1024²'s figures above are a firm lower bound on real throughput and probably an
underestimate of the true maximum; 2048²'s is close to it; 4096² was only affordable to test at one
concurrency level (pinned-memory budget) but already shows stalls, so it is at least a lower bound
too.

**Pixels/second, using the higher-concurrency row at each size (207.4–338.1 Mpx/s):**

| resolution | Mpx/s | vs 1024² |
|---|---|---|
| 512² | 334.5 | −1.1% |
| 1024² | 338.1 | peak |
| 2048² | 290.4 | −14.1% |
| 4096² | 207.4 | −38.7% |

**512² and 1024² are essentially tied, not a dip-then-peak.** The first pass's "512² underperforms
1024² because small frames don't amortize fixed overhead" narrative does not survive — it was the
window-cap artifact described above, not a real GPU characteristic (and 512²'s number here is
still probably a slight underestimate, since it was still visibly unsaturated even at concurrency
128). The genuine effect starts at 2048²: pixel throughput falls off past roughly 1M pixels/frame,
consistent with the working set outgrowing on-chip cache reuse.

`mean_kernel_ms` in this sweep is measured at `--streams 4` and is **not** a clean per-frame number
— with several frames resident the GPU time-slices other streams' kernels into this frame's
begin/end window (§3), so it is reported in the CSV but not analyzed here.

## 3. Tiling × streams — a headline finding that did not replicate

Neither Phase 6 nor Phase 7 ran the full 2×2 of tiling and multi-stream overlap. Showcase chain,
1024²:

| config | fps | vs naive/1 stream | mean kernel |
|---|---|---|---|
| naive, 1 stream | 272.8 | — | 2.473 ms |
| naive, 4 streams | 319.0 | 1.17x | 9.712 ms (time-sliced) |
| tile 16, 1 stream | 391.2 | 1.43x | 0.923 ms (2.68x faster) |
| **tile 16, 4 streams** | **504.2** | **1.85x** | 2.554 ms (time-sliced) |

**Session 1 measured this exact configuration at −3.8% and called it "nothing, not a regression."
Session 2 measures +28.9%, and tile16/streams4 is now the fastest full-chain configuration in the
whole matrix — faster than tile16/streams1, which session 1 had singled out as the best.** This is
not a small discrepancy inside a noise band; it is a sign flip on the headline claim of this
section, from two full, otherwise-consistent sessions.

**What's now known:** streams and tiling are not mutually exclusive wins — the naive kernel's
+17% from streams and the tiled kernel's 2.68x from tiling are not fighting each other the way
session 1's number implied. **What's not known:** whether +28.9% is closer to the true effect than
−3.8% was, or whether both are noisy draws from a wide distribution and a third session would land
somewhere else again. Two data points establish that the true effect is not reliably near zero —
they do not establish what it actually is. Reproducing this specific cell a third time before
citing a number for it (rather than a direction) is the honest next step; it is not done here.

One thing this reversal does *not* touch: tiling's own effect (2.65–2.68x kernel time across both
sessions) and streams' own effect on the naive kernel (+11–17%) both replicate closely on their
own. It is specifically the *combination* — going from tile16/streams1 to tile16/streams4 — that
disagreed in sign between sessions.

## 4. Fusion — replicates; the finding stands

Same method as before: the showcase chain's four ops fuse into two kernels; the unfused stand-in
is the same four ops as four separate one-op chains, compared by summed kernel time (four chains
are also four round trips, which fusion genuinely saves and this comparison excludes on purpose):

| chain | mean kernel |
|---|---|
| `grayscale` | 0.044 ms |
| `gaussian:1.4` | 1.701 ms |
| `sobel` | 0.170 ms |
| `threshold:0.3` | 0.078 ms |
| **sum of the four** | **1.993 ms** |
| **fused chain** | **2.516 ms** |

**1.26x slower fused**, against session 1's 1.28x — a close replication, well inside even the
stricter 9.2% kernel-time noise floor from §0. The cause is unchanged: `grayscale` sits in the
Gaussian's prologue and is re-executed at all 121 taps instead of once per pixel. Excess over
Gaussian+Sobel+threshold alone: 0.567 ms (session 1: 0.62 ms) — same order of magnitude, same
mechanism. `ARCHITECTURE.md`'s qualification of the fusion claim stands unchanged.

## 5. Chain length — replicates

`--streams 1`, 1024², each chain prewarmed:

| ops | chain | mean kernel | fps |
|---|---|---|---|
| 1 | `invert` | 0.053 ms | 566.9 |
| 2 | `grayscale,sobel` | 0.247 ms | 454.8 |
| 4 | showcase | 2.518 ms | 269.9 |
| 6 | showcase + `brightness` + `invert` | 3.108 ms | 231.3 |

Same shape as session 1: adding a 3×3 Sobel (1→2 ops) costs +0.194 ms; adding an 11×11 Gaussian
(2→4 ops) costs +2.271 ms — over 10x more for "two more ops"; adding two more pointwise ops in the
prologue (4→6) costs +0.590 ms, disproportionate for pointwise work for the same reason as §4.
Stencil radius, not op count, remains the cost driver.

## 6. Architectural claims mapped to numbers

`docs/PLAN.md`'s "Done when" for this phase, updated for both sessions:

| Claim | Measured | Confidence |
|---|---|---|
| NVRTC cold-start "~50–200 ms" | consistent with both sessions' raw cold/warm gaps | supported, exact per-compile figure not derivable — §1b |
| Kernel cache makes a repeat free | 4 compiles → 1 for 4 sigmas, both sessions | **confirmed** (discrete count) |
| Constant baking lets NVRTC unroll and constant-fold | baked 1.36–1.42x faster than parameterized, both sessions | **confirmed** |
| Fusion avoids launches and global round trips | true for epilogues; a prologue costs 1.26–1.28x, both sessions | **confirmed as qualified** |
| Shared-memory tiling helps large stencils | 2.65–2.68x kernel, both sessions | **confirmed** |
| Multi-stream overlap raises throughput (naive) | +11–17%, both sessions | **confirmed** |
| Multi-stream overlap on top of tiling | **−3.8% then +28.9%** — sign disagreement | **not established**, see §3 |
| Width/height never enter the kernel key | 3 resolutions, 1 compile, one backend | `phase3_gpu_kernel_cache` (not this matrix — see below) |

**The kernel-key row is a test result, not a matrix row.** Every resolution row in §2 reports one
compile, but each is a fresh server process — four processes compiling once each says nothing
about whether resolution is in the key. The actual proof is the ctest that runs three resolutions
through *one* backend and asserts the compile counter never moves
(`tests/test_cuda_backend.cpp`, "resolution is not part of the kernel identity").

## 7. What this run does not cover

- **Error injection** remains the outstanding Phase 8 checklist item: malformed protocol frames
  and a forced illegal access against a live GPU-backed server. Every row here reports
  `failures = 0` and zero context recreations — the healthy baseline it will be measured against.
- **The tile×streams interaction (§3) needs a third session** before it is reported as a number
  rather than a direction. This is the one open item this write-up cannot close by itself.
- **`--no-echo` throughout**, so no row includes the response payload on the wire.
- **Parameterized is naive-only** — not crossed with tiling, for the reason given in
  `docs/PLAN.md` Phase 8 (a tiled stage sizes its `__shared__` array from the stencil radius at
  compile time, which a parameterized kernel does not know until launch).
- **512² and 1024² in §2 are probable underestimates** — both were still visibly gaining
  throughput from added concurrency at the highest level tested (128), so the pipeline's true
  ceiling at those sizes is higher than the number reported.
