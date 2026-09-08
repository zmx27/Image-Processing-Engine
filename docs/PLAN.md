# Plan

Read this file at the start of every new session to see what's done and what's next — check
`docs/PLAN.md`'s boxes and any "next:" note before asking the user to re-explain state. See
`docs/ARCHITECTURE.md` for design rationale and `docs/PROTOCOL.md` for the wire format.

## Sequencing rationale

Two independently-verifiable tracks that merge at Phase 5. **Track G** (Phases 1, 3) proves JIT +
driver API through a file-in/file-out CLI with zero network code. **Track N** (Phases 2, 4) proves
protocol + concurrency + backpressure against the CPU backend with zero CUDA code. The payoff: a
bug found in Phase 4 is never confused with a CUDA bug, and a bug found in Phase 3 is never
confused with a race.

No Mac has an NVIDIA GPU (Intel or Apple Silicon — Apple dropped NVIDIA support system-wide, and
Apple Silicon GPUs aren't NVIDIA hardware either). Phases marked `env: Colab` cannot be built or
run locally; phases marked `env: Mac` are pure local development with no GPU dependency.

## Phase 0 — Scaffolding · env: Mac + Colab

- [x] `brew install cmake`
- [x] `git init`; push to GitHub
- [x] `CMakeLists.txt` with CUDA auto-detection → `IMGJIT_ENABLE_CUDA`; C++20; warnings-as-errors
- [x] Vendor stb_image.h / stb_image_write.h + Catch2 under `third_party/`
- [x] Add test image fixtures under `tests/testdata/`
- [x] Write `CLAUDE.md`, `docs/PLAN.md`, `docs/ARCHITECTURE.md`, `docs/PROTOCOL.md`
- [x] `colab/run.ipynb` skeleton that clones the repo and builds (with a `!git pull` cell at the
      top and an `nvidia-smi` / CUDA version print cell for drift visibility)
- **Done when:** clean CPU-only build on Mac, and the notebook produces a CUDA build on Colab

## Phase 1 — Driver API + JIT spike · env: Colab

**Done.** Verified on Colab (T4, driver 580.82.07, CUDA 13.0, nvcc 12.8.93, `compute_75`): both
ctest cases (`phase1_spike_checkerboard_16x16_rgb`, `phase1_spike_gradient_32x32_rgba`) passed,
and `imgjit-spike` reported exact match against the CPU oracle for both checkpoints.

Two checkpoints in one phase, so a failure is unambiguous about which layer broke:

- [x] **1a:** `cuInit` → `cuCtxCreate` → load a PTX blob produced by `nvcc --ptx` (ahead of time)
      → `cuModuleLoadData` → `cuLaunchKernel`. Proves driver plumbing with JIT out of the picture.
      Build that PTX with an `add_custom_command` invoking `${CUDAToolkit_NVCC_EXECUTABLE}`, or
      just check the `.ptx` in as a fixture — **do not** enable the CUDA language in CMake to get
      it. `project(imgjit LANGUAGES CXX)` is deliberate: NVRTC compiles at runtime, so a CUDA
      toolchain in the build is unnecessary, and enabling it invites `<<<>>>` and cudart linkage
      (invariant 8).image.pngimage.pngimage.pngimage.pngimage.pngimage.pngGo over every new file/edit that you made in this phase and walk me through on a high level what they do and their purpose. Assume that I am a complete beginner to cuda and image processing techniques.
- [x] **1b:** swap the AOT blob for `nvrtcCompileProgram` at runtime. Proves JIT plumbing on top
      of already-working driver plumbing.
- [x] `CU_CHECK` / `NVRTC_CHECK` macros; dump generated PTX to disk for inspection
- **Done when:** Colab inverts a real PNG on-GPU, output exactly matches scalar CPU inversion
      (exact is the right bar here — integer pointwise op, no float reassociation)

This phase is a **spike**, and it runs before Phase 2 deliberately: its job is to find out early
whether the Colab workflow is tolerable (see `ARCHITECTURE.md` → Open risk). Consequences:

- The oracle here is a throwaway inline inversion loop, **not** the Phase 2 CPU backend — that
  backend and the `Image` type do not exist yet. Do not block Phase 1 on building them.
- What must *survive* the spike into `src/backend/cuda/` is `CU_CHECK` / `NVRTC_CHECK` and the
  RAII wrappers for `CUcontext`, `CUmodule`, `CUdeviceptr`. The driver-plumbing scratch code
  around them is expendable.

## Phase 2 — Portable core · env: Mac

**Done.** Clean `-Werror` build on macOS with no CUDA toolkit; 42 test cases green plus the
`imgjit-cli` ctest case; invariant 1's grep silent over `src/` and `include/`.

- [x] `Image` type (`uint8`, 1/3/4 channels, row-major, explicit stride)
- [x] `OpChain` IR + parser for `"grayscale,gaussian:1.4,sobel,threshold:0.3"`
- [x] Canonicalizer + `KernelKey` hash — **excludes width/height** by construction
- [x] `IBackend` interface — see below; get the shape right here, not in Phase 6
- [x] CPU backend implementing the closed op set; stb load/save wrappers; `tools/imgjit-cli`
- [x] Tests: parser, canonicalizer, hash stability, each filter vs. checked-in fixtures
- **Done when:** `imgjit-cli --ops "grayscale,sobel" in.png out.png` works on the Mac; tests green

**Op semantics are binding on Phase 3's codegen and are defined once, in
`include/imgjit/core/op_chain.h`.** The plan fixes *which* six ops exist; it does not fix their
arithmetic, and a definitional difference between the generated kernel and the CPU oracle reads
as a correctness bug rather than as the disagreement it is. Settled in Phase 2:

- **Channel count is invariant across a chain** — `grayscale` writes luma into R, G and B rather
  than reducing to one channel. A channel-changing op would make every fusion boundary in Phase 3
  renegotiate the layout.
- **Alpha (channel 3 of RGBA) passes through every op untouched.**
- Float ops run on samples normalized to `[0,1]`, clamped and rounded back to `uint8` on store;
  `invert` stays integer, which is why it alone compares exactly. Stencils use clamped (replicate)
  edge addressing.
- Parameter ranges, which are what bound the baked stencil: `gaussian` σ ∈ [0.1, 4.0] (default
  1.0, radius = ⌈3σ⌉ ≤ 12), `threshold` ∈ [0,1] (default 0.5), `brightness` ∈ [-1,1] (default 0).
- The Gaussian oracle is a **single 2D pass over the outer product** of the 1D weights, not two
  separable passes — the generated kernel is one 2D stencil, and matching factorizations keeps
  float reassociation out of the GPU-vs-CPU diff.

Parameters are **quantized at parse time**, not only when hashing. Both collapse `gaussian:1.4`
and `gaussian:1.4000001` onto one cache entry; quantizing early additionally makes the op that
executes identical to the one the cached kernel baked, rather than merely colliding with it.

Tests assert hand-computed values and definitional properties (a normalized blur leaves a constant
image constant; a gradient operator is zero on one), plus an independently-written separable
Gaussian as a cross-check. Deliberately **no golden output PNGs**: regenerated from the code under
test, they prove only that behavior has not changed, and would have locked in a wrong luma
coefficient as happily as a right one.

**The op set is closed at six** — pointwise: `grayscale`, `invert`, `brightness`, `threshold`;
stencil: `gaussian`, `sobel`. Adding ops is the #1 over-scoping risk named in `ARCHITECTURE.md`;
the list is fixed here so later phases have a finite corpus to be complete against.

**Canonicalization is parameter normalization only — never reordering.** Op order is semantically
load-bearing (`gaussian→sobel` ≠ `sobel→gaussian`), so the canonicalizer only normalizes
whitespace, float formatting (`gaussian:1.40` → `gaussian:1.4`), and implicit defaults made
explicit. Additionally, **quantize σ before hashing** (e.g. to 2 decimals): `gaussian:1.4` and
`gaussian:1.4000001` bake identical weights, and without quantization they become two cache
entries for one kernel.

**`IBackend` is submit/poll from day one, even though the CPU backend has nothing to poll.**

    submit(FrameJob) -> JobHandle          // may complete immediately
    poll_completions() -> vector<Completion>

The tempting Phase 2 signature is a blocking `Image process(const Image&, const OpChain&)`. It is
fine for Phase 2, fine for Phase 5 — and impossible in Phase 6, where the worker keeps K frames
in flight and retires them by event poll. Adopting the blocking shape means redesigning both the
interface *and* the worker loop at precisely the phase that introduces the event-gated-return
invariant, which is the worst place in the project to be changing structure. The CPU backend
simply completes inside `submit` and returns the completion from the next `poll_completions()`;
the server loop is then a state machine from Phase 4 onward and Phase 6 changes only the backend.

The interface must also allocate the frame slots (see Phase 4):

    allocate_slots(count, bytes) -> std::byte*   // CPU: heap. CUDA: cuMemAllocHost on the worker.

Returning `std::byte*` keeps every CUDA type inside `src/backend/cuda/`, so invariant 1's grep
still passes with the pinned pool in place.

## Phase 3 — Codegen + kernel cache · env: Colab

**Done.** Verified on Colab (T4, driver reporting CUDA 12.8.93, `compute_75`): all 10 ctest cases
passed, including both gate tests — `phase3_gpu_oracle_diff` (6.78s: every corpus chain × {1,3,4}
channels within tolerance) and `phase3_gpu_kernel_cache` (1.71s: repeat-compiles and
resolution-independence both hold). `imgjit-cli --backend cuda --repeat 5` on
`grayscale,gaussian:1.4,sobel,threshold:0.3` showed run 1 at 73ms (cold NVRTC compile) and runs
2-5 at ~0.3ms (warm cache hit) — a ~250x difference from memoization alone — with
`NVRTC compiles: 1` confirming the cache held across all 5 runs.

As a pre-Colab sanity check, the generated source was also compiled as host C++ with the
CUDA-isms stubbed out and run against the oracle: all 17 corpus chains × {1,3,4} channels came
out bit-identical. That caught arithmetic/fusion bugs before ever reaching Colab, but proved
nothing about NVRTC or the driver — the real gate is the GPU run above.

- [x] `emit_cuda_source(KernelKey) → GeneratedProgram`; one kernel per stencil stage, pointwise
      ops fused. Takes the *key*, not the chain: the key is by definition the complete set of
      codegen inputs, so anything else in the signature would be an input the cache is not keyed by
- [x] `src/backend/cuda/kernel_prelude.h` — stable device helpers (clamp, luminance, `uint8`↔
      `float`, clamped indexing) as one raw-string literal; **no `kernels/` directory**, and the
      emitted source stays self-contained since NVRTC has no default include path
- [x] `--dump-source` flag alongside Phase 1's PTX dump — the generated kernel is what you read
      when debugging, not the fragments. Works on a Mac: codegen links no CUDA (see below)
- [x] Bake radius / weights / threshold / channels as literals; dimensions stay launch arguments
- [x] `KernelCache: KernelKey → {CUmodule, CUfunction}` with a compile counter for assertions
- [x] Multi-stage execution with intermediate device buffers
- [x] Measure cold-compile vs. warm-hit latency (`imgjit-cli --repeat n`)
- **Done when:** every chain in the test corpus matches CPU within tolerance, **and** a repeated
      chain provably compiles exactly once (compile counter unchanged), **and** the same chain at
      three different resolutions still compiles only once

**`src/backend/cuda/` builds as two targets, and only one of them needs a GPU.** `imgjit_codegen`
emits CUDA *source text*, which takes no CUDA headers, so it is compiled and unit-tested on macOS;
`imgjit_cuda` (context, cache, backend) is the half that calls the driver and stays behind
`IMGJIT_ENABLE_CUDA`. Both live in this directory, so invariant 1's grep still skips them as one.
The payoff is that a codegen typo is caught locally in seconds instead of one Colab round trip
later, and the codegen tests assert the structure a GPU test would struggle to attribute: the
fusion plan, which constants got baked, and that nothing resolution-dependent leaked into the
source.

**Fusion re-quantizes at every op boundary** — see `ARCHITECTURE.md` decision 5. The oracle stores
a `uint8` image between ops, so the fused kernel rounds at the same points; otherwise Sobel
amplifies the difference to ~4 LSB and a `threshold` after a stencil flips 0↔255, and the phase
gate's tolerance would have to be loosened per-chain to hide it.

**Every codegen input must be in `KernelKey`.** Two inputs arrive in later phases and are easy to
forget, because omitting them produces a stale cache hit rather than a failure — a wrong benchmark
number, not a crash: the **tile size** from Phase 7 (baked into `__shared__` array dimensions, so
a naive|tiled boolean is not sufficient) and the **baked|parameterized constants mode** from
Phase 8's A/B axis. Both are already reserved in the key as of Phase 2.

**The authoritative field list** (`CLAUDE.md` invariant 4 points here; one copy, deliberately —
`include/imgjit/core/kernel_key.h` implements exactly this and nothing else):

| Field | Since | Why it is a codegen input |
|---|---|---|
| canonical `OpChain` (kinds + quantized params, in order) | 2 | the emitted stages, their fusion boundaries and every baked literal |
| `channels` | 2 | baked as a literal; changes indexing and which channels the op touches |
| `tile` (naive\|tiled) | 7 (reserved in 2) | a different kernel body |
| `tile_size` | 7 (reserved in 2) | baked into the `__shared__` array dimensions |
| `constants` (baked\|parameterized) | 8 (reserved in 2) | literals versus kernel parameters |

**Not in the key, and never to be added: width and height.** They are launch arguments. This is
enforced structurally — the struct has no such field and the hash takes nothing but the struct.

**Invariant-2 exemption, scoped to this phase.** `CLAUDE.md` invariant 2 forbids `cuMemAlloc` in
the per-frame path. Phase 3 is a one-image-at-a-time CLI with no pipeline to serialize, so
allocating intermediate device buffers per invocation is acceptable *here only*. The device pool
arrives in Phase 6 and the invariant applies unconditionally from that point. This is a stated
exemption, not an oversight — do not "fix" it early, and do not let it leak into Phase 5.

## Phase 4 — Network layer · env: Mac (CPU backend)

- [ ] `docs/PROTOCOL.md` implemented as pure encode/decode functions
- [ ] `read_exact()` / `write_exact()`; explicit little-endian pack/unpack; header validation +
      error responses
- [ ] Codec unit tests including malformed input: bad magic, truncated header, length mismatch,
      oversized payload
- [ ] Validation order + per-error-code connection disposition (drain vs. close) per
      `docs/PROTOCOL.md`; desync test: an error frame followed by a valid frame on the same
      connection must still be served correctly
- [ ] `tools/imgjit-server`; acceptor + **reader and writer thread per connection**; bounded MPSC
      queue; per-connection outbox; slot pool with per-connection caps
- [ ] `tools/imgjit-client`; both result paths (echo, server-side write); blocking backpressure
      verified with an artificially throttled worker
- [ ] Pipelining test: client sends N frames before reading any response — the case that deadlocks
      if slot release is gated on the socket write
- [ ] Integration test: N clients × M frames, verified against the CPU oracle
- **Done when:** multi-client localhost run is correct and clean under **ASan and TSan**

**The slot pool owns indices, not storage.** Phase 5 replaces its backing memory with pinned
buffers that `cuMemAllocHost` must allocate *on the worker thread at startup* — a change of
lifetime owner, not a one-line swap. Written the natural way (pool `new`s its own storage in its
constructor), Phase 5 rewrites the pool. Written correctly, the pool holds only the free-list,
the per-connection caps, and the blocking claim, and receives its backing storage from
`IBackend::allocate_slots()` at init — so Phase 5 touches zero lines of pool logic.

**The client keeps a `seq_num → pending` map, and the integration test compares sets.** The
obvious Phase 4 client sends a frame and reads one response; that is correct now and silently
wrong from Phase 6 on, when multi-stream completions retire out of order — which is the entire
reason `seq_num` is in the protocol. Building the FIFO assumption in means rewriting the client
and the integration test during the async phase. Verify the *set* of responses and match each by
`seq_num`, never by arrival order.

**The response path is decided here, and it is `CLAUDE.md` invariant 10.** The worker never
writes to a socket, and a slot is released the moment its work completes — never after its
response has been written. Concretely: reader thread does framing and slot claim; worker
processes and, on completion, copies the result out of the slot, releases the slot, and pushes
the response to that connection's outbox; writer thread drains the outbox.

Both halves are load-bearing. A worker that writes sockets lets one slow client head-of-line-block
the whole pipeline (invisible until Phase 6, where it looks like a mystery throughput cliff). And
release-after-write deadlocks any pipelining client: the reader blocks claiming a slot for frame
N+1, so the response for frame 1 is never written, so its slot never frees. The copy-out is the
price of breaking that cycle, and it is why the pipelining test above exists.

**ASan and TSan are two build configurations, not one.** They cannot be linked into the same
binary; the gate is two ctest presets over the same test set.

## Phase 5 — Integration: GPU worker behind the server · env: Colab

- [ ] Swap the CPU worker for the GPU worker; `cuCtxCreate` once at startup on that thread
- [ ] Pinned slot pool allocated on the worker; connection threads `recv()` directly into slots
- [ ] **Single stream only** — isolate "is the plumbing correct" from async overlap
- [ ] Startup prewarm of a configured chain list — `ARCHITECTURE.md` claims it, and Phase 8's
      gate requires every such claim to map to a measured row
- [ ] Minimal timing harness (FPS, p50/p99 round-trip) and a **recorded baseline** committed to
      the repo
- **Done when:** concurrent loopback clients with differing chains all match the CPU oracle on
      Colab; TSan clean; no CUDA symbol reachable outside `src/backend/cuda/`

Phases 6 and 7 are both gated on being "measurably faster" — which requires a number recorded
*here*, with the harness that produced it. `bench/` in Phase 8 then widens the matrix rather than
inventing the measurement, and each phase gate from this point records its numbers under the same
harness so the comparisons are like-for-like.

## Phase 6 — Async multi-stream pipeline · env: Colab

- [ ] In-flight table across K streams (start K=4); `cuEventRecord` + poll-and-retire
- [ ] Event-gated buffer return; device memory pool replacing any per-frame `cuMemAlloc`
- [ ] Instrument queue depth, occupancy, stall counts
- [ ] **Context recreation** — teardown/rebuild as a unit: flush the kernel cache and device pool,
      fail every in-flight job with status 6, resume accepting work
- [ ] Stress test: sustained repeated runs with output checksums, specifically targeting
      premature slot reuse (`CLAUDE.md` invariant 3)
- **Done when:** measurably faster than the Phase 5 recorded baseline, Nsight Systems shows genuine
      H2D/compute/D2H overlap rather than serialized segments, and zero checksum drift under stress

Context recreation lands here, not in Phase 8, because it is a *design constraint on these
structures* rather than a feature bolted on afterwards: recovery means destroying the kernel
cache, the device pool, and the in-flight table together and rebuilding them. Retrofitting that
two phases after the in-flight table is built is invasive surgery on the most delicate code in the
project. Build them destroyable-as-a-unit now; Phase 8 only injects the fault and observes.

## Phase 7 — Shared-memory tiling · env: Colab

- [ ] Tiled stencil codegen variant: `__shared__` tile + halo/apron loads, bounds-clamped,
      parameterized by tile size
- [ ] Extend `KernelKey` with the naive|tiled variant **and the tile size** — the tile dimensions
      are baked into the `__shared__` array, so a boolean variant flag would collide two different
      kernels onto one key; both variants remain runtime-selectable
- **Done when:** tiled output matches naive and CPU within tolerance, and is measurably faster on
      both Gaussian and Sobel

## Phase 8 — Benchmarks + polish · env: Colab

- [ ] `bench/` CSV harness — widens the Phase 5 timing harness over the orthogonal matrix:
      naive|tiled × sync|async, plus fused-vs-unfused chain, cold-vs-warm cache,
      baked-vs-parameterized constants (the constants mode must already be a `KernelKey` input —
      see Phase 3, or the parameterized run silently reuses the baked kernel)
- [ ] Sweep 512² → 4K and chain lengths; report FPS, GB/s, p50/p99 round-trip latency
- [ ] Error-injection pass: malformed protocol, forced illegal access → confirm the Phase 6
      context recreation holds under fault, and that the server survives
- [ ] README results table; finalize `ARCHITECTURE.md` and `PROTOCOL.md` against actual behavior
- **Done when:** every architectural claim in `ARCHITECTURE.md` maps to a number in the table

## Stretch (explicitly not required for the three pillars)

- [ ] Off-thread NVRTC compilation — legal, since `nvrtcCompileProgram` needs no context and only
      `cuModuleLoadData` marshals back to the worker. Requires a cache-stampede guard: concurrent
      requests for the same uncompiled key must park on one in-flight compile, not launch N.
- [ ] `DropOldest` backpressure mode as a flag, for real-time semantics where stale frames are
      worthless
- [ ] Lock-free SPSC queues replacing mutex+condvar
