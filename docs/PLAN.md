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
| canonical `OpChain` (kinds + quantized params, in order) | 2 | the emitted stages, their fusion boundaries and every baked literal. **Params only when `constants` is baked** — parameterized kernels take them as launch arguments, so there they are not codegen inputs and leave the key |
| `channels` | 2 | baked as a literal; changes indexing and which channels the op touches |
| `tile` (naive\|tiled) | 7 (reserved in 2) | a different kernel body |
| `tile_size` | 7 (reserved in 2) | baked into the `__shared__` array dimensions |
| `constants` (baked\|parameterized) | 8 (reserved in 2) | literals versus kernel parameters |

**Not in the key, and never to be added: width and height.** They are launch arguments. This is
enforced structurally — the struct has no such field and the hash takes nothing but the struct.

**Invariant-2 exemption, scoped to this phase — and since closed.** `CLAUDE.md` invariant 2 forbids
`cuMemAlloc` in the per-frame path. Phase 3 is a one-image-at-a-time CLI with no pipeline to
serialize, so allocating intermediate device buffers per invocation was acceptable *here only*.
Phase 5 closed it: `allocate_slots()` now allocates the device ping-pong pair alongside the pinned
slots, and `submit()` has no allocating path left. It cost nothing, because `imgjit-cli` allocates
a slot sized to its one image and so was already taking the pooled path.

## Phase 4 — Network layer · env: Mac (CPU backend)

**Done.** Clean `-Werror` build on macOS with no CUDA toolkit; 8 ctest cases green in all three
configurations — plain, `-DIMGJIT_SANITIZER=address` and `-DIMGJIT_SANITIZER=thread` — with the
`[server]` suite additionally run 20× under each sanitizer to shake out flakiness. Invariant 1's
grep is silent over `src/` and `include/`.

- [x] `docs/PROTOCOL.md` implemented as pure encode/decode functions
- [x] `read_exact()` / `write_exact()`; explicit little-endian pack/unpack; header validation +
      error responses
- [x] Codec unit tests including malformed input: bad magic, truncated header, length mismatch,
      oversized payload
- [x] Validation order + per-error-code connection disposition (drain vs. close) per
      `docs/PROTOCOL.md`; desync test: an error frame followed by a valid frame on the same
      connection must still be served correctly
- [x] `tools/imgjit-server`; acceptor + **reader and writer thread per connection**; bounded MPSC
      queue; per-connection outbox; slot pool with per-connection caps
- [x] `tools/imgjit-client`; both result paths (echo, server-side write); blocking backpressure
      verified with an artificially throttled worker
- [x] Pipelining test: client sends N frames before reading any response — the case that deadlocks
      if slot release is gated on the socket write
- [x] Integration test: N clients × M frames, verified against the CPU oracle
- **Done when:** multi-client localhost run is correct and clean under **ASan and TSan**

**Two drain-accounting paths, not one.** An error can be detected before or after the chain bytes
have been read, and the number of bytes still outstanding differs between them: a chain rejected
from the *header* (`chain_len` over the cap) leaves chain **and** payload on the wire, while a
chain that was read and then failed to *parse* leaves only the payload. Both are status 4 and both
drain, so a single wrong subtraction is invisible until the next frame decodes from the middle of
this one. `tests/test_server.cpp` covers each separately.

**Backpressure is asserted on a counter, not a stopwatch.** `SlotPool::blocked_claims()` and the
queue's high-water mark are exposed through `Server` so the tests can prove the reader actually
parked, rather than timing a run and hoping. The pipelining test asserts it too: 8 frames through
2 slots must block, which is exactly the state release-after-write would deadlock in.

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

**Done.** Verified on Colab (T4, `compute_75`): all 14 ctest cases passed, including the gate —
`phase5_gpu_server` (0.99s: six concurrent connections, differing chains at differing channel
counts, every response diffed against the scalar CPU backend within 1 LSB). Locally: clean
`-Werror` build on macOS, all 8 portable ctest cases green in plain / ASan / TSan trees,
invariant 1's grep silent over `src/` and `include/`.

The baseline is `bench/baseline_phase5.csv`. At 1024²×3, 4 connections, one stream, synchronous:
~270 fps, p50 27 ms round trip. Prewarm's effect is exactly where the harness was built to show
it — first-frame latency 114 ms cold vs 41 ms prewarmed (the ~73 ms is the NVRTC compile, moved
to startup), with p50/p99/fps unchanged because prewarm only touches each connection's first
frame. That 270 fps / 27 ms is the number Phases 6 and 7 are measured against.

- [x] Swap the CPU worker for the GPU worker; `cuCtxCreate` once at startup on that thread
      (`imgjit-server --backend cuda`; the factory already ran on the worker thread, so this
      changed only which backend it returns)
- [x] Pinned slot pool allocated on the worker; connection threads `recv()` directly into slots.
      The pool half was already right from Phase 4 — what Phase 5 had to fix was the *other* end:
      `submit()` was copying the pinned slot into a pageable `Image` before the H2D, which quietly
      undid the whole point of the pool. The transfer now goes straight from the slot.
- [x] **Single stream only** — isolate "is the plumbing correct" from async overlap. An explicit
      `CudaStream` (RAII, `CU_STREAM_NON_BLOCKING`), async H2D → launches → D2H, one
      `cuStreamSynchronize`. Phase 6 makes it K of them and adds the events.
- [x] Startup prewarm of a configured chain list — `imgjit-server --prewarm "<chain>[@channels]"`,
      repeatable, executed inside the backend factory (the one place that is both on the worker
      thread and before `start()` returns). The channel suffix is not decoration: `channels` is
      baked into the kernel, so a chain is warmed for one channel count at a time.
- [x] Minimal timing harness (FPS, p50/p99 round-trip) — `bench/imgjit-bench`, closed-loop with a
      per-connection window, latency matched per `seq_num` off the client's pending map
- [x] **Recorded baseline** committed to the repo — `bench/baseline_phase5.csv`, three rows
      (`phase5_cuda_cold`, `phase5_cuda_prewarmed` from Colab; `phase5_cpu_mac` as the harness
      canary that reproduces on a Mac). See `bench/README.md`.
- **Done when:** concurrent loopback clients with differing chains all match the CPU oracle on
      Colab; TSan clean; no CUDA symbol reachable outside `src/backend/cuda/` — **all met.**

**TSan is gated on the Mac, not the CUDA build, deliberately.** The portable threading is
TSan-clean there, which is the part this project wrote. The driver brings its own threads and
TSan has no interceptors for them, so a report inside `libcuda` would be a tooling limitation,
not a finding — the Mac run stays the gate rather than suppressing driver frames into a false
green.

Phases 6 and 7 are both gated on being "measurably faster" — which requires a number recorded
*here*, with the harness that produced it. `bench/` in Phase 8 then widens the matrix rather than
inventing the measurement, and each phase gate from this point records its numbers under the same
harness so the comparisons are like-for-like.

## Phase 6 — Async multi-stream pipeline · env: Colab

**Done.** `phase6_gpu_async` and `phase6_gpu_recovery` pass on the T4; `phase6_gpu_stress` passes
after the `sobel` oracle-tolerance correction (`tests/test_gpu_pipeline.cpp`). Portable suite green
on macOS in all three trees (plain / ASan / TSan), invariant 1's grep silent. The `--streams 1`
vs `--streams 4` benchmark is in `bench/baseline_phase6.csv` and written up with the Nsight trace
in `bench/phase6_results.md`: **+18% throughput / −36% p99 on the showcase chain**, occupancy mean
2.61 (vs 1.00 serial), 539 ms of copy time overlapped with compute (≈ the whole copy budget),
83% GPU utilisation. Kernels do not overlap kernels — a 1024² kernel fills the T4, so the win is
copy/compute overlap and for a compute-bound chain that ceiling is ~the copy fraction.

- [x] In-flight table across K streams (K=4 by default, `--streams` to sweep it); `cuEventRecord`
      + poll-and-retire. `submit()` returns with the frame still on the GPU; `poll_completions()`
      retires on `cuEventQuery`
- [x] Event-gated buffer return; device memory pool replacing any per-frame `cuMemAlloc` — the
      pool is one device ping-pong pair plus a pinned staging buffer per stream slot, claimed and
      released as a unit
- [x] Instrument queue depth, occupancy, stall counts — `BackendStats` through `IBackend`,
      reported by `imgjit-server` at shutdown and asserted in the tests
- [x] **Context recreation** — teardown/rebuild as a unit: flush the kernel cache and device pool,
      fail every in-flight job with status 6, resume accepting work
- [x] Stress test: sustained repeated runs with output checksums, specifically targeting
      premature slot reuse (`CLAUDE.md` invariant 3)
- [x] Measurably faster than the serial pipeline: `--streams 4` vs `--streams 1` (same binary,
      same harness — the `--streams 1` row *is* the Phase 5 baseline re-measured) is +18%
      throughput and −36% p99 on the showcase chain. Zero checksum drift under stress.
- [x] Nsight Systems trace (`bench/phase6_timeline.png`, `bench/phase6_gpu_trace.csv`): H2D and
      D2H run concurrently with kernels on other streams — 17% of GPU-busy time has ≥2 ops, and
      that 17% accounts for essentially the entire copy budget being hidden

**The D2H destination has to be pinned, and that is not a micro-optimization.** An async
device-to-host copy into *pageable* memory is permitted to behave synchronously, and does. Phase 5
got away with it because one frame was in flight at a time; keeping it here would have serialized
the pipeline the streams exist to build, and the failure mode is a benchmark that improves by a
few percent while Nsight shows the same single-file timeline as before. Each stream slot therefore
owns a pinned staging buffer, and the worker copies out of it when the frame retires.

**Recreation cannot free the frame slots, which is why they stopped being `cuMemAllocHost`
memory.** `cuCtxDestroy` frees every allocation made in that context. The slot pool's base pointer
is held by the server for the life of the process and reader threads are `recv()`ing into
individual slots at the instant a recreation happens, so driver-owned slot memory would turn
recovery into a use-after-free across live connections — silent corruption, in the one code path
whose whole job is to survive a fault. The pages are ours and `cuMemHostRegister` pins them, so
recreation detaches and reattaches at an address that never moves and nothing outside
`src/backend/cuda/` learns that anything happened. See `CLAUDE.md` invariant 2.

**Two error classes, because only one of them is the context's fault.** A driver error may be
sticky — an illegal access poisons a context permanently and every later call returns it — so
`submit()` treats any `CudaError` as a recreation trigger. An `NvrtcError` is a chain that would
not compile, which never touched a context; recovering from one by tearing the GPU down would turn
a single client's bad request into every other client's failed frame.

> **Corrected by Phase 8's error injection.** The premise above is half wrong, and the half that
> is wrong is the interesting one: an illegal access poisons the whole **process**, not the
> context. `cuCtxCreate` after one returns `CUDA_ERROR_ILLEGAL_ADDRESS` too (measured on a T4),
> so the rebuild fails and no in-process recovery is possible. What the design delivers is
> therefore graceful degradation — status 6 for every later frame, server still up — which is
> exactly the fallback `recover()` was written for. The recreation attempt stays: it is right
> for per-context errors and costs one failed rebuild when the fault is process-wide. See
> `tests/test_gpu_faults.cpp` and `CLAUDE.md` invariant 1.

**The benchmark is two runs of one binary, not a comparison against a build that no longer
exists.** `--streams 1` is the Phase 5 pipeline (one frame on the GPU at a time), so the A/B is
like-for-like under the same harness — see `bench/README.md`.

**The benchmark must be run wide enough that the GPU is the bottleneck.** `imgjit-bench` is a
closed loop; at the Phase 5 baseline's `--window 2` / `--slots-per-conn 2` only 8 frames are ever
in the system, both `--streams` settings top out at the same `conns * window / round-trip` ceiling
(~300 fps for the showcase chain on a T4), and the comparison shows nothing. `--window 8` with 32
slots is the regime where `--streams` moves the number. First Colab run without this: streams-1
and streams-4 both reported ~290 fps while the occupancy counter correctly showed mean 1.0 vs 2.6.

**The showcase chain is compute-bound, so its overlap ceiling is ~1.2-1.4x, not 2x.**
`gaussian:1.4` is ~11x11 taps per pixel; the PCIe copy overlap can hide is ~20% of the frame.
`bench/baseline_phase6.csv` therefore also carries an `invert` row — one pointwise op, almost pure
copy — which is where the streams actually earn a large win.

**Occupancy is what separates the two halves of the gate.** "Measurably faster" and "genuinely
overlapped" are different claims, and the first can be had without the second. If `mean_in_flight`
sits near 1.0 with `--streams 4`, the pipeline serialized and the throughput came from somewhere
else, whatever Nsight is squinted at. That is why the number is printed and asserted rather than
inferred from FPS. It is also the fallback for the Nsight timeline itself: some Colab images ship
without `nsys`, and `mean_in_flight` ≈ 2.6 at 4 streams is the same claim the timeline makes.

Context recreation lands here, not in Phase 8, because it is a *design constraint on these
structures* rather than a feature bolted on afterwards: recovery means destroying the kernel
cache, the device pool, and the in-flight table together and rebuilding them. Retrofitting that
two phases after the in-flight table is built is invasive surgery on the most delicate code in the
project. Build them destroyable-as-a-unit now; Phase 8 only injects the fault and observes.

## Phase 7 — Shared-memory tiling · env: Colab

**Done, with a qualified gate.** Verified on Colab (T4, `compute_75`): `phase7_gpu_tiling` passes
— tiled matches the CPU oracle within 1 LSB and matches the naive kernel's own output at tile
8/16/32 across the full op corpus and {1,3,4} channels, the tile edge is confirmed part of the
kernel identity, and resolution stays a launch argument at every tile size. Portable suite green
in plain / ASan / TSan, invariant 1's grep silent. Locally, before ever reaching Colab: a host
simulation of the generated kernels (each block on real threads with a real barrier) came out
bit-identical to naive for all 144 tiled corpus runs at tile 8/16/32 and within 1 LSB of the
oracle — that proved the index arithmetic, not NVRTC or the driver.

Performance (`bench/baseline_phase7.csv`, `bench/phase7_kernel_ms.csv`, written up in
`bench/phase7_results.md`): Gaussian and the full showcase chain are large, unambiguous wins — up
to **3.33x** kernel time and **2.00x** FPS at tile 32. **Sobel is not faster** — a small,
repeatable ~3% *regression* in kernel time at both tile sizes tested, invisible in end-to-end FPS.
This falls short of the bar as originally written below; see the rationale after the checklist for
why the phase is still considered done.

- [x] Tiled stencil codegen variant: `__shared__` tile + halo/apron loads, bounds-clamped,
      parameterized by tile size
- [x] Extend `KernelKey` with the naive|tiled variant **and the tile size** — the tile dimensions
      are baked into the `__shared__` array, so a boolean variant flag would collide two different
      kernels onto one key; both variants remain runtime-selectable
- **Done when:** tiled output matches naive and CPU within tolerance (met), and is measurably
      faster on Gaussian and the fused chain (met, up to 3.33x kernel time); **not** met for Sobel
      alone, which is ~3% slower rather than faster at every tile size tried — see below.

**Sobel does not benefit from tiling, and that is the technique's own limit, not a bug.** Sobel's
radius is always 1 (a fixed 3x3), so the naive kernel already re-reads each input pixel at most 9
times — an order of magnitude less redundancy than Gaussian's 11x11 footprint — and the tiled
path's fixed cost (the strided cooperative load, the barrier, indexing through shared memory
instead of a register) is not paid back by eliminating that little reuse. Both tile sizes tested
land at the same 0.151 ms against naive's 0.146 ms: consistent, not noise. Phase 6 hit the same
shape of result for `invert` under `--streams 4` (a pointwise op with nothing to overlap) and was
marked done with that finding documented rather than as a failed gate; this follows the same call.
`--tile` defaults to naive either way, so nothing regresses for a caller who does not opt in.

**What is staged is the sample after the prologue, which is half the win.** The naive kernel
re-loads, re-converts (`/255`) and re-runs any fused prologue at every tap of every pixel —
121 times per sample for `gaussian:1.4`. The tiled load goes through the naive kernel's own tap
helper, once per staged sample, so edge clamping and prologue folding stay one piece of code and
the per-tap work becomes a shared-memory read. The accumulation around the fetch is the same
text in both variants (`test_codegen.cpp` asserts it), so tiling changes where operands come
from and never the arithmetic — tiled is expected to equal naive exactly, not merely within 1 LSB.

**The barrier comes before the bounds check.** A thread whose output pixel is past the image edge
still owns apron cells; returning early would leave them unloaded and strand the other threads at
`__syncthreads()`. The naive kernel's early return is correct only because it has no barrier.

**The launch shape is codegen's output, not the executor's choice.** A tiled stage's `__shared__`
array and index arithmetic are baked for exactly `tile_size` threads per side, so the block edge
travels with the program (`GeneratedStage::block_dim`) instead of being a constant in the backend
that could drift from it. Tiled stages also carry `__launch_bounds__(tile_size²)`: at tile 32 that
is 1024 threads, and without the cap the compiler may assign more registers than a block can hold,
which fails the launch — a `CudaError`, so a context recreation on every frame.

**The gate is kernel time, not FPS.** A 3×3 sobel is a sliver of a copy- and host-bound frame, so
end-to-end throughput cannot show it getting faster — the same problem Phase 6 had with overlap,
and the same answer: an instrument. `BackendStats::mean_kernel_ms` is read off two timing events
around the stages, excluding copies and compile, and is a clean per-frame number at `--streams 1`
(see `bench/README.md`).

## Phase 8 — Benchmarks + polish · env: Colab

- [x] Parameterized-constants codegen — the one new kernel variant this phase needs before
      its A/B can be measured. `--constants parameterized` on `imgjit-server` / `imgjit-cli`;
      `phase8_gpu_constants` verified on Colab (T4, 8.16s): parameterized kernels match the
      CPU oracle, match the baked kernel's own output, and one compile serves several
      parameter values. Portable suite green, invariant 1's grep silent.
- [x] `bench/` CSV harness — widens the Phase 5 timing harness over the orthogonal matrix:
      naive|tiled × sync|async, plus fused-vs-unfused chain, cold-vs-warm cache,
      baked-vs-parameterized constants (the constants mode must already be a `KernelKey` input —
      see Phase 3, or the parameterized run silently reuses the baked kernel)
- [x] Chain lengths (1, 2, 4, 6 ops); report FPS, p50/p99 round-trip latency
- [x] Sweep 512² → 4K, report FPS and GB/s. **First pass invalidated and rerun** — see below.

      **Two full sessions recorded on Colab (T4), written up in `bench/phase8_results.md`.**
      Cell 22 of `colab/run.ipynb` runs five focused sweeps (constants, cache, tile×streams,
      fusion, chain length — one axis each, the rest pinned) into `baseline_phase8.csv` /
      `phase8_matrix.csv`; cell 24 is the resolution sweep, into `phase8_resolution.csv` /
      `phase8_resolution_summary.csv`. `BackendStats` gained `jit_compiles` (compile count was
      only ever printed at prewarm time) and the resolution cell reads `submit_stalls` directly
      off the shutdown line. (`jit_compiles`, not `nvrtc_compiles` — named for the technique, not
      the compiler, since the latter in a portable header trips invariant 1's grep.)

      **The resolution sweep's first pass was invalidated by its own methodology and rerun.** It
      held concurrency fixed at 8 frames in flight for every resolution; `imgjit-bench` is a
      closed loop, so fps is mechanically tied to latency at fixed concurrency (Little's Law)
      whether or not the GPU is the bottleneck, and every recorded row landed within ~15% of that
      theoretical cap — the "peak Mpx/s at 1024²" shape drawn from it could not be told apart from
      "8 in flight did not saturate the pipeline at every size." Withdrawn rather than left
      standing. The rerun sizes the slot to each resolution's real frame and raises concurrency
      until `submit_stalls` goes nonzero — a direct signal of GPU-side contention, not an
      inference from an fps ratio. Confirmed GPU-bound at every resolution tested; the "peak at
      1024²" claim did not survive the fix — 512² and 1024² are now within 1.1% of each other, and
      the real falloff is at 2048²+ (`bench/phase8_results.md` §2).

      **A second full session also caught something a single run could not: a headline finding
      that reversed sign.** Session 1 measured tile16 at `--streams 4` as −3.8% versus tile16 at
      `--streams 1` and called it noise; session 2 measures the identical configuration at
      **+28.9%**, making tile16/streams4 the fastest full-chain config in the matrix rather than
      tile16/streams1. Both sessions' raw numbers are kept side by side in
      `bench/phase8_results.md` §0 and §3 rather than one silently overwriting the other, and the
      tile×streams interaction is recorded as **not established** — direction and magnitude both
      open — pending a third session. Five of the seven cross-session comparisons made *do*
      replicate closely (parameterized penalty, tiled speedup, fusion cost); this is the one that
      did not, and it is exactly the kind of thing `bench/phase8_results.md` §0 exists to catch.

      Headline numbers, both sessions agreeing unless noted: **parameterized constants cost
      1.36–1.42x kernel time** and **buy a 39–68% lower worst-first-frame latency** (a real but
      session-dependent magnitude — see below) on a cold cache with four distinct sigmas, which
      always compile 4x baked vs 1x parameterized (a discrete count, identical both sessions);
      **tiling gets 2.65–2.68x kernel time** on the showcase chain; **streams add +11–17% on the
      naive kernel** (replicates) but the streams×tiling interaction does **not** (see above);
      fusing a prologue before a stencil costs **1.26–1.28x** more kernel time, not less; an NVRTC
      compile costs tens-to-low-hundreds of ms, consistent with `ARCHITECTURE.md`'s "~50–200 ms"
      claim, though the specific per-compile figure derived in session 1 (~97 ms baked vs ~63 ms
      parameterized) reversed in session 2 (~43 ms vs ~84 ms) and is retracted as unreliable.
      The harness's own kernel-time repeatability is 3.5% *within* one session but 9.2% has now
      been observed *within* a second one — treat 9.2% as the current floor a difference must
      clear. This also qualifies Phase 7's ~3% tiled-Sobel regression further toward
      "not measurably faster" (already noted there after session 1; unchanged by session 2, which
      did not retest Sobel tiling).
- [ ] Error-injection pass: malformed protocol, forced illegal access → confirm the Phase 6
      context recreation holds under fault, and that the server survives

      **Written and run on Colab (T4), and it falsified a premise the project had carried since
      Phase 6** — see the correction under Phase 6 above. Recreation does *not* recover from an
      illegal access, because the fault is process-wide: `cuCtxCreate` afterwards returns the
      same error. What holds instead is the fallback `CudaBackend::recover()` already
      implemented: every later frame is answered with status 6, the worker does not die, and
      the server keeps accepting connections. The cases were rewritten to assert that contract
      (and to tolerate a driver that *does* recover, so they stay honest on other hardware)
      rather than the recovery that does not happen.

      Three ctest cases, **three processes, which is not optional**: an injected fault poisons
      the whole process, so the first version — all three under one tag — took its two innocent
      cases down with it in `cuCtxCreate`. `phase8_gpu_faults` (malformed protocol against a
      GPU-backed server, asserting `context_recreations == 0`: a protocol error must never
      become a GPU event), `phase8_gpu_fault_backend`, `phase8_gpu_fault_server`.
      **Next:** rerun all three on Colab now that they are split and assert the real contract.
- [ ] README results table; finalize `ARCHITECTURE.md` and `PROTOCOL.md` against actual behavior
- **Done when:** every architectural claim in `ARCHITECTURE.md` maps to a number in the table

**Parameterized means the gaussian's radius and weights, `brightness` and `threshold` become
kernel arguments; the channel count and sobel's 3x3 stay literals.** Those two are the
kernel's shape, not values a client picks. The weights travel by value as one
`kMaxGaussianTaps` (25) float struct — kernel arguments have a compile-time size — so they
sit in the parameter bank rather than behind a per-frame upload, and the A/B measures
unrolling and constant folding, not an extra copy.

**Parameter values leave the key in parameterized mode.** Invariant 4 is "every codegen input,
and nothing else", and there they are not codegen inputs: `gaussian:1.4` and `gaussian:3` are
one compile. The cost is that the launch must take its values from the frame's chain and never
from the key — the cached kernel holds whichever frame compiled it — and the GPU test diffs
each of several parameter values against the oracle to prove it does.

**Parameterized is naive-only.** A tiled stage sizes its `__shared__` array from the radius
at compile time, which a parameterized kernel does not know until launch. Dynamic shared
memory would lift that at the cost of reworking Phase 7's verified tile indexing; the plan
lists baked-vs-parameterized as its own axis rather than crossed with tiling, so the
combination is refused at startup instead, like a bad `--tile`.

**A prologue costs more than it saves, and that is a finding rather than a bug.** The fusion
sweep measured the showcase chain's fused kernels at **1.28x** the summed kernel time of the same
four ops unfused (`bench/phase8_results.md` §4), because `grayscale` sits in front of an 11x11
Gaussian and is therefore re-run at all 121 taps instead of once per output pixel. Fusion still
saves the launches and the global round trips, and an *epilogue* is unaffected — codegen's
existing preference for attaching pointwise runs backwards is exactly the right one, now with a
number behind it. `ARCHITECTURE.md`'s unconditional framing of fusion is qualified to match.
The obvious follow-up — emit a prologue as its own pointwise kernel once the following stencil's
radius is large enough — is **not** taken: it is a codegen change with its own correctness
surface, and the op set and fusion plan were settled in Phase 2. Recorded, not scheduled.

**The fault has to be real, which is what makes this pass different from Phase 6's `[recovery]`.**
Those cases call `recreate_context()` and feed the backend a null-input job; neither produces a
*sticky driver error*, which is the only state recovery actually exists for and the one that
cannot be reached by asking politely. `tests/test_gpu_faults.cpp` injects a null-pointer store
from a kernel — device address 0 is never mapped, so it is a genuine
`CUDA_ERROR_ILLEGAL_ADDRESS` on any device — and asserts the injection returned an error before
asserting anything about recovery, so an injection that quietly did nothing fails the test
instead of making every later assertion pass vacuously.

**Faulting a *running server* without breaking invariant 1 needed a specific trick.** The server
owns its backend on the worker thread, so reaching in from the test thread to poison the context
would be exactly the violation invariant 1 exists to prevent. But `IBackend::submit()` is already
called on the worker thread — so a test-only decorator around `CudaBackend` can inject the fault
from inside `submit()`, legally, with the test thread only ever flipping an atomic. **No
production code carries a test hook**, and the alternative (a debug flag on the server) was
rejected for that reason. The malformed-protocol half runs against a GPU-backed server rather
than Phase 4's CPU one, and its load-bearing assertion is `context_recreations == 0`: a protocol
error must be rejected in the network layer and never become a GPU event.

## Stretch (explicitly not required for the three pillars)

- [ ] Off-thread NVRTC compilation — legal, since `nvrtcCompileProgram` needs no context and only
      `cuModuleLoadData` marshals back to the worker. Requires a cache-stampede guard: concurrent
      requests for the same uncompiled key must park on one in-flight compile, not launch N.
- [ ] `DropOldest` backpressure mode as a flag, for real-time semantics where stale frames are
      worthless
- [ ] Lock-free SPSC queues replacing mutex+condvar
