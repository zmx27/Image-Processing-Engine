# imgjit — A Network-Accelerated, GPU-JIT Image Processing Engine

A native TCP server that takes a raw image frame and a filter chain (`"grayscale,gaussian:1.4,sobel,threshold:0.3"`),
JIT-compiles that exact chain into a CUDA kernel at runtime with NVRTC, and runs it on the GPU
through the raw CUDA Driver API — no `cudart`, no `<<<>>>`, no ahead-of-time kernel binary. The
network layer, the kernel cache, and the async execution pipeline are all built to keep the GPU
fed continuously instead of stalling on the transfer or the compile.

This is a from-scratch systems project, not a wrapper around an existing image library. Every
kernel, every RAII wrapper around a CUDA handle, and every byte of the wire protocol is written in
this repository.

## Table of contents

1. [Architectural motivation](#1-architectural-motivation--executive-summary)
2. [Systems innovations](#2-microarchitectural--systems-innovations)
3. [Repository layout](#3-repository--directory-architecture)
4. [Build & prerequisites](#4-build-system--prerequisites)
5. [Verification, testing & benchmarking](#5-verification-testing--benchmarking)

---

## 1. Architectural Motivation & Executive Summary

### The problem

Say you want a server that applies an arbitrary chain of image filters — grayscale, then a
Gaussian blur, then a Sobel edge detector, then a threshold — to frames coming in over the
network, as fast as a GPU can go. Two conventional approaches, both bad:

- **Precompile every kernel you might need.** Six ops, each with continuous parameters (blur
  sigma, threshold level), chained in any order and any length, is a combinatorially large set of
  kernels. You either ship a binary bloated with permutations nobody asked for, or you compile one
  generic kernel that branches on op type at runtime — which is a warp-divergence and
  register-pressure tax paid on every pixel, forever, because the compiler can never unroll a loop
  bound or fold a constant it doesn't know about at compile time.
- **Do the I/O synchronously.** Read a frame, process it, write the response, read the next frame.
  The GPU sits idle during every network read and PCIe transfer, and a single slow client stalls
  every other client behind it.

### The solution

Three pieces, fused together:

1. **A multi-threaded binary TCP ingest layer** that reads frames straight into pinned host memory
   and hands them to a worker over a bounded lock-protected queue, so backpressure is a property
   of the system (slot-pool exhaustion) rather of something bolted on with a rate limiter.
2. **An on-the-fly NVRTC JIT pipeline** that turns the *specific chain a client asked for* into one
   self-contained CUDA translation unit, with every constant (filter radius, blur weights,
   threshold, channel count) inlined as a literal, and caches the compiled `CUfunction` by a hash
   of exactly those inputs. A chain compiles once, no matter how many times or at how many
   resolutions it's requested afterward.
3. **Direct CUDA Driver API execution** — `cuCtxCreate`, `cuModuleLoadData`, `cuLaunchKernel`,
   `cuMemcpyHtoDAsync`/`cuMemcpyDtoHAsync`, `cuEventRecord` — on a pool of streams, so a frame's
   transfer-in, compute, and transfer-out can overlap with a *different* frame's transfer-in,
   compute, and transfer-out on another stream.

### End-to-end data path

```
                     ┌──────────────── portable core — builds on macOS, no GPU ────────────────┐
 client(s) ──TCP────►│ acceptor thread → one reader + one writer thread per connection           │
                     │                                                                           │
                     │  reader:  read_exact() → validate header → claim pinned slot (SlotPool,   │
                     │           blocks when the pool is empty — THIS is backpressure)           │
                     │           recv() payload DIRECTLY into the pinned slot, zero staging copy │
                     │                    │                                                      │
                     │                    ▼                                                      │
                     │           push(FrameJob{slot, seq_num, chain})                             │
                     │              bounded MPSC queue (mutex + condvar, capacity N)              │
                     └────────────────────┬──────────────────────────────────────────────────────┘
                                          │ pop
                     ┌──── GPU worker thread — the ONLY thread that ever touches the driver ─────┐
                     │                                                                           │
                     │  KernelKey = hash(op-chain, channels, tile, tile_size, constants-mode)     │
                     │      hit  → CUfunction, straight from KernelCache                          │
                     │      miss → emit_cuda_source() → nvrtcCompileProgram → PTX →                │
                     │             cuModuleLoadData → cuModuleGetFunction                          │
                     │                                                                           │
                     │  claim an idle CUstream slot (K=4 by default):                             │
                     │      cuMemcpyHtoDAsync (pinned host → device)                               │
                     │      cuLaunchKernel × (1 per stencil stage; pointwise ops fused in)         │
                     │      cuMemcpyDtoHAsync (device → pinned staging buffer)                     │
                     │      cuEventRecord                                                          │
                     │                                                                           │
                     │  poll_completions(): cuEventQuery each in-flight slot; on completion,       │
                     │      copy result out of the slot, release the slot, hand off the frame      │
                     └────────────────────┬──────────────────────────────────────────────────────┘
                                          │ per-connection outbox (never gated on a socket write)
                                          ▼
                     writer thread → send response (echo) and/or stb_image_write to disk
```

`IBackend` — the interface between the server and whatever does the actual pixel-crunching — is
implemented twice: once by the CUDA backend above, and once by a scalar CPU backend that runs the
identical op set in plain C++ loops. Same server binary, same protocol, same tests — swap
`--backend cpu` for `--backend cuda` and nothing else changes. That's what makes it possible to
develop and test the entire network layer, threading model, and protocol on a machine with no
NVIDIA GPU at all (every Mac, Intel or Apple Silicon), and it's also invariant 9: the CPU backend
is the correctness oracle every GPU output gets diffed against.

**Scope, stated plainly:** this is one process, one GPU, many client connections — not a
multi-node cluster. "Distributed" in the sense of horizontal scale-out was named early as an
over-scoping risk and deliberately left out; what's here is a single node doing genuine
network/compute/copy overlap, which is a big enough problem on its own.

---

## 2. Microarchitectural & Systems Innovations

### 2.1 Zero-AOT dynamic kernel JIT (NVRTC & the Driver API)

There's no `.cu` file compiled by `nvcc` anywhere in the runtime path, and no CUDA language
support enabled in CMake (`project(imgjit LANGUAGES CXX)` is deliberate). The pipeline from a
client's request string to a running kernel is entirely runtime:

```
"grayscale,gaussian:1.4,sobel,threshold:0.3"
        │  parse_op_chain()            — validate, quantize params to 2 decimals, canonicalize
        ▼
      OpChain                          — {kGrayscale}, {kGaussian, 1.40}, {kSobel}, {kThreshold, 0.30}
        │  + channels, tile, tile_size, constants-mode
        ▼
      KernelKey                        — every codegen input, hashed with 64-bit FNV-1a
        │  cache miss
        ▼
      emit_cuda_source(KernelKey)      — structural string assembly (not template substitution):
        │                                one self-contained translation unit, prelude + N stages,
        │                                every stencil weight/radius/threshold baked as a literal
        ▼
      nvrtcCompileProgram → PTX
        │
        ▼
      cuModuleLoadData → cuModuleGetFunction → CUfunction
        │  (KernelCache stores this keyed by the KernelKey's hash — never recompiled again)
        ▼
      cuLaunchKernel(width, height as launch-time arguments)
```

The kernel cache key is deliberately *every codegen input and nothing else* — see
`include/imgjit/core/kernel_key.h`. Width and height are conspicuously absent: they're launch-time
arguments, not part of the compiled identity, because baking them would turn every new resolution
into a fresh ~100 ms compile and defeat the entire point of caching. Measured on a T4: the first
run of a chain pays NVRTC's cold-compile cost (~73 ms for the showcase chain); every repeat,
at *any* resolution, is a cache hit at ~0.3 ms — roughly **250x**. NVRTC compiles are tracked by an
explicit counter (`jit_compiles` in `BackendStats`) so this is asserted in tests, not eyeballed.

Since NVRTC has no default include path, generated source can't `#include` anything — not even
`<cstdint>`. The stable device helpers (clamping, `uint8`↔`float` conversion, luma) live as one
raw-string literal in `src/backend/cuda/kernel_prelude.h`, and codegen concatenates that with
whatever the chain needs. `imgjit-cli --dump-source` writes the generated `.cu` text to disk, which
is what you actually read when you want to see what got compiled — not the fragments it was
assembled from.

**Phase 1 proved the driver plumbing before JIT was in the picture at all**, by loading a PTX blob
that `nvcc --ptx` compiled *ahead of time* (`tools/invert_aot.cu`, built by an `add_custom_command`,
never through `enable_language(CUDA)`) through the same `cuModuleLoadData` / `cuLaunchKernel` path.
That way, a Phase 1 failure could never be ambiguous between "the driver plumbing is wrong" and
"the JIT plumbing is wrong" — the AOT checkpoint isolates the first from the second before NVRTC
enters at all. `tools/imgjit-spike.cpp` still runs that checkpoint as a standalone smoke test.

**The CPU backend exists to verify all of this is correct**, not as a fallback feature.
`src/backend/cpu/ops.cpp` implements the identical closed six-op set in plain scalar C++ loops —
`grayscale`, `invert`, `brightness`, `threshold`, `gaussian`, `sobel` — using the same semantics
(channel-invariant chains, alpha passthrough, `[0,1]`-normalized float math, clamped/replicate edge
addressing) that bind the generated kernels. Every GPU chain, at every supported channel count, is
diffed against this oracle within tolerance (**≤1 LSB** on `uint8` output, since FMA contraction and
reassociation make bit-equality the wrong bar for stencils); `invert`, being purely integer
pointwise arithmetic, is the one op compared bit-exact. This is also what makes local development
possible without a GPU: `imgjit-cli --backend cpu` and the whole `[server]` test suite run on a
Mac with zero CUDA toolkit present.

### 2.2 Filter-chain fusion and constant baking

A naive translation of a 4-op chain is 4 kernel launches and 4 round trips through global memory.
Codegen instead emits **at most `#stencil_ops + 1` kernels**: pointwise ops (`grayscale`, `invert`,
`brightness`, `threshold`) fuse into the registers of whichever stencil stage (`gaussian`, `sobel`)
they sit next to, and only stencils — which need neighboring pixels — terminate a stage.

```
grayscale → gaussian:1.4 → sobel → threshold:0.3
└────┬────┘  └─────┬─────┘  └─┬─┘   └────┬─────┘
  prologue      stage 0      stage 1 (epilogue: threshold folds in)
 (re-run at
 every tap!)
```

A pointwise run always tries to attach *backwards*, as an epilogue, in preference to forward as a
prologue — an epilogue runs once per output pixel; a prologue gets re-executed at every stencil
tap. That's why only the *first* stage in a chain can ever carry a prologue.

Fusing across a boundary changes *where a value lives* (register vs. global memory) but must never
change the *numerics*: the CPU oracle materializes a rounded `uint8` between every op, so a fused
kernel re-quantizes at each fusion boundary (`imgjit_quantize`, one `roundf`) to match. Skipping
that isn't a rounding nicety — Sobel's coefficients sum to 8 in absolute value, so a half-LSB
difference per tap amplifies to ~4 LSB, and a threshold folded after a stencil flips a boundary
pixel 0↔255.

**Measured tradeoff** (`bench/phase8_results.md` §4): fusing an epilogue is free or better, but
fusing a *prologue* in front of an 11×11 Gaussian costs **1.26–1.28x** the kernel time of the same
four ops run as four separate kernels — because `grayscale` gets re-evaluated at all 121 taps
instead of once. Fusion still saves the launches and the round trips (through the network, too —
one chain is one request), it just isn't a free compute win when the fused op sits ahead of a wide
stencil. Codegen's backward-attachment preference already minimizes this; the benchmark just puts
a number on why that preference exists.

**Constant baking is a second, orthogonal axis.** Filter radius, Gaussian weights, threshold, and
channel count are normally emitted as compile-time literals, so NVRTC can fully unroll the stencil
loop and constant-fold. `--constants parameterized` takes the opposite tradeoff — the same
kernel structure, but every op parameter is a launch argument instead:

| | kernel speed | compiles for 4 distinct σ values |
|---|---|---|
| **baked** (default) | **1.36–1.42x faster** | 4 (one per distinct value) |
| **parameterized** | baseline | **1** |

Baked wins on raw speed but pays a fresh NVRTC compile for every new parameter value; parameterized
compiles once and serves any value from then on, worth a **39–68%** reduction in worst-case
first-frame latency on a cold cache across four different sigmas. The rule that falls out: bake for
a server with a handful of fixed presets, parameterize for one where a client picks an arbitrary
value on every request. Parameterized mode is naive-only — a tiled stage sizes its `__shared__`
array from the stencil radius at *compile* time, which a parameterized kernel doesn't know until
launch, so the combination is refused at startup rather than silently wrong.

### 2.3 Microarchitectural cache locality (shared-memory tiling)

The naive stencil kernel launches one thread per output pixel, and every thread independently
re-reads its own neighborhood from global memory. For an 11×11 Gaussian tap (`gaussian:1.4`,
radius 5), adjacent output pixels share about 10 of those 11 columns — the same bytes get pulled
across the PCIe-attached DRAM bus dozens of times over.

The tiled variant stages that redundancy through `__shared__` memory instead:

1. A block of `tile_size × tile_size` threads cooperatively loads a tile of that many output
   pixels **plus a radius-wide apron** (the halo) around it — through the exact same clamped tap
   helper the naive kernel calls, so edge handling and any fused prologue are one piece of code in
   both variants.
2. `__syncthreads()` — and critically, **the barrier comes before the bounds check**. A thread
   whose output pixel falls past the image edge still owns apron cells other threads need; an
   early return before the barrier would strand the rest of the block at it.
3. Every stencil tap after that reads shared memory instead of issuing a fresh global load.

What's staged is the sample *after* the prologue has already run on it — half the win, since the
naive kernel re-loads, re-converts, and re-runs the prologue at every one of the 121 taps, while
the tiled kernel does that once per input sample and then just reads it back out of shared memory
121 times. The accumulation math itself is identical text in both variants (asserted directly in
`tests/test_codegen.cpp`), so tiling only ever changes *where the operand comes from*.

The tile edge travels with the generated kernel rather than being an executor-side constant
(`GeneratedStage::block_dim`), and tiled stages carry `__launch_bounds__(tile_size²)` — at
`tile=32` that's 1024 threads per block, and without the hint the compiler is free to allocate more
registers than a block that size can actually hold, which fails the launch outright.

**Measured** (`bench/phase7_results.md`, kernel time only — no copy, no host work, no compile):

| chain | naive | tile 16 | tile 32 |
|---|---|---|---|
| `gaussian:1.4` (radius 5) | 1.581 ms | 0.808 ms (**1.96x**) | 0.475 ms (**3.33x**) |
| `sobel` (radius 1) | 0.146 ms | 0.151 ms (0.97x) | 0.151 ms (0.97x) |
| full showcase chain | 2.348 ms | 0.927 ms (**2.53x**) | 0.617 ms (**3.81x**) |

Sobel doesn't benefit — its radius is always 1, so the naive kernel already re-reads each pixel at
most 9 times, an order of magnitude less redundancy than Gaussian's footprint, and the tiled path's
fixed overhead (the strided cooperative load, the barrier, indexing through shared memory instead
of a register) isn't paid back. The ~3% gap is small enough that repeated benchmark runs put the
harness's own noise floor at up to 9.2% on kernel time — the honest read is "not measurably faster
on Sobel," not "a regression." `--tile` defaults to naive, so nothing regresses for a caller who
doesn't opt in.

### 2.4 Asynchronous latency hiding (streams & pinned memory)

Two host-side allocation habits make the overlap possible, and both are enforced structurally, not
by convention:

- **Frame slots are pinned once, at worker startup, never per frame.** `cuMemAllocHost` requires a
  current context, so only the worker thread can call it, and it implicitly synchronizes — doing it
  per frame would serialize the exact pipeline it exists to parallelize. The frame slot pool is
  actually backed by `cuMemHostRegister` on process-owned pages (`RegisteredHostBuffer`) rather than
  driver-owned `cuMemAllocHost` memory — see [2.5](#25-fault-isolation--context-recreation) for why.
- **The device-to-host copy must land in pinned memory, full stop.** An async D2H copy into
  *pageable* memory is permitted by the driver to run synchronously, and it does — every call still
  *looks* asynchronous, but the pipeline silently serializes anyway. Each stream slot owns its own
  small pinned staging buffer for exactly this reason.

`CudaBackend::StreamSlot` is the unit of everything a frame touches on the GPU: a `CUstream`, a
completion `CUevent`, a device ping-pong buffer pair (stages read one, write the other), and the
pinned staging buffer. `submit()` claims an idle slot, queues H2D → N kernel launches → D2H on it,
records the event, and **returns immediately** while the GPU is still working — `poll_completions()`
retires whichever slots' events have since fired, via `cuEventQuery`, in whatever order the GPU
finished them (not submission order, which is exactly why the wire protocol carries a client-chosen
`seq_num` instead of relying on FIFO).

**A buffer returns to its pool only after `cuEventQuery` confirms its event complete — never on the
return of the launch or the copy call that used it.** This is the most important invariant in the
codebase: get it backwards and a slot gets reused while its DMA is still
in flight, which corrupts data silently instead of crashing. It's structural here rather than a
rule to remember — there is no code path that hands out a device buffer except by claiming a
`StreamSlot`, and `busy` is cleared in exactly one place.

**Measured** (`bench/phase6_results.md`, T4, `K=4` streams, showcase chain vs. the Phase-5
single-stream baseline):

| | mean frames in flight | peak | fps | p50 | p99 |
|---|---|---|---|---|---|
| `--streams 1` (serial) | 1.00 | 1 | 263.4 | 111.5 ms | 202.1 ms |
| `--streams 4` | **2.61** | 4 | **310.6** (+18%) | 102.5 ms | **129.3 ms** (−36%) |

An Nsight Systems trace over the same run confirms genuine overlap rather than a better number that
happens to look like one: H2D + D2H total ~612 ms of the run, and 539 ms of that runs concurrently
with a kernel on another stream — essentially the whole copy budget hidden, at 83% overall GPU
utilization. Kernels never overlap *kernels*, since one 1024² image already fills every SM on a T4
— the overlap is strictly copy/compute, which is why the win tracks the copy fraction of the frame
(a compute-bound chain like the showcase one caps out around 1.2–1.4x; a nearly-pure-copy op like
`invert` is where streaming would earn the most, but at that point host-side overhead — polling
four events, sweeping four slots — outweighs the copy it would hide, so `invert` measures ~6%
*slower* at 4 streams. `K=4` is tuned for real multi-op chains, not a single passthrough op.)

### 2.5 Fault isolation & context recreation

Async CUDA errors are sticky: they frequently surface at the *next* driver call rather than the one
that caused them, and an illegal memory access can poison a `CUcontext` such that every subsequent
call on it also fails. The backend distinguishes two error types precisely because only one of
them is the context's fault: an `NvrtcError` is a compile failure — never touched the context, and
tearing the GPU down for a single client's malformed request would fail every *other* client's
frame for no reason. A `CudaError` out of the driver, on the other hand, triggers
`CudaBackend::recreate_context()` — destroy the context, the kernel cache, and every stream slot as
one unit, and rebuild them, because a `CUmodule` belongs to its context and none of these
structures' lifetimes can be untangled from each other.

The frame slot pool survives a recreation on purpose. It's registered with `cuMemHostRegister`
against process-owned pages rather than allocated by the driver with `cuMemAllocHost`, specifically
because `cuCtxDestroy` frees every allocation the *driver* made in that context — and reader
threads are actively `recv()`ing into individual slots at the instant a recreation might run.
Driver-owned slot memory would turn recovery into a use-after-free on every live connection. Owning
the pages ourselves makes recreation a detach/reattach at an address that never moves.

**What Phase 8's fault-injection testing actually found is the honest limit of this design**, and
it's worth stating plainly rather than glossing over: for the fault that matters most — a genuine
illegal memory access — recreation **does not succeed**. The access poisons the whole *process*,
not just the context, so on a T4 the `cuCtxCreate` inside the rebuild itself returns
`CUDA_ERROR_ILLEGAL_ADDRESS`. `recover()` already has a path for that: it swallows the failed
rebuild and every subsequent frame gets answered with protocol status 6 (internal error) instead of
the connection or the process going down. What's actually guaranteed is **a live server that
degrades gracefully and keeps answering other clients' malformed-input errors correctly** — not
*transparent recovery* where work resumes. `tests/test_gpu_faults.cpp` injects a real null-pointer
store from a kernel (not a synthetic call to `recreate_context()`) against a live server with
concurrent traffic on other connections, and asserts exactly that contract — including that a frame
which *does* come back `kOk` is never silently corrupted, and that a protocol error from malformed
input never triggers a context recreation in the first place (`context_recreations == 0`).

### 2.6 Backpressure and the network threading model

The server has no rate limiter and drops no frames silently. Backpressure is two structural
mechanisms instead:

- **`SlotPool`** (`include/imgjit/util/slot_pool.h`) owns a free-list of fixed-size slot indices,
  not the backing storage itself — the CPU backend hands it heap memory, the CUDA backend hands it
  pinned memory, and not a line of pool logic changes between them. A connection's reader thread
  blocks on `claim()` when the pool (or that connection's own per-owner cap) is exhausted, which
  stops it `recv()`ing, which fills the client's TCP send buffer through ordinary flow control.
- **`BoundedQueue<FrameJob>`** between the reader threads and the single GPU worker is a fixed-capacity
  mutex/condvar queue; `push()` blocks the same way when it's full.

Both counters are exposed and asserted in tests (`SlotPool::blocked_claims()`, the queue's
`high_water()`), rather than inferred from a timing that would be flaky.

**The worker thread never touches a socket, and a slot is never held past its response being
written** (invariant 10) — two separate failure modes make this structural rather than
stylistic. If the worker wrote responses directly, one slow client's `send()` would head-of-line
block every other client behind the same worker. And if a slot were released only *after* its
response was written, a pipelining client (frames 1..N sent before any response is read — the
default mode of `imgjit-client`) would deadlock its own connection: the reader blocks claiming a
slot for frame N+1, so frame 1's response is never written, so frame 1's slot is never freed.
Completions are copied out of the slot immediately, the slot is released, and only *then* is the
response pushed to a per-connection outbox that a separate writer thread drains.

**Wire format** (`docs/PROTOCOL.md`), packed field-by-field, little-endian, never `memcpy`'d as a
struct:

| Request field | bytes | | Response field | bytes |
|---|---|---|---|---|
| magic (`0xDEADBEEF`) | 4 | | magic | 4 |
| version | 1 | | version | 1 |
| flags (bit 0 = echo, bit 1 = server-write) | 1 | | status (0 = ok) | 1 |
| seq_num | 4 | | seq_num | 4 |
| width, height | 4+4 | | width, height | 4+4 |
| channels | 1 | | channels | 1 |
| payload_len | 4 | | payload_len | 4 |
| chain_len + chain | 2+n | | payload (if requested & ok) | payload_len |
| payload | payload_len | | | |

`seq_num` exists specifically because completions retire out of submission order once multiple
streams are in flight — the client keeps a `seq_num → pending` map and matches responses by it,
never by arrival order. Every declared length is checked against a hard cap *before* anything is
allocated (`kMaxPayloadBytes` = 64 MiB, `kMaxChainLen` = 256 bytes by default), with the
dimension product computed in `uint64_t` specifically because `65535 * 65535 * 4` overflows
`uint32_t` — a wrapped product that happened to equal a small `payload_len` would otherwise pass
validation and then get used as an indexing bound.

---

## 3. Repository & Directory Architecture

```
image-processing/
├── CMakeLists.txt                # root; auto-detects CUDA → IMGJIT_ENABLE_CUDA
├── docs/
│   ├── PLAN.md                   # phased build log — what's done, what's next
│   ├── ARCHITECTURE.md           # design rationale, the six load-bearing decisions
│   └── PROTOCOL.md               # wire format spec, byte-for-byte
├── colab/
│   └── run.ipynb                 # clones, builds, runs the full test + benchmark matrix on a T4
│
├── include/imgjit/                          # public headers — mirrors src/ layout
│   ├── core/
│   │   ├── image.h                #   Image: uint8, 1/3/4 channel, row-major, explicit stride
│   │   ├── op_chain.h              #   the closed 6-op IR + parser/canonicalizer
│   │   └── kernel_key.h            #   KernelKey — the ONLY codegen-cache identity
│   ├── backend/
│   │   ├── backend.h                #   IBackend: submit()/poll_completions(), identical CPU/CUDA
│   │   └── cpu/{cpu_backend.h,ops.h} #   the correctness oracle's public surface
│   ├── net/
│   │   ├── protocol.h              #   wire codec as pure functions — no sockets, no I/O
│   │   ├── server.h                #   Server: acceptor + reader/writer threads + worker
│   │   ├── client.h                #   reference client, seq_num → pending map
│   │   └── socket.h                #   thin RAII socket wrapper
│   └── util/
│       ├── slot_pool.h              #   pinned/heap frame-slot free list (owns indices, not storage)
│       ├── bounded_queue.h          #   mutex+condvar MPSC queue between readers and the worker
│       └── image_io.h               #   PNG load/save (CLI/test convenience — never on the wire)
│
├── src/
│   ├── core/                      # Image, OpChain, KernelKey — portable, no CUDA
│   ├── net/                       # server.cpp, client.cpp, protocol.cpp, socket.cpp — portable
│   ├── util/                      # slot_pool.cpp, stb wrapper — portable
│   ├── backend/
│   │   ├── cpu/                   # ops.cpp, cpu_backend.cpp — the scalar reference (portable)
│   │   └── cuda/                  # the ONLY directory allowed to #include <cuda.h> / <nvrtc.h>
│   │       ├── codegen.{h,cpp}     #   KernelKey → CUDA source text (portable! no CUDA headers)
│   │       ├── kernel_prelude.h    #   stable device helpers as one raw-string literal
│   │       ├── cuda_raii.{h}       #   RAII: CudaContext, CudaModule, DeviceBuffer, CudaStream,
│   │       │                       #   CudaEvent, PinnedHostBuffer, RegisteredHostBuffer
│   │       ├── cuda_check.h        #   CU_CHECK / NVRTC_CHECK — throw on any driver/NVRTC failure
│   │       ├── kernel_cache.{h,cpp}#   KernelKey → {CUmodule, CUfunction}, with a compile counter
│   │       └── cuda_backend.{h,cpp}#   IBackend over K streams; context recreation on fault
│   └── CMakeLists.txt
│
├── tools/
│   ├── imgjit-server.cpp          # the frame server binary
│   ├── imgjit-client.cpp          # reference client — pipelines by default
│   ├── imgjit-cli.cpp             # file in, file out, through the real IBackend (no network)
│   ├── imgjit-spike.cpp           # Phase 1: AOT PTX + NVRTC JIT driver-plumbing smoke test
│   └── invert_aot.cu              # the one .cu file in the repo — compiled by nvcc --ptx only
│
├── bench/
│   ├── imgjit-bench.cpp           # closed-loop timing harness: fps, p50/p99, CSV output
│   ├── README.md                  # how every recorded benchmark was produced, and why
│   └── phase{6,7,8}_results.md    # write-ups: streams, tiling, and the full matrix
│
├── tests/
│   ├── test_{op_chain,kernel_key,cpu_ops,cpu_backend,image_io}.cpp   # portable unit tests
│   ├── test_{protocol,server}.cpp                                    # wire codec + integration
│   ├── test_slot_pool.cpp                                            # backpressure assertions
│   ├── test_codegen.cpp                                              # generated-source structure
│   ├── test_cuda_backend.cpp                                         # \  GPU-only: oracle diff,
│   ├── test_gpu_pipeline.cpp                                         #  | cache correctness,
│   ├── test_gpu_server.cpp                                           #  | async residency + stress,
│   └── test_gpu_faults.cpp                                           # /  fault injection
│
└── third_party/                  # vendored stb_image / stb_image_write, Catch2 (amalgamated)
```

**Static library graph** (see each directory's `CMakeLists.txt`): `imgjit_core` (Image/OpChain/
KernelKey, zero dependencies) ← `imgjit_util` and `imgjit_cpu` ← `imgjit_net` (adds sockets and
threading) and `imgjit_codegen` (source-text generation, still no CUDA headers) ← `imgjit_cuda`
(the only target that links `CUDA::cuda_driver` / `CUDA::nvrtc`, and only exists when a toolkit is
found). `imgjit_io` (stb wrappers) sits off to the side and is linked only where PNG files are
actually read or written — no backend links an image codec.

---

## 4. Build System & Prerequisites

| Dependency | Needed for | Notes |
|---|---|---|
| CMake ≥ 3.24 | everything | `find_package(CUDAToolkit)` (3.24 baseline) |
| A C++20 compiler (Clang or GCC) | everything | build is `-Wall -Wextra -Werror` |
| POSIX threads | everything | `find_package(Threads REQUIRED)` |
| NVIDIA CUDA Toolkit (Driver API + NVRTC) | the GPU backend only | auto-detected; its absence silently builds the CPU-only core |
| An NVIDIA GPU | running the GPU backend, or GPU tests | **no Mac ships one** — Apple dropped NVIDIA support system-wide, and Apple Silicon GPUs aren't NVIDIA hardware. Colab or a rented Linux GPU box only. |

CMake auto-detects the toolkit with `find_package(CUDAToolkit) QUIET` and sets
`IMGJIT_ENABLE_CUDA` accordingly — there's no flag to flip by hand. The link line for anything CUDA
is `CUDA::cuda_driver` and `CUDA::nvrtc`; `CUDA::cudart` is never linked anywhere, on purpose (the
Runtime API brings an implicit primary context that would fight the one the worker thread owns
explicitly). The one and only `.cu` file in the repo (`tools/invert_aot.cu`) is compiled by an
`add_custom_command` invoking `nvcc --ptx` directly — `enable_language(CUDA)` is never turned on,
so the project needs no CUDA *toolchain* integrated into the build, only the toolkit's headers and
libraries.

### Local build (macOS or Linux, CPU backend only on a Mac)

```sh
cmake -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build -j
ctest --test-dir build --output-on-failure
```

On a Mac this builds the portable core, the CPU backend, the full network layer, and every
CPU-only tool and test — no CUDA toolkit required or expected.

### Sanitizer builds (two separate trees — ASan and TSan cannot be linked into one binary)

```sh
cmake -B build-asan -DIMGJIT_SANITIZER=address && cmake --build build-asan -j
ctest --test-dir build-asan --output-on-failure

cmake -B build-tsan -DIMGJIT_SANITIZER=thread  && cmake --build build-tsan -j
ctest --test-dir build-tsan --output-on-failure
```

### Clean

```sh
rm -rf build build-asan build-tsan
```

### GPU build (Colab)

Since no local machine here has an NVIDIA GPU, the CUDA backend is built and tested exclusively on
Google Colab, driven by `colab/run.ipynb`. The notebook is a straight-line runner, not a tutorial:

1. **Toolchain / GPU sanity check** — `!nvidia-smi`, so a session that came up without a GPU
   attached fails loud immediately instead of three cells later.
2. **Configure + build** — the same `cmake -B build && cmake --build build -j` as above;
   `find_package(CUDAToolkit)` finds Colab's toolkit automatically and `IMGJIT_ENABLE_CUDA` flips
   on, no notebook-side configuration needed.
3. **Run tests** — the full `ctest` suite, portable and GPU alike.
4. **Phase-specific cells** — one section per phase (codegen/cache, the GPU-backed server, the
   async multi-stream pipeline plus an Nsight Systems capture, shared-memory tiling, and the
   Phase 8 benchmark matrix including the resolution sweep), each runnable independently once the
   build cell has run.

Colab's network is sandboxed — you can reach a server running inside the same container, but not
from outside it — so the GPU demo is loopback-only within the notebook. That's sufficient for
everything here, since the "network" half of this project is already fully proven on the CPU
backend against real multi-machine-shaped traffic (concurrent connections, pipelining, malformed
input) before the GPU is ever involved. If Colab's ephemeral sessions or GPU availability become a
blocker, the fallback is a rented Linux GPU instance, which changes only this notebook and nothing
about the architecture.

---

## 5. Verification, Testing & Benchmarking

### Running the test suites

```sh
ctest --test-dir build --output-on-failure          # everything the current build tree supports
```

Tests are registered per Catch2 tag rather than as one opaque binary, so a failure names the layer
that broke:

| ctest name | Tag | Needs a GPU? | What it covers |
|---|---|---|---|
| `imgjit_unit_core` | `[core]` | no | `Image`, `OpChain` parsing/canonicalization, `KernelKey` hashing |
| `imgjit_unit_cpu` | `[cpu]` | no | scalar op semantics — the correctness oracle itself |
| `imgjit_unit_io` | `[io]` | no | PNG load/save round trips |
| `imgjit_unit_codegen` | `[codegen]` | no | generated CUDA *source text* — fusion plan, baked constants, no resolution leakage |
| `imgjit_unit_net` | `[net]` | no | wire codec, `SlotPool`/`BoundedQueue` concurrency primitives |
| `imgjit_unit_server` | `[server]` | no | full server integration: multi-client, pipelining, malformed frames, backpressure — the suite required clean under **both** ASan and TSan |
| `phase3_gpu_oracle_diff` | `[oracle]` | yes | every corpus chain × {1,3,4} channels vs. the CPU oracle |
| `phase3_gpu_kernel_cache` | `[cache]` | yes | repeat-compile and resolution-independence assertions |
| `phase5_gpu_server` | `[gpu-server]` | yes | concurrent connections, differing chains, real server + real GPU backend |
| `phase6_gpu_async` | `[async]` | yes | genuine multi-stream residency, per-handle completion routing |
| `phase6_gpu_stress` | `[stress]` | yes | sustained load with output checksums — targets premature slot reuse |
| `phase6_gpu_recovery` | `[recovery]` | yes | context recreation flushes cache + pool, resumes, frame slots survive |
| `phase7_gpu_tiling` | `[tiling]` | yes | tiled vs. naive vs. CPU oracle at tile 8/16/32, full op corpus |
| `phase8_gpu_constants` | `[constants]` | yes | parameterized kernels vs. oracle vs. baked; one compile serves several parameter values |
| `phase8_gpu_faults` | `[faults]` | yes | malformed protocol against a GPU-backed server — asserts zero context recreations |
| `phase8_gpu_fault_backend` | `[fault-backend]` | yes | a real injected illegal access, direct against the backend |
| `phase8_gpu_fault_server` | `[fault-server]` | yes | the same fault against the live server + socket |
| `phase8_gpu_fault_concurrent` | `[fault-concurrent]` | yes | two live connections, neither the "cause," both see the fault's blast radius |

The three fault-injection cases run as **three separate ctest processes**, deliberately — an
injected illegal access poisons the whole process, so running them under one tag would take two
innocent test cases down with the one that intentionally faults.

### Standalone fixtures

```sh
# File in, file out, no network — exercises the real IBackend directly
build/tools/imgjit-cli --ops "grayscale,sobel" input.png output.png

# See the generated kernel without needing a GPU — codegen is portable
build/tools/imgjit-cli --ops "grayscale,gaussian:1.4,sobel" \
    --dump-source chain.cu input.png output.png

# On Colab: run on the GPU, dump PTX too, and show cold-compile vs. warm-cache timing
build/tools/imgjit-cli --backend cuda --ops "grayscale,gaussian:1.4,sobel" --repeat 5 \
    --dump-source chain.cu --dump-ptx chain.ptx input.png output.png

# Try the tiled variant and parameterized constants
build/tools/imgjit-cli --backend cuda --tile 16 --constants parameterized \
    --ops "gaussian:1.4" input.png output.png
```

### Launching the server and streaming frames

```sh
# CPU backend, works on a Mac
build/tools/imgjit-server --port 9000 --slots 8 --slots-per-conn 2 &
build/tools/imgjit-client --port 9000 --ops "grayscale,sobel" input.png output.png

# GPU backend, tuned like the benchmark configurations (Colab)
build/tools/imgjit-server --port 9000 --backend cuda --streams 4 --tile 16 \
    --slots 32 --slots-per-conn 8 --queue 32 --max-payload 16777216 \
    --prewarm "grayscale,gaussian:1.4,sobel,threshold:0.3" &

# --repeat pipelines N frames before reading any response — the load that would
# deadlock a server that released slots only after writing (see §2.6)
build/tools/imgjit-client --port 9000 --ops "gaussian:1.4" --repeat 8 --server-write input.png
```

### Benchmarking

`bench/imgjit-bench` is a **closed-loop** harness — each connection keeps at most `--window` frames
in flight and only sends a replacement once a response comes back, which is what makes its latency
numbers meaningful (a fire-and-forget client like `--repeat` above measures something real, but
every frame after the first few reports queueing delay for the whole run, not an actual round
trip).

```sh
build/bench/imgjit-bench --port 9000 --connections 4 --window 8 --frames 300 --no-echo \
    --width 1024 --height 1024 --channels 3 \
    --ops "grayscale,gaussian:1.4,sobel,threshold:0.3" \
    --label my_run --csv results.csv
```

| CSV column | Meaning |
|---|---|
| `fps` | total frames ÷ wall time, across all connections |
| `p50_ms` / `p99_ms` / `max_ms` | round-trip latency, matched per `seq_num` — never by arrival order |
| `cold_first_ms` | each connection's first frame, reported separately and excluded from the percentiles (it pays the NVRTC compile unless prewarmed) |
| `failures` | frames answered with a non-zero status; a healthy baseline reports 0 |

Full methodology — why each sweep is shaped the way it is, and the pitfalls that produced a
withdrawn and re-run resolution sweep — is written up in `bench/README.md`.

### Measured results (Colab T4, `compute_75`, 2 vCPUs)

| Optimization | Configuration | Result |
|---|---|---|
| Kernel cache | cold NVRTC compile vs. cache hit | **~250x** (73 ms → 0.3 ms) |
| Constant baking | baked vs. parameterized kernels | **1.36–1.42x** faster |
| Constant baking | cold cache, 4 distinct σ | 4 compiles (baked) vs. **1** (parameterized); **39–68%** lower worst-first-frame latency |
| Filter-chain fusion | prologue ahead of an 11×11 Gaussian | **1.26–1.28x slower** kernel — a real cost, offset by launch/round-trip savings |
| Shared-memory tiling | `gaussian:1.4`, tile 32 vs. naive | **3.33x** kernel time, **2.00x** end-to-end fps (full chain) |
| Shared-memory tiling | `sobel` (radius 1) | not measurably faster — within the ~9% run-to-run noise floor |
| Multi-stream overlap | 4 streams vs. 1, showcase chain | **+18% fps**, **−36% p99** latency |
| Multi-stream overlap | mean frames resident on the GPU | **2.61 of 4** (vs. 1.00 serial) |
| Copy/compute overlap | H2D + D2H hidden behind compute | **539 of 612 ms** copy budget overlapped (~88%) |
| Resolution scaling | 512²/1024² vs. 2048²/4096² throughput | within **1.1%** of each other; **−14.1%** at 2048², **−38.7%** at 4096² |

Every row above is session-to-session variable by several percent (kernel-time repeatability was
measured at up to 9.2% across two independent runs) — treat these as directional confirmations of
the architecture, not fixed constants. The full write-ups, including the one finding that reversed
sign between two independent sessions (streams stacked on top of tiling), live in
`bench/phase6_results.md`, `bench/phase7_results.md`, and `bench/phase8_results.md`.

### Profiling

**Nsight Systems** is already wired into the benchmark workflow — it's how the copy/compute overlap
claim in [§2.4](#24-asynchronous-latency-hiding-streams--pinned-memory) is verified rather than
just asserted from a throughput number:

```sh
nsys profile -o phase6_trace --trace=cuda \
    build/tools/imgjit-server --port 9000 --backend cuda --streams 4 \
    --prewarm "grayscale,gaussian:1.4,sobel,threshold:0.3"
```

The timeline needs to show H2D, kernel, and D2H bars genuinely interleaved across streams, not one
file of segments with gaps between them. `BackendStats::mean_in_flight` is the software proxy for
the same claim on a Colab image that doesn't ship `nsys`.

Every kernel-level number reported above (`mean_kernel_ms`) comes from CUDA event timestamps
recorded either side of the stages, not from a separate profiling pass — that's what keeps it a
clean per-frame number that the test suite and the benchmark harness can both assert on directly.
