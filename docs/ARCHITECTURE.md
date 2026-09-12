# Architecture

## Vision

A native server that receives image frames over TCP, dynamically generates and compiles CUDA
kernels at runtime via NVRTC based on the requested filter chain, and executes them on the GPU
via the CUDA Driver API with overlapped I/O. Three pillars: GPU driver-level control, runtime
compilation, and high-throughput network I/O. This is a portfolio/demonstration project —
architectural clarity and defensible benchmark numbers matter more than feature count.

## Why JIT actually earns its place

A JIT that only picks between kernels it could have precompiled is a gimmick. Two things make
runtime compilation genuinely load-bearing here:

**1. Filter-chain fusion.** A client requests a *chain*, e.g.
`grayscale → gaussian(σ=1.4) → sobel → threshold(0.3)`. Executed conventionally that's 4 kernel
launches and 4 round-trips through global memory. Codegen instead emits **one kernel per stencil
stage**, inlining adjacent pointwise ops into registers around it. The number of possible chains
is combinatorial in the op set, so ahead-of-time compilation of every permutation is exactly the
binary bloat this design avoids. This is also why the wire protocol carries an op *chain*, not a
single op code — a single op per frame would be a weak justification for JIT, since a handful of
ops could just be precompiled.

*Measured, and only half confirmed, over two sessions* (`bench/phase8_results.md` §4): the saving
is real for a pointwise op fused **after** a stencil, which runs once per output pixel either way.
Fused **before** one it is not — a prologue is re-executed at every stencil tap, so `grayscale` in
front of an 11x11 Gaussian costs 121 evaluations per pixel instead of one, and the showcase
chain's fused kernels take **1.26–1.28x the kernel time** of the same four ops run unfused,
replicated closely across two independent runs. Fusion still saves the launches, the global round
trips and (through the server) the network round trips; what it does not do is unconditionally
reduce compute. Codegen already prefers attaching pointwise runs backwards for this reason — the
measurement is what puts a number on the preference.

**2. Constant baking.** Filter radius, Gaussian weights, threshold, and channel count are emitted
as compile-time literals, letting NVRTC fully unroll stencil loops and constant-fold. The
alternative — passing them as kernel parameters — leaves dynamic bounds in the inner loop. This
gives a clean A/B benchmark axis: same kernel, baked vs. parameterized.

*Measured, over two sessions* (`bench/phase8_results.md` §1): baked kernels run **1.36-1.42x
faster**, replicated closely both times. Parameterized ones compile **once for every parameter
value** instead of once per value — four distinct sigmas are 4 compiles baked and 1 parameterized,
identically in both sessions, since it is a discrete count rather than a timing measurement. What
that difference is worth in latency on a cold cache is real but not a fixed number: a **39–68%**
reduction in worst-first-frame latency across the two runs. A specific per-compile-ms cost and a
frames-to-break-even figure were derived from session 1 alone and are **not reported here**: the
same derivation applied to session 2 reversed which mode compiles faster, so neither number is
trustworthy standing alone. The direction that survives both sessions is the rule the flag exists
to make choosable: `--constants parameterized` for a per-request slider, the default for presets.

**Do not bake width/height.** They are launch-time kernel arguments. Baking them would make every
new resolution a cache miss and a fresh ~100 ms compile, growing the cache unboundedly and
defeating the memoization pillar. See `CLAUDE.md` invariant 4.

## Component map

```
                  ┌──────────── Portable core (builds on macOS) ────────────┐
  client(s) ──TCP─┤ acceptor thread → per conn: reader + writer thread      │
                  │        ├─ read_exact() framing, header validation       │
                  │        ├─ claim pinned slot from free-list (blocking)   │
                  │        └─ recv() payload DIRECTLY into pinned slot      │
                  │                   ↓ push(FrameJob{slot, seq, chain})    │
                  │           bounded MPSC queue (mutex + condvar)          │
                  └────────────────────────┬───────────────────────────────-┘
                                           ↓ pop
                  ┌──── GPU worker thread — SOLE owner of CUcontext ────────┐
                  │  in-flight table (depth = #streams)                     │
                  │    KernelKey ← hash(every codegen input — the           │
                  │                     authoritative list is PLAN §3)      │
                  │    KernelCache: hit → CUfunction | miss → NVRTC → PTX   │
                  │                        → cuModuleLoadData               │
                  │    cuMemcpyHtoDAsync → cuLaunchKernel → DtoHAsync       │
                  │    cuEventRecord; poll events; retire ONLY on complete  │
                  └────────────────────────┬───────────────────────────────-┘
                                           ↓ retire (event confirmed complete)
                  copy result out of slot → release pinned slot → free-list
                                           ↓
                       per-connection outbox (response tagged with seq)
                                           ↓
                  writer thread → echo to conn and/or stb_image_write
```

The CPU backend implements the same `IBackend` interface as the CUDA backend, so the server is
identical in both builds and the entire network path is testable on a Mac with no GPU at all.

## The six load-bearing decisions

**1. One thread owns the CUcontext, forever.** The GPU worker calls `cuCtxCreate` once at startup
and never releases or migrates it. No other thread makes any CUDA call. This sidesteps the whole
class of push/pop lifetime bugs and is auditable by grep (see `CLAUDE.md` invariant 1).

**2. Pinned memory is a fixed pool allocated at startup, never per frame.** `cuMemAllocHost`
requires a current context (so only the worker can allocate) and implicitly synchronizes (so
per-frame allocation would serialize the pipeline it exists to parallelize). `kNumSlots`
fixed-size pinned buffers are allocated on the worker at init; connection threads take indices
from a free-list and `recv()` straight into pinned memory — no staging copy on ingest.

Deliberate consequences: fixed slots force the `kMaxPayloadBytes` cap (oversized frames are
rejected before any allocation — never trust an attacker-controlled length), and **free-list
exhaustion is the backpressure mechanism**. A connection thread blocks on slot claim, which stops
it `recv()`ing, which fills the client's send buffer through TCP flow control. That's correct
end-to-end behavior, not a limitation. Per-connection slot caps prevent one loud client starving
the others.

**3. Buffers return to the pool only after their `CUevent` is confirmed complete.** Not when the
launch call returns, not when the async copy call returns. Reusing a slot early is a
write-into-in-flight-DMA bug that corrupts data silently rather than crashing — the sharpest
hazard in the design. It has a dedicated checksum stress test (Phase 6).

**4. The worker keeps N frames in flight, not one.** The easiest place to accidentally destroy
the project's point: a `pop → copy → launch → sync → respond` loop serializes everything and
makes multiple streams worthless. The worker is a state machine — fill idle stream slots from the
queue, record an event per slot, poll and retire each iteration.

Landed in Phase 6, and it changed nothing above `CudaBackend` — the payoff for making `IBackend`
submit/poll in Phase 2 rather than the blocking shape that was tempting then. One further trap
sits inside it: **the device-to-host copy must land in pinned memory**, because an async copy into
a pageable destination is allowed to run synchronously and does, which serializes the pipeline
while every call still looks asynchronous. Each stream slot owns its own staging buffer.

**5. Stencil ops are the fusion boundary.** Pointwise ops (`grayscale`, `invert`, `brightness`,
`threshold`) fuse into registers at zero memory cost. Stencil ops (`gaussian`, `sobel`) need
neighbors and terminate a stage. Codegen emits at most `#stencil_ops + 1` kernels, with pointwise
runs folded into the adjacent stage's prologue/epilogue. The op set is **closed at these six**
(see `docs/PLAN.md` Phase 2) — that cap is what keeps the fusion story honest.

A pointwise run attaches *backwards* where it can — an epilogue runs once per output pixel, a
prologue once per stencil tap — so only the first stage ever carries a prologue.

**Fusion is a memory optimization and must not become a numerics change.** The CPU oracle
materializes a `uint8` image between every pair of ops; a fused kernel keeps the value in a
register, so codegen re-quantizes (`round(clamp(v,0,1)*255)/255`) at each fused boundary, rounding
exactly where the oracle rounds. Skipping that is not a rounding nicety: Sobel's coefficients sum
to 8 in absolute value, so a half-LSB difference per tap amplifies to ~4 LSB, and a `threshold`
folded after a stencil flips 0↔255 for any sample that straddles it — both far outside the ≤1 LSB
bar the GPU is held to. The cost is one `roundf` per boundary.

**6. The worker never touches a socket, and slot release is never gated on a socket write.**
Completions land in a per-connection outbox that a writer thread drains; the reader thread does
framing and slot claim only. Two distinct failures make this structural rather than stylistic.

If the *worker* wrote responses, a single slow client would head-of-line-block the entire GPU
pipeline — the whole point of decision 4 undone by one `send()`, and it would surface as an
inexplicable throughput cliff in Phase 6 rather than as an obvious bug.

If a slot were released only *after* its response was written, a pipelining client would deadlock
its connection. The client sends frames 1..N before reading any response (which the `seq_num`
design explicitly invites); the reader thread blocks claiming a slot for frame N+1; the response
for frame 1 is never written, so its slot is never freed, so the reader never unblocks. Releasing
on event completion breaks the cycle — which means **the echoed payload must be copied out of the
slot before release**, and that copy is the price of the property. It is worth it: the alternative
is a deadlock that only appears under pipelining, which is exactly the load the benchmarks run.

## Pitfalls to plan around

- **TCP is a byte stream.** Short `recv()` is the #1 bug in hand-rolled protocol code. One
  `read_exact()` helper, used everywhere.
- **No struct-memcpy on the wire.** Padding and endianness make it non-portable. Pack field by
  field (see `docs/PROTOCOL.md`).
- **Async CUDA errors are sticky.** They often surface at the next sync point, not the failing
  call, and an illegal-access error can poison the context permanently. Scope decision: tear down
  and recreate the context (flushing the kernel cache and device pool with it), reject in-flight
  work, resume. This is a deliberate, documented limitation, not an oversight. Implemented in
  Phase 6 as `CudaBackend::recreate_context()`. Because the error surfaces at a later call than
  the one that caused it, there is no attributing it to a frame: everything in flight fails with
  status 6, which is the documented cost. The *frame slots* deliberately survive — they are
  process-owned pages pinned with `cuMemHostRegister` rather than `cuMemAllocHost` allocations,
  precisely so that recovery cannot free memory the connection threads are `recv()`ing into.
- **NVRTC source must be self-contained, and there is no `kernels/` directory.** NVRTC has no
  default include path, so generated source cannot `#include <cstdint>` (and must not include
  `<cuda_runtime.h>` — that is invariant 8). The stable device helpers therefore live in
  `src/backend/cuda/kernel_prelude.h` as a single raw-string literal, not as `.cu` template files.
  Codegen is structural string assembly, not template substitution — stage count, fusion
  boundaries and unrolled weights all vary — so only ~50 lines are actually stable text, and
  putting those in files would buy syntax highlighting at the cost of a runtime path dependency
  (or a CMake embed step) plus fragments no compiler can check. The readability need is served
  from the other end: a `--dump-source` flag prints the *generated* kernel, which is what you
  actually read when debugging.
- **No CUDA Runtime API, anywhere.** Not `cudaMalloc`, not `<<<>>>`, and no Runtime-backed
  dependency (Thrust/CUB) — they create an implicit primary context that conflicts with our
  explicit one.
- **Colab's network is sandboxed.** You cannot reach a Colab server from outside it. The GPU demo
  is loopback *within the container*; genuine multi-machine testing happens against the CPU
  backend instead. Stated plainly here rather than implied.
- **stb is not part of the wire protocol.** Raw contiguous pixels go over the socket, and the
  server never *decodes* PNG/JPEG from network input — `stb_image` is a CLI/test-side file
  convenience only. Encoding is the exception: the server-side write path (`flags` bit 1) does use
  `stb_image_write` to emit a PNG, which is a file-output concern and never touches the wire.
- **NVRTC cold-start is ~50–200 ms.** Exactly why the cache exists; cold-vs-warm is a legitimate
  benchmark line. A configured chain list is prewarmed at startup (Phase 5) so this claim maps to
  a measured row rather than staying an aspiration.
- **Transport `uint8`, compute `float`.** Convert in-kernel, write back `uint8`. Halves PCIe
  traffic versus float transport.
- **Compare with tolerance, never bit-equality**, for stencil ops — FMA contraction and
  reassociation make exactness the wrong bar. Use ≤1 LSB on `uint8` output, or PSNR > 50 dB.
  Pointwise integer ops (e.g. inversion) are the exception and may be compared exactly.
- **Scope discipline.** The two live over-scoping risks are adding filter ops and re-expanding
  "distributed" beyond one node / one GPU / many client connections. Both are bounded by design;
  hold the line.

## Repository layout

Built incrementally as phases need it, not scaffolded all at once.

```
image-processing/
├── CLAUDE.md                 # build commands, conventions, architectural invariants
├── CMakeLists.txt            # auto-detects CUDA → IMGJIT_ENABLE_CUDA
├── README.md                 # what it is, results table, how to run
├── colab/run.ipynb           # clones repo, installs deps, builds, runs tests + benchmarks
├── docs/
│   ├── PLAN.md               # phased roadmap
│   ├── ARCHITECTURE.md       # this file
│   └── PROTOCOL.md           # wire format spec
├── include/imgjit/           # public headers, mirrors src/ layout
├── src/
│   ├── core/                 # Image, OpChain IR, parser, canonicalizer      (portable)
│   ├── net/                  # framing, server, reader/writer threads, client (portable)
│   ├── util/                 # queue, slot pool, logging, stb wrappers       (portable)
│   ├── backend/cpu/          # scalar reference implementation               (portable)
│   └── backend/cuda/         # driver-API context, NVRTC codegen, cache, RAII (CUDA only)
│       └── kernel_prelude.h  #   stable device helpers as one raw-string literal
├── tests/                    # unit + integration, with testdata/ fixtures
├── bench/                    # benchmark harness and result CSVs
├── tools/                    # imgjit-server, imgjit-client, imgjit-cli
└── third_party/              # vendored stb_image, stb_image_write, Catch2
```

## Verification strategy

- **Correctness:** the CPU backend is the oracle; every GPU path is diffed against it within
  tolerance, per filter and per full chain.
- **Cache:** assert via a compile counter that a repeated chain compiles once — including across
  differing resolutions, which is the regression test for invariant 4.
- **Concurrency:** TSan on the multi-client integration test, ASan across the suite.
- **Memory safety:** a checksum stress test specifically targeting premature slot reuse
  (invariant 3).
- **Overlap:** Nsight Systems timeline must show real concurrency, not merely a better number.
- **Performance:** every claim in this document maps to a row in the Phase 8 benchmark table —
  `bench/phase8_results.md` §6 is that mapping. Two full sessions were run rather than one
  specifically to catch claims that don't replicate: five of seven cross-session comparisons
  agree closely (parameterized penalty, tiled speedup, fusion cost, streams on the naive kernel,
  cache compile count); the streams×tiling interaction reversed sign between sessions and is
  recorded as unresolved rather than reported as a number (§3); the resolution sweep's first pass
  was invalidated by its own methodology and rerun (§2).

## Open risk

Colab sessions are ephemeral and GPU allocation is not guaranteed. Phase 1 exists partly to find
out early whether that workflow is tolerable. If it isn't, the fallback is a rented Linux GPU
instance — which changes `colab/run.ipynb` and nothing about the architecture.
