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
- [ ] Vendor stb_image.h / stb_image_write.h + Catch2 under `third_party/`
- [ ] Add test image fixtures under `tests/testdata/`
- [x] Write `CLAUDE.md`, `docs/PLAN.md`, `docs/ARCHITECTURE.md`, `docs/PROTOCOL.md`
- [x] `colab/run.ipynb` skeleton that clones the repo and builds (with a `!git pull` cell at the
      top and an `nvidia-smi` / CUDA version print cell for drift visibility)
- **Done when:** clean CPU-only build on Mac, and the notebook produces a CUDA build on Colab

## Phase 1 — Driver API + JIT spike · env: Colab

Two checkpoints in one phase, so a failure is unambiguous about which layer broke:

- [ ] **1a:** `cuInit` → `cuCtxCreate` → load a PTX blob produced by `nvcc --ptx` (ahead of time)
      → `cuModuleLoadData` → `cuLaunchKernel`. Proves driver plumbing with JIT out of the picture.
- [ ] **1b:** swap the AOT blob for `nvrtcCompileProgram` at runtime. Proves JIT plumbing on top
      of already-working driver plumbing.
- [ ] `CU_CHECK` / `NVRTC_CHECK` macros; dump generated PTX to disk for inspection
- **Done when:** Colab inverts a real PNG on-GPU, output exactly matches scalar CPU inversion
      (exact is the right bar here — integer pointwise op, no float reassociation)

## Phase 2 — Portable core · env: Mac

- [ ] `Image` type (`uint8`, 1/3/4 channels, row-major, explicit stride)
- [ ] `OpChain` IR + parser for `"grayscale,gaussian:1.4,sobel,threshold:0.3"`
- [ ] Canonicalizer + `KernelKey` hash — **excludes width/height** by construction
- [ ] CPU backend implementing all ops; stb load/save wrappers; `tools/imgjit-cli`
- [ ] Tests: parser, canonicalizer, hash stability, each filter vs. checked-in fixtures
- **Done when:** `imgjit-cli --ops "grayscale,sobel" in.png out.png` works on the Mac; tests green

## Phase 3 — Codegen + kernel cache · env: Colab

- [ ] `emit_cuda_source(OpChain) → std::string`; one kernel per stencil stage, pointwise ops fused
- [ ] Bake radius / weights / threshold / channels as literals; dimensions stay launch arguments
- [ ] `KernelCache: KernelKey → {CUmodule, CUfunction}` with a compile counter for assertions
- [ ] Multi-stage execution with intermediate device buffers
- [ ] Measure cold-compile vs. warm-hit latency
- **Done when:** every chain in the test corpus matches CPU within tolerance, **and** a repeated
      chain provably compiles exactly once (compile counter unchanged), **and** the same chain at
      three different resolutions still compiles only once

## Phase 4 — Network layer · env: Mac (CPU backend)

- [ ] `docs/PROTOCOL.md` implemented as pure encode/decode functions
- [ ] `read_exact()` / `write_exact()`; explicit little-endian pack/unpack; header validation +
      error responses
- [ ] Codec unit tests including malformed input: bad magic, truncated header, length mismatch,
      oversized payload
- [ ] Acceptor + per-connection threads; bounded MPSC queue; slot pool with per-connection caps
- [ ] `tools/imgjit-client`; both result paths (echo, server-side write); blocking backpressure
      verified with an artificially throttled worker
- [ ] Integration test: N clients × M frames, verified against the CPU oracle
- **Done when:** multi-client localhost run is correct and clean under **ASan and TSan**

## Phase 5 — Integration: GPU worker behind the server · env: Colab

- [ ] Swap the CPU worker for the GPU worker; `cuCtxCreate` once at startup on that thread
- [ ] Pinned slot pool allocated on the worker; connection threads `recv()` directly into slots
- [ ] **Single stream only** — isolate "is the plumbing correct" from async overlap
- **Done when:** concurrent loopback clients with differing chains all match the CPU oracle on
      Colab; TSan clean; no CUDA symbol reachable outside `src/backend/cuda/`

## Phase 6 — Async multi-stream pipeline · env: Colab

- [ ] In-flight table across K streams (start K=4); `cuEventRecord` + poll-and-retire
- [ ] Event-gated buffer return; device memory pool replacing any per-frame `cuMemAlloc`
- [ ] Instrument queue depth, occupancy, stall counts
- [ ] Stress test: sustained repeated runs with output checksums, specifically targeting
      premature slot reuse (`CLAUDE.md` invariant 3)
- **Done when:** measurably faster than the Phase 5 sync path, Nsight Systems shows genuine
      H2D/compute/D2H overlap rather than serialized segments, and zero checksum drift under stress

## Phase 7 — Shared-memory tiling · env: Colab

- [ ] Tiled stencil codegen variant: `__shared__` tile + halo/apron loads, bounds-clamped,
      parameterized by tile size
- [ ] Extend `KernelKey` with the naive|tiled variant; both remain runtime-selectable
- **Done when:** tiled output matches naive and CPU within tolerance, and is measurably faster on
      both Gaussian and Sobel

## Phase 8 — Benchmarks + polish · env: Colab

- [ ] `bench/` CSV harness over the orthogonal matrix: naive|tiled × sync|async, plus
      fused-vs-unfused chain, cold-vs-warm cache, baked-vs-parameterized constants
- [ ] Sweep 512² → 4K and chain lengths; report FPS, GB/s, p50/p99 round-trip latency
- [ ] Error-injection pass: malformed protocol, forced illegal access → confirm context
      recreation rather than crash
- [ ] README results table; finalize `ARCHITECTURE.md` and `PROTOCOL.md` against actual behavior
- **Done when:** every architectural claim in `ARCHITECTURE.md` maps to a number in the table

## Stretch (explicitly not required for the three pillars)

- [ ] Off-thread NVRTC compilation — legal, since `nvrtcCompileProgram` needs no context and only
      `cuModuleLoadData` marshals back to the worker. Requires a cache-stampede guard: concurrent
      requests for the same uncompiled key must park on one in-flight compile, not launch N.
- [ ] `DropOldest` backpressure mode as a flag, for real-time semantics where stale frames are
      worthless
- [ ] Lock-free SPSC queues replacing mutex+condvar
