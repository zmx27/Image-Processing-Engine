# CLAUDE.md

GPU-JIT image processing server: NVRTC-compiled CUDA kernels executed via the CUDA Driver API,
fed by a multi-threaded TCP protocol server through a bounded queue. See `docs/ARCHITECTURE.md`
for design rationale and `docs/PLAN.md` for the phased build order — read `docs/PLAN.md` at the
start of every session to see what's done and what's next.

## Build

Local (macOS — CPU backend only; **no Mac has an NVIDIA GPU**, Intel or Apple Silicon):

    cmake -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
    cmake --build build -j
    ctest --test-dir build --output-on-failure

Colab (CUDA auto-detected via `find_package(CUDAToolkit)`): see `colab/run.ipynb`.

Link `CUDA::cuda_driver` and `CUDA::nvrtc`. Never link `CUDA::cudart`.

## Architectural invariants — do not violate without updating this file

1. **CUcontext ownership**: only the GPU worker thread ever calls a CUDA Driver API function.
   `cuCtxCreate` runs once at startup on that thread and is never released or migrated. No file
   outside `src/backend/cuda/` may `#include <cuda.h>`.
   Verify: `grep -rE '\bcu[A-Z]' src/ | grep -v backend/cuda/`
2. **Pools, not per-frame allocation.** Pinned and device buffers come from pools allocated once
   at worker startup. Never `cuMemAllocHost`/`cuMemAlloc` in the per-frame path — both require a
   current context and the pinned call implicitly synchronizes, serializing the pipeline.
3. **Event-gated buffer return.** A buffer returns to its pool only after its `CUevent` is
   confirmed complete — never on return of the async copy or launch that used it. This is the
   most important invariant in the codebase: violating it silently corrupts data instead of
   crashing.
4. **Kernel cache key = hash(canonicalized op chain + baked constants + channels + naive/tiled
   variant).** It must NOT include width/height — dimensions are launch-time arguments, not part
   of the compiled kernel identity. Baking them would make every new resolution a cache miss and
   grow the cache unboundedly, defeating memoization. Any new codegen input must be added to the
   key or the cache goes stale.
5. **Backpressure is slot-pool exhaustion.** Connection threads block on slot claim, which stops
   them `recv()`ing, which backs up through TCP flow control to the client. Never drop frames
   silently; never grow the pool at runtime.
6. **Wire format is little-endian, packed field by field** (see `docs/PROTOCOL.md`). Never
   `memcpy` a struct onto a socket. Validate and cap declared length BEFORE allocating anything.
7. **All socket reads go through `read_exact()`.** A bare `recv()` is a bug — TCP is a byte
   stream and short reads are the norm, not the exception.
8. **No CUDA Runtime API anywhere** — no `cudaMalloc`, no `<<<>>>`, no Runtime-API-backed
   dependency (Thrust/CUB). They create an implicit primary context that conflicts with the one
   the GPU worker owns explicitly.
9. **Every kernel has a scalar CPU equivalent** in `src/backend/cpu/` — it is the correctness
   oracle for every GPU test.

## Conventions

- C++20, warnings-as-errors. `#pragma once` for header guards.
- Naming: types `PascalCase`, functions/free functions `snake_case`, member variables
  `trailing_underscore_`, constants `kPascalCase`.
- RAII wrappers for every driver resource (`CUdeviceptr`, pinned host pointer, `CUstream`,
  `CUevent`, `CUmodule`). No raw `new`/`delete`, no manual `cuMemFree` in logic code.
- Every Driver API / NVRTC call is wrapped in `CU_CHECK(...)` / `NVRTC_CHECK(...)`, which throws
  on failure. No unchecked driver calls.
- Protocol errors produce an error response; they never abort the server.
- GPU/CPU comparison uses a tolerance (≤1 LSB on `uint8` output), never bit-equality — FMA
  contraction and reassociation make exactness the wrong bar for stencil ops. Pointwise integer
  ops (e.g. inversion) are the exception and may be compared exactly.
- Portable code (`core/`, `net/`, `util/`, `backend/cpu/`) must compile with no CUDA toolkit
  present — this is what makes local Mac development possible at all.
- Tests use Catch2 (vendored under `third_party/`).
