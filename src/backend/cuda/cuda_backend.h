#pragma once

// The CUDA backend: IBackend over NVRTC codegen, the kernel cache and the driver API.
//
// Phase 5 shape: ONE STREAM, and submit() still runs the chain to completion before
// returning, handing the result back from the next poll_completions() exactly as
// CpuBackend does. That is a degenerate implementation of the async contract, and
// deliberately so (docs/PLAN.md Phase 5) — a single stream isolates "is the plumbing
// correct" from "does async overlap work", so the phase that first puts a GPU behind
// the server cannot fail ambiguously between the two. The interface is already the one
// Phase 6 needs, so the K streams, the in-flight table and the event-gated buffer
// return land inside this class rather than rippling through everything above it.
//
// TWO ALLOCATIONS HAPPEN ONCE, IN allocate_slots(), AND NEVER PER FRAME (CLAUDE.md
// invariant 2): the pinned host slots the connection threads recv() straight into, and
// the device ping-pong pair the stages read and write. Phase 3's exemption from that
// invariant — a file-in/file-out CLI may allocate per invocation, having no pipeline to
// serialize — is CLOSED as of Phase 5: submit() has no allocating path left, so a
// caller that never allocated slots gets an error instead of a quiet per-frame
// cuMemAlloc. Nothing had to give that up, since imgjit-cli allocates a slot sized to
// its one image and so was already taking the pooled path.
//
// This header lives in src/backend/cuda/ rather than include/, because it includes
// <cuda.h> transitively and CLAUDE.md invariant 1 scans include/ precisely because
// public headers leak the furthest. Tools reach for it under IMGJIT_ENABLE_CUDA, which
// is how tools/imgjit-spike.cpp already works.
//
// THREADING: worker-thread-only, like every IBackend. Here that is not a convention
// but invariant 1 — the CUcontext this owns is created on the constructing thread and
// never migrates.

#include <cstddef>
#include <optional>
#include <vector>

#include "backend/cuda/cuda_raii.h"
#include "backend/cuda/kernel_cache.h"
#include "imgjit/backend/backend.h"

namespace imgjit {

class CudaBackend final : public IBackend {
 public:
  // Creates the context. Constructing this object is the "cuCtxCreate once at startup
  // on the worker thread" step; there is no other entry point that creates one.
  explicit CudaBackend(int device_ordinal = 0);

  std::byte* allocate_slots(std::size_t count, std::size_t bytes) override;
  JobHandle submit(const FrameJob& job) override;
  std::vector<Completion> poll_completions() override;

  // Compiles `chain` at `channels` now, so the first client to ask for it hits the
  // cache instead of paying the ~50-200 ms NVRTC cold start (docs/ARCHITECTURE.md,
  // "NVRTC cold-start"). Called from the server's backend factory, which already runs
  // on the worker thread — which is why this is a CudaBackend method and not a virtual
  // on IBackend: prewarming is meaningless for a backend that never compiles anything.
  void prewarm(const OpChain& chain, int channels);

  // The phase gate's instrument (docs/PLAN.md Phase 3): a repeated chain, and the same
  // chain at three resolutions, must leave this unchanged.
  std::size_t compile_count() const { return cache_.compile_count(); }
  cuda::KernelCache& cache() { return cache_; }
  const cuda::CudaContext& context() const { return context_; }

 private:
  // Takes the job rather than an Image: the input is pinned slot memory and the whole
  // point of the pool is that it reaches the driver without being copied first.
  Image run_chain(const FrameJob& job);

  // Declaration order is load-bearing: cache_ is constructed from context_'s compute
  // capability, and every CUmodule it holds belongs to that context, so it must also
  // be destroyed before it. The same goes for every buffer and the stream below.
  cuda::CudaContext context_;
  cuda::KernelCache cache_;
  cuda::CudaStream stream_;
  std::optional<cuda::PinnedHostBuffer> slots_;
  // Ping-pong for multi-stage execution, sized to one slot. Two is the whole
  // requirement while a single stream runs one frame at a time; Phase 6 replaces the
  // pair with a pool of K and gates their return on an event.
  std::optional<cuda::DeviceBuffer> device_front_;
  std::optional<cuda::DeviceBuffer> device_back_;
  std::vector<Completion> completed_;
  JobHandle next_handle_{1};
};

}  // namespace imgjit
