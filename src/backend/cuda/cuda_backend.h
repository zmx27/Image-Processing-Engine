#pragma once

// The CUDA backend: IBackend over NVRTC codegen, the kernel cache and the driver API.
//
// Phase 3 shape: submit() runs the chain to completion synchronously and hands the
// result back from the next poll_completions(), exactly as CpuBackend does. That is a
// degenerate implementation of the async contract, and deliberately so — the interface
// is already the one Phase 6 needs, so the streams and the in-flight table land inside
// this class rather than rippling through everything above it.
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

  // The phase gate's instrument (docs/PLAN.md Phase 3): a repeated chain, and the same
  // chain at three resolutions, must leave this unchanged.
  std::size_t compile_count() const { return cache_.compile_count(); }
  cuda::KernelCache& cache() { return cache_; }
  const cuda::CudaContext& context() const { return context_; }

 private:
  Image run_chain(const Image& input, const OpChain& chain);

  // Declaration order is load-bearing: cache_ is constructed from context_'s compute
  // capability, and every CUmodule it holds belongs to that context, so it must also
  // be destroyed before it.
  cuda::CudaContext context_;
  cuda::KernelCache cache_;
  std::optional<cuda::PinnedHostBuffer> slots_;
  std::vector<Completion> completed_;
  JobHandle next_handle_{1};
};

}  // namespace imgjit
