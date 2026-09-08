#pragma once

// KernelKey -> compiled module. The memoization the whole project exists to
// demonstrate: NVRTC cold-start is ~50-200 ms, so the difference between hitting and
// missing here is the difference between a frame budget and a stall.
//
// KEYED BY THE FULL KernelKey, NOT BY ITS HASH. hash_kernel_key is used as the bucket
// function, and equality still compares every field. A 64-bit collision is unlikely,
// but the failure it would cause — silently running someone else's kernel — is exactly
// the class of bug invariant 4 is written to prevent, and comparing a handful of ints
// costs nothing on a path that is about to launch a kernel anyway.
//
// The counter is the phase gate. "Provably compiles exactly once" (docs/PLAN.md
// Phase 3) is only provable if something counts, and a stale hit produces a wrong
// number rather than a crash, so the count is asserted by tests rather than eyeballed.

#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

#include <cuda.h>

#include "backend/cuda/codegen.h"
#include "backend/cuda/cuda_raii.h"
#include "imgjit/core/kernel_key.h"

namespace imgjit::cuda {

struct CompiledChain {
  // Kept alongside the module so `--dump-source` / `--dump-ptx` can print exactly what
  // was compiled, rather than regenerating it and hoping the two agree.
  GeneratedProgram program;
  std::string ptx;
  CudaModule module;
  // Parallel to program.stages; launched front to back.
  std::vector<CUfunction> stage_functions;
};

class KernelCache {
 public:
  explicit KernelCache(std::string gpu_architecture)
      : gpu_architecture_(std::move(gpu_architecture)) {}

  // Compiles on a miss, returns the existing entry on a hit. The reference stays valid
  // until clear() — unordered_map does not invalidate references on rehash.
  const CompiledChain& get(const KernelKey& key);

  // Number of NVRTC compiles since construction. Never reset by clear(), so a test can
  // assert across a flush.
  std::size_t compile_count() const { return compile_count_; }
  std::size_t size() const { return entries_.size(); }

  // Phase 6 context recreation destroys the cache, the device pool and the in-flight
  // table together; this is the cache's half of that. Present now because every
  // CUmodule here belongs to the context, so the two lifetimes are already coupled.
  void clear() { entries_.clear(); }

 private:
  struct KeyHash {
    std::size_t operator()(const KernelKey& key) const {
      return static_cast<std::size_t>(hash_kernel_key(key));
    }
  };

  std::string gpu_architecture_;
  std::unordered_map<KernelKey, CompiledChain, KeyHash> entries_;
  std::size_t compile_count_{0};
};

}  // namespace imgjit::cuda
