#include "backend/cuda/kernel_cache.h"

#include <utility>

#include "backend/cuda/nvrtc_compile.h"

namespace imgjit::cuda {

const CompiledChain& KernelCache::get(const KernelKey& key) {
  const auto found = entries_.find(key);
  if (found != entries_.end()) {
    return found->second;
  }

  GeneratedProgram program = emit_cuda_source(key);
  std::string ptx = compile_to_ptx(program.source, "imgjit_chain.cu", gpu_architecture_);
  CudaModule module(ptx.c_str());

  std::vector<CUfunction> functions;
  functions.reserve(program.stages.size());
  for (const GeneratedStage& stage : program.stages) {
    functions.push_back(module.get_function(stage.kernel_name.c_str()));
  }

  // Counted only once everything above has succeeded: a failed compile is not a cache
  // entry, and counting it would make the phase gate's "compiles exactly once"
  // assertion pass for the wrong reason.
  ++compile_count_;

  CompiledChain entry{std::move(program), std::move(ptx), std::move(module), std::move(functions)};
  return entries_.emplace(key, std::move(entry)).first->second;
}

}  // namespace imgjit::cuda
