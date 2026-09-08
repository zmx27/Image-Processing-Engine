#pragma once

// Phase 3 codegen: a KernelKey in, one self-contained CUDA translation unit out.
//
// This is structural string assembly, not template substitution — the stage count,
// the fusion boundaries and every baked literal vary with the chain, so there is no
// fixed skeleton to fill in (docs/ARCHITECTURE.md).
//
// PORTABLE ON PURPOSE. Generating CUDA source needs no CUDA headers, so this target
// links neither the driver nor NVRTC and builds on macOS with no toolkit present.
// That is what lets the emitted source be unit-tested locally instead of only on
// Colab, and it is why `imgjit-cli --dump-source` works on a Mac. Its neighbours in
// this directory (kernel_cache, cuda_backend) are still gated behind
// IMGJIT_ENABLE_CUDA — they are the ones that actually call the driver.
//
// The input is a KernelKey rather than an OpChain because the key IS the complete
// list of codegen inputs (CLAUDE.md invariant 4): taking anything else would mean a
// codegen input that the cache is not keyed by, which shows up as a stale cache hit —
// a wrong benchmark number, not a crash. Note what the signature therefore cannot
// take: width and height. They are launch arguments.

#include <string>
#include <vector>

#include "imgjit/core/kernel_key.h"

namespace imgjit::cuda {

struct GeneratedStage {
  // The `extern "C" __global__` entry point, e.g. "imgjit_stage0".
  std::string kernel_name;
  // True when this stage is built around a gaussian or sobel. Not needed to launch
  // it — every stage has the same signature — but it is what the codegen tests assert
  // the fusion plan against.
  bool is_stencil{false};
};

struct GeneratedProgram {
  // One translation unit: the prelude, then every stage, in execution order.
  std::string source;
  // Stages run front to back, each reading the previous one's output buffer. Empty
  // for an empty chain, which is an identity pass and needs no kernel at all.
  std::vector<GeneratedStage> stages;
};

// FUSION PLAN. Pointwise ops fuse into registers; stencil ops need neighbours and so
// terminate a stage (docs/ARCHITECTURE.md decision 5). Concretely, walking the chain
// left to right, a pointwise op joins the current stage's epilogue if that stage
// already has its stencil, and its prologue otherwise; a stencil op either claims the
// current stage or opens a new one. That yields at most `#stencil_ops + 1` kernels,
// and — because a pointwise run prefers to attach backwards as an epilogue — only the
// first stage can ever carry a prologue, so the expensive folding (a prologue is
// recomputed at every stencil tap) happens at most once in a chain.
//
// Throws std::invalid_argument on a key whose channel count is unsupported, or whose
// tile/constants fields are set to the variants reserved for Phases 7 and 8.
GeneratedProgram emit_cuda_source(const KernelKey& key);

}  // namespace imgjit::cuda
