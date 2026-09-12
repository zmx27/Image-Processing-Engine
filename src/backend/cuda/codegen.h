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

#include <cstddef>
#include <string>
#include <vector>

#include "imgjit/core/kernel_key.h"

namespace imgjit::cuda {

// The block edge every naive stage launches with: 16x16 = 256 threads, the usual
// starting point for a 2D image kernel.
inline constexpr int kNaiveBlockDim = 16;

// Phase 7's tile edge bound. A tiled stage launches one thread per output pixel of its
// tile, so tile_size^2 threads per block, and 32^2 = 1024 is the hardware maximum.
inline constexpr int kMaxTileSize = 32;

// The tile half of a KernelKey that codegen can emit: naive with tile_size 0, or tiled
// with tile_size in [1, kMaxTileSize]. Naive with a nonzero size is refused rather than
// ignored, so one kernel can never sit under two keys.
bool is_supported_tile(TileVariant tile, int tile_size);

// The widest gaussian the op set allows: sigma 4 is radius 12, so 2*12 + 1 weights per
// axis. A parameterized gaussian takes its weights as one struct of this many floats,
// because a kernel argument's size is fixed when the kernel is compiled.
inline constexpr int kMaxGaussianTaps = 25;

// The constants half of a KernelKey that codegen can emit. Parameterized is naive-only:
// a tiled stage sizes its __shared__ array from the stencil radius at compile time, and
// a parameterized kernel does not learn the radius until launch.
bool is_supported_constants(ConstantsMode constants, TileVariant tile);

struct GeneratedStage {
  // The `extern "C" __global__` entry point, e.g. "imgjit_stage0".
  std::string kernel_name;
  // True when this stage is built around a gaussian or sobel. Not needed to launch
  // it, but it is what the codegen tests assert the fusion plan against.
  bool is_stencil{false};
  // The square block edge this stage MUST be launched with. Decided here rather than by
  // the executor because a tiled stage's correctness depends on it: its __shared__ tile
  // and its index arithmetic are baked for exactly this many threads per side, and a
  // launch with any other block would read apron cells that were never loaded.
  int block_dim{kNaiveBlockDim};
  // Parameterized mode only; empty when baked. Indices into the key's chain whose
  // parameters this stage takes as extra kernel arguments after (src, dst, width,
  // height), in signature order. A gaussian index is two arguments — `int radius`, then
  // a struct of kMaxGaussianTaps floats holding its 1D weights — and any other index is
  // one `float`. The values must come from each frame's own chain at launch: the cached
  // kernel was compiled for whichever frame first asked for this chain shape.
  std::vector<std::size_t> param_ops;
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
// TILED VARIANT (docs/PLAN.md Phase 7). Every stencil stage stages its input through a
// __shared__ tile: the block cooperatively loads tile_size^2 pixels plus a radius-wide
// apron — through the same clamped tap helper the naive kernel calls, so edge handling
// and the prologue are one piece of code in both — synchronizes, and then every tap
// reads shared memory. What is stored is the sample AFTER the prologue, which is half
// the win: the naive kernel re-loads, re-converts and re-runs the prologue at every
// tap of every pixel, the tiled one does it once per input sample. The accumulation
// around the fetch is the same text in both variants, so tiling changes where operands
// come from and never the arithmetic. Pointwise-only stages have no neighbours to share
// and are emitted naive in either variant.
//
// PARAMETERIZED CONSTANTS (docs/PLAN.md Phase 8). The same kernels, with every op
// parameter — the gaussian's radius and weights, brightness, threshold — read from a
// kernel argument instead of a literal, so the gaussian's tap loop has runtime bounds
// NVRTC cannot unroll. The source then contains no parameter value at all, which is
// why such keys ignore parameters: one compile serves every sigma. The channel count
// and sobel's fixed 3x3 coefficients stay literals — they are the kernel's shape, not
// something a client chooses.
//
// Throws std::invalid_argument on a key whose channel count or tile is unsupported
// (is_supported_tile), or whose constants mode cannot be combined with its tile
// (is_supported_constants).
GeneratedProgram emit_cuda_source(const KernelKey& key);

}  // namespace imgjit::cuda
