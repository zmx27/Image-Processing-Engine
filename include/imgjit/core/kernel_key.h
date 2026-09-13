#pragma once

// The kernel cache key: every codegen input, and nothing else (invariant 4).
//
// WIDTH AND HEIGHT ARE ABSENT BY CONSTRUCTION — there is no field to set and no
// overload that takes them. They are launch-time kernel arguments; baking them would
// make every new resolution a cache miss and a fresh ~100 ms NVRTC compile, growing
// the cache unboundedly and defeating the memoization this project exists to show.
//
// The converse failure is quieter and therefore worse: a codegen input MISSING from
// this key produces a stale cache hit, which surfaces as a wrong benchmark number
// rather than a crash. So the two inputs that arrive in later phases are reserved
// here, already hashed, rather than retrofitted onto a key the cache is already
// keyed by. The authoritative field list lives in docs/PLAN.md Phase 3.
//
// Portable: no CUDA, builds on macOS with no toolkit present.

#include <cstdint>

#include "imgjit/core/op_chain.h"

namespace imgjit {

// Phase 7. Not a bare naive|tiled boolean: the tile dimensions are baked into the
// __shared__ array, so two tile sizes are two different kernels and a boolean would
// collide them onto one key.
enum class TileVariant : std::uint8_t { kNaive = 0, kTiled = 1 };

// Phase 8's A/B axis. Same kernel, op parameters (gaussian sigma, brightness,
// threshold) baked as literals versus passed as kernel arguments at launch — if this is
// not in the key, the parameterized run silently reuses the baked kernel and the
// benchmark measures nothing.
//
// Parameterized, the values are no longer codegen inputs, so invariant 4's "and nothing
// else" takes them OUT of the key: gaussian:1.4 and gaussian:3 share one kernel. Op
// kinds and order still count in both modes — they decide the kernel's structure.
enum class ConstantsMode : std::uint8_t { kBaked = 0, kParameterized = 1 };

struct KernelKey {
  OpChain chain;
  int channels{0};

  // Phase 7: the stencil kernels' variant. tile_size is the tile edge in pixels, and 0
  // for naive — codegen refuses a naive key with a size, so one kernel has one key.
  TileVariant tile{TileVariant::kNaive};
  int tile_size{0};
  // Phase 8; the default is what every earlier phase emits.
  ConstantsMode constants{ConstantsMode::kBaked};

  // Not defaulted: a parameterized key ignores the op parameters, and equality has to
  // agree with the hash or the cache would hold two entries for one kernel.
  friend bool operator==(const KernelKey& lhs, const KernelKey& rhs);
};

// 64-bit FNV-1a over a byte encoding of the fields above. Parameters are encoded from
// their quantized integer hundredths, not their float bits, so +0.0 and -0.0 cannot
// produce two entries for one kernel — and are skipped entirely for a parameterized key.
std::uint64_t hash_kernel_key(const KernelKey& key);

}  // namespace imgjit
