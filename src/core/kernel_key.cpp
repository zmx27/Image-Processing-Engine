#include "imgjit/core/kernel_key.h"

#include <cmath>
#include <cstddef>

namespace imgjit {
namespace {

constexpr std::uint64_t kFnvOffsetBasis = 1469598103934665603ULL;
constexpr std::uint64_t kFnvPrime = 1099511628211ULL;

void hash_byte(std::uint64_t& state, std::uint8_t byte) {
  state ^= byte;
  state *= kFnvPrime;
}

void hash_i32(std::uint64_t& state, std::int32_t value) {
  const auto bits = static_cast<std::uint32_t>(value);
  for (int shift = 0; shift < 32; shift += 8) {
    hash_byte(state, static_cast<std::uint8_t>((bits >> shift) & 0xFFU));
  }
}

// The one encoding of a parameter that both equality and the hash read, so the two
// cannot disagree about whether two keys name one kernel.
std::int32_t param_hundredths(float param) {
  return static_cast<std::int32_t>(std::lround(static_cast<double>(param) * 100.0));
}

}  // namespace

bool operator==(const KernelKey& lhs, const KernelKey& rhs) {
  if (lhs.channels != rhs.channels || lhs.tile != rhs.tile || lhs.tile_size != rhs.tile_size ||
      lhs.constants != rhs.constants || lhs.chain.ops.size() != rhs.chain.ops.size()) {
    return false;
  }
  const bool keyed_by_params = lhs.constants == ConstantsMode::kBaked;
  for (std::size_t i = 0; i < lhs.chain.ops.size(); ++i) {
    const Op& left = lhs.chain.ops[i];
    const Op& right = rhs.chain.ops[i];
    if (left.kind != right.kind ||
        (keyed_by_params && param_hundredths(left.param) != param_hundredths(right.param))) {
      return false;
    }
  }
  return true;
}

std::uint64_t hash_kernel_key(const KernelKey& key) {
  std::uint64_t state = kFnvOffsetBasis;

  const bool keyed_by_params = key.constants == ConstantsMode::kBaked;
  hash_i32(state, static_cast<std::int32_t>(key.chain.ops.size()));
  for (const Op& op : key.chain.ops) {
    hash_byte(state, static_cast<std::uint8_t>(op.kind));
    if (keyed_by_params) {
      hash_i32(state, param_hundredths(op.param));
    }
  }

  hash_i32(state, key.channels);
  hash_byte(state, static_cast<std::uint8_t>(key.tile));
  hash_i32(state, key.tile_size);
  hash_byte(state, static_cast<std::uint8_t>(key.constants));
  return state;
}

}  // namespace imgjit
