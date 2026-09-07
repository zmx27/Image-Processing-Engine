#include "imgjit/core/kernel_key.h"

#include <cmath>

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

}  // namespace

std::uint64_t hash_kernel_key(const KernelKey& key) {
  std::uint64_t state = kFnvOffsetBasis;

  hash_i32(state, static_cast<std::int32_t>(key.chain.ops.size()));
  for (const Op& op : key.chain.ops) {
    hash_byte(state, static_cast<std::uint8_t>(op.kind));
    hash_i32(state, static_cast<std::int32_t>(std::lround(static_cast<double>(op.param) * 100.0)));
  }

  hash_i32(state, key.channels);
  hash_byte(state, static_cast<std::uint8_t>(key.tile));
  hash_i32(state, key.tile_size);
  hash_byte(state, static_cast<std::uint8_t>(key.constants));
  return state;
}

}  // namespace imgjit
