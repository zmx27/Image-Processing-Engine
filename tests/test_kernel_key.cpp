// KernelKey hashing (docs/PLAN.md Phase 2, CLAUDE.md invariant 4).

#include <string_view>
#include <unordered_set>

#include "catch_amalgamated.hpp"
#include "imgjit/core/kernel_key.h"

using imgjit::ConstantsMode;
using imgjit::hash_kernel_key;
using imgjit::KernelKey;
using imgjit::parse_op_chain;
using imgjit::TileVariant;

namespace {

KernelKey key_for(std::string_view chain, int channels = 3) {
  const auto parsed = parse_op_chain(chain);
  REQUIRE(parsed.has_value());
  KernelKey key;
  key.chain = *parsed;
  key.channels = channels;
  return key;
}

}  // namespace

TEST_CASE("one chain gives one key, whatever the frame", "[core]") {
  // Invariant 4 is enforced structurally rather than by a check here: KernelKey has
  // no dimension field and hash_kernel_key takes nothing but the key, so no
  // resolution change can become a cache miss. There is nothing at runtime to assert
  // about a field that does not exist — the real regression test is Phase 3's, where
  // the same chain at three resolutions must leave the compile counter at one.
  CHECK(hash_kernel_key(key_for("gaussian:1.4,sobel")) ==
        hash_kernel_key(key_for("gaussian:1.4,sobel")));
}

TEST_CASE("hashing is deterministic and equality-consistent", "[core]") {
  const KernelKey key = key_for("grayscale,gaussian:1.4,sobel,threshold:0.3");
  const KernelKey same = key_for("grayscale,gaussian:1.4,sobel,threshold:0.3");
  CHECK(key == same);
  CHECK(hash_kernel_key(key) == hash_kernel_key(key));
  CHECK(hash_kernel_key(key) == hash_kernel_key(same));
}

TEST_CASE("canonically equivalent spellings share one cache entry", "[core]") {
  // The whole point of canonicalizing before hashing: these are one compile, not five.
  const std::uint64_t expected = hash_kernel_key(key_for("gaussian:1.4"));
  CHECK(hash_kernel_key(key_for("gaussian:1.40")) == expected);
  CHECK(hash_kernel_key(key_for(" gaussian : 1.4 ")) == expected);
  CHECK(hash_kernel_key(key_for("gaussian:1.4000001")) == expected);
  CHECK(hash_kernel_key(key_for("gaussian:1.399")) == expected);
}

TEST_CASE("every codegen input changes the key", "[core]") {
  // A missing input is the failure mode that does not crash: it returns a stale
  // kernel and reports a wrong benchmark number.
  std::unordered_set<std::uint64_t> hashes;
  const auto distinct = [&hashes](const KernelKey& key) {
    CHECK(hashes.insert(hash_kernel_key(key)).second);
  };

  distinct(key_for("gaussian:1.4,sobel"));
  distinct(key_for("sobel,gaussian:1.4"));   // order
  distinct(key_for("gaussian:1.5,sobel"));   // parameter
  distinct(key_for("gaussian:1.4"));         // chain length
  distinct(key_for("gaussian:1.4,sobel", 1));  // channel count
  distinct(key_for("gaussian:1.4,sobel", 4));

  KernelKey tiled = key_for("gaussian:1.4,sobel");
  tiled.tile = TileVariant::kTiled;
  tiled.tile_size = 16;
  distinct(tiled);

  // Phase 7: the tile size is baked into the __shared__ array, so two tile sizes are
  // two kernels. A naive|tiled boolean would collide them here.
  KernelKey tiled_32 = tiled;
  tiled_32.tile_size = 32;
  distinct(tiled_32);

  // Phase 8's A/B axis: same chain, constants passed rather than baked.
  KernelKey parameterized = key_for("gaussian:1.4,sobel");
  parameterized.constants = ConstantsMode::kParameterized;
  distinct(parameterized);
}

TEST_CASE("an empty chain hashes stably", "[core]") {
  CHECK(hash_kernel_key(key_for("")) == hash_kernel_key(key_for("")));
  CHECK(hash_kernel_key(key_for("")) != hash_kernel_key(key_for("invert")));
}
