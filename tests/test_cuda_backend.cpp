// Phase 3's gate (docs/PLAN.md), and Colab-only — every test here needs a real NVIDIA
// GPU. Two independent claims, registered as two ctest cases so a failure names which:
//
//   [oracle]  every chain in the corpus matches the scalar CPU backend, which is
//             invariant 9's whole purpose.
//   [cache]   a repeated chain compiles exactly once, and the same chain at three
//             resolutions still compiles once.
//
// The second is not a performance nicety. A codegen input missing from KernelKey
// returns a STALE KERNEL rather than failing, so the only thing standing between this
// project and a benchmark that quietly measures the wrong kernel is a counter that
// someone asserts on.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "backend/cuda/cuda_backend.h"
#include "catch_amalgamated.hpp"
#include "imgjit/backend/cpu/ops.h"
#include "imgjit/core/image.h"
#include "imgjit/core/kernel_key.h"
#include "imgjit/core/op_chain.h"

using imgjit::Image;
using imgjit::KernelKey;
using imgjit::OpChain;
using imgjit::parse_op_chain;
using imgjit::TileVariant;
using imgjit::ConstantsMode;

namespace {

// Deterministic, so a failure is reproducible and a threshold that lands on a knife
// edge does so identically on every run rather than flaking.
Image make_image(int width, int height, int channels) {
  Image image(width, height, channels);
  std::uint32_t state = 0x9e3779b9U;
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      for (int c = 0; c < channels; ++c) {
        state = state * 1664525U + 1013904223U;
        image.at(x, y, c) = static_cast<std::uint8_t>(state >> 24U);
      }
    }
  }
  return image;
}

OpChain chain_for(std::string_view text) {
  const auto parsed = parse_op_chain(text);
  REQUIRE(parsed.has_value());
  return *parsed;
}

int max_abs_difference(const Image& lhs, const Image& rhs) {
  REQUIRE(lhs.width() == rhs.width());
  REQUIRE(lhs.height() == rhs.height());
  REQUIRE(lhs.channels() == rhs.channels());
  REQUIRE(lhs.byte_count() == rhs.byte_count());

  int worst = 0;
  for (std::size_t i = 0; i < lhs.byte_count(); ++i) {
    const int difference = static_cast<int>(lhs.data()[i]) - static_cast<int>(rhs.data()[i]);
    worst = std::max(worst, difference < 0 ? -difference : difference);
  }
  return worst;
}

// One backend, one context, one slot allocation — matching how the server will use it,
// and how invariant 2 says slots are allocated: once, at startup.
class Gpu {
 public:
  explicit Gpu(std::size_t max_slot_bytes, TileVariant tile = TileVariant::kNaive,
               int tile_size = 0, ConstantsMode constants = ConstantsMode::kBaked)
      : backend_(0, imgjit::CudaBackend::kDefaultStreams, tile, tile_size, constants),
        slot_(backend_.allocate_slots(1, max_slot_bytes)),
        max_bytes_(max_slot_bytes) {}

  Image run(const Image& input, const OpChain& chain) {
    REQUIRE(input.byte_count() <= max_bytes_);
    std::memcpy(slot_, input.data(), input.byte_count());

    imgjit::FrameJob job;
    job.input = slot_;
    job.width = input.width();
    job.height = input.height();
    job.channels = input.channels();
    job.stride = input.stride();
    job.chain = chain;

    const imgjit::JobHandle handle = backend_.submit(job);
    // Poll until it comes back. From Phase 6 submit() returns while the frame is still
    // on the GPU, so a single poll is a race rather than a shortcut — and the interface
    // has said so since Phase 2 ("may complete immediately").
    std::vector<imgjit::Completion> completions;
    while (completions.empty()) {
      completions = backend_.poll_completions();
    }
    REQUIRE(completions.size() == 1);
    REQUIRE(completions.front().handle == handle);
    INFO("backend error: " << completions.front().error);
    REQUIRE(completions.front().status == imgjit::JobStatus::kOk);
    return completions.front().output;
  }

  void prewarm(const OpChain& chain, int channels) { backend_.prewarm(chain, channels); }

  std::size_t compiles() const { return backend_.compile_count(); }
  imgjit::BackendStats stats() const { return backend_.stats(); }

 private:
  imgjit::CudaBackend backend_;
  std::byte* slot_;
  std::size_t max_bytes_{0};
};

// The corpus the phase gate is "complete against" — which is only a meaningful phrase
// because the op set is closed at six (docs/PLAN.md Phase 2). It covers each op alone,
// both fusion directions, both stencils in one chain, and the interleaving that
// exercises multi-stage execution with intermediate device buffers.
constexpr std::string_view kCorpus[] = {
    "",
    "invert",
    "grayscale",
    "brightness:0.2",
    "brightness:-0.35",
    "threshold:0.4",
    "gaussian:0.5",
    "gaussian:1.4",
    "gaussian:4",
    "sobel",
    "grayscale,sobel",
    "sobel,invert",
    "gaussian:1.4,sobel",
    "grayscale,gaussian:1.4,sobel",
    "invert,gaussian:1,invert,sobel,invert",
    "grayscale,gaussian:1.4,sobel,threshold:0.3",
    "grayscale,invert,brightness:0.1,threshold:0.6",
};

}  // namespace

TEST_CASE("every chain in the corpus matches the CPU oracle", "[oracle]") {
  // Odd dimensions on purpose: they leave a partial thread block on both axes, so the
  // bounds check and the clamped (replicate) edge addressing are exercised rather than
  // assumed.
  constexpr int kWidth = 37;
  constexpr int kHeight = 23;
  Gpu gpu(static_cast<std::size_t>(kWidth) * kHeight * 4);

  for (const int channels : {1, 3, 4}) {
    const Image input = make_image(kWidth, kHeight, channels);
    for (const std::string_view text : kCorpus) {
      CAPTURE(text, channels);
      const OpChain chain = chain_for(text);
      const Image expected = imgjit::cpu::apply_chain(input, chain);
      const Image actual = gpu.run(input, chain);

      // <=1 LSB, never bit-equality: FMA contraction and reassociation happen on both
      // sides (Apple Silicon contracts the oracle's own multiply-adds), so exactness is
      // the wrong bar for anything with a float in it.
      //
      // Fusion does NOT widen this. The generated kernel re-quantizes at every op
      // boundary, exactly where the oracle stores a uint8 image, so a fused chain and
      // an unfused one round at the same points (src/backend/cuda/kernel_prelude.h).
      //
      // The one place this bound is fragile by nature is a threshold immediately after
      // a stencil: it is a discontinuity, so a sub-ulp disagreement becomes a 255
      // difference for a sample that lands on the knife edge. The inputs here are
      // fixed and deterministic, so that is reproducible rather than flaky — if it
      // ever trips, read it as a straddling pixel, not as a broken kernel.
      CHECK(max_abs_difference(expected, actual) <= 1);
    }
  }
}

TEST_CASE("integer pointwise chains match the oracle exactly", "[oracle]") {
  // docs/PLAN.md Phase 2: inversion is the one op with no float round trip in the
  // oracle, so it is the one that may be compared exactly. The generated kernel writes
  // it as 1.0f - v and still lands on the same byte — the true result is an integer and
  // the float error is orders of magnitude below the rounding.
  Gpu gpu(64 * 64 * 4);
  for (const int channels : {1, 3, 4}) {
    const Image input = make_image(64, 64, channels);
    for (const std::string_view text : {"invert", "invert,invert"}) {
      CAPTURE(text, channels);
      const OpChain chain = chain_for(text);
      CHECK(gpu.run(input, chain) == imgjit::cpu::apply_chain(input, chain));
    }
  }
}

TEST_CASE("a repeated chain compiles exactly once", "[cache]") {
  Gpu gpu(64 * 64 * 3);
  const Image input = make_image(64, 64, 3);
  const OpChain chain = chain_for("grayscale,gaussian:1.4,sobel");

  gpu.run(input, chain);
  REQUIRE(gpu.compiles() == 1);
  for (int i = 0; i < 4; ++i) {
    gpu.run(input, chain);
  }
  CHECK(gpu.compiles() == 1);
}

TEST_CASE("resolution is not part of the kernel identity", "[cache]") {
  // Invariant 4, as a runtime assertion rather than a structural one. If
  // width or height ever became a codegen input, this is where it would show: three
  // compiles instead of one, and a cache that grows with every frame size a client
  // happens to send.
  Gpu gpu(256 * 256 * 3);
  const OpChain chain = chain_for("gaussian:1.4,sobel");

  gpu.run(make_image(64, 64, 3), chain);
  gpu.run(make_image(128, 96, 3), chain);
  gpu.run(make_image(256, 256, 3), chain);
  CHECK(gpu.compiles() == 1);
}

TEST_CASE("the channel count is part of the kernel identity", "[cache]") {
  // The converse of the test above, and the reason `channels` is in KernelKey: it is
  // baked as a literal, so reusing an RGB kernel for RGBA would index the wrong bytes.
  Gpu gpu(64 * 64 * 4);
  const OpChain chain = chain_for("grayscale,sobel");

  gpu.run(make_image(64, 64, 3), chain);
  gpu.run(make_image(64, 64, 4), chain);
  CHECK(gpu.compiles() == 2);
}

TEST_CASE("canonically equivalent spellings share one compile", "[cache]") {
  // What quantizing at parse time buys: these are one cache entry and one kernel, not
  // four near-identical compiles of the same stencil.
  Gpu gpu(64 * 64 * 3);
  const Image input = make_image(64, 64, 3);
  for (const std::string_view text : {"gaussian:1.4", "gaussian:1.40", " gaussian : 1.4 ",
                                      "gaussian:1.4000001"}) {
    gpu.run(input, chain_for(text));
  }
  CHECK(gpu.compiles() == 1);
}

TEST_CASE("an empty chain never reaches the compiler", "[cache]") {
  // An identity pass has no kernel to generate, so it is short-circuited rather than
  // compiled as a copy.
  Gpu gpu(64 * 64 * 3);
  const Image input = make_image(64, 64, 3);
  CHECK(gpu.run(input, chain_for("")) == input);
  CHECK(gpu.compiles() == 0);
}

TEST_CASE("a prewarmed chain costs its first frame no compile", "[cache]") {
  // docs/PLAN.md Phase 5: the server warms a configured chain list at startup so the
  // first client to ask for one hits the cache instead of stalling ~50-200 ms on NVRTC.
  // The claim is only worth making if the warmed entry is the SAME cache entry the
  // frame looks up, which is what the compile counter shows here.
  Gpu gpu(64 * 64 * 4);
  const OpChain chain = chain_for("grayscale,gaussian:1.4,sobel");

  gpu.prewarm(chain, 3);
  REQUIRE(gpu.compiles() == 1);
  gpu.run(make_image(64, 64, 3), chain);
  CHECK(gpu.compiles() == 1);

  // And the converse, which is invariant 4 again: `channels` is baked as a
  // literal, so warming a chain warms it for ONE channel count. A 4-channel frame of
  // the same chain is a different kernel and must still compile.
  gpu.run(make_image(64, 64, 4), chain);
  CHECK(gpu.compiles() == 2);
}

TEST_CASE("prewarming an empty chain compiles nothing", "[cache]") {
  // An identity pass is short-circuited rather than compiled, so asking to warm one is
  // a no-op instead of an NVRTC invocation on a source file with no __global__ in it.
  Gpu gpu(64 * 64 * 3);
  gpu.prewarm(chain_for(""), 3);
  CHECK(gpu.compiles() == 0);
}

TEST_CASE("a malformed job is an error completion, not a throw", "[cache]") {
  // docs/PROTOCOL.md status 6: the server stays alive. Checked here because the CUDA
  // backend has failure modes the CPU one does not — an NVRTC error is a runtime event
  // on a path the network layer will call.
  imgjit::CudaBackend backend;
  imgjit::FrameJob job;
  job.input = nullptr;

  const imgjit::JobHandle handle = backend.submit(job);
  const std::vector<imgjit::Completion> completions = backend.poll_completions();
  REQUIRE(completions.size() == 1);
  CHECK(completions.front().handle == handle);
  CHECK(completions.front().status == imgjit::JobStatus::kError);
  CHECK_FALSE(completions.front().error.empty());
}

// ---------------------------------------------------------------------------------------
// Phase 7 (docs/PLAN.md): the tiled variant. Registered as its own ctest case,
// phase7_gpu_tiling, so a tiling failure is never confused with the naive gate above.
// ---------------------------------------------------------------------------------------

TEST_CASE("the tiled variant matches the CPU oracle at every tile size", "[tiling]") {
  // The same corpus, the same input and the same <=1 LSB bar as the naive gate, so a
  // tiled kernel is held to exactly what the naive one was.
  //
  // 37x23 earns more here than it did there. At every tile size it leaves a partial tile
  // on at least one axis, so threads past the image edge must still load their apron
  // cells and reach the barrier. At tile 8, gaussian:4's radius-12 apron is wider than
  // the tile itself, so the cooperative load loops 16 times per thread. And the image is
  // smaller than the widest apron, so every edge of every tile goes through the clamp.
  constexpr int kWidth = 37;
  constexpr int kHeight = 23;
  for (const int tile_size : {8, 16, 32}) {
    Gpu gpu(static_cast<std::size_t>(kWidth) * kHeight * 4, TileVariant::kTiled, tile_size);
    for (const int channels : {1, 3, 4}) {
      const Image input = make_image(kWidth, kHeight, channels);
      for (const std::string_view text : kCorpus) {
        CAPTURE(tile_size, text, channels);
        const OpChain chain = chain_for(text);
        CHECK(max_abs_difference(imgjit::cpu::apply_chain(input, chain), gpu.run(input, chain)) <=
              1);
      }
    }
  }
}

TEST_CASE("tiled output matches the naive kernel's", "[tiling]") {
  // Several tiles across at every size, so interior tiles — whose apron needs no clamping
  // at all — are exercised, not only edge tiles.
  //
  // The two variants run the same arithmetic on the same operands: codegen emits the
  // same accumulation text around a different fetch (test_codegen.cpp asserts that), and
  // every staged sample is exactly the value the naive tap computes, since the load IS
  // the naive tap. So they are expected to agree exactly; <=1 is the plan's bar.
  constexpr int kWidth = 131;
  constexpr int kHeight = 97;
  const std::size_t bytes = static_cast<std::size_t>(kWidth) * kHeight * 4;
  constexpr std::string_view kChains[] = {
      "gaussian:1.4",
      "gaussian:4",
      "sobel",
      "grayscale,gaussian:1.4,sobel,threshold:0.3",
      "invert,gaussian:1,invert,sobel,invert",
  };

  // Every naive output, computed and the backend closed out BEFORE any tiled backend —
  // and so any tiled CUcontext — exists. Only one CudaBackend may be alive on this
  // thread at a time: cuCtxCreate makes the new context current on the calling thread
  // without popping the old one, so a second live backend does not coexist with the
  // first, it SHADOWS it — every CUfunction the first compiled belongs to a context
  // that is no longer current, and launching one is CUDA_ERROR_INVALID_HANDLE. The real
  // server never does this (it owns exactly one CudaBackend for its whole life); this is
  // a rule about the test, not about CudaBackend.
  std::vector<Image> expected;
  double naive_kernel_ms = 0.0;
  {
    Gpu naive(bytes);
    for (const int channels : {1, 3, 4}) {
      const Image input = make_image(kWidth, kHeight, channels);
      for (const std::string_view text : kChains) {
        expected.push_back(naive.run(input, chain_for(text)));
      }
    }
    naive_kernel_ms = naive.stats().mean_kernel_ms;
  }
  CHECK(naive_kernel_ms > 0.0);

  for (const int tile_size : {8, 16, 32}) {
    Gpu tiled(bytes, TileVariant::kTiled, tile_size);
    std::size_t index = 0;
    for (const int channels : {1, 3, 4}) {
      const Image input = make_image(kWidth, kHeight, channels);
      for (const std::string_view text : kChains) {
        CAPTURE(tile_size, text, channels);
        const OpChain chain = chain_for(text);
        CHECK(max_abs_difference(expected[index++], tiled.run(input, chain)) <= 1);
      }
    }
    // The benchmark's instrument reads something: every frame above ran kernels.
    CHECK(tiled.stats().mean_kernel_ms > 0.0);
  }
}

TEST_CASE("the tile configuration is part of the kernel identity", "[tiling]") {
  // Invariant 4, asserted at the cache by key rather than through a backend
  // (a backend has one tile configuration for life). Naive, tile 16 and tile 32 are
  // three kernels — the tile edge is baked into the __shared__ array, so a naive|tiled
  // flag alone would have collided the last two — and asking for one again compiles
  // nothing.
  imgjit::CudaBackend backend;
  KernelKey key;
  key.chain = chain_for("gaussian:1.4,sobel");
  key.channels = 3;

  backend.cache().get(key);
  key.tile = TileVariant::kTiled;
  key.tile_size = 16;
  backend.cache().get(key);
  key.tile_size = 32;
  backend.cache().get(key);
  CHECK(backend.compile_count() == 3);

  key.tile_size = 16;
  backend.cache().get(key);
  CHECK(backend.compile_count() == 3);
}

TEST_CASE("a tiled backend still compiles once across resolutions", "[tiling]") {
  // The tile edge is baked; the image size is still a launch argument. If tiling had
  // leaked a dimension into the source — the apron origin is computed from blockIdx,
  // which is exactly where one could — this would compile three times.
  Gpu gpu(256 * 256 * 3, TileVariant::kTiled, 16);
  const OpChain chain = chain_for("gaussian:1.4,sobel");
  gpu.run(make_image(64, 64, 3), chain);
  gpu.run(make_image(128, 96, 3), chain);
  gpu.run(make_image(256, 256, 3), chain);
  CHECK(gpu.compiles() == 1);
}

TEST_CASE("a tile codegen cannot emit is refused when the backend is built", "[tiling]") {
  // At construction, which for the server is startup: otherwise a bad --tile is a
  // server that accepts connections and answers every frame with status 6.
  CHECK_THROWS_AS(imgjit::CudaBackend(0, 1, TileVariant::kTiled, 64), std::invalid_argument);
  CHECK_THROWS_AS(imgjit::CudaBackend(0, 1, TileVariant::kNaive, 16), std::invalid_argument);
}

// ---------------------------------------------------------------------------------------
// Phase 8 (docs/PLAN.md): parameterized constants. Registered as phase8_gpu_constants.
// The question is not only "is it right" but "are the values really the frame's": the
// cached kernel is shared by every parameter value, so a launch that passed stale values
// would still produce a plausible image — just the wrong one.
// ---------------------------------------------------------------------------------------

TEST_CASE("parameterized kernels match the CPU oracle", "[constants]") {
  // The same corpus, input and <=1 LSB bar as the baked gate.
  constexpr int kWidth = 37;
  constexpr int kHeight = 23;
  Gpu gpu(static_cast<std::size_t>(kWidth) * kHeight * 4, TileVariant::kNaive, 0,
          ConstantsMode::kParameterized);
  for (const int channels : {1, 3, 4}) {
    const Image input = make_image(kWidth, kHeight, channels);
    for (const std::string_view text : kCorpus) {
      CAPTURE(text, channels);
      const OpChain chain = chain_for(text);
      CHECK(max_abs_difference(imgjit::cpu::apply_chain(input, chain), gpu.run(input, chain)) <=
            1);
    }
  }
}

TEST_CASE("parameterized output matches the baked kernel's", "[constants]") {
  // Both modes convolve with the same weights (the backend computes them with the very
  // function codegen bakes from) through the same arithmetic, so they are expected to
  // agree exactly; <=1 is the plan's bar.
  //
  // Baked outputs first, and that backend closed out before the parameterized one is
  // built — one live CudaBackend per thread (see "tiled output matches the naive
  // kernel's").
  constexpr int kWidth = 131;
  constexpr int kHeight = 97;
  const std::size_t bytes = static_cast<std::size_t>(kWidth) * kHeight * 4;
  constexpr std::string_view kChains[] = {
      "gaussian:0.5",
      "gaussian:4",
      "brightness:-0.35",
      "grayscale,gaussian:1.4,sobel,threshold:0.3",
      "brightness:0.1,gaussian:1,threshold:0.6,sobel,brightness:-0.2",
  };

  std::vector<Image> expected;
  double baked_kernel_ms = 0.0;
  {
    Gpu baked(bytes);
    for (const int channels : {1, 3, 4}) {
      const Image input = make_image(kWidth, kHeight, channels);
      for (const std::string_view text : kChains) {
        expected.push_back(baked.run(input, chain_for(text)));
      }
    }
    baked_kernel_ms = baked.stats().mean_kernel_ms;
  }
  CHECK(baked_kernel_ms > 0.0);

  Gpu parameterized(bytes, TileVariant::kNaive, 0, ConstantsMode::kParameterized);
  std::size_t index = 0;
  for (const int channels : {1, 3, 4}) {
    const Image input = make_image(kWidth, kHeight, channels);
    for (const std::string_view text : kChains) {
      CAPTURE(text, channels);
      CHECK(max_abs_difference(expected[index++], parameterized.run(input, chain_for(text))) <= 1);
    }
  }
  CHECK(parameterized.stats().mean_kernel_ms > 0.0);
}

TEST_CASE("one parameterized compile serves every parameter value", "[constants]") {
  // Invariant 4 with the values out of the key: three sigmas and three brightnesses, one
  // compile. Each frame is also diffed against the oracle for ITS OWN values, which is
  // what proves the launch passed this frame's parameters rather than the ones the
  // cached kernel was first compiled for.
  Gpu gpu(64 * 64 * 4, TileVariant::kNaive, 0, ConstantsMode::kParameterized);
  const Image input = make_image(64, 64, 3);
  for (const std::string_view text :
       {"brightness:0.1,gaussian:0.5,brightness:-0.1", "brightness:-0.2,gaussian:1.4,brightness:0.3",
        "brightness:0.3,gaussian:4,brightness:0.05"}) {
    CAPTURE(text);
    const OpChain chain = chain_for(text);
    CHECK(max_abs_difference(imgjit::cpu::apply_chain(input, chain), gpu.run(input, chain)) <= 1);
  }
  CHECK(gpu.compiles() == 1);

  // The shape still counts: a different op order, and a different channel count.
  gpu.run(input, chain_for("gaussian:1.4,brightness:0.3,brightness:0.1"));
  CHECK(gpu.compiles() == 2);
  gpu.run(make_image(64, 64, 4), chain_for("brightness:0.1,gaussian:0.5,brightness:-0.1"));
  CHECK(gpu.compiles() == 3);
}

TEST_CASE("the constants mode is part of the kernel identity", "[constants]") {
  imgjit::CudaBackend backend;
  KernelKey key;
  key.chain = chain_for("gaussian:1.4,sobel");
  key.channels = 3;

  backend.cache().get(key);
  key.constants = ConstantsMode::kParameterized;
  backend.cache().get(key);
  CHECK(backend.compile_count() == 2);

  // Parameterized, a new sigma is the same kernel.
  key.chain = chain_for("gaussian:3,sobel");
  backend.cache().get(key);
  CHECK(backend.compile_count() == 2);
}

TEST_CASE("a tiled parameterized backend is refused when it is built", "[constants]") {
  CHECK_THROWS_AS(
      imgjit::CudaBackend(0, 1, TileVariant::kTiled, 16, ConstantsMode::kParameterized),
      std::invalid_argument);
}
