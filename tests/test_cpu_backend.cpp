// The CPU backend behind IBackend (docs/PLAN.md Phase 2).
//
// What matters here is not the arithmetic — test_cpu_ops.cpp covers that — but that
// the submit/poll contract the Phase 4 server and the Phase 6 GPU worker are written
// against actually holds.

#include <cstddef>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>

#include "catch_amalgamated.hpp"
#include "imgjit/backend/cpu/cpu_backend.h"
#include "imgjit/backend/cpu/ops.h"
#include "imgjit/util/image_io.h"

using imgjit::Completion;
using imgjit::CpuBackend;
using imgjit::FrameJob;
using imgjit::Image;
using imgjit::JobHandle;
using imgjit::JobStatus;

namespace {

Image fixture(const std::string& name) {
  return imgjit::load_png(std::string(IMGJIT_TESTDATA_DIR) + "/" + name);
}

FrameJob job_for(const Image& image, const std::byte* slot, std::string_view chain_text) {
  const auto chain = imgjit::parse_op_chain(chain_text);
  REQUIRE(chain.has_value());
  FrameJob job;
  job.input = slot;
  job.width = image.width();
  job.height = image.height();
  job.channels = image.channels();
  job.stride = image.stride();
  job.chain = *chain;
  return job;
}

}  // namespace

TEST_CASE("a submitted frame comes back from poll_completions", "[cpu]") {
  const Image input = fixture("checkerboard_16x16_rgb.png");
  CpuBackend backend;
  std::byte* slot = backend.allocate_slots(1, input.byte_count());
  std::memcpy(slot, input.data(), input.byte_count());

  const JobHandle handle = backend.submit(job_for(input, slot, "grayscale,sobel"));
  const std::vector<Completion> completions = backend.poll_completions();

  REQUIRE(completions.size() == 1);
  CHECK(completions[0].handle == handle);
  CHECK(completions[0].status == JobStatus::kOk);
  CHECK(completions[0].error.empty());
  CHECK(completions[0].output ==
        imgjit::cpu::apply_chain(input, *imgjit::parse_op_chain("grayscale,sobel")));
}

TEST_CASE("a drained queue polls empty", "[cpu]") {
  const Image input = fixture("solid_4x4_rgb.png");
  CpuBackend backend;
  std::byte* slot = backend.allocate_slots(1, input.byte_count());
  std::memcpy(slot, input.data(), input.byte_count());

  CHECK(backend.poll_completions().empty());
  backend.submit(job_for(input, slot, "invert"));
  CHECK(backend.poll_completions().size() == 1);
  CHECK(backend.poll_completions().empty());
}

TEST_CASE("handles are unique and completions are matched by handle, not order",
          "[cpu]") {
  // Phase 6 retires multi-stream completions out of submission order, so nothing
  // above IBackend may assume poll order. Matching by handle from the start is what
  // keeps the Phase 4 server and its integration test from baking in FIFO.
  const Image input = fixture("solid_4x4_rgb.png");
  CpuBackend backend;
  const std::size_t bytes = input.byte_count();
  std::byte* slots = backend.allocate_slots(3, bytes);
  for (std::size_t i = 0; i < 3; ++i) {
    std::memcpy(slots + i * bytes, input.data(), bytes);
  }

  const JobHandle a = backend.submit(job_for(input, slots + 0 * bytes, "invert"));
  const JobHandle b = backend.submit(job_for(input, slots + 1 * bytes, "grayscale"));
  const JobHandle c = backend.submit(job_for(input, slots + 2 * bytes, "threshold:0.5"));
  CHECK(a != b);
  CHECK(b != c);

  const std::vector<Completion> completions = backend.poll_completions();
  REQUIRE(completions.size() == 3);
  for (const Completion& completion : completions) {
    REQUIRE(completion.status == JobStatus::kOk);
    const JobHandle handle = completion.handle;
    const std::string_view chain = handle == a   ? "invert"
                                   : handle == b ? "grayscale"
                                                 : "threshold:0.5";
    CHECK(completion.output ==
          imgjit::cpu::apply_chain(input, *imgjit::parse_op_chain(chain)));
  }
}

TEST_CASE("slots are contiguous and independent", "[cpu]") {
  // The Phase 4 slot pool owns indices into exactly this block and never storage of
  // its own, so slot i must start at base + i * bytes.
  CpuBackend backend;
  std::byte* slots = backend.allocate_slots(4, 64);
  std::memset(slots, 0xAB, 4 * 64);
  std::memset(slots + 2 * 64, 0xCD, 64);
  CHECK(static_cast<unsigned char>(slots[2 * 64]) == 0xCD);
  CHECK(static_cast<unsigned char>(slots[2 * 64 - 1]) == 0xAB);
  CHECK(static_cast<unsigned char>(slots[3 * 64]) == 0xAB);
}

TEST_CASE("slots are allocated once, at startup", "[cpu]") {
  // A second call would mean a per-frame allocation path exists, which is invariant 2
  // in the CUDA build — and there it would also implicitly synchronize.
  CpuBackend backend;
  backend.allocate_slots(2, 32);
  CHECK_THROWS(backend.allocate_slots(2, 32));
}

TEST_CASE("a bad job completes with an error rather than throwing", "[cpu]") {
  // docs/PROTOCOL.md status 6: the server stays alive.
  CpuBackend backend;
  backend.allocate_slots(1, 16);
  FrameJob job;
  job.input = nullptr;
  job.width = 4;
  job.height = 4;
  job.channels = 1;
  job.stride = 4;

  const JobHandle handle = backend.submit(job);
  const std::vector<Completion> completions = backend.poll_completions();
  REQUIRE(completions.size() == 1);
  CHECK(completions[0].handle == handle);
  CHECK(completions[0].status == JobStatus::kError);
  CHECK_FALSE(completions[0].error.empty());
  CHECK(completions[0].output.empty());
}
