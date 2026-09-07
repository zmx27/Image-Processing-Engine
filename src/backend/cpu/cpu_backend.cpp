#include "imgjit/backend/cpu/cpu_backend.h"

#include <exception>
#include <stdexcept>
#include <utility>

#include "imgjit/backend/cpu/ops.h"

namespace imgjit {

std::byte* CpuBackend::allocate_slots(std::size_t count, std::size_t bytes) {
  if (!slots_.empty()) {
    throw std::logic_error("CpuBackend::allocate_slots: slots are allocated once, at startup");
  }
  slots_.resize(count * bytes);
  return slots_.data();
}

JobHandle CpuBackend::submit(const FrameJob& job) {
  const JobHandle handle = next_handle_++;
  Completion completion;
  completion.handle = handle;

  // A malformed job is a completion with an error status, never a throw out of
  // submit(): docs/PROTOCOL.md status 6 says the server stays alive.
  try {
    if (job.input == nullptr) {
      throw std::invalid_argument("FrameJob has no input");
    }
    const Image input =
        Image::from_bytes(job.width, job.height, job.channels,
                          reinterpret_cast<const std::uint8_t*>(job.input), job.stride);
    completion.output = cpu::apply_chain(input, job.chain);
  } catch (const std::exception& error) {
    completion.status = JobStatus::kError;
    completion.output = Image{};
    completion.error = error.what();
  }

  completed_.push_back(std::move(completion));
  return handle;
}

std::vector<Completion> CpuBackend::poll_completions() {
  return std::exchange(completed_, {});
}

}  // namespace imgjit
