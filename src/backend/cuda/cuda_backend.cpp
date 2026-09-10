#include "backend/cuda/cuda_backend.h"

#include <algorithm>
#include <cstdint>
#include <exception>
#include <stdexcept>
#include <string>
#include <utility>

namespace imgjit {
namespace {

// 16x16 = 256 threads, the usual starting point for a 2D image kernel. Phase 7 makes
// the tile a codegen input and sweeps it; until then it is a launch detail and stays
// out of KernelKey.
constexpr unsigned int kBlockDim = 16;

unsigned int grid_dim(int extent) {
  return (static_cast<unsigned int>(extent) + kBlockDim - 1) / kBlockDim;
}

}  // namespace

CudaBackend::CudaBackend(int device_ordinal)
    : context_(device_ordinal), cache_(context_.compute_arch()) {}

std::byte* CudaBackend::allocate_slots(std::size_t count, std::size_t bytes) {
  if (slots_.has_value()) {
    throw std::logic_error("CudaBackend::allocate_slots: slots are allocated once, at startup");
  }
  slots_.emplace(count * bytes);

  // The device side of the same "allocate once" rule. `bytes` is the slot size, and the
  // server caps every payload at it (docs/PROTOCOL.md validates
  // width*height*channels == payload_len <= max_payload_bytes), so any frame that
  // passes validation fits these buffers and the per-frame path below never allocates.
  device_front_.emplace(bytes);
  device_back_.emplace(bytes);
  return slots_->data();
}

void CudaBackend::prewarm(const OpChain& chain, int channels) {
  if (chain.ops.empty()) {
    return;  // an identity pass is short-circuited, so there is no kernel to warm
  }
  KernelKey key;
  key.chain = chain;
  key.channels = channels;
  cache_.get(key);
}

JobHandle CudaBackend::submit(const FrameJob& job) {
  const JobHandle handle = next_handle_++;
  Completion completion;
  completion.handle = handle;

  // A bad job, a failed NVRTC compile and a driver error all become an error
  // completion rather than a throw: docs/PROTOCOL.md status 6 says the server survives.
  try {
    completion.output = run_chain(job);
  } catch (const std::exception& error) {
    completion.status = JobStatus::kError;
    completion.output = Image{};
    completion.error = error.what();
  }

  completed_.push_back(std::move(completion));
  return handle;
}

std::vector<Completion> CudaBackend::poll_completions() {
  return std::exchange(completed_, {});
}

Image CudaBackend::run_chain(const FrameJob& job) {
  if (job.input == nullptr) {
    throw std::invalid_argument("FrameJob has no input");
  }
  if (job.width <= 0 || job.height <= 0 || !is_supported_channel_count(job.channels)) {
    throw std::invalid_argument("FrameJob has unusable dimensions");
  }
  const std::size_t row_bytes =
      static_cast<std::size_t>(job.width) * static_cast<std::size_t>(job.channels);
  // Packed rows only, which is what every caller supplies: the wire payload is packed
  // (docs/PROTOCOL.md) and Image always allocates packed rows. Rejecting a padded
  // stride keeps the copy below a single contiguous transfer straight out of the slot;
  // the alternative is a row-by-row staging copy, which is the thing the pinned pool
  // exists to avoid.
  if (job.stride != row_bytes) {
    throw std::invalid_argument("FrameJob rows must be packed");
  }
  const std::size_t bytes = row_bytes * static_cast<std::size_t>(job.height);
  const auto* input_bytes = reinterpret_cast<const std::uint8_t*>(job.input);

  // An empty chain is an identity pass (imgjit/core/op_chain.h). Short-circuited here
  // rather than compiled as a copy kernel: there is nothing for NVRTC to be asked for,
  // and a translation unit with no __global__ in it is not worth relying on.
  if (job.chain.ops.empty()) {
    return Image::from_bytes(job.width, job.height, job.channels, input_bytes, job.stride);
  }

  KernelKey key;
  key.chain = job.chain;
  key.channels = job.channels;
  const cuda::CompiledChain& compiled = cache_.get(key);

  // CLAUDE.md invariant 2, enforced rather than merely intended: there is no per-frame
  // allocation path here at all, so the only way to get a device buffer is to have
  // asked for one at startup. Phase 3's exemption — a file-in/file-out CLI may allocate
  // per invocation, having no pipeline to serialize — is CLOSED as of Phase 5, and it
  // closed for free: imgjit-cli allocates a slot sized to its one image, so it takes
  // the pooled path like everything else.
  //
  // The size check cannot fail through the server: docs/PROTOCOL.md validates
  // width*height*channels == payload_len <= max_payload_bytes, and max_payload_bytes IS
  // the slot size these buffers were sized to. It is a loud error rather than a silent
  // allocation because the alternative to noticing here is corrupting the frame.
  if (!device_front_.has_value() || device_front_->size() < bytes) {
    throw std::logic_error(
        "CudaBackend: no device buffer for a " + std::to_string(bytes) +
        " byte frame — allocate_slots() must be called at startup with at least that slot size");
  }
  CUdeviceptr src = device_front_->get();
  CUdeviceptr dst = device_back_->get();

  // Straight out of the pinned slot, with no staging copy on ingest — the entire reason
  // the pool exists (docs/ARCHITECTURE.md decision 2). Async, on the one stream: the
  // copies and the launches are ordered by the stream rather than by the host, so
  // Phase 6 adds streams and events here instead of changing the shape of this code.
  CU_CHECK(cuMemcpyHtoDAsync(src, input_bytes, bytes, stream_.get()));

  // Every stage has the same signature, so multi-stage execution is a ping-pong: read
  // `src`, write `dst`, swap. After the loop `src` names the buffer the last stage
  // wrote.
  int width = job.width;
  int height = job.height;
  for (const CUfunction stage : compiled.stage_functions) {
    void* arguments[] = {&src, &dst, &width, &height};
    CU_CHECK(cuLaunchKernel(stage, grid_dim(width), grid_dim(height), 1, kBlockDim, kBlockDim, 1, 0,
                            stream_.get(), arguments, nullptr));
    std::swap(src, dst);
  }

  Image output(job.width, job.height, job.channels);
  CU_CHECK(cuMemcpyDtoHAsync(output.data(), src, bytes, stream_.get()));
  // The single sync point, and the only reason submit() can complete a job inline. The
  // destination is pageable, so the driver stages this copy — that costs nothing while
  // one frame is in flight at a time, and Phase 6 is where it starts to matter.
  CU_CHECK(cuStreamSynchronize(stream_.get()));
  return output;
}

}  // namespace imgjit
