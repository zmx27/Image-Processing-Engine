#include "backend/cuda/cuda_backend.h"

#include <algorithm>
#include <exception>
#include <stdexcept>
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
  return slots_->data();
}

JobHandle CudaBackend::submit(const FrameJob& job) {
  const JobHandle handle = next_handle_++;
  Completion completion;
  completion.handle = handle;

  // A bad job, a failed NVRTC compile and a driver error all become an error
  // completion rather than a throw: docs/PROTOCOL.md status 6 says the server survives.
  try {
    if (job.input == nullptr) {
      throw std::invalid_argument("FrameJob has no input");
    }
    const Image input =
        Image::from_bytes(job.width, job.height, job.channels,
                          reinterpret_cast<const std::uint8_t*>(job.input), job.stride);
    completion.output = run_chain(input, job.chain);
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

Image CudaBackend::run_chain(const Image& input, const OpChain& chain) {
  // An empty chain is an identity pass (imgjit/core/op_chain.h). Short-circuited here
  // rather than compiled as a copy kernel: there is nothing for NVRTC to be asked for,
  // and a translation unit with no __global__ in it is not worth relying on.
  if (chain.ops.empty()) {
    return input;
  }

  KernelKey key;
  key.chain = chain;
  key.channels = input.channels();
  const cuda::CompiledChain& compiled = cache_.get(key);

  // PHASE 3 EXEMPTION from CLAUDE.md invariant 2, scoped to this phase and stated in
  // docs/PLAN.md: a file-in/file-out CLI has no pipeline for a per-frame cuMemAlloc to
  // serialize. The device pool arrives in Phase 6 and the invariant applies
  // unconditionally from there. Do not let this leak into Phase 5.
  const std::size_t bytes = input.byte_count();
  cuda::DeviceBuffer front(bytes);
  cuda::DeviceBuffer back(bytes);
  CU_CHECK(cuMemcpyHtoD(front.get(), input.data(), bytes));

  // Every stage has the same signature, so multi-stage execution is a ping-pong: read
  // `src`, write `dst`, swap. After the loop `src` names the buffer the last stage
  // wrote.
  CUdeviceptr src = front.get();
  CUdeviceptr dst = back.get();
  int width = input.width();
  int height = input.height();
  for (const CUfunction stage : compiled.stage_functions) {
    void* arguments[] = {&src, &dst, &width, &height};
    CU_CHECK(cuLaunchKernel(stage, grid_dim(width), grid_dim(height), 1, kBlockDim, kBlockDim, 1, 0,
                            nullptr, arguments, nullptr));
    std::swap(src, dst);
  }
  CU_CHECK(cuCtxSynchronize());

  Image output(input.width(), input.height(), input.channels());
  CU_CHECK(cuMemcpyDtoH(output.data(), src, bytes));
  return output;
}

}  // namespace imgjit
