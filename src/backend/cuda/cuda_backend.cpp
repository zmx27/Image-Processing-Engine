#include "backend/cuda/cuda_backend.h"

#include <algorithm>
#include <cstdint>
#include <exception>
#include <limits>
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

CudaBackend::CudaBackend(int device_ordinal, std::size_t stream_count)
    : context_(std::in_place, device_ordinal),
      cache_(context_->compute_arch()),
      device_ordinal_(device_ordinal),
      stream_count_(stream_count == 0 ? 1 : stream_count) {}

std::byte* CudaBackend::allocate_slots(std::size_t count, std::size_t bytes) {
  if (host_slots_.has_value()) {
    throw std::logic_error("CudaBackend::allocate_slots: slots are allocated once, at startup");
  }
  if (count == 0 || bytes == 0) {
    throw std::invalid_argument("CudaBackend::allocate_slots: needs at least one non-empty slot");
  }

  // Page-locked in place rather than allocated by the driver, so that the pointer this
  // returns outlives a context recreation — the server's slot pool holds it forever and
  // reader threads recv() into it concurrently (src/backend/cuda/cuda_raii.h).
  host_slots_.emplace(count * bytes);

  // The device side of the same "allocate once" rule. `bytes` is the slot size, and the
  // server caps every payload at it (docs/PROTOCOL.md validates
  // width*height*channels == payload_len <= max_payload_bytes), so any frame that
  // passes validation fits these buffers and the per-frame path never allocates.
  slot_bytes_ = bytes;
  build_stream_slots();
  return host_slots_->data();
}

void CudaBackend::build_stream_slots() {
  streams_.clear();
  if (slot_bytes_ == 0) {
    return;  // no allocate_slots() yet; submit() will refuse rather than allocate
  }
  streams_.reserve(stream_count_);
  for (std::size_t i = 0; i < stream_count_; ++i) {
    streams_.emplace_back(slot_bytes_);
  }
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

  // Three outcomes, and the difference between the last two is the whole reason
  // NvrtcError exists (src/backend/cuda/cuda_check.h).
  try {
    launch(job, handle);
  } catch (const cuda::NvrtcError& error) {
    // A chain that will not compile is an ordinary bad job. The context is untouched —
    // NVRTC never had one — so every other client keeps running.
    fail_job(handle, error.what());
  } catch (const cuda::CudaError& error) {
    // A driver error may be sticky, in which case every later call returns it and the
    // context is unusable. There is no way to tell cheaply and no way back except a
    // recreation, so this recovers unconditionally: an unnecessary rebuild costs one
    // NVRTC warm-up, while a missed one costs every frame from here on.
    fail_job(handle, error.what());
    recover(std::string("driver error: ") + error.what());
  } catch (const std::exception& error) {
    // A malformed job. docs/PROTOCOL.md status 6: the server stays alive.
    fail_job(handle, error.what());
  }

  return handle;
}

void CudaBackend::launch(const FrameJob& job, const JobHandle handle) {
  if (!context_.has_value()) {
    throw std::runtime_error("CudaBackend: the context could not be rebuilt after a driver error");
  }
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
  // and a translation unit with no __global__ in it is not worth relying on. It also
  // never touches a stream, so it completes inline and retires from the next poll.
  if (job.chain.ops.empty()) {
    Completion completion;
    completion.handle = handle;
    completion.output =
        Image::from_bytes(job.width, job.height, job.channels, input_bytes, job.stride);
    completed_.push_back(std::move(completion));
    ++stats_.frames_submitted;
    return;
  }

  KernelKey key;
  key.chain = job.chain;
  key.channels = job.channels;
  const cuda::CompiledChain& compiled = cache_.get(key);

  // CLAUDE.md invariant 2, enforced rather than merely intended: there is no per-frame
  // allocation path here at all, so the only way to get a device buffer is to have
  // asked for one at startup. Phase 3's exemption — a file-in/file-out CLI may allocate
  // per invocation, having no pipeline to serialize — closed in Phase 5 and has stayed
  // closed: imgjit-cli allocates a slot sized to its one image, so it takes the pooled
  // path like everything else.
  //
  // The size check cannot fail through the server: docs/PROTOCOL.md validates
  // width*height*channels == payload_len <= max_payload_bytes, and max_payload_bytes IS
  // the slot size these buffers were sized to. It is a loud error rather than a silent
  // allocation because the alternative to noticing here is corrupting the frame.
  if (streams_.empty() || slot_bytes_ < bytes) {
    throw std::logic_error(
        "CudaBackend: no device buffer for a " + std::to_string(bytes) +
        " byte frame — allocate_slots() must be called at startup with at least that slot size");
  }

  // Blocks only when all K frames' worth of GPU resources are occupied, which is the
  // backend's own backpressure and is counted rather than hidden.
  const std::size_t index = claim_stream();
  StreamSlot& slot = streams_[index];

  CUdeviceptr source = slot.front.get();
  CUdeviceptr destination = slot.back.get();
  const CUstream stream = slot.stream.get();

  // Straight out of the pinned frame slot, with no staging copy on ingest — the entire
  // reason the pool exists (docs/ARCHITECTURE.md decision 2). Everything below is
  // queued on this one stream, so the copies and launches are ordered against each
  // other while running concurrently with the other K-1 streams.
  CU_CHECK(cuMemcpyHtoDAsync(source, input_bytes, bytes, stream));

  // Every stage has the same signature, so multi-stage execution is a ping-pong: read
  // `source`, write `destination`, swap. After the loop `source` names the buffer the
  // last stage wrote.
  int width = job.width;
  int height = job.height;
  for (const CUfunction stage : compiled.stage_functions) {
    void* arguments[] = {&source, &destination, &width, &height};
    CU_CHECK(cuLaunchKernel(stage, grid_dim(width), grid_dim(height), 1, kBlockDim, kBlockDim, 1, 0,
                            stream, arguments, nullptr));
    std::swap(source, destination);
  }

  // Into PINNED memory, which is what makes this copy actually asynchronous — an async
  // D2H into a pageable destination is permitted to block, and would serialize the
  // pipeline the streams exist to build. The worker copies it out at retire time.
  CU_CHECK(cuMemcpyDtoHAsync(slot.staging.data(), source, bytes, stream));
  CU_CHECK(cuEventRecord(slot.done.get(), stream));

  // Marked busy only now, and this is deliberate. If anything above threw, the slot was
  // never claimed by a frame — and it does not matter that work may already be queued
  // on its stream, because every driver error takes the recreation path, which destroys
  // the stream and its buffers outright.
  slot.busy = true;
  slot.handle = handle;
  slot.submit_order = next_submit_order_++;
  slot.width = job.width;
  slot.height = job.height;
  slot.channels = job.channels;

  ++stats_.frames_submitted;

  // Sampled per launch: the number that says whether K streams bought real overlap or
  // just a deeper queue in front of a serialized pipeline (docs/PLAN.md Phase 6).
  const auto in_flight = static_cast<std::uint64_t>(
      std::count_if(streams_.begin(), streams_.end(), [](const StreamSlot& s) { return s.busy; }));
  occupancy_sum_ += static_cast<double>(in_flight);
  ++occupancy_samples_;
  stats_.max_in_flight = std::max(stats_.max_in_flight, in_flight);
}

std::size_t CudaBackend::claim_stream() {
  for (;;) {
    for (std::size_t i = 0; i < streams_.size(); ++i) {
      if (!streams_[i].busy) {
        return i;
      }
    }

    ++stats_.submit_stalls;

    // Every slot is occupied, so block in the driver on the OLDEST outstanding event
    // rather than spinning on cuEventQuery. The streams run concurrently so completion
    // order is not guaranteed to be submission order, but the oldest is the best
    // available guess and correctness does not depend on the guess: the retire pass
    // below sweeps every slot, so whichever finished first is the one that frees.
    constexpr std::uint64_t kNotBusy = std::numeric_limits<std::uint64_t>::max();
    const auto age = [](const StreamSlot& slot) { return slot.busy ? slot.submit_order : kNotBusy; };
    const auto oldest =
        std::min_element(streams_.begin(), streams_.end(),
                         [&age](const StreamSlot& lhs, const StreamSlot& rhs) {
                           return age(lhs) < age(rhs);
                         });
    CU_CHECK(cuEventSynchronize(oldest->done.get()));
    retire_ready_streams();
  }
}

void CudaBackend::retire_ready_streams() {
  for (StreamSlot& slot : streams_) {
    if (!slot.busy) {
      continue;
    }
    const CUresult status = cuEventQuery(slot.done.get());
    if (status == CUDA_ERROR_NOT_READY) {
      continue;  // still on the GPU; nothing this frame touched may be reused yet
    }
    CU_CHECK(status);  // anything but success-or-not-ready is a driver failure

    // CLAUDE.md invariant 3, and this is the only place in the project that decides a
    // frame is finished. Past this point — and not one line before it — the device
    // pair, the staging buffer and (through the completion the server is about to
    // route) the pinned input slot are all free to be handed to another frame.
    Completion completion;
    completion.handle = slot.handle;
    completion.output = Image::from_bytes(
        slot.width, slot.height, slot.channels,
        reinterpret_cast<const std::uint8_t*>(slot.staging.data()),
        static_cast<std::size_t>(slot.width) * static_cast<std::size_t>(slot.channels));
    completed_.push_back(std::move(completion));

    slot.busy = false;
  }
}

std::vector<Completion> CudaBackend::poll_completions() {
  try {
    retire_ready_streams();
  } catch (const cuda::CudaError& error) {
    // A sticky error surfaces at the next call rather than at the failing one, so there
    // is no way to attribute it to one frame. Everything in flight dies with the
    // context, which is exactly what the recreation reports.
    recover(std::string("driver error: ") + error.what());
  }
  return std::exchange(completed_, {});
}

void CudaBackend::recover(const std::string& reason) noexcept {
  // NOTHROW, and that is the whole reason this wrapper exists rather than a bare call.
  // The server's worker loop calls poll_completions() outside any try block, so an
  // exception escaping it would take the worker thread — and with it every connection
  // waiting on a job to retire — instead of costing the frames the fault hit.
  //
  // If the rebuild itself fails the GPU is genuinely gone, and the honest response is a
  // backend that answers every later frame with status 6 rather than one that crashes:
  // launch() refuses while the context is empty. Phase 8's error injection is what
  // watches this happen.
  try {
    recreate_context(reason);
  } catch (const std::exception&) {
    // Swallowed deliberately; the in-flight jobs were already failed before anything
    // was torn down, so nothing is lost that has not been reported.
  }
}

void CudaBackend::recreate_context(const std::string& reason) {
  // 1. Every frame on the GPU is lost. Reported before anything is destroyed, so the
  //    handles are still there to report.
  for (StreamSlot& slot : streams_) {
    if (slot.busy) {
      fail_job(slot.handle, reason);
      slot.busy = false;
    }
  }

  // 2. Teardown, in dependency order, and every step of it is nothrow by construction —
  //    a poisoned context makes each of these calls return the sticky error, and RAII
  //    teardown already reports-and-swallows (src/backend/cuda/cuda_check.h).
  streams_.clear();  // events, streams, device pairs, staging buffers
  cache_.clear();    // every CUmodule belongs to the context about to go
  if (host_slots_.has_value()) {
    host_slots_->detach();  // the PAGES stay ours; only the registration is dropped
  }
  context_.reset();

  ++stats_.context_recreations;

  // 3. Rebuild. If the GPU is genuinely gone this throws and leaves context_ empty, in
  //    which case launch() refuses every subsequent frame with an error completion —
  //    a server that answers status 6 forever beats one that crashes, and it is the
  //    difference Phase 8's error injection is there to observe.
  context_.emplace(device_ordinal_);
  if (host_slots_.has_value()) {
    host_slots_->attach();  // same address, so the server's slot pool is none the wiser
  }
  build_stream_slots();
}

void CudaBackend::fail_job(const JobHandle handle, const std::string& reason) {
  Completion completion;
  completion.handle = handle;
  completion.status = JobStatus::kError;
  completion.output = Image{};
  completion.error = reason;
  completed_.push_back(std::move(completion));
}

BackendStats CudaBackend::stats() const {
  BackendStats snapshot = stats_;
  snapshot.mean_in_flight =
      occupancy_samples_ == 0 ? 0.0 : occupancy_sum_ / static_cast<double>(occupancy_samples_);
  return snapshot;
}

}  // namespace imgjit
