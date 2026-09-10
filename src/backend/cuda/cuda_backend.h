#pragma once

// The CUDA backend: IBackend over NVRTC codegen, the kernel cache and the driver API.
//
// PHASE 6 SHAPE: K STREAMS, GENUINELY ASYNCHRONOUS. submit() claims an idle stream slot,
// queues H2D -> stages -> D2H on it, records a CUevent and RETURNS while the GPU is
// still working; poll_completions() retires the frames whose events have come back.
// Phase 5's single stream with a cuStreamSynchronize at the end of submit() is gone —
// it was a deliberate degenerate implementation of this same interface, so that the
// phase which first put a GPU behind the server could not fail ambiguously between
// "the plumbing is wrong" and "the overlap is wrong" (docs/PLAN.md Phase 5).
//
// Because the interface was submit/poll from day one, nothing above this class had to
// change to get here: the server's worker loop was already a state machine that pops
// work, polls completions and routes them by handle (docs/PLAN.md Phase 2).
//
// A STREAM SLOT IS THE UNIT OF EVERYTHING PER-FRAME. Each holds a stream, an event, the
// device ping-pong pair the stages read and write, and a pinned staging buffer for the
// result. That is what makes CLAUDE.md invariant 3 structural rather than remembered:
// there is no way to hand a frame a device buffer except by claiming a slot, and a slot
// is returned to the free list ONLY after cuEventQuery confirms its event complete.
// Reuse-before-completion is not a mistake that can be made one line at a time here.
//
// The staging buffer is not incidental either. An async device-to-host copy into
// PAGEABLE memory is allowed to behave synchronously, and does — Phase 5 got away with
// it because one frame was in flight at a time, but it would have quietly serialized
// the pipeline this phase exists to build. The D2H lands in pinned memory and the
// worker copies out of it when the frame retires.
//
// ALLOCATION HAPPENS ONCE, IN allocate_slots(), AND NEVER PER FRAME (CLAUDE.md
// invariant 2): the host frame slots, plus every stream slot's device pair and staging
// buffer. There is no allocating path in submit() at all — a frame with no device
// buffer to run in is an error, not a quiet cuMemAlloc.
//
// CONTEXT RECREATION (docs/PLAN.md Phase 6). An illegal access poisons a context
// permanently: every later driver call returns the same sticky error, so there is no
// recovery short of destroying it. recreate_context() does that as ONE operation —
// in-flight jobs fail with an error completion, the stream slots and the kernel cache
// and the context all go, and a fresh set comes back — because these structures'
// lifetimes are genuinely coupled (every CUmodule belongs to the context) and bolting
// recovery on after the fact would be surgery on the most delicate code in the project.
//
// This header lives in src/backend/cuda/ rather than include/, because it includes
// <cuda.h> transitively and CLAUDE.md invariant 1 scans include/ precisely because
// public headers leak the furthest. Tools reach for it under IMGJIT_ENABLE_CUDA, which
// is how tools/imgjit-spike.cpp already works.
//
// THREADING: worker-thread-only, like every IBackend. Here that is not a convention
// but invariant 1 — the CUcontext this owns is created on the constructing thread, is
// only ever destroyed and rebuilt by that same thread, and never migrates.

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "backend/cuda/cuda_raii.h"
#include "backend/cuda/kernel_cache.h"
#include "imgjit/backend/backend.h"

namespace imgjit {

class CudaBackend final : public IBackend {
 public:
  // docs/PLAN.md Phase 6 says "start K=4", and 4 is enough to overlap the three stages
  // of a frame (H2D, compute, D2H) with a spare; the flag exists so the benchmark can
  // sweep it and so --streams 1 reproduces the Phase 5 baseline for a like-for-like A/B.
  static constexpr std::size_t kDefaultStreams = 4;

  // Creates the context. Constructing this object is the "cuCtxCreate on the worker
  // thread" step; there is no other entry point that creates one.
  explicit CudaBackend(int device_ordinal = 0, std::size_t stream_count = kDefaultStreams);

  std::byte* allocate_slots(std::size_t count, std::size_t bytes) override;
  JobHandle submit(const FrameJob& job) override;
  std::vector<Completion> poll_completions() override;
  BackendStats stats() const override;

  // Compiles `chain` at `channels` now, so the first client to ask for it hits the
  // cache instead of paying the ~50-200 ms NVRTC cold start (docs/ARCHITECTURE.md,
  // "NVRTC cold-start"). Called from the server's backend factory, which already runs
  // on the worker thread — which is why this is a CudaBackend method and not a virtual
  // on IBackend: prewarming is meaningless for a backend that never compiles anything.
  void prewarm(const OpChain& chain, int channels);

  // Destroys the context, the kernel cache and every stream slot, then rebuilds them.
  // Every frame on the GPU when this runs comes back from the next poll_completions()
  // as an error completion, which the server turns into a status 6 response — the
  // documented contract that a driver fault costs the frames it hit and nothing else
  // (docs/PROTOCOL.md).
  //
  // Public because Phase 8's error-injection pass drives it directly, and because a
  // recovery path with no way to invoke it deliberately is a recovery path nobody has
  // ever seen run. The backend also calls it itself on any driver error out of submit()
  // or poll_completions().
  //
  // The frame slots survive: their pages belong to this process rather than to the
  // context, so the pointer allocate_slots() handed the server stays valid across this
  // call (src/backend/cuda/cuda_raii.h, RegisteredHostBuffer).
  void recreate_context(const std::string& reason = "context recreated on request");

  // The phase gate's instrument (docs/PLAN.md Phase 3): a repeated chain, and the same
  // chain at three resolutions, must leave this unchanged. Never reset by a recreation,
  // so a test can assert that recovery flushed the cache by watching it climb again.
  std::size_t compile_count() const { return cache_.compile_count(); }
  cuda::KernelCache& cache() { return cache_; }
  const cuda::CudaContext& context() const { return *context_; }
  std::size_t stream_count() const { return stream_count_; }

 private:
  // One frame's worth of GPU resources, and the reason invariant 3 cannot be violated
  // by accident: `busy` is cleared in exactly one place, immediately after cuEventQuery
  // reports `done` complete.
  struct StreamSlot {
    explicit StreamSlot(std::size_t bytes) : front(bytes), back(bytes), staging(bytes) {}

    cuda::CudaStream stream;
    cuda::CudaEvent done;
    // Ping-pong for multi-stage execution. Two is the whole requirement at any chain
    // length: each stage reads one and writes the other.
    cuda::DeviceBuffer front;
    cuda::DeviceBuffer back;
    // Pinned destination for the D2H, so the copy is genuinely asynchronous.
    cuda::PinnedHostBuffer staging;

    bool busy{false};
    JobHandle handle{0};
    std::uint64_t submit_order{0};
    int width{0};
    int height{0};
    int channels{0};
  };

  // Queues one frame on a stream and records its event. Throws rather than reporting:
  // submit() is the single place that decides what an exception means.
  void launch(const FrameJob& job, JobHandle handle);

  // Blocks until a stream slot is free, retiring completions while it waits. Waits on
  // the oldest outstanding event rather than spinning — the worker thread has nothing
  // else to do until a slot frees, and a spin would burn a core the connection threads
  // need.
  std::size_t claim_stream();

  // Moves every slot whose event has completed onto `completed_`. The one place a slot
  // is returned to the free list.
  void retire_ready_streams();

  void build_stream_slots();
  void fail_job(JobHandle handle, const std::string& reason);

  // recreate_context() that cannot throw, for the two call sites that must not: an
  // exception out of poll_completions() would take the server's worker thread, which
  // the loop calls it from without a try block.
  void recover(const std::string& reason) noexcept;

  // Declaration order is load-bearing. cache_ is constructed from context_'s compute
  // capability and every CUmodule it holds belongs to that context; the frame slots'
  // registration and every stream slot's resources belong to it too. Destruction runs
  // in reverse, so the context outlives all of them.
  std::optional<cuda::CudaContext> context_;
  cuda::KernelCache cache_;
  std::optional<cuda::RegisteredHostBuffer> host_slots_;
  std::vector<StreamSlot> streams_;

  std::vector<Completion> completed_;
  int device_ordinal_{0};
  std::size_t stream_count_{kDefaultStreams};
  std::size_t slot_bytes_{0};
  JobHandle next_handle_{1};
  std::uint64_t next_submit_order_{1};

  BackendStats stats_;
  double occupancy_sum_{0.0};
  std::uint64_t occupancy_samples_{0};
};

}  // namespace imgjit
