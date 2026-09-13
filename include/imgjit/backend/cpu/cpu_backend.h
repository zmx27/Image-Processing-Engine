#pragma once

// The CPU backend: IBackend over the scalar oracle in cpu/ops.h.
//
// It completes each job inside submit() and hands the result back from the next
// poll_completions(). That is a degenerate but honest implementation of the async
// contract — the server loop is a submit/poll state machine from Phase 4 onward, so
// Phase 6 changes the backend and nothing above it.
//
// Not thread-safe, deliberately: IBackend is a worker-thread-only interface (see
// backend.h), so a lock here would be protecting against a call that would already be
// violating invariant 1 in the CUDA build.
//
// Portable: no CUDA, builds on macOS with no toolkit present.

#include <cstddef>
#include <vector>

#include "imgjit/backend/backend.h"

namespace imgjit {

class CpuBackend final : public IBackend {
 public:
  std::byte* allocate_slots(std::size_t count, std::size_t bytes) override;
  JobHandle submit(const FrameJob& job) override;
  std::vector<Completion> poll_completions() override;

 private:
  std::vector<std::byte> slots_;
  std::vector<Completion> completed_;
  JobHandle next_handle_{1};
};

}  // namespace imgjit
