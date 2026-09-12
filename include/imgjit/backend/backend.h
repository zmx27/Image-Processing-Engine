#pragma once

// The interface the server talks to, identical in the CPU and CUDA builds.
//
// SUBMIT/POLL FROM DAY ONE, even though the CPU backend has nothing to poll. The
// tempting Phase 2 signature is a blocking `Image process(const Image&, const
// OpChain&)`; it is fine now, fine in Phase 5, and impossible in Phase 6, where the
// worker keeps K frames in flight and retires them by polling a completion event.
// Adopting the blocking shape means redesigning the interface AND the worker loop at
// exactly the
// phase that introduces the event-gated-return invariant — the worst place in this
// project to be changing structure (docs/PLAN.md Phase 2).
//
// THREADING: every method here is called on the worker thread and only there. That
// is not an incidental property of the CPU backend, it is CLAUDE.md invariant 1 for
// the CUDA one — which is why no implementation needs a lock.
//
// Portable: no CUDA, builds on macOS with no toolkit present. `allocate_slots`
// returning std::byte* is what keeps it that way — the CUDA backend hands back pinned
// host memory through the same signature, so invariant 1's grep still passes with the
// pinned pool in place. (That grep is why no comment in this file spells a driver
// entry point: a false positive in a header is exactly what teaches people to stop
// running the check.)

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "imgjit/core/image.h"
#include "imgjit/core/op_chain.h"

namespace imgjit {

// Backend-assigned. The server keeps handle -> {connection, seq_num, slot} and
// releases the slot when the matching completion arrives — never before, and never
// after the response is written (CLAUDE.md invariants 3 and 10).
using JobHandle = std::uint64_t;

struct FrameJob {
  // Input pixels, living in a slot from allocate_slots(). Readable until the job's
  // completion is returned from poll_completions().
  const std::byte* input{nullptr};
  int width{0};
  int height{0};
  int channels{0};
  std::size_t stride{0};
  OpChain chain;
};

enum class JobStatus : std::uint8_t {
  kOk = 0,
  // Maps to protocol status 6 (docs/PROTOCOL.md): the server stays alive.
  kError = 1,
};

struct Completion {
  JobHandle handle{0};
  JobStatus status{JobStatus::kOk};
  Image output;       // empty unless status == kOk
  std::string error;  // empty when status == kOk
};

// Phase 6's instrumentation, in portable terms so the server can report it without
// knowing what a stream is (docs/PLAN.md Phase 6, "instrument queue depth, occupancy,
// stall counts" — queue depth is the server's own Server::max_queue_depth()).
//
// Occupancy is the number the phase gate actually turns on. "Measurably faster" and
// "genuine overlap rather than serialized segments" are two different claims, and a
// throughput win can be had without the second: if mean_in_flight sits near 1 while K
// is 4, the pipeline is serialized and the extra streams bought nothing, whatever the
// FPS says.
struct BackendStats {
  std::uint64_t frames_submitted{0};
  // Times a submit found every stream busy and had to wait for one to retire. Nonzero
  // is healthy — it means the GPU is the bottleneck rather than the queue feeding it.
  std::uint64_t submit_stalls{0};
  std::uint64_t max_in_flight{0};
  // Driver-error recoveries. A run that serves traffic normally must report 0.
  std::uint64_t context_recreations{0};
  // Mean frames on the GPU, sampled once per launch.
  double mean_in_flight{0.0};
  // Phase 7: mean GPU time per frame from the start of its first stage to the end of its
  // last, read off the GPU's own clock — the kernels alone, with no copy, host work or
  // compile in it. This is what the naive|tiled A/B turns on: a 3x3 sobel is a sliver
  // of a copy-bound frame, so end-to-end FPS cannot show it getting faster.
  //
  // A clean per-frame kernel time only at ONE stream. With several, the GPU time-slices
  // other frames' kernels into this frame's window (kernels do not overlap kernels on a
  // full-size image — docs/PLAN.md Phase 6), so the number then includes the wait.
  double mean_kernel_ms{0.0};
  // Phase 8: runtime kernel compiles since startup, the prewarm included. Both of this
  // phase's own axes are read off it — a load that varies one op's parameter compiles
  // once per distinct value when constants are baked, and once in total when they are
  // parameterized (docs/PLAN.md Phase 8). 0 on a backend that compiles nothing.
  //
  // Deliberately not named for the compiler that does it: this is the portable
  // interface, and invariant 1's grep scans include/ for exactly that spelling.
  std::uint64_t jit_compiles{0};
};

class IBackend {
 public:
  virtual ~IBackend() = default;

  IBackend() = default;
  IBackend(const IBackend&) = delete;
  IBackend& operator=(const IBackend&) = delete;

  // Allocates `count` contiguous slots of `bytes` each and returns the base pointer;
  // slot i starts at base + i * bytes. Called once at worker startup — CPU: heap,
  // CUDA: pinned host memory, allocated on the worker. The storage is owned by the backend
  // and stays valid for its lifetime, which is what lets the Phase 4 slot pool own
  // only indices and never storage.
  virtual std::byte* allocate_slots(std::size_t count, std::size_t bytes) = 0;

  // May complete the job immediately (the CPU backend does) or merely enqueue it.
  // Either way the result is observed through poll_completions().
  virtual JobHandle submit(const FrameJob& job) = 0;

  // Returns every job finished since the last call, in no guaranteed order — from
  // Phase 6 multi-stream completions genuinely retire out of submission order, which
  // is the entire reason seq_num exists in the protocol.
  virtual std::vector<Completion> poll_completions() = 0;

  // Counters accumulated since construction. Not pure: a backend with one frame in
  // flight at a time has nothing to say here, and the CPU backend's defaults are the
  // honest answer rather than a stub.
  virtual BackendStats stats() const { return {}; }
};

}  // namespace imgjit
