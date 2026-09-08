#pragma once

// The frame-slot free list. Connection reader threads claim a slot, recv() the payload
// straight into it, and the worker releases it when the job completes.
//
// THE POOL OWNS INDICES, NOT STORAGE — this is the whole design of the type and the
// reason it is written this way in Phase 4 rather than the obvious way (docs/PLAN.md
// Phase 4). The obvious pool new[]s its own buffer in its constructor; Phase 5 then has
// to rewrite it, because pinned host memory must be allocated by
// IBackend::allocate_slots() ON THE WORKER THREAD (CLAUDE.md invariants 1 and 2). By
// taking a base pointer it does not own, this class is identical in both phases: the
// CPU backend hands it heap memory, the CUDA backend hands it pinned memory, and not a
// line of pool logic changes.
//
// Blocking claim IS the backpressure mechanism (CLAUDE.md invariant 5). A reader that
// cannot get a slot stops recv()ing, which backs up through TCP flow control to the
// client. Frames are never dropped and the pool never grows at runtime.
//
// Portable: no CUDA, builds on macOS with no toolkit present.

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <vector>

namespace imgjit {

class SlotPool {
 public:
  // `base` must point at `count * bytes` bytes owned by someone else for the pool's
  // whole lifetime — in the server, by the backend. `per_owner_cap` bounds how many
  // slots any one connection may hold at once, so a single loud client cannot starve
  // the others (docs/ARCHITECTURE.md decision 2).
  SlotPool(std::byte* base, std::size_t count, std::size_t bytes, std::size_t per_owner_cap);

  SlotPool(const SlotPool&) = delete;
  SlotPool& operator=(const SlotPool&) = delete;

  // Blocks until a slot is free AND `owner` is under its cap. Returns nullopt only once
  // close() has been called, which is how a reader thread parked here is released at
  // shutdown instead of deadlocking the join.
  std::optional<std::size_t> claim(std::uint64_t owner);

  void release(std::size_t index);

  std::byte* slot(std::size_t index);

  void close();

  std::size_t count() const { return count_; }
  std::size_t slot_bytes() const { return bytes_; }
  std::size_t available() const;

  // How many claims had to wait. The backpressure test asserts this is non-zero against
  // a throttled worker: without it, "the reader blocked" is an assumption about a path
  // that may never have been taken.
  std::uint64_t blocked_claims() const;

 private:
  bool can_claim(std::uint64_t owner) const;

  mutable std::mutex mutex_;
  std::condition_variable available_;
  std::byte* base_{nullptr};
  std::size_t count_{0};
  std::size_t bytes_{0};
  std::size_t per_owner_cap_{0};
  std::vector<std::size_t> free_list_;
  std::vector<std::uint64_t> owner_of_;
  std::unordered_map<std::uint64_t, std::size_t> held_by_owner_;
  std::uint64_t blocked_claims_{0};
  bool closed_{false};
};

}  // namespace imgjit
