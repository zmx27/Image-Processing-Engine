#include "imgjit/util/slot_pool.h"

#include <stdexcept>

namespace imgjit {
namespace {

constexpr std::uint64_t kNoOwner = ~std::uint64_t{0};

}  // namespace

SlotPool::SlotPool(std::byte* base, std::size_t count, std::size_t bytes,
                   std::size_t per_owner_cap)
    : base_(base),
      count_(count),
      bytes_(bytes),
      per_owner_cap_(per_owner_cap == 0 ? count : per_owner_cap),
      owner_of_(count, kNoOwner) {
  if (base == nullptr || count == 0 || bytes == 0) {
    throw std::invalid_argument("SlotPool: needs a non-empty block of backing storage");
  }
  free_list_.reserve(count);
  // Handed out from the back, so slot 0 goes first and a single-connection run reuses
  // one slot — which makes a premature-reuse bug show up early rather than after the
  // pool has cycled.
  for (std::size_t i = count; i > 0; --i) {
    free_list_.push_back(i - 1);
  }
}

bool SlotPool::can_claim(std::uint64_t owner) const {
  if (free_list_.empty()) {
    return false;
  }
  const auto held = held_by_owner_.find(owner);
  return held == held_by_owner_.end() || held->second < per_owner_cap_;
}

std::optional<std::size_t> SlotPool::claim(std::uint64_t owner) {
  std::unique_lock<std::mutex> lock(mutex_);
  if (!closed_ && !can_claim(owner)) {
    ++blocked_claims_;
    available_.wait(lock, [this, owner] { return closed_ || can_claim(owner); });
  }
  if (closed_) {
    return std::nullopt;
  }
  const std::size_t index = free_list_.back();
  free_list_.pop_back();
  owner_of_[index] = owner;
  ++held_by_owner_[owner];
  return index;
}

void SlotPool::release(std::size_t index) {
  {
    const std::lock_guard<std::mutex> lock(mutex_);
    if (index >= count_ || owner_of_[index] == kNoOwner) {
      throw std::logic_error("SlotPool::release: slot is not currently claimed");
    }
    const auto held = held_by_owner_.find(owner_of_[index]);
    if (held != held_by_owner_.end() && --held->second == 0) {
      held_by_owner_.erase(held);
    }
    owner_of_[index] = kNoOwner;
    free_list_.push_back(index);
  }
  // notify_all, not notify_one: waiters are not interchangeable. A freed slot may be
  // unusable to the connection at the front of the queue (it is at its cap) and usable
  // to the next one, and waking only that first waiter would park the pool with a free
  // slot and a runnable claimant.
  available_.notify_all();
}

std::byte* SlotPool::slot(std::size_t index) {
  if (index >= count_) {
    throw std::out_of_range("SlotPool::slot: index past the end of the pool");
  }
  return base_ + index * bytes_;
}

void SlotPool::close() {
  {
    const std::lock_guard<std::mutex> lock(mutex_);
    closed_ = true;
  }
  available_.notify_all();
}

std::size_t SlotPool::available() const {
  const std::lock_guard<std::mutex> lock(mutex_);
  return free_list_.size();
}

std::uint64_t SlotPool::blocked_claims() const {
  const std::lock_guard<std::mutex> lock(mutex_);
  return blocked_claims_;
}

}  // namespace imgjit
