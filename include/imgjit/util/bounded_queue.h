#pragma once

// The bounded MPSC queue between the connection reader threads and the single worker
// (docs/ARCHITECTURE.md component map). Mutex + condvar; the lock-free version is a
// stretch item and would be a premature optimization against an untested design.
//
// BOUNDED is the operative word. The capacity is the second half of the backpressure
// story — the first being slot-pool exhaustion (CLAUDE.md invariant 5): when the worker
// falls behind, `push` blocks the reader thread, the reader stops recv()ing, and the
// client's send buffer fills through TCP flow control. An unbounded queue would convert
// that into unbounded memory growth and call it throughput.
//
// `close()` is what makes shutdown terminate rather than deadlock: it wakes every
// blocked producer and consumer, after which `push` fails and `pop` drains what is left
// before reporting the queue is finished.
//
// Portable: no CUDA, builds on macOS with no toolkit present.

#include <algorithm>
#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <queue>
#include <utility>

namespace imgjit {

template <typename T>
class BoundedQueue {
 public:
  explicit BoundedQueue(std::size_t capacity) : capacity_(capacity == 0 ? 1 : capacity) {}

  BoundedQueue(const BoundedQueue&) = delete;
  BoundedQueue& operator=(const BoundedQueue&) = delete;

  // Blocks while the queue is full. Returns false only if the queue was closed, in
  // which case `value` is left untouched and the caller still owns whatever it holds —
  // which matters, because for the server that is a claimed slot to release.
  bool push(T value) {
    std::unique_lock<std::mutex> lock(mutex_);
    not_full_.wait(lock, [this] { return closed_ || items_.size() < capacity_; });
    if (closed_) {
      return false;
    }
    items_.push(std::move(value));
    high_water_ = std::max(high_water_, items_.size());
    lock.unlock();
    not_empty_.notify_one();
    return true;
  }

  // Blocks until an item is available. Returns false once the queue is both closed and
  // drained — closing does not discard work already queued.
  bool pop(T& out) {
    std::unique_lock<std::mutex> lock(mutex_);
    not_empty_.wait(lock, [this] { return closed_ || !items_.empty(); });
    if (items_.empty()) {
      return false;
    }
    out = std::move(items_.front());
    items_.pop();
    lock.unlock();
    not_full_.notify_one();
    return true;
  }

  // Never blocks. The worker uses this when it has jobs in flight to poll: from Phase 6
  // it must keep retiring completions rather than parking in pop().
  bool try_pop(T& out) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (items_.empty()) {
      return false;
    }
    out = std::move(items_.front());
    items_.pop();
    lock.unlock();
    not_full_.notify_one();
    return true;
  }

  void close() {
    {
      const std::lock_guard<std::mutex> lock(mutex_);
      closed_ = true;
    }
    not_empty_.notify_all();
    not_full_.notify_all();
  }

  std::size_t size() const {
    const std::lock_guard<std::mutex> lock(mutex_);
    return items_.size();
  }

  // The deepest the queue ever got. Reported by the server so "backpressure engaged"
  // is an observation rather than an assertion about code that was not run.
  std::size_t high_water() const {
    const std::lock_guard<std::mutex> lock(mutex_);
    return high_water_;
  }

  std::size_t capacity() const { return capacity_; }

 private:
  mutable std::mutex mutex_;
  std::condition_variable not_full_;
  std::condition_variable not_empty_;
  std::queue<T> items_;
  std::size_t capacity_;
  std::size_t high_water_{0};
  bool closed_{false};
};

}  // namespace imgjit
