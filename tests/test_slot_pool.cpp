// The slot pool and the bounded queue (docs/PLAN.md Phase 4).
//
// These two types are where backpressure actually lives (CLAUDE.md invariant 5): a full
// pool blocks the reader, a full queue blocks it too, and neither ever drops a frame or
// grows at runtime. Tested here in isolation, with threads but without sockets, so a
// failure names the mechanism rather than the whole server.

#include <atomic>
#include <chrono>
#include <cstddef>
#include <optional>
#include <thread>
#include <vector>

#include "catch_amalgamated.hpp"
#include "imgjit/util/bounded_queue.h"
#include "imgjit/util/slot_pool.h"

using imgjit::BoundedQueue;
using imgjit::SlotPool;

namespace {

constexpr std::size_t kSlotBytes = 64;

}  // namespace

TEST_CASE("the pool hands out distinct slots into storage it does not own", "[net]") {
  // Storage-not-owned is the whole design: Phase 5 replaces this vector with pinned
  // memory from IBackend::allocate_slots() and no line of pool logic changes
  // (docs/PLAN.md Phase 4).
  std::vector<std::byte> storage(4 * kSlotBytes);
  SlotPool pool(storage.data(), 4, kSlotBytes, 4);

  const std::optional<std::size_t> first = pool.claim(1);
  const std::optional<std::size_t> second = pool.claim(1);
  REQUIRE(first.has_value());
  REQUIRE(second.has_value());
  CHECK(*first != *second);
  CHECK(pool.slot(*first) == storage.data() + *first * kSlotBytes);
  CHECK(pool.slot(*second) == storage.data() + *second * kSlotBytes);
  CHECK(pool.available() == 2);

  pool.release(*first);
  pool.release(*second);
  CHECK(pool.available() == 4);
}

TEST_CASE("a per-connection cap stops one owner taking the pool", "[net]") {
  // docs/ARCHITECTURE.md decision 2: per-connection caps keep one loud client from
  // starving the others. The second owner must still be servable while the first is
  // parked at its cap.
  std::vector<std::byte> storage(4 * kSlotBytes);
  SlotPool pool(storage.data(), 4, kSlotBytes, 2);

  const std::optional<std::size_t> a = pool.claim(1);
  const std::optional<std::size_t> b = pool.claim(1);
  REQUIRE(a.has_value());
  REQUIRE(b.has_value());

  std::atomic<bool> third_claimed{false};
  std::thread greedy([&] {
    const std::optional<std::size_t> c = pool.claim(1);  // at its cap: must block
    third_claimed.store(true);
    if (c.has_value()) {
      pool.release(*c);
    }
  });

  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  CHECK_FALSE(third_claimed.load());

  // A different connection is not blocked by owner 1's cap, even though owner 1 is
  // parked ahead of it.
  const std::optional<std::size_t> other = pool.claim(2);
  REQUIRE(other.has_value());
  pool.release(*other);

  pool.release(*a);  // frees owner 1 to proceed
  greedy.join();
  CHECK(third_claimed.load());
  CHECK(pool.blocked_claims() >= 1);
  pool.release(*b);
}

TEST_CASE("a claim blocked on an empty pool resumes when a slot is released", "[net]") {
  std::vector<std::byte> storage(kSlotBytes);
  SlotPool pool(storage.data(), 1, kSlotBytes, 1);

  const std::optional<std::size_t> only = pool.claim(1);
  REQUIRE(only.has_value());

  std::atomic<bool> got_slot{false};
  std::thread waiter([&] {
    const std::optional<std::size_t> next = pool.claim(2);
    got_slot.store(next.has_value());
  });

  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  CHECK_FALSE(got_slot.load());
  pool.release(*only);
  waiter.join();
  CHECK(got_slot.load());
  CHECK(pool.blocked_claims() == 1);
}

TEST_CASE("closing the pool releases a parked claim", "[net]") {
  // Without this, shutdown deadlocks: a reader thread blocked on claim() would never
  // return, so its connection thread would never join.
  std::vector<std::byte> storage(kSlotBytes);
  SlotPool pool(storage.data(), 1, kSlotBytes, 1);
  const std::optional<std::size_t> only = pool.claim(1);
  REQUIRE(only.has_value());

  std::atomic<bool> finished{false};
  std::optional<std::size_t> result;
  std::thread waiter([&] {
    result = pool.claim(2);
    finished.store(true);
  });

  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  CHECK_FALSE(finished.load());
  pool.close();
  waiter.join();
  CHECK(finished.load());
  CHECK_FALSE(result.has_value());
  pool.release(*only);
}

TEST_CASE("releasing a slot nobody holds is a logic error", "[net]") {
  std::vector<std::byte> storage(2 * kSlotBytes);
  SlotPool pool(storage.data(), 2, kSlotBytes, 2);
  CHECK_THROWS(pool.release(0));
  CHECK_THROWS(pool.release(99));

  const std::optional<std::size_t> claimed = pool.claim(1);
  REQUIRE(claimed.has_value());
  pool.release(*claimed);
  CHECK_THROWS(pool.release(*claimed));  // a double release is premature reuse
}

TEST_CASE("the queue is FIFO and drains after close", "[net]") {
  BoundedQueue<int> queue(4);
  CHECK(queue.push(1));
  CHECK(queue.push(2));
  CHECK(queue.size() == 2);

  queue.close();
  CHECK_FALSE(queue.push(3));  // closed to new work...

  int value = 0;
  CHECK(queue.pop(value));
  CHECK(value == 1);
  CHECK(queue.pop(value));
  CHECK(value == 2);
  CHECK_FALSE(queue.pop(value));  // ...but only reports empty once it has drained
}

TEST_CASE("a full queue blocks the producer", "[net]") {
  // This is the second half of backpressure: when the worker falls behind, the reader
  // thread parks here instead of buffering frames without bound.
  BoundedQueue<int> queue(2);
  CHECK(queue.push(10));
  CHECK(queue.push(20));

  std::atomic<bool> pushed{false};
  std::thread producer([&] {
    queue.push(30);
    pushed.store(true);
  });

  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  CHECK_FALSE(pushed.load());

  int value = 0;
  REQUIRE(queue.pop(value));
  producer.join();
  CHECK(pushed.load());
  CHECK(queue.high_water() == 2);
}

TEST_CASE("try_pop never blocks", "[net]") {
  // The worker needs this from Phase 6: with frames in flight it must keep polling
  // completion events rather than parking in pop().
  BoundedQueue<int> queue(2);
  int value = 0;
  CHECK_FALSE(queue.try_pop(value));
  CHECK(queue.push(5));
  CHECK(queue.try_pop(value));
  CHECK(value == 5);
  CHECK_FALSE(queue.try_pop(value));
}

TEST_CASE("closing the queue wakes a blocked consumer", "[net]") {
  BoundedQueue<int> queue(2);
  std::atomic<bool> finished{false};
  bool popped = true;
  std::thread consumer([&] {
    int value = 0;
    popped = queue.pop(value);
    finished.store(true);
  });

  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  CHECK_FALSE(finished.load());
  queue.close();
  consumer.join();
  CHECK_FALSE(popped);
}
