#pragma once

// The frame server: an acceptor thread, a reader and a writer thread per connection,
// one bounded queue, and one worker thread that owns the backend.
//
// The thread layout is CLAUDE.md invariant 10, and both halves of it are load-bearing
// rather than stylistic (docs/ARCHITECTURE.md decision 6):
//
//   * THE WORKER NEVER TOUCHES A SOCKET. Completions go to a per-connection outbox that
//     a writer thread drains. If the worker wrote responses, one slow client would
//     head-of-line-block the entire pipeline — invisible until Phase 6, where it would
//     look like an inexplicable throughput cliff rather than a bug.
//   * A SLOT IS RELEASED WHEN ITS WORK COMPLETES, never after its response is written.
//     Release-after-write deadlocks any pipelining client: the reader blocks claiming a
//     slot for frame N+1, so the response for frame 1 is never written, so its slot is
//     never freed. The result is therefore copied out of the slot before release, and
//     that copy is the price of breaking the cycle.
//
// The backend is created BY the worker thread, not handed to it, which is why this
// takes a factory rather than a unique_ptr. Phase 5 needs the driver context created on
// the thread that will own it forever (CLAUDE.md invariant 1), and slot storage comes
// from allocate_slots() on that same thread (invariant 2). Making that the shape now
// means Phase 5 swaps the factory and changes nothing else in this file.
//
// No driver entry point is named anywhere in this file, deliberately: invariant 1 is
// audited by grep over src/ and include/, and a comment that trips it is exactly what
// teaches people to stop running the check.
//
// Portable: no CUDA, builds on macOS with no toolkit present.

#include <cstdint>
#include <functional>
#include <memory>
#include <string>

#include "imgjit/backend/backend.h"

namespace imgjit::net {

struct ServerConfig {
  // 0 asks the kernel for an ephemeral port; read it back with Server::port().
  std::uint16_t port{0};

  // Where flags-bit-1 results are written, as frame_<conn_id>_<seq_num>.png. The
  // protocol deliberately carries no filename: accepting a client-supplied path would
  // be a directory-traversal hole for zero demonstrative value (docs/PROTOCOL.md).
  std::string output_dir{"."};

  // The pool is allocated once, at worker startup, and never grows (invariant 2).
  // max_payload_bytes doubles as the slot size, so any frame that passes validation is
  // guaranteed to fit the slot it is read into.
  std::size_t num_slots{8};
  std::uint32_t max_payload_bytes{8U * 1024U * 1024U};

  // Per-connection cap, so one loud client cannot take the whole pool.
  std::size_t slots_per_connection{2};

  std::size_t queue_capacity{16};
  int backlog{16};
};

class Server {
 public:
  using BackendFactory = std::function<std::unique_ptr<IBackend>()>;

  Server(ServerConfig config, BackendFactory backend_factory);
  ~Server();

  Server(const Server&) = delete;
  Server& operator=(const Server&) = delete;

  // Starts the worker, waits for it to construct the backend and allocate the slot
  // pool, then starts accepting. Throws if the backend or the listening socket fails —
  // including any exception the factory threw, rethrown here on the caller's thread.
  void start();

  // Idempotent, and also run by the destructor. Ordering matters and is explained in
  // the implementation: the queue is closed only after every connection thread has
  // joined, because a connection cannot finish until its in-flight jobs retire.
  void stop();

  std::uint16_t port() const;

  std::uint64_t frames_completed() const;
  std::uint64_t frames_failed() const;
  std::uint64_t connections_accepted() const;

  // Backpressure observability: claims that had to wait for a free slot, and the
  // deepest the job queue ever got. The Phase 4 backpressure test asserts on these
  // rather than on a timing, which would be flaky.
  std::uint64_t slot_waits() const;
  std::size_t max_queue_depth() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace imgjit::net
