#include "imgjit/net/server.h"

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <exception>
#include <filesystem>
#include <future>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "imgjit/core/op_chain.h"
#include "imgjit/net/protocol.h"
#include "imgjit/net/socket.h"
#include "imgjit/util/bounded_queue.h"
#include "imgjit/util/image_io.h"
#include "imgjit/util/slot_pool.h"

namespace imgjit::net {
namespace {

// What the writer thread needs in order to answer one request. It carries the pixels by
// value: the slot the input arrived in is released the moment the job completes, so
// nothing downstream of the worker may point into it (CLAUDE.md invariant 10).
struct Response {
  std::uint32_t seq_num{0};
  Status status{Status::kOk};
  bool echo{false};
  bool write_file{false};
  Image image;
};

// Per-connection response queue, drained by that connection's writer thread.
//
// Unbounded on purpose, and it is the one place in the design that is: the worker must
// never block here, since blocking would hand one slow client the head-of-line stall
// that invariant 10 exists to prevent. In practice the depth is bounded anyway — a
// connection can have at most `slots_per_connection` frames in flight, and a client
// that never reads stops the writer, then the reader, then itself via TCP flow control.
class Outbox {
 public:
  void push(Response response) {
    {
      const std::lock_guard<std::mutex> lock(mutex_);
      if (closed_) {
        return;  // the connection is finished; the response has nowhere to go
      }
      items_.push_back(std::move(response));
    }
    not_empty_.notify_one();
  }

  // Blocks until an item arrives. Returns false once closed AND drained — closing must
  // not discard responses that are already queued, or an error frame reported just
  // before the client hung up would vanish.
  bool pop(Response& out) {
    std::unique_lock<std::mutex> lock(mutex_);
    not_empty_.wait(lock, [this] { return closed_ || !items_.empty(); });
    if (items_.empty()) {
      return false;
    }
    out = std::move(items_.front());
    items_.pop_front();
    return true;
  }

  void close() {
    {
      const std::lock_guard<std::mutex> lock(mutex_);
      closed_ = true;
    }
    not_empty_.notify_all();
  }

 private:
  std::mutex mutex_;
  std::condition_variable not_empty_;
  std::deque<Response> items_;
  bool closed_{false};
};

struct Connection {
  Connection(std::uint64_t connection_id, Socket connected)
      : id(connection_id), socket(std::move(connected)) {}

  void job_submitted() {
    const std::lock_guard<std::mutex> lock(mutex);
    ++outstanding;
  }

  void job_retired() {
    const std::lock_guard<std::mutex> lock(mutex);
    --outstanding;
    close_outbox_if_done();
  }

  void reader_finished() {
    const std::lock_guard<std::mutex> lock(mutex);
    reader_done = true;
    close_outbox_if_done();
  }

  std::uint64_t id{0};
  Socket socket;
  Outbox outbox;
  std::atomic<bool> finished{false};

 private:
  // The writer outlives the reader by exactly as long as it takes the worker to retire
  // the frames already submitted. Closing the outbox the moment the reader stops would
  // drop their responses on the floor.
  void close_outbox_if_done() {
    if (reader_done && outstanding == 0) {
      outbox.close();
    }
  }

  std::mutex mutex;
  std::size_t outstanding{0};
  bool reader_done{false};
};

// One frame's worth of work, from a reader thread to the worker. The payload is not
// here — it is already sitting in slot `slot`, recv()'d straight into it.
struct QueuedFrame {
  std::shared_ptr<Connection> connection;
  std::uint32_t seq_num{0};
  std::uint8_t flags{0};
  std::size_t slot{0};
  int width{0};
  int height{0};
  int channels{0};
  OpChain chain;
};

// The worker's in-flight table: everything needed to route a completion back to the
// connection that asked for it, keyed by the handle the backend assigned. Never by
// arrival order — from Phase 6 completions retire out of submission order.
struct InFlight {
  std::shared_ptr<Connection> connection;
  std::uint32_t seq_num{0};
  std::uint8_t flags{0};
  std::size_t slot{0};
};

struct ConnectionRecord {
  std::shared_ptr<Connection> connection;
  std::thread thread;
};

}  // namespace

struct Server::Impl {
  Impl(ServerConfig configuration, Server::BackendFactory factory)
      : config(std::move(configuration)),
        backend_factory(std::move(factory)),
        queue(config.queue_capacity) {
    limits.max_payload_bytes = config.max_payload_bytes;
  }

  void start();
  void stop();

  void worker_main(std::promise<void> ready);
  void worker_loop();
  void submit(QueuedFrame& frame, std::unordered_map<JobHandle, InFlight>& in_flight);
  void retire(Completion& completion, std::unordered_map<JobHandle, InFlight>& in_flight);

  void acceptor_loop();
  void connection_main(const std::shared_ptr<Connection>& connection);
  void reader_loop(const std::shared_ptr<Connection>& connection);
  void writer_loop(const std::shared_ptr<Connection>& connection);
  void reap_finished_locked();

  ServerConfig config;
  Server::BackendFactory backend_factory;
  Limits limits;

  Socket listener;
  std::thread acceptor;
  std::thread worker;
  std::atomic<bool> running{false};

  // Touched only by the worker thread, from its first line to its last. That is not a
  // convention here, it is CLAUDE.md invariant 1 for the CUDA build.
  std::unique_ptr<IBackend> backend;

  // Published by the worker before it signals ready, read by every reader thread
  // afterwards; the promise/future handoff in start() is the synchronization.
  std::optional<SlotPool> slots;
  BoundedQueue<QueuedFrame> queue;

  std::mutex connections_mutex;
  std::vector<ConnectionRecord> connections;
  std::uint64_t next_connection_id{1};

  std::atomic<std::uint64_t> frames_completed{0};
  std::atomic<std::uint64_t> frames_failed{0};
  std::atomic<std::uint64_t> connections_accepted{0};
};

void Server::Impl::start() {
  if (running.load()) {
    throw std::logic_error("Server::start: already started");
  }
  if (config.num_slots == 0 || config.max_payload_bytes == 0) {
    throw std::invalid_argument("Server::start: needs at least one non-empty slot");
  }
  if (!config.output_dir.empty()) {
    std::filesystem::create_directories(config.output_dir);
  }

  listener = listen_on(config.port, config.backlog);

  // The backend is constructed ON the worker thread, and start() does not return until
  // that succeeded — so a factory that throws (Phase 5: no GPU, no driver) surfaces as
  // an exception from start() on the caller's thread rather than as a server that
  // accepts connections and then cannot serve them.
  std::promise<void> ready;
  std::future<void> ready_future = ready.get_future();
  worker = std::thread([this, ready = std::move(ready)]() mutable {
    worker_main(std::move(ready));
  });

  try {
    ready_future.get();
  } catch (...) {
    queue.close();
    if (worker.joinable()) {
      worker.join();
    }
    listener.close();
    throw;
  }

  running.store(true);
  acceptor = std::thread([this] { acceptor_loop(); });
}

void Server::Impl::stop() {
  if (!running.exchange(false)) {
    return;
  }

  // 1. The acceptor notices `running` within one poll timeout and returns. Joining it
  //    before closing the listener avoids closing a descriptor another thread is
  //    polling on.
  if (acceptor.joinable()) {
    acceptor.join();
  }
  listener.close();

  // 2. Unblock every reader: shutdown() wakes one parked in recv(), closing the pool
  //    wakes one parked claiming a slot. Neither destroys anything the reader is still
  //    naming.
  {
    const std::lock_guard<std::mutex> lock(connections_mutex);
    for (ConnectionRecord& record : connections) {
      record.connection->socket.shutdown_both();
    }
  }
  if (slots.has_value()) {
    slots->close();
  }

  // 3. Connection threads join BEFORE the queue closes, and the order is not
  //    interchangeable: a connection thread ends by joining its writer, the writer ends
  //    when the outbox closes, and the outbox closes when the last in-flight job
  //    retires — which only the worker can do. Close the queue first and those jobs are
  //    never retired, so the join never returns.
  {
    const std::lock_guard<std::mutex> lock(connections_mutex);
    for (ConnectionRecord& record : connections) {
      if (record.thread.joinable()) {
        record.thread.join();
      }
    }
    connections.clear();
  }

  queue.close();
  if (worker.joinable()) {
    worker.join();
  }
}

void Server::Impl::worker_main(std::promise<void> ready) {
  try {
    backend = backend_factory();
    if (backend == nullptr) {
      throw std::runtime_error("Server: the backend factory returned nothing");
    }
    // Allocated once, on this thread, and never again (CLAUDE.md invariant 2). The pool
    // borrows this storage; the backend owns it for its whole lifetime.
    std::byte* base = backend->allocate_slots(config.num_slots, config.max_payload_bytes);
    slots.emplace(base, config.num_slots, config.max_payload_bytes, config.slots_per_connection);
    ready.set_value();
  } catch (...) {
    ready.set_exception(std::current_exception());
    return;
  }

  worker_loop();

  // Destroyed here rather than in ~Impl: from Phase 5 this is where the driver context
  // is torn down, and it must be torn down by the thread that created it.
  backend.reset();
}

void Server::Impl::worker_loop() {
  std::unordered_map<JobHandle, InFlight> in_flight;

  // A state machine, not a pop/process/respond loop. With the CPU backend `in_flight`
  // is empty by the end of every iteration, so this always parks in the blocking pop —
  // but the shape is what Phase 6 needs, where the worker must keep polling events for
  // K frames already on the GPU instead of blocking for a K+1th (docs/PLAN.md Phase 2,
  // docs/ARCHITECTURE.md decision 4).
  for (;;) {
    QueuedFrame frame;
    const bool idle = in_flight.empty();
    const bool got_work = idle ? queue.pop(frame) : queue.try_pop(frame);
    if (got_work) {
      submit(frame, in_flight);
    } else if (idle) {
      break;  // queue closed and drained, nothing outstanding
    }

    std::vector<Completion> completions = backend->poll_completions();
    for (Completion& completion : completions) {
      retire(completion, in_flight);
    }

    if (!got_work && !in_flight.empty() && completions.empty()) {
      // Unreachable with a backend that completes inside submit(). Phase 6 replaces it
      // with a real event poll rather than leaving a spin here.
      std::this_thread::yield();
    }
  }
}

void Server::Impl::submit(QueuedFrame& frame, std::unordered_map<JobHandle, InFlight>& in_flight) {
  InFlight entry;
  entry.connection = frame.connection;
  entry.seq_num = frame.seq_num;
  entry.flags = frame.flags;
  entry.slot = frame.slot;

  FrameJob job;
  job.input = slots->slot(frame.slot);
  job.width = frame.width;
  job.height = frame.height;
  job.channels = frame.channels;
  job.stride = static_cast<std::size_t>(frame.width) * static_cast<std::size_t>(frame.channels);
  job.chain = std::move(frame.chain);

  try {
    const JobHandle handle = backend->submit(job);
    in_flight.emplace(handle, std::move(entry));
  } catch (const std::exception&) {
    // IBackend's contract is that a bad job comes back as a failed completion, not as a
    // throw. Honouring it here anyway costs three lines and means a backend that breaks
    // the contract degrades to one error response (status 6) instead of killing the
    // worker thread and hanging every connection.
    slots->release(frame.slot);
    Response response;
    response.seq_num = frame.seq_num;
    response.status = Status::kInternalError;
    response.echo = (frame.flags & kFlagEcho) != 0;
    frame.connection->outbox.push(std::move(response));
    frame.connection->job_retired();
    ++frames_failed;
  }
}

void Server::Impl::retire(Completion& completion,
                          std::unordered_map<JobHandle, InFlight>& in_flight) {
  const auto entry = in_flight.find(completion.handle);
  if (entry == in_flight.end()) {
    return;  // a handle we never submitted; nothing to route
  }
  InFlight job = std::move(entry->second);
  in_flight.erase(entry);

  Response response;
  response.seq_num = job.seq_num;
  response.echo = (job.flags & kFlagEcho) != 0;
  if (completion.status == JobStatus::kOk) {
    response.status = Status::kOk;
    response.write_file = (job.flags & kFlagServerWrite) != 0;
    response.image = std::move(completion.output);
    ++frames_completed;
  } else {
    response.status = Status::kInternalError;
    ++frames_failed;
  }

  // Released HERE — on completion, before the response goes anywhere near a socket.
  // Completion::output is backend-owned storage rather than a view into the slot, so
  // the copy-out invariant 10 requires has already happened. Releasing after the write
  // instead would deadlock any pipelining client (docs/ARCHITECTURE.md decision 6).
  slots->release(job.slot);

  job.connection->outbox.push(std::move(response));
  job.connection->job_retired();
}

void Server::Impl::acceptor_loop() {
  while (running.load()) {
    // Polled with a timeout rather than blocked in accept(): closing a listening socket
    // does not reliably wake a blocked accept() on both macOS and Linux, and a shutdown
    // path that depends on which one it is would be a shutdown path that hangs on the
    // other.
    Socket accepted = accept_with_timeout(listener, 50);
    if (!accepted.valid()) {
      const std::lock_guard<std::mutex> lock(connections_mutex);
      reap_finished_locked();
      continue;
    }

    auto connection = std::make_shared<Connection>(next_connection_id++, std::move(accepted));
    ++connections_accepted;

    const std::lock_guard<std::mutex> lock(connections_mutex);
    reap_finished_locked();
    connections.push_back(ConnectionRecord{
        connection, std::thread([this, connection] { connection_main(connection); })});
  }
}

void Server::Impl::reap_finished_locked() {
  for (auto record = connections.begin(); record != connections.end();) {
    if (record->connection->finished.load()) {
      if (record->thread.joinable()) {
        record->thread.join();
      }
      record = connections.erase(record);
    } else {
      ++record;
    }
  }
}

void Server::Impl::connection_main(const std::shared_ptr<Connection>& connection) {
  std::thread writer([this, connection] { writer_loop(connection); });
  reader_loop(connection);
  connection->reader_finished();
  writer.join();

  // shutdown(), not close(): the peer needs the FIN now, but the descriptor number must
  // stay reserved until the Connection is destroyed. stop() may be calling
  // shutdown_both() on this same socket concurrently, and shutdown is idempotent and
  // mutates nothing — close() would write the descriptor out from under it.
  connection->socket.shutdown_both();
  connection->finished.store(true);
}

void Server::Impl::reader_loop(const std::shared_ptr<Connection>& connection) {
  const int descriptor = connection->socket.descriptor();
  std::vector<char> chain_buffer;

  for (;;) {
    std::array<std::byte, kRequestHeaderBytes> header_bytes{};
    if (read_exact(descriptor, header_bytes.data(), header_bytes.size()) != IoResult::kOk) {
      return;  // clean close, or a connection that died mid-header
    }
    // Fails only on a short buffer, which read_exact has just ruled out.
    const RequestHeader header = *decode_request_header(header_bytes.data(), header_bytes.size());

    // What the client is still streaming behind this header. Kept up to date as bytes
    // are consumed, because an error detected mid-frame is only recoverable if the
    // exact remainder is drained (docs/PROTOCOL.md, validation order).
    std::uint64_t unread =
        static_cast<std::uint64_t>(header.chain_len) + static_cast<std::uint64_t>(header.payload_len);

    Status status = validate_request_header(header, limits);
    OpChain chain;
    if (status == Status::kOk) {
      chain_buffer.resize(header.chain_len);
      if (header.chain_len > 0 &&
          read_exact(descriptor, chain_buffer.data(), header.chain_len) != IoResult::kOk) {
        return;
      }
      unread -= header.chain_len;
      const std::optional<OpChain> parsed =
          parse_op_chain(std::string_view(chain_buffer.data(), header.chain_len));
      if (parsed.has_value()) {
        chain = *parsed;
      } else {
        status = Status::kBadChain;
      }
    }

    if (status != Status::kOk) {
      Response response;
      response.seq_num = header.seq_num;
      response.status = status;
      connection->outbox.push(std::move(response));

      // A protocol error produces an error response; it never aborts the server
      // (CLAUDE.md conventions). Whether it ends the CONNECTION depends on whether the
      // rest of the frame can be safely skipped.
      if (disposition_for(status) == Disposition::kClose) {
        return;
      }
      if (drain_exactly(descriptor, unread) != IoResult::kOk) {
        return;
      }
      continue;
    }

    // Blocks when the pool is empty or this connection is at its cap. That block is the
    // backpressure: it stops this thread recv()ing, which fills the client's send
    // buffer through TCP flow control (CLAUDE.md invariant 5). Nothing is ever dropped.
    const std::optional<std::size_t> slot = slots->claim(connection->id);
    if (!slot.has_value()) {
      return;  // the pool was closed: the server is shutting down
    }

    // Straight into the slot — no staging copy on ingest. From Phase 5 this slot is
    // pinned host memory and this recv() is the whole reason the pool exists.
    if (read_exact(descriptor, slots->slot(*slot), header.payload_len) != IoResult::kOk) {
      slots->release(*slot);
      return;
    }

    QueuedFrame frame;
    frame.connection = connection;
    frame.seq_num = header.seq_num;
    frame.flags = header.flags;
    frame.slot = *slot;
    frame.width = static_cast<int>(header.width);
    frame.height = static_cast<int>(header.height);
    frame.channels = static_cast<int>(header.channels);
    frame.chain = std::move(chain);

    connection->job_submitted();
    if (!queue.push(std::move(frame))) {
      connection->job_retired();
      slots->release(*slot);
      return;  // queue closed: shutting down
    }
  }
}

void Server::Impl::writer_loop(const std::shared_ptr<Connection>& connection) {
  const int descriptor = connection->socket.descriptor();
  bool socket_alive = true;

  Response response;
  while (connection->outbox.pop(response)) {
    if (response.write_file && response.status == Status::kOk) {
      // File output, never wire format: the protocol carries no filename, so the name
      // is entirely server-assigned (docs/PROTOCOL.md, "Server-side write").
      const std::string path = config.output_dir + "/frame_" +
                               std::to_string(connection->id) + "_" +
                               std::to_string(response.seq_num) + ".png";
      try {
        save_png(path, response.image);
      } catch (const std::exception&) {
        response.status = Status::kInternalError;
      }
    }

    if (!socket_alive) {
      continue;  // keep draining: the reader may still be queueing responses
    }

    const bool ok = response.status == Status::kOk;
    ResponseHeader header;
    header.version = kVersion;
    header.status = response.status;
    header.seq_num = response.seq_num;
    if (ok) {
      header.width = static_cast<std::uint32_t>(response.image.width());
      header.height = static_cast<std::uint32_t>(response.image.height());
      header.channels = static_cast<std::uint8_t>(response.image.channels());
      if (response.echo) {
        header.payload_len = static_cast<std::uint32_t>(response.image.byte_count());
      }
    }

    const std::array<std::byte, kResponseHeaderBytes> encoded = encode_response_header(header);
    IoResult result = write_exact(descriptor, encoded.data(), encoded.size());
    if (result == IoResult::kOk && header.payload_len > 0) {
      result = write_exact(descriptor, response.image.data(), header.payload_len);
    }
    if (result != IoResult::kOk) {
      // The client went away mid-response. Shut the socket down so a reader thread
      // parked in recv() on it wakes up instead of waiting for a peer that is gone.
      socket_alive = false;
      connection->socket.shutdown_both();
    }
  }
}

Server::Server(ServerConfig config, BackendFactory backend_factory)
    : impl_(std::make_unique<Impl>(std::move(config), std::move(backend_factory))) {}

Server::~Server() {
  impl_->stop();
}

void Server::start() {
  impl_->start();
}

void Server::stop() {
  impl_->stop();
}

std::uint16_t Server::port() const {
  return local_port(impl_->listener);
}

std::uint64_t Server::frames_completed() const {
  return impl_->frames_completed.load();
}

std::uint64_t Server::frames_failed() const {
  return impl_->frames_failed.load();
}

std::uint64_t Server::connections_accepted() const {
  return impl_->connections_accepted.load();
}

std::uint64_t Server::slot_waits() const {
  return impl_->slots.has_value() ? impl_->slots->blocked_claims() : 0;
}

std::size_t Server::max_queue_depth() const {
  return impl_->queue.high_water();
}

}  // namespace imgjit::net
