#pragma once

// The BSD-socket layer, kept to the smallest surface the server and client need: an
// RAII descriptor, listen/accept/connect, and the two loops that make TCP's byte-stream
// nature survivable.
//
// `read_exact` is CLAUDE.md invariant 7: a bare recv() is a bug, not a shortcut. TCP
// makes no promise that a 25-byte header arrives in one segment, and the failure mode
// of assuming it does is a header that decodes from a payload's bytes — silent
// corruption, not an error. Every read in this project goes through here.
//
// Portable: no CUDA, builds on macOS with no toolkit present. POSIX sockets only, which
// covers macOS and Colab's Linux container.

#include <cstddef>
#include <cstdint>
#include <string>

namespace imgjit::net {

enum class IoResult {
  kOk,
  kEof,    // peer closed; a partial frame is still kEof, since the connection is dead
  kError,  // errno was set; the caller drops the connection either way
};

// Owns a file descriptor and closes it exactly once. Move-only: two owners of one fd is
// the double-close bug, and a double close is not merely noisy — the number can already
// have been handed to another thread's accept() by then.
class Socket {
 public:
  Socket() = default;
  explicit Socket(int descriptor) : descriptor_(descriptor) {}
  ~Socket();

  Socket(Socket&& other) noexcept;
  Socket& operator=(Socket&& other) noexcept;
  Socket(const Socket&) = delete;
  Socket& operator=(const Socket&) = delete;

  int descriptor() const { return descriptor_; }
  bool valid() const { return descriptor_ >= 0; }

  // Wakes a thread blocked in recv() on this socket without closing the descriptor.
  // That distinction is what makes shutdown safe to call from another thread: close()
  // would free the number for reuse while the reader is still naming it.
  void shutdown_both();

  void close();

 private:
  int descriptor_{-1};
};

// Throws std::runtime_error on failure. Port 0 asks the kernel for an ephemeral port —
// which is how the tests run many servers at once without colliding; read it back with
// local_port().
Socket listen_on(std::uint16_t port, int backlog);
std::uint16_t local_port(const Socket& socket);
Socket connect_to(const std::string& host, std::uint16_t port);

// Waits up to `timeout_ms` for a connection. Returns an invalid Socket on timeout or
// once the listener is closed — the acceptor loop polls rather than blocking forever in
// accept(), because closing a listening socket does not reliably wake accept() on both
// macOS and Linux.
Socket accept_with_timeout(const Socket& listener, int timeout_ms);

// Loop until all `bytes` are transferred, EINTR is retried, and a short transfer is
// resumed rather than treated as completion.
IoResult read_exact(int descriptor, void* buffer, std::size_t bytes);
IoResult write_exact(int descriptor, const void* buffer, std::size_t bytes);

// Reads and discards exactly `bytes` bytes through a small fixed scratch buffer — never
// an allocation sized by the client (docs/PROTOCOL.md, "Draining reads through
// read_exact()"). This is what resynchronizes a connection after a drainable error.
IoResult drain_exactly(int descriptor, std::uint64_t bytes);

}  // namespace imgjit::net
