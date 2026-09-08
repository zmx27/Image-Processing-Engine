#include "imgjit/net/socket.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>

namespace imgjit::net {
namespace {

std::runtime_error socket_error(const std::string& what) {
  return std::runtime_error(what + ": " + std::strerror(errno));
}

// A write to a socket the peer has closed raises SIGPIPE, whose default action is to
// kill the process — a client that hangs up mid-response would take the server with it.
// Linux suppresses it per-call with MSG_NOSIGNAL; macOS has no such flag and suppresses
// it per-socket with SO_NOSIGPIPE instead.
#ifdef MSG_NOSIGNAL
constexpr int kSendFlags = MSG_NOSIGNAL;
#else
constexpr int kSendFlags = 0;
#endif

void suppress_sigpipe(int descriptor) {
#ifdef SO_NOSIGPIPE
  const int on = 1;
  ::setsockopt(descriptor, SOL_SOCKET, SO_NOSIGPIPE, &on, sizeof(on));
#else
  (void)descriptor;
#endif
}

}  // namespace

Socket::~Socket() {
  close();
}

Socket::Socket(Socket&& other) noexcept : descriptor_(std::exchange(other.descriptor_, -1)) {}

Socket& Socket::operator=(Socket&& other) noexcept {
  if (this != &other) {
    close();
    descriptor_ = std::exchange(other.descriptor_, -1);
  }
  return *this;
}

void Socket::shutdown_both() {
  if (descriptor_ >= 0) {
    ::shutdown(descriptor_, SHUT_RDWR);
  }
}

void Socket::close() {
  if (descriptor_ >= 0) {
    ::close(descriptor_);
    descriptor_ = -1;
  }
}

Socket listen_on(std::uint16_t port, int backlog) {
  Socket socket(::socket(AF_INET, SOCK_STREAM, 0));
  if (!socket.valid()) {
    throw socket_error("socket()");
  }
  // Without SO_REUSEADDR a server restarted inside TIME_WAIT's two minutes fails to
  // bind — which in a test suite reads as a flaky failure rather than as the kernel
  // doing exactly what it documents.
  const int on = 1;
  ::setsockopt(socket.descriptor(), SOL_SOCKET, SO_REUSEADDR, &on, sizeof(on));
  suppress_sigpipe(socket.descriptor());

  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  address.sin_port = htons(port);
  if (::bind(socket.descriptor(), reinterpret_cast<const sockaddr*>(&address), sizeof(address)) !=
      0) {
    throw socket_error("bind(port " + std::to_string(port) + ")");
  }
  if (::listen(socket.descriptor(), backlog) != 0) {
    throw socket_error("listen()");
  }
  return socket;
}

std::uint16_t local_port(const Socket& socket) {
  sockaddr_in address{};
  socklen_t length = sizeof(address);
  if (::getsockname(socket.descriptor(), reinterpret_cast<sockaddr*>(&address), &length) != 0) {
    throw socket_error("getsockname()");
  }
  return ntohs(address.sin_port);
}

Socket connect_to(const std::string& host, std::uint16_t port) {
  Socket socket(::socket(AF_INET, SOCK_STREAM, 0));
  if (!socket.valid()) {
    throw socket_error("socket()");
  }
  suppress_sigpipe(socket.descriptor());

  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_port = htons(port);
  if (::inet_pton(AF_INET, host.c_str(), &address.sin_addr) != 1) {
    throw std::runtime_error("not a dotted-quad IPv4 address: " + host);
  }
  if (::connect(socket.descriptor(), reinterpret_cast<const sockaddr*>(&address),
                sizeof(address)) != 0) {
    throw socket_error("connect(" + host + ":" + std::to_string(port) + ")");
  }
  // Responses are small and latency is what the benchmarks measure, so Nagle's
  // coalescing delay is pure cost here.
  const int on = 1;
  ::setsockopt(socket.descriptor(), IPPROTO_TCP, TCP_NODELAY, &on, sizeof(on));
  return socket;
}

Socket accept_with_timeout(const Socket& listener, int timeout_ms) {
  pollfd descriptors{};
  descriptors.fd = listener.descriptor();
  descriptors.events = POLLIN;
  const int ready = ::poll(&descriptors, 1, timeout_ms);
  if (ready <= 0) {
    return Socket{};  // timeout, EINTR, or the listener went away
  }
  Socket accepted(::accept(listener.descriptor(), nullptr, nullptr));
  if (accepted.valid()) {
    suppress_sigpipe(accepted.descriptor());
    const int on = 1;
    ::setsockopt(accepted.descriptor(), IPPROTO_TCP, TCP_NODELAY, &on, sizeof(on));
  }
  return accepted;
}

IoResult read_exact(int descriptor, void* buffer, std::size_t bytes) {
  auto* cursor = static_cast<std::uint8_t*>(buffer);
  std::size_t remaining = bytes;
  while (remaining > 0) {
    const ssize_t received = ::recv(descriptor, cursor, remaining, 0);
    if (received == 0) {
      return IoResult::kEof;
    }
    if (received < 0) {
      if (errno == EINTR) {
        continue;
      }
      return IoResult::kError;
    }
    cursor += received;
    remaining -= static_cast<std::size_t>(received);
  }
  return IoResult::kOk;
}

IoResult write_exact(int descriptor, const void* buffer, std::size_t bytes) {
  const auto* cursor = static_cast<const std::uint8_t*>(buffer);
  std::size_t remaining = bytes;
  while (remaining > 0) {
    const ssize_t sent = ::send(descriptor, cursor, remaining, kSendFlags);
    if (sent < 0) {
      if (errno == EINTR) {
        continue;
      }
      return IoResult::kError;
    }
    cursor += sent;
    remaining -= static_cast<std::size_t>(sent);
  }
  return IoResult::kOk;
}

IoResult drain_exactly(int descriptor, std::uint64_t bytes) {
  std::array<std::byte, 4096> scratch{};
  while (bytes > 0) {
    const std::size_t chunk =
        static_cast<std::size_t>(std::min<std::uint64_t>(bytes, scratch.size()));
    const IoResult result = read_exact(descriptor, scratch.data(), chunk);
    if (result != IoResult::kOk) {
      return result;
    }
    bytes -= chunk;
  }
  return IoResult::kOk;
}

}  // namespace imgjit::net
