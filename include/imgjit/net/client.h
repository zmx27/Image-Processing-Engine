#pragma once

// The reference client: one connection, and a seq_num -> pending map.
//
// THE MAP IS THE POINT (docs/PLAN.md Phase 4). The obvious client sends a frame and
// reads its response — correct today, and silently wrong from Phase 6 on, where
// multi-stream completions retire out of submission order. That is the entire reason
// `seq_num` is in the protocol (docs/PROTOCOL.md). Building the FIFO assumption in now
// would mean rewriting the client and the integration test during the async phase, so
// responses are matched by sequence number here from the first line: `receive()` looks
// the arriving seq_num up in the map, and the test compares SETS.
//
// Portable: no CUDA, builds on macOS with no toolkit present.

#include <cstdint>
#include <map>
#include <string>
#include <string_view>
#include <vector>

#include "imgjit/core/image.h"
#include "imgjit/net/protocol.h"
#include "imgjit/net/socket.h"

namespace imgjit::net {

// What was sent, kept so a response can be checked against its own request rather than
// against whatever the caller happens to still have in scope.
struct PendingFrame {
  std::uint32_t seq_num{0};
  std::string chain;
  int width{0};
  int height{0};
  int channels{0};
  std::uint8_t flags{0};
};

struct ClientResponse {
  std::uint8_t version{0};
  Status status{Status::kOk};
  std::uint32_t seq_num{0};
  Image image;  // empty on error, and on a request that did not ask for an echo

  // False if the server answered a seq_num this client never sent — which would mean
  // the connection had desynchronized, so it is worth asserting on rather than
  // silently tolerating.
  bool matched{false};
  PendingFrame request;
};

class Client {
 public:
  Client(const std::string& host, std::uint16_t port);

  Client(const Client&) = delete;
  Client& operator=(const Client&) = delete;

  // Sends one frame and records it as pending. Returns the assigned seq_num. Does not
  // wait for a response — pipelining is the normal case, not a special mode.
  std::uint32_t send(const Image& image, std::string_view chain, std::uint8_t flags = kFlagEcho);

  // Sends exactly the bytes described, bypassing every client-side check: this is how
  // the malformed-input tests produce a bad magic, an over-long chain or a length that
  // disagrees with the dimensions. `header.chain_len` and `header.payload_len` are
  // written to the wire as given, while the chain and payload actually sent are the
  // ones passed here — which is what makes a lying header expressible.
  void send_raw(const RequestHeader& header, std::string_view chain,
                const std::vector<std::uint8_t>& payload);

  // Blocks for the next response and matches it to its request by seq_num. Throws
  // std::runtime_error if the connection closes or the frame is malformed.
  ClientResponse receive();

  std::size_t pending_count() const { return pending_.size(); }

  void close() { socket_.close(); }

 private:
  Socket socket_;
  std::uint32_t next_seq_num_{1};
  std::map<std::uint32_t, PendingFrame> pending_;
};

}  // namespace imgjit::net
