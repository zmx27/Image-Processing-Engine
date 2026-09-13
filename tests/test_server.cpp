// The server end to end (docs/PLAN.md Phase 4): real sockets, real threads, the CPU
// backend as the oracle.
//
// Track N's payoff is that everything here is proven with zero CUDA in the process, so
// a failure is a protocol or concurrency bug and never a GPU one. Four of these cases
// exist because of a specific way the design can be got wrong, and each says which:
// desync after an error frame, the pipelining deadlock, backpressure that drops instead
// of blocking, and responses matched by arrival order instead of seq_num.

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <memory>
#include <set>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "catch_amalgamated.hpp"
#include "imgjit/backend/cpu/cpu_backend.h"
#include "imgjit/backend/cpu/ops.h"
#include "imgjit/net/client.h"
#include "imgjit/net/protocol.h"
#include "imgjit/net/server.h"
#include "imgjit/util/image_io.h"

using imgjit::Image;
using imgjit::net::Client;
using imgjit::net::ClientResponse;
using imgjit::net::RequestHeader;
using imgjit::net::Server;
using imgjit::net::ServerConfig;
using imgjit::net::Status;

namespace {

Image fixture(const std::string& name) {
  return imgjit::load_png(std::string(IMGJIT_TESTDATA_DIR) + "/" + name);
}

Image oracle(const Image& input, std::string_view chain_text) {
  const auto chain = imgjit::parse_op_chain(chain_text);
  REQUIRE(chain.has_value());
  return imgjit::cpu::apply_chain(input, *chain);
}

Server::BackendFactory cpu_factory() {
  return [] { return std::make_unique<imgjit::CpuBackend>(); };
}

ServerConfig test_config() {
  ServerConfig config;
  config.port = 0;  // ephemeral: the suite runs many servers without colliding
  config.num_slots = 2;
  config.slots_per_connection = 2;
  config.max_payload_bytes = 64 * 1024;
  config.queue_capacity = 2;
  config.output_dir = std::string(IMGJIT_TEST_TMP_DIR) + "/phase4_out";
  return config;
}

// A backend that takes a fixed time per frame, so backpressure can be provoked
// deterministically rather than by hoping the worker happens to fall behind.
class ThrottledBackend final : public imgjit::IBackend {
 public:
  explicit ThrottledBackend(std::chrono::milliseconds delay) : delay_(delay) {}

  std::byte* allocate_slots(std::size_t count, std::size_t bytes) override {
    return inner_.allocate_slots(count, bytes);
  }

  imgjit::JobHandle submit(const imgjit::FrameJob& job) override {
    std::this_thread::sleep_for(delay_);
    return inner_.submit(job);
  }

  std::vector<imgjit::Completion> poll_completions() override {
    return inner_.poll_completions();
  }

 private:
  imgjit::CpuBackend inner_;
  std::chrono::milliseconds delay_;
};

// A well-formed request header for `image`, which the raw-send tests then spoil in
// exactly one way.
RequestHeader header_for(const Image& image, std::uint32_t seq_num, std::size_t chain_len) {
  RequestHeader header;
  header.flags = imgjit::net::kFlagEcho;
  header.seq_num = seq_num;
  header.width = static_cast<std::uint32_t>(image.width());
  header.height = static_cast<std::uint32_t>(image.height());
  header.channels = static_cast<std::uint8_t>(image.channels());
  header.payload_len = static_cast<std::uint32_t>(image.byte_count());
  header.chain_len = static_cast<std::uint16_t>(chain_len);
  return header;
}

std::vector<std::uint8_t> payload_of(const Image& image) {
  return std::vector<std::uint8_t>(image.data(), image.data() + image.byte_count());
}

}  // namespace

TEST_CASE("a frame round-trips and matches the CPU oracle", "[server]") {
  const Image input = fixture("checkerboard_16x16_rgb.png");
  Server server(test_config(), cpu_factory());
  server.start();

  Client client("127.0.0.1", server.port());
  const std::uint32_t seq = client.send(input, "grayscale,sobel");
  const ClientResponse response = client.receive();

  CHECK(response.matched);
  CHECK(response.seq_num == seq);
  CHECK(response.status == Status::kOk);
  CHECK(response.version == imgjit::net::kVersion);
  CHECK(response.image == oracle(input, "grayscale,sobel"));
  CHECK(client.pending_count() == 0);
  CHECK(server.frames_completed() == 1);
}

TEST_CASE("every chain in the corpus is served correctly", "[server]") {
  const Image input = fixture("gradient_32x32_rgba.png");
  Server server(test_config(), cpu_factory());
  server.start();
  Client client("127.0.0.1", server.port());

  const std::vector<std::string> chains{
      "", "invert", "grayscale", "brightness:0.25", "threshold:0.4", "gaussian:1.4", "sobel",
      "grayscale,gaussian:1.4,sobel,threshold:0.3"};
  for (const std::string& chain : chains) {
    const std::uint32_t seq = client.send(input, chain);
    const ClientResponse response = client.receive();
    INFO("chain: \"" << chain << "\"");
    REQUIRE(response.matched);
    CHECK(response.seq_num == seq);
    REQUIRE(response.status == Status::kOk);
    CHECK(response.image == oracle(input, chain));
  }
}

TEST_CASE("flags bit 1 writes the result server-side", "[server]") {
  const Image input = fixture("solid_4x4_rgb.png");
  ServerConfig config = test_config();
  config.output_dir = std::string(IMGJIT_TEST_TMP_DIR) + "/phase4_write";
  std::filesystem::remove_all(config.output_dir);

  Server server(config, cpu_factory());
  server.start();
  Client client("127.0.0.1", server.port());

  // Write only, no echo: the response must still report the dimensions but carry no
  // payload (docs/PROTOCOL.md response table).
  const std::uint32_t seq = client.send(input, "invert", imgjit::net::kFlagServerWrite);
  const ClientResponse response = client.receive();
  REQUIRE(response.matched);
  CHECK(response.seq_num == seq);
  CHECK(response.status == Status::kOk);
  CHECK(response.image.empty());

  // The name is entirely server-assigned — the protocol carries no filename, because
  // accepting a client-supplied path would be a directory-traversal hole.
  const std::filesystem::path written = std::filesystem::path(config.output_dir) / "frame_1_1.png";
  REQUIRE(std::filesystem::exists(written));
  CHECK(imgjit::load_png(written.string()) == oracle(input, "invert"));
}

TEST_CASE("both result paths can be requested at once", "[server]") {
  const Image input = fixture("solid_4x4_rgb.png");
  ServerConfig config = test_config();
  config.output_dir = std::string(IMGJIT_TEST_TMP_DIR) + "/phase4_both";
  std::filesystem::remove_all(config.output_dir);

  Server server(config, cpu_factory());
  server.start();
  Client client("127.0.0.1", server.port());

  const std::uint8_t both = imgjit::net::kFlagEcho | imgjit::net::kFlagServerWrite;
  client.send(input, "grayscale", both);
  const ClientResponse response = client.receive();
  REQUIRE(response.status == Status::kOk);
  CHECK(response.image == oracle(input, "grayscale"));
  CHECK(std::filesystem::exists(std::filesystem::path(config.output_dir) / "frame_1_1.png"));
}

TEST_CASE("an unparseable chain is answered and the connection stays in sync", "[server]") {
  // THE DESYNC CASE. The error is detected from the header while the client is already
  // streaming the payload; if those bytes are not drained, the next header read starts
  // mid-payload and every later frame on this connection is garbage.
  const Image input = fixture("checkerboard_16x16_rgb.png");
  Server server(test_config(), cpu_factory());
  server.start();
  Client client("127.0.0.1", server.port());

  const std::string bad_chain = "definitely-not-an-op";
  client.send_raw(header_for(input, 1, bad_chain.size()), bad_chain, payload_of(input));
  const ClientResponse rejected = client.receive();
  CHECK(rejected.matched);
  CHECK(rejected.seq_num == 1);
  CHECK(rejected.status == Status::kBadChain);
  CHECK(rejected.image.empty());

  // The chain here was READ before it failed to parse, so only the payload was
  // outstanding. The next case covers the other accounting path.
  const std::uint32_t good = client.send(input, "invert");
  const ClientResponse served = client.receive();
  REQUIRE(served.matched);
  CHECK(served.seq_num == good);
  REQUIRE(served.status == Status::kOk);
  CHECK(served.image == oracle(input, "invert"));
}

TEST_CASE("an over-long chain is drained along with its payload", "[server]") {
  // The other drain accounting: chain_len is rejected from the header, so the chain
  // bytes were never read and BOTH the chain and the payload are still outstanding.
  // Draining only the payload here would leave 300 bytes of 'x' where the next header
  // should be.
  const Image input = fixture("checkerboard_16x16_rgb.png");
  Server server(test_config(), cpu_factory());
  server.start();
  Client client("127.0.0.1", server.port());

  const std::string long_chain(imgjit::net::kMaxChainLen + 44, 'x');
  client.send_raw(header_for(input, 1, long_chain.size()), long_chain, payload_of(input));
  const ClientResponse rejected = client.receive();
  CHECK(rejected.status == Status::kBadChain);

  const std::uint32_t good = client.send(input, "grayscale");
  const ClientResponse served = client.receive();
  REQUIRE(served.matched);
  CHECK(served.seq_num == good);
  REQUIRE(served.status == Status::kOk);
  CHECK(served.image == oracle(input, "grayscale"));
}

TEST_CASE("a length that disagrees with the dimensions is drained by the declared length",
          "[server]") {
  const Image input = fixture("checkerboard_16x16_rgb.png");
  Server server(test_config(), cpu_factory());
  server.start();
  Client client("127.0.0.1", server.port());

  // The declared payload_len is honest about how many bytes follow; the DIMENSIONS are
  // the lie. Draining by payload_len (bounded, already validated) rather than by the
  // dimension product is what keeps the connection recoverable.
  RequestHeader header = header_for(input, 1, 6);
  header.width = input.width() + 1;
  client.send_raw(header, "invert", payload_of(input));
  const ClientResponse rejected = client.receive();
  CHECK(rejected.status == Status::kLengthMismatch);

  const std::uint32_t good = client.send(input, "invert");
  const ClientResponse served = client.receive();
  REQUIRE(served.matched);
  CHECK(served.seq_num == good);
  REQUIRE(served.status == Status::kOk);
  CHECK(served.image == oracle(input, "invert"));
}

TEST_CASE("a bad magic is answered and then closes the connection", "[server]") {
  // Framing is unrecoverable: there is no trustworthy length to resynchronize on, so
  // the disposition is close (docs/PROTOCOL.md).
  const Image input = fixture("solid_4x4_rgb.png");
  Server server(test_config(), cpu_factory());
  server.start();
  Client client("127.0.0.1", server.port());

  RequestHeader header = header_for(input, 1, 6);
  header.magic = 0x0BADF00D;
  client.send_raw(header, "invert", payload_of(input));

  const ClientResponse rejected = client.receive();
  CHECK(rejected.status == Status::kBadMagic);
  CHECK_THROWS(client.receive());  // the server hung up
  CHECK(server.frames_completed() == 0);
}

TEST_CASE("an unsupported version is answered with the server's own version", "[server]") {
  // Status 2 is how a client learns what is actually supported, so the version field of
  // the RESPONSE is the server's, not an echo of the request's.
  const Image input = fixture("solid_4x4_rgb.png");
  Server server(test_config(), cpu_factory());
  server.start();
  Client client("127.0.0.1", server.port());

  RequestHeader header = header_for(input, 1, 6);
  header.version = 99;
  client.send_raw(header, "invert", payload_of(input));

  const ClientResponse rejected = client.receive();
  CHECK(rejected.status == Status::kUnsupportedVersion);
  CHECK(rejected.version == imgjit::net::kVersion);
  CHECK_THROWS(client.receive());
}

TEST_CASE("an oversized payload is rejected before anything is allocated", "[server]") {
  // The cap is checked against the DECLARED length, before a byte of it is read — never
  // trust an attacker-controlled length for an allocation size. Draining a length we
  // have already called unreasonable would itself be the DoS, so this one closes.
  const Image input = fixture("solid_4x4_rgb.png");
  ServerConfig config = test_config();
  config.max_payload_bytes = 4096;
  Server server(config, cpu_factory());
  server.start();
  Client client("127.0.0.1", server.port());

  RequestHeader header = header_for(input, 1, 6);
  header.width = 4096;
  header.height = 4096;
  header.payload_len = 4096U * 4096U * 3U;
  client.send_raw(header, "invert", {});  // the 48 MiB it promises never arrives

  const ClientResponse rejected = client.receive();
  CHECK(rejected.status == Status::kPayloadTooLarge);
  CHECK_THROWS(client.receive());
}

TEST_CASE("a pipelining client does not deadlock", "[server]") {
  // THE DEADLOCK CASE (docs/ARCHITECTURE.md decision 6). The client sends every frame
  // before reading any response, and sends more of them than the pool holds. If a slot
  // were released only after its response had been written, the reader would block
  // claiming a slot for frame N+1, so frame 1's response would never be written, so its
  // slot would never free — and this test would hang rather than fail.
  // The worker is throttled ON PURPOSE, and the test is worthless without it. With the
  // bare CPU backend a 16x16 frame is processed in microseconds, so the worker keeps
  // freeing slots faster than the reader refills them and the reader may never park at
  // all — the test would pass while never once reaching the state it exists to check.
  // 5 ms a frame makes the reader's block on slot 3 a certainty rather than a race.
  const Image input = fixture("checkerboard_16x16_rgb.png");
  Server server(test_config(), [] {  // 2 slots, cap 2, queue 2
    return std::make_unique<ThrottledBackend>(std::chrono::milliseconds(5));
  });
  server.start();
  Client client("127.0.0.1", server.port());

  constexpr int kFrames = 8;
  std::set<std::uint32_t> sent;
  for (int i = 0; i < kFrames; ++i) {
    sent.insert(client.send(input, "grayscale,sobel"));
  }

  const Image expected = oracle(input, "grayscale,sobel");
  std::set<std::uint32_t> received;
  for (int i = 0; i < kFrames; ++i) {
    const ClientResponse response = client.receive();
    REQUIRE(response.matched);
    REQUIRE(response.status == Status::kOk);
    CHECK(response.image == expected);
    received.insert(response.seq_num);
  }
  // Sets, never arrival order: from Phase 6 completions genuinely retire out of order.
  CHECK(received == sent);
  CHECK(client.pending_count() == 0);
  CHECK(server.frames_completed() == kFrames);
  // Proof the test actually reached the dangerous state rather than passing because the
  // worker kept up: the reader parked on a slot claim while responses were still
  // unwritten, which is precisely the moment release-after-write would deadlock.
  CHECK(server.slot_waits() > 0);
}

TEST_CASE("backpressure blocks the reader rather than dropping frames", "[server]") {
  // One slot, one per connection, a worker that takes 20 ms a frame: the reader is
  // guaranteed to park on slot claim. What must NOT happen is a dropped frame — an
  // unconfigured build never drops (invariant 5).
  const Image input = fixture("solid_4x4_rgb.png");
  ServerConfig config = test_config();
  config.num_slots = 1;
  config.slots_per_connection = 1;
  config.queue_capacity = 1;

  Server server(config, [] {
    return std::make_unique<ThrottledBackend>(std::chrono::milliseconds(20));
  });
  server.start();
  Client client("127.0.0.1", server.port());

  constexpr int kFrames = 6;
  std::set<std::uint32_t> sent;
  for (int i = 0; i < kFrames; ++i) {
    sent.insert(client.send(input, "invert"));
  }

  std::set<std::uint32_t> received;
  for (int i = 0; i < kFrames; ++i) {
    const ClientResponse response = client.receive();
    REQUIRE(response.matched);
    REQUIRE(response.status == Status::kOk);
    received.insert(response.seq_num);
  }

  CHECK(received == sent);
  CHECK(server.frames_completed() == kFrames);
  // Asserted on the counter rather than on a timing, which would be flaky: without
  // this, "the reader blocked" would be a claim about a path that may never have run.
  CHECK(server.slot_waits() > 0);
  CHECK(server.max_queue_depth() <= config.queue_capacity);
}

TEST_CASE("N clients x M frames all match the oracle", "[server]") {
  const Image input = fixture("checkerboard_16x16_rgb.png");
  ServerConfig config = test_config();
  config.num_slots = 4;
  config.slots_per_connection = 2;
  config.queue_capacity = 4;

  Server server(config, cpu_factory());
  server.start();
  const std::uint16_t port = server.port();

  constexpr int kClients = 4;
  constexpr int kFrames = 5;
  const std::vector<std::string> chains{"invert", "grayscale,sobel", "gaussian:1.4",
                                        "brightness:0.2,threshold:0.4"};
  std::vector<Image> expected;
  expected.reserve(chains.size());
  for (const std::string& chain : chains) {
    expected.push_back(oracle(input, chain));
  }

  // Catch2's assertion macros are not thread-safe, so the threads only record what went
  // wrong and main() does the asserting.
  std::vector<std::string> failures(kClients);
  std::vector<std::thread> clients;
  clients.reserve(kClients);
  for (int index = 0; index < kClients; ++index) {
    clients.emplace_back([&, index] {
      try {
        Client client("127.0.0.1", port);
        std::set<std::uint32_t> sent;
        for (int frame = 0; frame < kFrames; ++frame) {
          sent.insert(client.send(input, chains[index]));
        }
        std::set<std::uint32_t> received;
        for (int frame = 0; frame < kFrames; ++frame) {
          const ClientResponse response = client.receive();
          if (!response.matched) {
            failures[index] = "a response arrived for a seq_num this client never sent";
            return;
          }
          if (response.status != Status::kOk) {
            failures[index] = "status " + std::string(status_message(response.status));
            return;
          }
          if (!(response.image == expected[index])) {
            failures[index] = "pixels differ from the CPU oracle";
            return;
          }
          received.insert(response.seq_num);
        }
        if (received != sent) {
          failures[index] = "the set of responses does not equal the set of requests";
        }
      } catch (const std::exception& error) {
        failures[index] = error.what();
      }
    });
  }
  for (std::thread& client : clients) {
    client.join();
  }

  for (int index = 0; index < kClients; ++index) {
    INFO("client " << index << " (" << chains[index] << "): " << failures[index]);
    CHECK(failures[index].empty());
  }
  CHECK(server.frames_completed() == kClients * kFrames);
  CHECK(server.connections_accepted() == kClients);
}

TEST_CASE("a client that hangs up mid-stream does not disturb the others", "[server]") {
  const Image input = fixture("solid_4x4_rgb.png");
  Server server(test_config(), cpu_factory());
  server.start();

  {
    Client rude("127.0.0.1", server.port());
    rude.send(input, "invert");
    rude.close();  // walks away without reading the response
  }

  Client polite("127.0.0.1", server.port());
  const std::uint32_t seq = polite.send(input, "grayscale");
  const ClientResponse response = polite.receive();
  REQUIRE(response.matched);
  CHECK(response.seq_num == seq);
  REQUIRE(response.status == Status::kOk);
  CHECK(response.image == oracle(input, "grayscale"));
}

TEST_CASE("stop() is idempotent and joins every thread", "[server]") {
  const Image input = fixture("solid_4x4_rgb.png");
  Server server(test_config(), cpu_factory());
  server.start();
  {
    Client client("127.0.0.1", server.port());
    client.send(input, "invert");
    CHECK(client.receive().status == Status::kOk);
  }
  server.stop();
  server.stop();  // the destructor calls it again too
}
