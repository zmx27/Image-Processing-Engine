#pragma once

// docs/PROTOCOL.md as pure functions: pack, unpack, validate. No sockets, no
// allocation, no I/O — which is what makes every malformed-input case a unit test
// rather than something you can only provoke over a socket.
//
// Two properties this file exists to enforce (CLAUDE.md invariant 6):
//
//   * Fields are packed ONE AT A TIME, little-endian. Never memcpy a struct onto a
//     socket: `RequestHeader` below is 25 bytes on the wire and larger in memory,
//     because the compiler pads it. The struct is a decoded view, not the frame.
//   * A declared length is validated and capped BEFORE anything is sized by it.
//     `validate_request_header` runs the checks in the order docs/PROTOCOL.md fixes
//     them — magic, version, payload_len, chain_len, dimensions — so the server never
//     reads or allocates more than it has already justified.
//
// Portable: no CUDA, builds on macOS with no toolkit present.

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string_view>

namespace imgjit::net {

inline constexpr std::uint32_t kMagic = 0xDEADBEEFU;
inline constexpr std::uint8_t kVersion = 1;

// 25 and 23 are spelled out in docs/PROTOCOL.md's tables. They are NOT sizeof() of
// anything — see the padding note above.
inline constexpr std::size_t kRequestHeaderBytes = 25;
inline constexpr std::size_t kResponseHeaderBytes = 23;

// Defaults for the caps in docs/PROTOCOL.md → Constants. A server may configure a
// smaller payload cap (imgjit-server sets it to its slot size, so a frame that passes
// validation always fits the slot it will be read into); it may never configure a
// larger one than it can hold.
inline constexpr std::uint32_t kMaxPayloadBytes = 64U * 1024U * 1024U;
inline constexpr std::uint16_t kMaxChainLen = 256;

inline constexpr std::uint8_t kFlagEcho = 1U << 0U;
inline constexpr std::uint8_t kFlagServerWrite = 1U << 1U;

enum class Status : std::uint8_t {
  kOk = 0,
  kBadMagic = 1,
  kUnsupportedVersion = 2,
  kPayloadTooLarge = 3,
  kBadChain = 4,
  kLengthMismatch = 5,
  kInternalError = 6,
};

// What the connection does after an error response, per docs/PROTOCOL.md's disposition
// table. This is the resynchronization rule, and it is the whole reason an error is not
// simply "reply and carry on": the client is already streaming the payload when the
// header is rejected, so those bytes must be either drained or the connection dropped.
// Get it wrong and the next header read starts mid-payload — every later frame on that
// connection is garbage.
enum class Disposition {
  kContinue,  // nothing outstanding to consume (the payload was already read)
  kDrain,     // discard chain + payload, then keep serving this connection
  kClose,     // no trustworthy length to resynchronize on, or draining is itself the DoS
};

Disposition disposition_for(Status status);

// Human-readable reason, for the server log and the client tool.
std::string_view status_message(Status status);

// A decoded request header. Field order matches the wire, but the layout does not.
struct RequestHeader {
  std::uint32_t magic{kMagic};
  std::uint8_t version{kVersion};
  std::uint8_t flags{0};
  std::uint32_t seq_num{0};
  std::uint32_t width{0};
  std::uint32_t height{0};
  std::uint8_t channels{0};
  std::uint32_t payload_len{0};
  std::uint16_t chain_len{0};
};

struct ResponseHeader {
  std::uint32_t magic{kMagic};
  std::uint8_t version{kVersion};
  Status status{Status::kOk};
  std::uint32_t seq_num{0};
  std::uint32_t width{0};
  std::uint32_t height{0};
  std::uint8_t channels{0};
  std::uint32_t payload_len{0};
};

struct Limits {
  std::uint32_t max_payload_bytes{kMaxPayloadBytes};
  std::uint16_t max_chain_len{kMaxChainLen};
};

std::array<std::byte, kRequestHeaderBytes> encode_request_header(const RequestHeader& header);
std::array<std::byte, kResponseHeaderBytes> encode_response_header(const ResponseHeader& header);

// Unpack only — no judgement. A wrong magic decodes fine and is rejected by
// `validate_request_header`, because docs/PROTOCOL.md validates fields in read order
// and the decode step has no business short-circuiting that. Returns nullopt only when
// `size` is not exactly the header length, i.e. a truncated read.
std::optional<RequestHeader> decode_request_header(const std::byte* bytes, std::size_t size);
std::optional<ResponseHeader> decode_response_header(const std::byte* bytes, std::size_t size);

// Everything checkable from the header alone, in docs/PROTOCOL.md's order. The chain
// PARSE is not here — it needs the chain bytes, which the caller only reads once
// `chain_len` has been bounded by this function. A parse failure is Status::kBadChain,
// same as an over-long chain, and shares its drain disposition.
//
// The dimension product is computed in uint64_t deliberately: 65535 * 65535 * 4
// overflows uint32, and a wrapped product comparing equal to a small payload_len would
// pass validation and then be used as an indexing bound (docs/PROTOCOL.md).
Status validate_request_header(const RequestHeader& header, const Limits& limits = {});

}  // namespace imgjit::net
