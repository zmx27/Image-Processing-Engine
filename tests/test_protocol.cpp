// The wire codec (docs/PLAN.md Phase 4, docs/PROTOCOL.md).
//
// These are pure-function tests on purpose: every malformed frame the server has to
// survive is expressible here, without a socket, a thread or a timeout. What is being
// asserted is not only "does it round-trip" but the two properties the server's safety
// rests on — that the byte layout is fixed by specification rather than by this
// machine's struct layout, and that validation happens in docs/PROTOCOL.md's order, so
// nothing is ever sized by a length that has not been capped yet.

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "catch_amalgamated.hpp"
#include "imgjit/net/protocol.h"

using imgjit::net::decode_request_header;
using imgjit::net::decode_response_header;
using imgjit::net::Disposition;
using imgjit::net::disposition_for;
using imgjit::net::encode_request_header;
using imgjit::net::encode_response_header;
using imgjit::net::kMagic;
using imgjit::net::kMaxChainLen;
using imgjit::net::kMaxPayloadBytes;
using imgjit::net::kRequestHeaderBytes;
using imgjit::net::kResponseHeaderBytes;
using imgjit::net::kVersion;
using imgjit::net::Limits;
using imgjit::net::RequestHeader;
using imgjit::net::ResponseHeader;
using imgjit::net::Status;
using imgjit::net::validate_request_header;

namespace {

// A header that passes every check, so each test can spoil exactly one field.
RequestHeader valid_header() {
  RequestHeader header;
  header.magic = kMagic;
  header.version = kVersion;
  header.flags = imgjit::net::kFlagEcho;
  header.seq_num = 7;
  header.width = 16;
  header.height = 8;
  header.channels = 3;
  header.payload_len = 16 * 8 * 3;
  header.chain_len = 9;
  return header;
}

std::uint8_t byte_at(const std::array<std::byte, kRequestHeaderBytes>& bytes, std::size_t index) {
  return std::to_integer<std::uint8_t>(bytes[index]);
}

}  // namespace

TEST_CASE("a request header round-trips through encode and decode", "[net]") {
  const RequestHeader original = valid_header();
  const auto encoded = encode_request_header(original);
  const auto decoded = decode_request_header(encoded.data(), encoded.size());

  REQUIRE(decoded.has_value());
  CHECK(decoded->magic == original.magic);
  CHECK(decoded->version == original.version);
  CHECK(decoded->flags == original.flags);
  CHECK(decoded->seq_num == original.seq_num);
  CHECK(decoded->width == original.width);
  CHECK(decoded->height == original.height);
  CHECK(decoded->channels == original.channels);
  CHECK(decoded->payload_len == original.payload_len);
  CHECK(decoded->chain_len == original.chain_len);
}

TEST_CASE("request fields are packed little-endian at the offsets in PROTOCOL.md", "[net]") {
  // Hand-written expected bytes, not a re-derivation from the encoder: the point is to
  // pin the layout to the spec, so a change to either has to be deliberate. This is
  // also what catches a struct-memcpy regression — padding would move every field after
  // `version`.
  RequestHeader header = valid_header();
  header.seq_num = 0x01020304;
  header.width = 0x0000ABCD;
  header.height = 0x00001234;
  header.payload_len = 0x00ABCDEF;
  header.chain_len = 0x0102;

  const auto bytes = encode_request_header(header);
  REQUIRE(bytes.size() == 25);

  // magic 0xDEADBEEF, least significant byte first
  CHECK(byte_at(bytes, 0) == 0xEF);
  CHECK(byte_at(bytes, 1) == 0xBE);
  CHECK(byte_at(bytes, 2) == 0xAD);
  CHECK(byte_at(bytes, 3) == 0xDE);
  CHECK(byte_at(bytes, 4) == kVersion);
  CHECK(byte_at(bytes, 5) == imgjit::net::kFlagEcho);
  CHECK(byte_at(bytes, 6) == 0x04);
  CHECK(byte_at(bytes, 9) == 0x01);
  CHECK(byte_at(bytes, 10) == 0xCD);
  CHECK(byte_at(bytes, 11) == 0xAB);
  CHECK(byte_at(bytes, 14) == 0x34);
  CHECK(byte_at(bytes, 15) == 0x12);
  CHECK(byte_at(bytes, 18) == 3);
  CHECK(byte_at(bytes, 19) == 0xEF);
  CHECK(byte_at(bytes, 20) == 0xCD);
  CHECK(byte_at(bytes, 21) == 0xAB);
  CHECK(byte_at(bytes, 23) == 0x02);
  CHECK(byte_at(bytes, 24) == 0x01);
}

TEST_CASE("a response header round-trips and is 23 bytes", "[net]") {
  ResponseHeader original;
  original.version = kVersion;
  original.status = Status::kOk;
  original.seq_num = 4242;
  original.width = 64;
  original.height = 32;
  original.channels = 4;
  original.payload_len = 64 * 32 * 4;

  const auto encoded = encode_response_header(original);
  REQUIRE(encoded.size() == kResponseHeaderBytes);
  REQUIRE(encoded.size() == 23);

  const auto decoded = decode_response_header(encoded.data(), encoded.size());
  REQUIRE(decoded.has_value());
  CHECK(decoded->magic == kMagic);
  CHECK(decoded->version == kVersion);
  CHECK(decoded->status == Status::kOk);
  CHECK(decoded->seq_num == 4242);
  CHECK(decoded->width == 64);
  CHECK(decoded->height == 32);
  CHECK(decoded->channels == 4);
  CHECK(decoded->payload_len == 64U * 32U * 4U);
}

TEST_CASE("a truncated header does not decode", "[net]") {
  // The server can only reach decode through read_exact, so this should be
  // unreachable there — which is exactly why it is asserted here rather than trusted.
  const auto encoded = encode_request_header(valid_header());
  CHECK_FALSE(decode_request_header(encoded.data(), encoded.size() - 1).has_value());
  CHECK_FALSE(decode_request_header(encoded.data(), 0).has_value());
  CHECK_FALSE(decode_request_header(nullptr, kRequestHeaderBytes).has_value());
  CHECK_FALSE(decode_response_header(encoded.data(), kResponseHeaderBytes - 1).has_value());
}

TEST_CASE("decoding passes judgement on nothing", "[net]") {
  // A bad magic must survive decode intact and be rejected by validation instead.
  // Short-circuiting in the decoder would quietly reorder docs/PROTOCOL.md's validation
  // sequence, which is the one thing the error codes depend on.
  RequestHeader header = valid_header();
  header.magic = 0x12345678;
  const auto encoded = encode_request_header(header);
  const auto decoded = decode_request_header(encoded.data(), encoded.size());
  REQUIRE(decoded.has_value());
  CHECK(decoded->magic == 0x12345678);
  CHECK(validate_request_header(*decoded) == Status::kBadMagic);
}

TEST_CASE("a valid header validates", "[net]") {
  CHECK(validate_request_header(valid_header()) == Status::kOk);
}

TEST_CASE("validation runs in the order PROTOCOL.md fixes", "[net]") {
  // Each case below is wrong in several ways at once; the expected status is whichever
  // check comes FIRST. This is what stops the server from, say, capping a payload
  // length it read out of a frame whose version it does not even support.
  RequestHeader header = valid_header();
  header.magic = 0;
  header.version = 99;
  header.payload_len = kMaxPayloadBytes + 1;
  header.chain_len = kMaxChainLen + 1;
  CHECK(validate_request_header(header) == Status::kBadMagic);

  header.magic = kMagic;
  CHECK(validate_request_header(header) == Status::kUnsupportedVersion);

  header.version = kVersion;
  CHECK(validate_request_header(header) == Status::kPayloadTooLarge);

  header.payload_len = 16 * 8 * 3;
  CHECK(validate_request_header(header) == Status::kBadChain);

  header.chain_len = kMaxChainLen;
  CHECK(validate_request_header(header) == Status::kOk);
}

TEST_CASE("the caps are boundaries, not approximations", "[net]") {
  RequestHeader header = valid_header();
  header.chain_len = kMaxChainLen;
  CHECK(validate_request_header(header) == Status::kOk);
  header.chain_len = kMaxChainLen + 1;
  CHECK(validate_request_header(header) == Status::kBadChain);

  // A server may configure a smaller cap than the protocol default — imgjit-server sets
  // it to its slot size, so a frame that validates always fits the slot it is read into.
  const Limits tight{.max_payload_bytes = 16 * 8 * 3, .max_chain_len = kMaxChainLen};
  header = valid_header();
  CHECK(validate_request_header(header, tight) == Status::kOk);
  header.width = 17;
  header.payload_len = 17 * 8 * 3;
  CHECK(validate_request_header(header, tight) == Status::kPayloadTooLarge);
}

TEST_CASE("a payload length that disagrees with the dimensions is rejected", "[net]") {
  RequestHeader header = valid_header();
  header.payload_len = 16 * 8 * 3 - 1;
  CHECK(validate_request_header(header) == Status::kLengthMismatch);
}

TEST_CASE("the dimension product is computed in 64 bits", "[net]") {
  // docs/PROTOCOL.md: 65535 * 65535 * 4 overflows uint32. In 32-bit arithmetic
  // 65536 * 65536 * 1 wraps to exactly 0, which would compare EQUAL to a payload_len of
  // 0 and pass validation — after which those dimensions get used for indexing. Widen
  // first, and the frame is rejected instead.
  RequestHeader header = valid_header();
  header.width = 65536;
  header.height = 65536;
  header.channels = 1;
  header.payload_len = 0;
  CHECK(validate_request_header(header) == Status::kLengthMismatch);

  header.channels = 3;  // 3 * 2^32 also wraps to 0
  CHECK(validate_request_header(header) == Status::kLengthMismatch);

  // And a wrap that lands somewhere plausible rather than on zero: 65536 * 16385 * 4 is
  // 2^32 + 262144, so 32-bit arithmetic would produce exactly the 262144 declared here
  // and wave the frame through — with dimensions 16 GiB apart from the payload that
  // actually arrives.
  header.width = 65536;
  header.height = 16385;
  header.channels = 4;
  header.payload_len = 262144;
  CHECK(validate_request_header(header) == Status::kLengthMismatch);
}

TEST_CASE("a channel count the protocol does not define is a dimension error", "[net]") {
  // 2-channel gray+alpha is the realistic mistake; the protocol has no such format.
  RequestHeader header = valid_header();
  header.channels = 2;
  header.payload_len = 16 * 8 * 2;
  CHECK(validate_request_header(header) == Status::kLengthMismatch);

  header = valid_header();
  header.width = 0;
  header.payload_len = 0;
  CHECK(validate_request_header(header) == Status::kLengthMismatch);
}

TEST_CASE("every status carries PROTOCOL.md's connection disposition", "[net]") {
  // The disposition table is the resynchronization rule. Getting one row wrong means
  // the next header read starts mid-payload and every later frame is garbage — which
  // is why it is asserted row by row rather than exercised only end to end.
  CHECK(disposition_for(Status::kOk) == Disposition::kContinue);
  CHECK(disposition_for(Status::kInternalError) == Disposition::kContinue);
  CHECK(disposition_for(Status::kBadChain) == Disposition::kDrain);
  CHECK(disposition_for(Status::kLengthMismatch) == Disposition::kDrain);
  CHECK(disposition_for(Status::kBadMagic) == Disposition::kClose);
  CHECK(disposition_for(Status::kUnsupportedVersion) == Disposition::kClose);
  CHECK(disposition_for(Status::kPayloadTooLarge) == Disposition::kClose);
}
