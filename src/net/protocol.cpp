#include "imgjit/net/protocol.h"

#include "imgjit/core/image.h"

namespace imgjit::net {
namespace {

// Field-by-field little-endian packing. Written out rather than reinterpret_cast so the
// result does not depend on the host's byte order or on struct padding (CLAUDE.md
// invariant 6). Every environment this targets is little-endian, so these are no-ops on
// the hardware — the point is that they are no-ops by specification, not by accident.
void put_u8(std::byte* out, std::uint8_t value) {
  out[0] = static_cast<std::byte>(value);
}

void put_u16(std::byte* out, std::uint16_t value) {
  out[0] = static_cast<std::byte>(value & 0xFFU);
  out[1] = static_cast<std::byte>((value >> 8U) & 0xFFU);
}

void put_u32(std::byte* out, std::uint32_t value) {
  out[0] = static_cast<std::byte>(value & 0xFFU);
  out[1] = static_cast<std::byte>((value >> 8U) & 0xFFU);
  out[2] = static_cast<std::byte>((value >> 16U) & 0xFFU);
  out[3] = static_cast<std::byte>((value >> 24U) & 0xFFU);
}

std::uint8_t get_u8(const std::byte* in) {
  return std::to_integer<std::uint8_t>(in[0]);
}

std::uint16_t get_u16(const std::byte* in) {
  return static_cast<std::uint16_t>(static_cast<std::uint16_t>(get_u8(in)) |
                                    static_cast<std::uint16_t>(get_u8(in + 1) << 8U));
}

std::uint32_t get_u32(const std::byte* in) {
  return static_cast<std::uint32_t>(get_u8(in)) |
         (static_cast<std::uint32_t>(get_u8(in + 1)) << 8U) |
         (static_cast<std::uint32_t>(get_u8(in + 2)) << 16U) |
         (static_cast<std::uint32_t>(get_u8(in + 3)) << 24U);
}

}  // namespace

Disposition disposition_for(Status status) {
  switch (status) {
    case Status::kOk:
    case Status::kInternalError:
      // The payload was fully read before processing began; nothing is outstanding.
      return Disposition::kContinue;
    case Status::kBadChain:
    case Status::kLengthMismatch:
      // payload_len and chain_len are both already bounded by this point, so discarding
      // exactly that many bytes is safe and bounded work.
      return Disposition::kDrain;
    case Status::kBadMagic:
    case Status::kUnsupportedVersion:
    case Status::kPayloadTooLarge:
      // 1 and 2: framing itself is in question, so there is no length to trust. 3:
      // draining a length we have already declared unreasonable *is* the DoS.
      return Disposition::kClose;
  }
  return Disposition::kClose;
}

std::string_view status_message(Status status) {
  switch (status) {
    case Status::kOk:
      return "ok";
    case Status::kBadMagic:
      return "bad magic";
    case Status::kUnsupportedVersion:
      return "unsupported version";
    case Status::kPayloadTooLarge:
      return "payload exceeds the configured maximum";
    case Status::kBadChain:
      return "op chain too long or malformed";
    case Status::kLengthMismatch:
      return "payload length does not match width * height * channels";
    case Status::kInternalError:
      return "internal error";
  }
  return "unknown status";
}

std::array<std::byte, kRequestHeaderBytes> encode_request_header(const RequestHeader& header) {
  std::array<std::byte, kRequestHeaderBytes> out{};
  put_u32(out.data() + 0, header.magic);
  put_u8(out.data() + 4, header.version);
  put_u8(out.data() + 5, header.flags);
  put_u32(out.data() + 6, header.seq_num);
  put_u32(out.data() + 10, header.width);
  put_u32(out.data() + 14, header.height);
  put_u8(out.data() + 18, header.channels);
  put_u32(out.data() + 19, header.payload_len);
  put_u16(out.data() + 23, header.chain_len);
  return out;
}

std::array<std::byte, kResponseHeaderBytes> encode_response_header(const ResponseHeader& header) {
  std::array<std::byte, kResponseHeaderBytes> out{};
  put_u32(out.data() + 0, header.magic);
  put_u8(out.data() + 4, header.version);
  put_u8(out.data() + 5, static_cast<std::uint8_t>(header.status));
  put_u32(out.data() + 6, header.seq_num);
  put_u32(out.data() + 10, header.width);
  put_u32(out.data() + 14, header.height);
  put_u8(out.data() + 18, header.channels);
  put_u32(out.data() + 19, header.payload_len);
  return out;
}

std::optional<RequestHeader> decode_request_header(const std::byte* bytes, std::size_t size) {
  if (bytes == nullptr || size != kRequestHeaderBytes) {
    return std::nullopt;
  }
  RequestHeader header;
  header.magic = get_u32(bytes + 0);
  header.version = get_u8(bytes + 4);
  header.flags = get_u8(bytes + 5);
  header.seq_num = get_u32(bytes + 6);
  header.width = get_u32(bytes + 10);
  header.height = get_u32(bytes + 14);
  header.channels = get_u8(bytes + 18);
  header.payload_len = get_u32(bytes + 19);
  header.chain_len = get_u16(bytes + 23);
  return header;
}

std::optional<ResponseHeader> decode_response_header(const std::byte* bytes, std::size_t size) {
  if (bytes == nullptr || size != kResponseHeaderBytes) {
    return std::nullopt;
  }
  ResponseHeader header;
  header.magic = get_u32(bytes + 0);
  header.version = get_u8(bytes + 4);
  header.status = static_cast<Status>(get_u8(bytes + 5));
  header.seq_num = get_u32(bytes + 6);
  header.width = get_u32(bytes + 10);
  header.height = get_u32(bytes + 14);
  header.channels = get_u8(bytes + 18);
  header.payload_len = get_u32(bytes + 19);
  return header;
}

Status validate_request_header(const RequestHeader& header, const Limits& limits) {
  if (header.magic != kMagic) {
    return Status::kBadMagic;
  }
  if (header.version != kVersion) {
    return Status::kUnsupportedVersion;
  }
  if (header.payload_len > limits.max_payload_bytes) {
    return Status::kPayloadTooLarge;
  }
  if (header.chain_len > limits.max_chain_len) {
    return Status::kBadChain;
  }
  // A channel count the protocol does not define, and a zero dimension, are both
  // dimension errors and share status 5's drain disposition: payload_len is already
  // bounded above, so the connection can still be resynchronized.
  if (!is_supported_channel_count(static_cast<int>(header.channels)) || header.width == 0 ||
      header.height == 0) {
    return Status::kLengthMismatch;
  }
  const std::uint64_t product = static_cast<std::uint64_t>(header.width) *
                                static_cast<std::uint64_t>(header.height) *
                                static_cast<std::uint64_t>(header.channels);
  if (product != static_cast<std::uint64_t>(header.payload_len)) {
    return Status::kLengthMismatch;
  }
  return Status::kOk;
}

}  // namespace imgjit::net
