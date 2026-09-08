#include "imgjit/net/client.h"

#include <algorithm>
#include <array>
#include <optional>
#include <stdexcept>
#include <utility>

namespace imgjit::net {

Client::Client(const std::string& host, std::uint16_t port) : socket_(connect_to(host, port)) {}

std::uint32_t Client::send(const Image& image, std::string_view chain, std::uint8_t flags) {
  RequestHeader header;
  header.flags = flags;
  header.seq_num = next_seq_num_++;
  header.width = static_cast<std::uint32_t>(image.width());
  header.height = static_cast<std::uint32_t>(image.height());
  header.channels = static_cast<std::uint8_t>(image.channels());
  header.payload_len = static_cast<std::uint32_t>(image.byte_count());
  header.chain_len = static_cast<std::uint16_t>(chain.size());

  const std::array<std::byte, kRequestHeaderBytes> encoded = encode_request_header(header);
  if (write_exact(socket_.descriptor(), encoded.data(), encoded.size()) != IoResult::kOk ||
      write_exact(socket_.descriptor(), chain.data(), chain.size()) != IoResult::kOk ||
      write_exact(socket_.descriptor(), image.data(), image.byte_count()) != IoResult::kOk) {
    throw std::runtime_error("imgjit client: connection closed while sending frame " +
                             std::to_string(header.seq_num));
  }

  pending_.emplace(header.seq_num, PendingFrame{header.seq_num, std::string(chain), image.width(),
                                                image.height(), image.channels(), flags});
  return header.seq_num;
}

void Client::send_raw(const RequestHeader& header, std::string_view chain,
                      const std::vector<std::uint8_t>& payload) {
  const std::array<std::byte, kRequestHeaderBytes> encoded = encode_request_header(header);
  if (write_exact(socket_.descriptor(), encoded.data(), encoded.size()) != IoResult::kOk ||
      write_exact(socket_.descriptor(), chain.data(), chain.size()) != IoResult::kOk ||
      write_exact(socket_.descriptor(), payload.data(), payload.size()) != IoResult::kOk) {
    throw std::runtime_error("imgjit client: connection closed while sending raw frame");
  }

  pending_.emplace(header.seq_num,
                   PendingFrame{header.seq_num, std::string(chain),
                                static_cast<int>(header.width), static_cast<int>(header.height),
                                static_cast<int>(header.channels), header.flags});
  next_seq_num_ = std::max(next_seq_num_, header.seq_num + 1);
}

ClientResponse Client::receive() {
  std::array<std::byte, kResponseHeaderBytes> header_bytes{};
  if (read_exact(socket_.descriptor(), header_bytes.data(), header_bytes.size()) !=
      IoResult::kOk) {
    throw std::runtime_error("imgjit client: connection closed while awaiting a response");
  }
  const std::optional<ResponseHeader> header =
      decode_response_header(header_bytes.data(), header_bytes.size());
  if (!header.has_value() || header->magic != kMagic) {
    throw std::runtime_error("imgjit client: response frame has a bad magic");
  }

  ClientResponse response;
  response.version = header->version;
  response.status = header->status;
  response.seq_num = header->seq_num;

  if (header->payload_len > 0) {
    // Trust the server no further than the client trusts its own arithmetic: a payload
    // that does not match the dimensions it came with is a broken frame, not something
    // to allocate for.
    const std::uint64_t expected = static_cast<std::uint64_t>(header->width) *
                                   static_cast<std::uint64_t>(header->height) *
                                   static_cast<std::uint64_t>(header->channels);
    if (expected != static_cast<std::uint64_t>(header->payload_len) ||
        header->payload_len > kMaxPayloadBytes) {
      throw std::runtime_error("imgjit client: response payload disagrees with its dimensions");
    }
    Image image(static_cast<int>(header->width), static_cast<int>(header->height),
                static_cast<int>(header->channels));
    if (read_exact(socket_.descriptor(), image.data(), header->payload_len) != IoResult::kOk) {
      throw std::runtime_error("imgjit client: connection closed mid-payload");
    }
    response.image = std::move(image);
  }

  const auto entry = pending_.find(response.seq_num);
  if (entry != pending_.end()) {
    response.matched = true;
    response.request = std::move(entry->second);
    pending_.erase(entry);
  }
  return response;
}

}  // namespace imgjit::net
