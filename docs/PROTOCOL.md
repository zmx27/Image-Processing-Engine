# Wire Protocol

One frame in, one response out. All multi-byte fields are **little-endian**, packed field by
field — never `memcpy` a struct onto the socket (padding and endianness make that non-portable).
Little-endian, not network byte order, is a deliberate choice: every environment this project
targets (x86_64 Mac, Apple Silicon Mac, Colab's Linux x86_64 container) is already little-endian,
so big-endian would mean byte-swapping on both ends for zero benefit.

## Request frame

| Offset | Field | Size (bytes) | Notes |
|---|---|---|---|
| 0 | magic | 4 | `0xDEADBEEF` |
| 4 | version | 1 | starts at `1` |
| 5 | flags | 1 | bit 0 = echo result to client, bit 1 = write result server-side. Both may be set. |
| 6 | seq_num | 4 | client-assigned, echoed back verbatim in the response |
| 10 | width | 4 | pixels |
| 14 | height | 4 | pixels |
| 18 | channels | 1 | `1` (gray), `3` (RGB), or `4` (RGBA) |
| 19 | chain_len | 2 | byte length of the `chain` field that follows |
| 21 | chain | `chain_len` | ASCII op-chain descriptor, e.g. `grayscale,gaussian:1.4,sobel,threshold:0.3` |
| 21 + chain_len | payload | `width * height * channels` | raw contiguous pixels, row-major, `uint8` per channel |

## Response frame

| Offset | Field | Size (bytes) | Notes |
|---|---|---|---|
| 0 | magic | 4 | `0xDEADBEEF` |
| 4 | version | 1 | matches request |
| 5 | status | 1 | `0` = ok, non-zero = error code (see below) |
| 6 | seq_num | 4 | echoed from the request |
| 10 | width | 4 | pixels (0 if status != 0) |
| 14 | height | 4 | pixels (0 if status != 0) |
| 18 | channels | 1 | (0 if status != 0) |
| 19 | payload_len | 4 | byte length of payload that follows |
| 23 | payload | `payload_len` | processed pixels (present only if flags bit 0 was set and status == 0); empty on error or write-only requests |

## Why `seq_num` is not optional

Once multiple CUDA streams are in flight (Phase 6+), GPU completions finish out of order relative
to submission — even within a single connection. Echoing a client-assigned sequence number lets
the client match responses itself, so the server never needs a reorder buffer.

## Constants

- `kMaxPayloadBytes` (default 64 MiB): a request declaring `width * height * channels` above this
  is rejected with an error response **before any buffer is allocated**. Never trust an
  attacker-controlled length for allocation size.
- `kMaxChainLen` (default 256 bytes): bounds the op-chain descriptor length similarly.

## Error status codes

| Code | Meaning |
|---|---|
| 0 | success |
| 1 | bad magic |
| 2 | unsupported version |
| 3 | payload exceeds `kMaxPayloadBytes` |
| 4 | chain exceeds `kMaxChainLen` or fails to parse |
| 5 | declared payload length does not match `width * height * channels` |
| 6 | internal error (GPU/compile failure) — server remains alive; context may be recreated (see `ARCHITECTURE.md`) |

## Explicit non-goals

- **No JPEG/PNG decoding over the wire.** The payload is always raw contiguous pixels. `stb_image`
  is a CLI/test-side file convenience only, never used by the server on network input.
- **No authentication or TLS.** Out of scope for this project (see `ARCHITECTURE.md` error
  handling scope).
