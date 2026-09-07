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
| 19 | payload_len | 4 | byte length of the `payload` field; must equal `width * height * channels` |
| 23 | chain_len | 2 | byte length of the `chain` field that follows |
| 25 | chain | `chain_len` | ASCII op-chain descriptor, e.g. `grayscale,gaussian:1.4,sobel,threshold:0.3` |
| 25 + chain_len | payload | `payload_len` | raw contiguous pixels, row-major, `uint8` per channel |

The fixed header is 25 bytes. `payload_len` is redundant with the dimensions by design: it makes
framing self-describing (the server knows how many bytes to drain or skip without first trusting
the dimension fields), and the cross-check against `width * height * channels` is itself a
validation step — a mismatch is error code 5.

## Response frame

| Offset | Field | Size (bytes) | Notes |
|---|---|---|---|
| 0 | magic | 4 | `0xDEADBEEF` |
| 4 | version | 1 | the **server's** version — normally equal to the request's; on status 2 this is how the client learns what is actually supported |
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

- `kMaxPayloadBytes` (default 64 MiB): a request declaring a `payload_len` above this is rejected
  with an error response **before any buffer is allocated**. Never trust an attacker-controlled
  length for allocation size.
- `kMaxChainLen` (default 256 bytes): bounds the op-chain descriptor length similarly.

**Compute the dimension product in `uint64_t`.** `width * height * channels` overflows `uint32`
for values a client can trivially send (`65535 * 65535 * 4`), and a wrapped product that compares
equal to a small `payload_len` would pass validation and then be used as a dimension for indexing.
Widen first, compare against `kMaxPayloadBytes` and against `payload_len`, and only then narrow.

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

## Validation order, and what happens to the payload on error

Header fields are validated in the order they are read, so the server never allocates or reads
more than it has already justified: magic → version → `payload_len` against `kMaxPayloadBytes` →
`chain_len` against `kMaxChainLen` → dimension product against `payload_len` → chain parse.

The subtlety is that an error is detected from the header while the client is *already streaming
the payload*. Those bytes are still in flight; if the server does not account for them, the next
header read starts mid-payload and every subsequent frame on that connection is garbage. Each
error code therefore carries a connection disposition:

| Code | Disposition |
|---|---|
| 1 (bad magic) | **Close.** Framing is unrecoverable — there is no trustworthy length to resynchronize on. |
| 2 (unsupported version) | **Close.** The frame layout itself is what is in question. |
| 3 (payload exceeds `kMaxPayloadBytes`) | **Close**, after sending the error response. Draining a length the client chose and we already declared unreasonable *is* the DoS. |
| 4 (chain too long / parse failure) | **Drain and continue.** `payload_len` is already validated and bounded, so discarding exactly that many bytes is safe and bounded work. |
| 5 (payload length / dimension mismatch) | **Drain and continue**, using `payload_len` (bounded) rather than the dimension product. |
| 6 (internal error) | **Continue.** The payload was fully read before processing began; nothing to drain. |

Draining reads through `read_exact()` into a small fixed scratch buffer — never into an allocation
sized by the client.

## Server-side write (flags bit 1)

The protocol carries no filename: accepting a client-supplied path would be a directory-traversal
hole for zero demonstrative value. The server writes to
`<output_dir>/frame_<conn_id>_<seq_num>.png`, where `output_dir` is server configuration and
`conn_id` is server-assigned. A client that needs a specific name uses bit 0 and writes the file
itself.

## Explicit non-goals

- **No JPEG/PNG decoding over the wire.** The payload is always raw contiguous pixels. `stb_image`
  is a CLI/test-side file convenience only, never used to decode network input. The server-side
  write path (`flags` bit 1) does use `stb_image_write` to emit a PNG — that is file output, not
  wire format, and nothing is ever decoded on the way in.
- **No authentication or TLS.** Out of scope for this project (see `ARCHITECTURE.md` error
  handling scope).
