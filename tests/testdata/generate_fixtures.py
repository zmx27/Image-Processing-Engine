#!/usr/bin/env python3
"""Regenerates the PNG fixtures in this directory. Run: python3 generate_fixtures.py

Uses only the standard library (struct/zlib/binascii) so it has zero dependency on
Pillow or any other image library ever being installed on the dev machine or in CI.
"""
import struct
import zlib
from pathlib import Path

HERE = Path(__file__).parent


def write_png(path: Path, width: int, height: int, channels: int, pixel_fn):
    color_type = {1: 0, 3: 2, 4: 6}[channels]

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))

    raw = bytearray()
    for y in range(height):
        raw.append(0)  # no filter
        for x in range(width):
            raw.extend(pixel_fn(x, y))

    png = bytearray(b"\x89PNG\r\n\x1a\n")
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(bytes(raw), 9))
    png += chunk(b"IEND", b"")

    path.write_bytes(png)
    print(f"wrote {path.name} ({width}x{height}, {channels}ch)")


def solid_rgb(x, y):
    return (200, 40, 40)


def gradient_gray(x, y):
    return (int(255 * x / 7),)


def checkerboard_rgb(x, y):
    on = (x // 4 + y // 4) % 2 == 0
    return (230, 230, 230) if on else (20, 20, 20)


def gradient_rgba(x, y):
    v = int(255 * (x + y) / 62)
    return (v, 255 - v, 128, 255 if (x + y) % 2 == 0 else 180)


if __name__ == "__main__":
    write_png(HERE / "solid_4x4_rgb.png", 4, 4, 3, solid_rgb)
    write_png(HERE / "gradient_8x8_gray.png", 8, 8, 1, gradient_gray)
    write_png(HERE / "checkerboard_16x16_rgb.png", 16, 16, 3, checkerboard_rgb)
    write_png(HERE / "gradient_32x32_rgba.png", 32, 32, 4, gradient_rgba)
