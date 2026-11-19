"""core/images: format and size come from the bytes, not from what was asked for."""

from __future__ import annotations

import base64

import pytest

from media_engine.core.images import NotAnImage, data_url, inspect_image
from tests.unit.harness.fakes import image_bytes


@pytest.mark.parametrize(
    ("fmt", "mime"), [("PNG", "image/png"), ("JPEG", "image/jpeg"), ("WEBP", "image/webp")]
)
def test_inspect_reads_format_and_size(fmt, mime):
    info = inspect_image(image_bytes(320, 200, fmt=fmt))
    assert (info.mime, info.width, info.height) == (mime, 320, 200)


@pytest.mark.parametrize("data", [b"", b"<html>502 Bad Gateway</html>", image_bytes(fmt="GIF")])
def test_non_images_and_unsupported_formats_are_rejected(data):
    with pytest.raises(NotAnImage):
        inspect_image(data)


def test_data_url_round_trips():
    data = image_bytes()
    url = data_url(data, "image/png")
    assert url.startswith("data:image/png;base64,")
    assert base64.b64decode(url.split(",", 1)[1]) == data


def _png_header(width: int, height: int) -> bytes:
    """A PNG that only *claims* a size (signature + IHDR + IEND): no pixel data to allocate."""
    import struct
    import zlib

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
        )

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IEND", b"")


@pytest.mark.parametrize("side", [9000, 20000])  # over our cap; 20000^2 is past Pillow's bomb error
def test_oversized_images_are_rejected_without_decoding(side):
    with pytest.raises(NotAnImage):
        inspect_image(_png_header(side, side))


def test_pillow_limit_is_set_explicitly():
    from PIL import Image

    from media_engine.core.images import MAX_IMAGE_PIXELS

    assert Image.MAX_IMAGE_PIXELS == MAX_IMAGE_PIXELS == 8192 * 8192


def test_downscale_keeps_aspect_and_small_images_untouched():
    from media_engine.core.images import downscale

    small = image_bytes(300, 200)
    assert downscale(small, 768, 90) == (small, "image/png")

    data, mime = downscale(image_bytes(1600, 900), 768, 90)
    info = inspect_image(data)
    assert (mime, info.width, info.height) == ("image/jpeg", 768, 432)
