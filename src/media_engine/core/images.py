"""Read what an image actually is from its bytes. In-memory only: Pillow parses the header, so
`inspect_image` is cheap. `downscale` decodes and re-encodes, which is real CPU work: call it
via `asyncio.to_thread` from async code."""

from __future__ import annotations

import base64
import io
import warnings
from typing import NamedTuple

from PIL import Image, UnidentifiedImageError

# Formats the ArtifactStore accepts (LocalArtifactStore maps these to file extensions).
MIME_BY_FORMAT = {"PNG": "image/png", "JPEG": "image/jpeg", "WEBP": "image/webp"}

# Largest image we will decode: 8192 x 8192 (above any output we produce). Set explicitly so
# the decompression-bomb limit doesn't depend on Pillow's default. Pillow only WARNS between
# this and 2x it (and raises above), so inspect_image turns the warning into a rejection too:
# an oversized image is always a NotAnImage, never a log warning.
MAX_IMAGE_PIXELS = 8192 * 8192
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS


class ImageInfo(NamedTuple):
    mime: str
    width: int
    height: int


class NotAnImage(ValueError):
    """The bytes are not a PNG, JPEG or WebP image we are willing to decode."""


def inspect_image(data: bytes) -> ImageInfo:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as img:
                fmt, (width, height) = img.format, img.size
    except (
        UnidentifiedImageError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
        OSError,
        ValueError,
    ) as exc:
        raise NotAnImage(f"unreadable image ({len(data)} bytes)") from exc
    if width * height > MAX_IMAGE_PIXELS:
        raise NotAnImage(f"image too large ({width}x{height})")
    mime = MIME_BY_FORMAT.get(fmt or "")
    if mime is None:
        raise NotAnImage(f"unsupported image format {fmt!r}")
    return ImageInfo(mime, width, height)


def downscale(data: bytes, max_side: int, jpeg_quality: int) -> tuple[bytes, str]:
    """Shrink so the long side is at most `max_side`, re-encoded as JPEG. Returns the original
    bytes and mime unchanged when the image is already small enough. Blocking: use to_thread."""
    info = inspect_image(data)
    if max(info.width, info.height) <= max_side:
        return data, info.mime
    with Image.open(io.BytesIO(data)) as img:
        small = img.convert("RGB")
        small.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        small.save(buf, "JPEG", quality=jpeg_quality)
    return buf.getvalue(), "image/jpeg"


def data_url(data: bytes, mime: str) -> str:
    """`data:` URL for an OpenAI-style `image_url` content part. Only ever inside a model
    request: never in an Envelope, an event or a log."""
    return f"data:{mime};base64,{base64.b64encode(data).decode()}"
