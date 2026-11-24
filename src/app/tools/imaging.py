"""Plain image helpers: read the real file type, make thumbnails. No LLM."""

import io
from dataclasses import dataclass

from PIL import Image, UnidentifiedImageError

from app.core.errors import InvalidModelOutput

_MIME = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}
_EXT = {"image/jpeg": "jpg", "image/png": "png", "image/webp": "webp"}
THUMB_SIZE = 256


@dataclass(frozen=True)
class ImageInfo:
    mime: str
    ext: str
    width: int
    height: int


def sniff(data: bytes) -> ImageInfo:
    """Read the file type from the bytes. Never assume PNG."""
    try:
        with Image.open(io.BytesIO(data)) as im:
            fmt = (im.format or "").upper()
            width, height = im.size
    except (UnidentifiedImageError, OSError) as exc:
        raise InvalidModelOutput("provider returned something that is not an image") from exc
    mime = _MIME.get(fmt)
    if mime is None:
        raise InvalidModelOutput(f"provider returned an unsupported image type: {fmt}")
    return ImageInfo(mime=mime, ext=_EXT[mime], width=width, height=height)


def make_thumbnail(data: bytes, size: int = THUMB_SIZE) -> bytes:
    """A JPEG thumbnail whose longest side is `size` px."""
    with Image.open(io.BytesIO(data)) as im:
        rgb = im.convert("RGB")
        rgb.thumbnail((size, size))
        out = io.BytesIO()
        rgb.save(out, format="JPEG", quality=85)
    return out.getvalue()
