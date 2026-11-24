"""Image prep for the vision check (pure). Spec section 7.3.

The output is only for the safety model. The generator gets the original bytes.
"""

import base64
import hashlib
import io
from dataclasses import dataclass

from PIL import Image, ImageOps

from app.core.errors import BadInput
from app.guardrails.policy import (
    IMAGE_LONG_SIDE,
    IMAGES_PER_CALL,
    JPEG_QUALITY,
    MAX_IMAGE_BYTES,
    MAX_IMAGE_PIXELS,
    MAX_IMAGES_PER_REQUEST,
)


@dataclass(frozen=True)
class PreparedImage:
    data_url: str
    sha256: str  # of the original bytes, for the audit log
    width: int
    height: int


def _detect_format(data: bytes) -> str | None:
    if data.startswith(b"\xff\xd8\xff"):
        return "JPEG"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "PNG"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "WEBP"
    return None


def prep_image(data: bytes) -> PreparedImage:
    """Validate, fix rotation, strip metadata, downscale, re-encode as JPEG."""
    if len(data) > MAX_IMAGE_BYTES:
        raise BadInput("Image is larger than 20 MB.")
    fmt = _detect_format(data)
    if fmt is None:
        raise BadInput("Image must be JPEG, PNG or WEBP.")
    try:
        with Image.open(io.BytesIO(data)) as img:
            if img.format != fmt:
                raise BadInput("Image type does not match its content.")
            if img.width * img.height > MAX_IMAGE_PIXELS:
                raise BadInput("Image is larger than 50 megapixels.")
            img.load()
            fixed = ImageOps.exif_transpose(img)
            rgb = _to_rgb(fixed)
    except BadInput:
        raise
    except Exception as exc:  # corrupt or truncated file, decompression bomb, ...
        raise BadInput("Image could not be read.") from exc

    longest = max(rgb.size)
    if longest > IMAGE_LONG_SIDE:  # never upscale
        scale = IMAGE_LONG_SIDE / longest
        size = (max(1, round(rgb.width * scale)), max(1, round(rgb.height * scale)))
        rgb = rgb.resize(size, Image.Resampling.LANCZOS)

    out = io.BytesIO()
    rgb.save(out, format="JPEG", quality=JPEG_QUALITY)  # new file: no EXIF carried over
    encoded = base64.b64encode(out.getvalue()).decode("ascii")
    return PreparedImage(
        data_url=f"data:image/jpeg;base64,{encoded}",
        sha256=hashlib.sha256(data).hexdigest(),
        width=rgb.width,
        height=rgb.height,
    )


def _to_rgb(img: Image.Image) -> Image.Image:
    if img.mode in ("RGBA", "LA", "PA") or "transparency" in img.info:
        rgba = img.convert("RGBA")
        white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(white, rgba).convert("RGB")
    return img.convert("RGB")


def prep_images(images: list[bytes]) -> list[PreparedImage]:
    if len(images) > MAX_IMAGES_PER_REQUEST:
        raise BadInput(f"At most {MAX_IMAGES_PER_REQUEST} images per request.")
    return [prep_image(data) for data in images]


def batches(items: list[PreparedImage]) -> list[list[PreparedImage]]:
    """Groups of up to 5 images, one vision call each."""
    return [items[i : i + IMAGES_PER_CALL] for i in range(0, len(items), IMAGES_PER_CALL)]
