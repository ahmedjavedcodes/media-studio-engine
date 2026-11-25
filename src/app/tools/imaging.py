"""Plain image helpers: read the real file type, make thumbnails. No LLM.

Formats: JPEG, PNG, WEBP, GIF (first frame for thumbnails) and SVG (vector models). An SVG is
checked for scripts, because it is served to a browser, and rasterised with resvg for thumbnails,
so the per-image safety check always sees pixels.
"""

import io
import re
from dataclasses import dataclass

import resvg_py
from PIL import Image, UnidentifiedImageError

from app.core.errors import InvalidModelOutput

_MIME = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp", "GIF": "image/gif"}
_EXT = {
    "image/jpeg": "jpg",
    "image/png": "png",
    "image/webp": "webp",
    "image/gif": "gif",
    "image/svg+xml": "svg",
}
SVG_MIME = "image/svg+xml"
THUMB_SIZE = 256
MAX_SVG_BYTES = 5 * 1024 * 1024

# Anything that can run code or load another file when the SVG is opened in a browser.
_SVG_ACTIVE = re.compile(
    rb"<\s*script|javascript:|\son[a-z]+\s*=|<\s*foreignObject|<\s*iframe|<\s*embed|"
    rb"<\s*object|<!ENTITY|xlink:href\s*=\s*[\"'](?!#)|href\s*=\s*[\"']\s*(?:https?:|data:)",
    re.IGNORECASE,
)
_SVG_HEAD = re.compile(rb"^\s*(?:<\?xml[^>]*\?>\s*)?(?:<!--.*?-->\s*)*<svg[\s>]", re.DOTALL)
_NUM = r"([0-9.]+)"


@dataclass(frozen=True)
class ImageInfo:
    mime: str
    ext: str
    width: int
    height: int


def is_svg(data: bytes) -> bool:
    return bool(_SVG_HEAD.match(data[:2048]))


def _svg_size(data: bytes) -> tuple[int, int]:
    """Width and height from the root tag: `width`/`height`, else the viewBox."""
    head = data[:4096].decode("utf-8", "ignore")
    tag = re.search(r"<svg\b[^>]*>", head, re.IGNORECASE | re.DOTALL)
    text = tag.group(0) if tag else head
    w = re.search(rf'\bwidth\s*=\s*"{_NUM}', text)
    h = re.search(rf'\bheight\s*=\s*"{_NUM}', text)
    if w and h:
        return round(float(w.group(1))), round(float(h.group(1)))
    box = re.search(rf'viewBox\s*=\s*"[^"]*?\s{_NUM}\s+{_NUM}\s*"', text)
    if box:
        return round(float(box.group(1))), round(float(box.group(2)))
    return 1024, 1024  # a vector has no fixed size. This is only a layout hint


def sniff(data: bytes, media_type: str | None = None) -> ImageInfo:
    """Read the file type from the bytes. Never assume PNG. `media_type` is only a hint."""
    if is_svg(data) or media_type == SVG_MIME:
        if not is_svg(data):
            raise InvalidModelOutput("provider returned something that is not an SVG")
        if len(data) > MAX_SVG_BYTES:
            raise InvalidModelOutput("provider returned an SVG that is too large")
        if _SVG_ACTIVE.search(data):
            raise InvalidModelOutput("provider returned an SVG with active content")
        w, h = _svg_size(data)
        return ImageInfo(mime=SVG_MIME, ext="svg", width=w, height=h)
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


def _rasterise_svg(data: bytes, size: int) -> Image.Image:
    try:
        png = bytes(resvg_py.svg_to_bytes(svg_string=data.decode("utf-8", "replace"), width=size))
        with Image.open(io.BytesIO(png)) as im:
            rgba = im.convert("RGBA")
    except Exception as exc:  # noqa: BLE001 - a bad SVG is an invalid model output
        raise InvalidModelOutput("provider returned an SVG that cannot be drawn") from exc
    white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))  # a vector has no background
    return Image.alpha_composite(white, rgba).convert("RGB")


def make_thumbnail(data: bytes, size: int = THUMB_SIZE) -> bytes:
    """A JPEG thumbnail whose longest side is `size` px. An SVG is drawn first, a GIF uses its
    first frame."""
    if is_svg(data):
        rgb = _rasterise_svg(data, size)
    else:
        with Image.open(io.BytesIO(data)) as im:
            rgb = im.convert("RGB")
    rgb.thumbnail((size, size))
    out = io.BytesIO()
    rgb.save(out, format="JPEG", quality=85)
    return out.getvalue()
