import base64
import io
import zlib

import pytest
from PIL import Image

from app.core.errors import BadInput
from app.guardrails.image_prep import batches, prep_image, prep_images


def make(fmt: str, size: tuple[int, int] = (100, 50), mode: str = "RGB", **kw: object) -> bytes:
    buf = io.BytesIO()
    Image.new(mode, size, "red").save(buf, format=fmt, **kw)  # type: ignore[arg-type]
    return buf.getvalue()


def png_claiming(width: int, height: int) -> bytes:
    """A PNG whose header says it is huge. Only the header is read before the size check."""
    data = bytearray(make("PNG", (4, 4)))
    ihdr = data[12:29]  # type + 13 bytes of data
    ihdr[4:8] = width.to_bytes(4, "big")
    ihdr[8:12] = height.to_bytes(4, "big")
    data[12:29] = ihdr
    data[29:33] = zlib.crc32(bytes(ihdr)).to_bytes(4, "big")
    return bytes(data)


def decode(data_url: str) -> Image.Image:
    assert data_url.startswith("data:image/jpeg;base64,")
    return Image.open(io.BytesIO(base64.b64decode(data_url.split(",", 1)[1])))


@pytest.mark.parametrize("fmt", ["JPEG", "PNG", "WEBP"])
def test_accepts_three_formats(fmt: str) -> None:
    out = prep_image(make(fmt))
    assert decode(out.data_url).format == "JPEG"
    assert len(out.sha256) == 64


def test_rejects_other_formats_and_junk() -> None:
    with pytest.raises(BadInput):
        prep_image(make("GIF"))
    with pytest.raises(BadInput):
        prep_image(b"not an image at all")


def test_rejects_truncated_file() -> None:
    with pytest.raises(BadInput):
        prep_image(make("PNG", (400, 400))[:200])


def test_rejects_oversize_bytes() -> None:
    with pytest.raises(BadInput):
        prep_image(b"\xff\xd8\xff" + b"0" * (20 * 1024 * 1024))


def test_rejects_too_many_pixels() -> None:
    with pytest.raises(BadInput):
        prep_image(png_claiming(8000, 6500))  # 52 MP


def test_downscale_keeps_ratio_and_never_upscales() -> None:
    big = prep_image(make("PNG", (1536, 768)))
    assert (big.width, big.height) == (768, 384)
    small = prep_image(make("PNG", (200, 100)))
    assert (small.width, small.height) == (200, 100)


def test_alpha_goes_on_white() -> None:
    buf = io.BytesIO()
    Image.new("RGBA", (10, 10), (0, 0, 0, 0)).save(buf, format="PNG")
    px = decode(prep_image(buf.getvalue()).data_url).getpixel((5, 5))
    assert isinstance(px, tuple)
    assert min(px) > 250


def test_exif_rotation_applied_and_metadata_stripped() -> None:
    img = Image.new("RGB", (100, 50), "blue")
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90
    exif[0x010F] = "SecretCamera"
    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=exif.tobytes())
    out = prep_image(buf.getvalue())
    assert (out.width, out.height) == (50, 100)
    result = decode(out.data_url)
    assert not result.getexif()
    assert "exif" not in result.info


def test_count_limit_and_batches() -> None:
    one = make("PNG")
    with pytest.raises(BadInput):
        prep_images([one] * 11)
    prepared = prep_images([one] * 7)
    assert [len(b) for b in batches(prepared)] == [5, 2]
    assert batches([]) == []
