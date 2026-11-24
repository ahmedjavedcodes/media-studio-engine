"""Image sizes from an aspect ratio. Social sizes and 4K print come later."""

import math

ASPECT_RATIOS = ("1:1", "4:5", "5:4", "3:4", "4:3", "2:3", "3:2", "9:16", "16:9", "21:9")

# Target area in pixels. 1K means about one megapixel. 2K comes from the upscale tool later.
_AREA = {"1K": 1024 * 1024}


def parse_ratio(aspect_ratio: str) -> tuple[int, int]:
    try:
        w, h = aspect_ratio.split(":")
        a, b = int(w), int(h)
    except ValueError as exc:
        raise ValueError(f"bad aspect ratio: {aspect_ratio!r}") from exc
    if a <= 0 or b <= 0:
        raise ValueError(f"bad aspect ratio: {aspect_ratio!r}")
    return a, b


def dimensions(aspect_ratio: str, resolution: str = "1K", multiple: int = 32) -> tuple[int, int]:
    """(width, height) for the ratio at about 1 megapixel, each side a multiple of `multiple`."""
    a, b = parse_ratio(aspect_ratio)
    area = _AREA.get(resolution, _AREA["1K"])
    width = math.sqrt(area * a / b)
    height = width * b / a

    def snap(v: float) -> int:
        return max(multiple, round(v / multiple) * multiple)

    return snap(width), snap(height)
