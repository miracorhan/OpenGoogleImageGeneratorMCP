# Developer: Mirac Orhan (mirac.orhan@gmail.com)
# License: Open Source (MIT License)
import os
from io import BytesIO
from typing import Tuple

from PIL import Image

SUPPORTED_FORMATS = ("png", "jpeg", "webp", "avif")

_PIL_FORMAT = {"jpeg": "JPEG", "webp": "WEBP", "avif": "AVIF"}
_MIME = {
    "png": "image/png",
    "jpeg": "image/jpeg",
    "webp": "image/webp",
    "avif": "image/avif",
}
_EXT = {"png": ".png", "jpeg": ".jpg", "webp": ".webp", "avif": ".avif"}


def convert_image_bytes(image_bytes: bytes, to_format: str) -> Tuple[bytes, str]:
    """Convert image bytes to the target format. Returns (converted_bytes, mime_type)."""
    fmt = to_format.lower()
    if fmt not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported format: {fmt!r}. Valid options: {', '.join(SUPPORTED_FORMATS)}"
        )
    if fmt == "png":
        return image_bytes, _MIME["png"]
    if fmt == "avif":
        try:
            import pillow_avif  # noqa: F401
        except ImportError:
            raise RuntimeError(
                "AVIF output requires pillow-avif-plugin. "
                "Install with: pip install pillow-avif-plugin"
            )
    img = Image.open(BytesIO(image_bytes))
    if fmt == "jpeg" and img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGB")
    buf = BytesIO()
    img.save(buf, format=_PIL_FORMAT[fmt], quality=95)
    return buf.getvalue(), _MIME[fmt]


def save_with_format(image_bytes: bytes, base_output_path: str, to_format: str) -> Tuple[str, str]:
    """Convert and save image bytes. Returns (final_path, mime_type).

    Extension in base_output_path is replaced to match to_format.
    """
    fmt = to_format.lower()
    converted, mime = convert_image_bytes(image_bytes, fmt)
    base, _ = os.path.splitext(base_output_path)
    final_path = base + _EXT[fmt]
    os.makedirs(os.path.dirname(os.path.abspath(final_path)), exist_ok=True)
    with open(final_path, "wb") as f:
        f.write(converted)
    return final_path, mime
