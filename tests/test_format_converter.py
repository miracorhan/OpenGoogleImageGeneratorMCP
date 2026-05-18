import pytest
from io import BytesIO
from PIL import Image


def _png_bytes(mode: str = "RGB") -> bytes:
    color = (100, 150, 200, 255) if mode == "RGBA" else (100, 150, 200)
    img = Image.new(mode, (4, 4), color=color)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---- convert_image_bytes -----------------------------------------------

def test_png_passthrough():
    from format_converter import convert_image_bytes
    data = _png_bytes()
    result, mime = convert_image_bytes(data, "png")
    assert result is data
    assert mime == "image/png"


def test_to_webp_returns_webp_bytes():
    from format_converter import convert_image_bytes
    result, mime = convert_image_bytes(_png_bytes(), "webp")
    assert mime == "image/webp"
    img = Image.open(BytesIO(result))
    assert img.format == "WEBP"


def test_to_jpeg_strips_alpha():
    from format_converter import convert_image_bytes
    result, mime = convert_image_bytes(_png_bytes("RGBA"), "jpeg")
    assert mime == "image/jpeg"
    img = Image.open(BytesIO(result))
    assert img.mode == "RGB"


def test_unsupported_format_raises_value_error():
    from format_converter import convert_image_bytes
    with pytest.raises(ValueError, match="Unsupported format"):
        convert_image_bytes(b"data", "svg")


def test_case_insensitive_format():
    from format_converter import convert_image_bytes
    result, mime = convert_image_bytes(_png_bytes(), "WEBP")
    assert mime == "image/webp"


# ---- save_with_format --------------------------------------------------

def test_save_with_format_webp_creates_file(tmp_path):
    from format_converter import save_with_format
    data = _png_bytes()
    path, mime = save_with_format(data, str(tmp_path / "out.png"), "webp")
    assert path.endswith(".webp")
    assert mime == "image/webp"
    img = Image.open(path)
    assert img.format == "WEBP"


def test_save_with_format_jpeg(tmp_path):
    from format_converter import save_with_format
    data = _png_bytes()
    path, mime = save_with_format(data, str(tmp_path / "photo.png"), "jpeg")
    assert path.endswith(".jpg")
    assert mime == "image/jpeg"


def test_save_with_format_png_passthrough(tmp_path):
    from format_converter import save_with_format
    data = _png_bytes()
    path, mime = save_with_format(data, str(tmp_path / "img.png"), "png")
    assert path.endswith(".png")
    assert mime == "image/png"
    with open(path, "rb") as f:
        assert f.read() == data
