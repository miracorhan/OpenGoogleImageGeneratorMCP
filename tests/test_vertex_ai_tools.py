import base64
import pytest
from unittest.mock import MagicMock, patch, AsyncMock

from vertex_ai_tools import (
    generate_image, edit_image, analyze_image,
    upscale_image, remove_background,
    _encode_base64, _save_image_bytes,
)

FAKE_PNG = b"\x89PNG\r\n\x1a\nfake-image-payload"
FAKE_B64 = base64.b64encode(FAKE_PNG).decode("utf-8")


@pytest.fixture(autouse=True)
def _clear_caches():
    from vertex_ai_tools import _gemini_model_cache
    _gemini_model_cache.clear()
    yield
    _gemini_model_cache.clear()


def test_encode_base64():
    assert _encode_base64(FAKE_PNG) == base64.b64encode(FAKE_PNG).decode("utf-8")


def test_save_image_bytes(tmp_path):
    out = str(tmp_path / "out.png")
    _save_image_bytes(FAKE_PNG, out)
    with open(out, "rb") as f:
        assert f.read() == FAKE_PNG


@pytest.mark.asyncio
@patch("vertex_ai_tools._imagen_predict")
async def test_generate_image_success(mock_predict, tmp_path):
    mock_predict.return_value = {"predictions": [{"bytesBase64Encoded": FAKE_B64, "mimeType": "image/png"}]}
    out = str(tmp_path / "img.png")
    result = await generate_image(
        prompt="a cat",
        output_path=out,
        model_name="imagen-4.0-fast-generate-001",
        return_base64=True,
    )
    assert result["success"] is True
    assert result["results"][0]["base64"] == FAKE_B64
    assert result["results"][0]["path"] == out


@pytest.mark.asyncio
@patch("vertex_ai_tools._imagen_predict")
async def test_generate_image_failure(mock_predict):
    mock_predict.side_effect = RuntimeError("HTTP 500: boom")
    result = await generate_image(prompt="a cat", model_name="imagen-4.0-fast-generate-001")
    assert result["success"] is False
    assert "boom" in result["error"]


@pytest.mark.asyncio
async def test_generate_image_rejects_non_imagen_model():
    result = await generate_image(prompt="x", model_name="gemini-2.5-flash")
    assert result["success"] is False
    assert "not a supported image-generation model" in result["error"]


@pytest.mark.asyncio
@patch("vertex_ai_tools.GenerativeModel")
async def test_analyze_image_success(mock_gen_model, tmp_path):
    img_path = tmp_path / "test.jpg"
    img_path.write_bytes(b"fake-data")
    mock_instance = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "This is a cat."
    mock_instance.generate_content_async = AsyncMock(return_value=mock_response)
    mock_gen_model.return_value = mock_instance

    result = await analyze_image(prompt="what is this?", image_path=str(img_path))
    assert result["success"] is True
    assert result["analysis"] == "This is a cat."


@pytest.mark.asyncio
async def test_analyze_image_file_not_found():
    result = await analyze_image(prompt="test", image_path="non_existent.jpg")
    assert result["success"] is False
    assert "not found" in result["error"]


@pytest.mark.asyncio
@patch("vertex_ai_tools._imagen_predict")
async def test_edit_image_success(mock_predict, tmp_path):
    mock_predict.return_value = {"predictions": [{"bytesBase64Encoded": FAKE_B64}]}
    base = tmp_path / "base.png"
    base.write_bytes(FAKE_PNG)
    out = str(tmp_path / "edited.png")

    result = await edit_image(
        prompt="add a hat",
        base_image_path=str(base),
        output_path=out,
        model_name="imagen-3.0-generate-002",
        return_base64=True,
    )
    assert result["success"] is True
    assert "base64" in result["results"][0]


@pytest.mark.asyncio
@patch("vertex_ai_tools._imagen_predict")
async def test_upscale_image_success(mock_predict, tmp_path):
    mock_predict.return_value = {"predictions": [{"bytesBase64Encoded": FAKE_B64}]}
    base = tmp_path / "low.png"
    base.write_bytes(FAKE_PNG)

    result = await upscale_image(
        base_image_path=str(base),
        model_name="imagen-3.0-generate-002",
        return_base64=True,
    )
    assert result["success"] is True


@pytest.mark.asyncio
@patch("vertex_ai_tools._imagen_predict")
async def test_remove_background_success(mock_predict, tmp_path):
    mock_predict.return_value = {"predictions": [{"bytesBase64Encoded": FAKE_B64}]}
    base = tmp_path / "subj.png"
    base.write_bytes(FAKE_PNG)

    result = await remove_background(
        base_image_path=str(base),
        model_name="imagen-3.0-generate-002",
        return_base64=True,
    )
    assert result["success"] is True
