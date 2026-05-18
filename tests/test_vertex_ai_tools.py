import base64
import urllib.error
from io import BytesIO
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

import vertex_ai_tools
from vertex_ai_tools import (
    generate_image, edit_image, transform_image, analyze_image,
    upscale_image, remove_background,
    _encode_base64, _save_image_bytes,
    _handle_vertex_http_error, _build_validation_error,
    _validate_output_path,
    SUPPORTED_EDIT_MODES,
)

FAKE_PNG = b"\x89PNG\r\n\x1a\nfake-image-payload"
FAKE_B64 = base64.b64encode(FAKE_PNG).decode("utf-8")


@pytest.fixture(autouse=True)
def _clear_caches():
    from vertex_ai_tools import _gemini_model_cache
    _gemini_model_cache.clear()
    yield
    _gemini_model_cache.clear()


# ---- helpers ----------------------------------------------------------------

def _http_error(code: int, body: str = "{}", headers=None) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        url="https://example/test", code=code, msg="Test",
        hdrs=headers or {}, fp=BytesIO(body.encode("utf-8")),
    )


# ---- basic helpers ----------------------------------------------------------

def test_encode_base64():
    assert _encode_base64(FAKE_PNG) == base64.b64encode(FAKE_PNG).decode("utf-8")


def test_save_image_bytes(tmp_path):
    out = str(tmp_path / "out.png")
    _save_image_bytes(FAKE_PNG, out)
    with open(out, "rb") as f:
        assert f.read() == FAKE_PNG


# ---- generate_image ---------------------------------------------------------

@pytest.mark.asyncio
@patch("vertex_ai_tools._imagen_predict")
async def test_generate_image_success(mock_predict, tmp_path):
    mock_predict.return_value = {"predictions": [{"bytesBase64Encoded": FAKE_B64, "mimeType": "image/png"}]}
    out = str(tmp_path / "img.png")
    result = await generate_image(
        prompt="a cat", output_path=out,
        model_name="imagen-4.0-fast-generate-001", return_base64=True,
    )
    assert result["success"] is True
    assert result["results"][0]["base64"] == FAKE_B64
    assert result["results"][0]["path"] == out


@pytest.mark.asyncio
@patch("vertex_ai_tools._imagen_predict")
async def test_generate_image_http_404(mock_predict):
    from vertex_ai_tools import VertexAPIError
    err_dict = _handle_vertex_http_error(_http_error(404, '{"error":{"message":"not found"}}'),
                                          "imagen-4.0-fast-generate-001", ":predict", 0.4)
    mock_predict.side_effect = VertexAPIError(err_dict)
    result = await generate_image(prompt="a cat", model_name="imagen-4.0-fast-generate-001")
    assert result["success"] is False
    assert result["error"]["code"] == 404
    assert result["error"]["model"] == "imagen-4.0-fast-generate-001"
    assert "hint" in result["error"]


@pytest.mark.asyncio
async def test_generate_image_rejects_non_imagen_model():
    result = await generate_image(prompt="x", model_name="gemini-2.5-flash")
    assert result["success"] is False
    assert result["error"]["code"] == "VALIDATION"
    assert "not a supported image-generation model" in result["error"]["message"]


# ---- edit_image (Imagen 3 Capability) --------------------------------------

@pytest.mark.asyncio
@patch("vertex_ai_tools._imagen_predict")
async def test_edit_image_capability_default_mode(mock_predict, tmp_path):
    mock_predict.return_value = {"predictions": [{"bytesBase64Encoded": FAKE_B64}]}
    base = tmp_path / "base.png"; base.write_bytes(FAKE_PNG)
    out = str(tmp_path / "edited.png")

    result = await edit_image(
        prompt="make it snowy", base_image_path=str(base),
        output_path=out, model_name="imagen-3.0-capability-001",
        return_base64=True,
    )
    assert result["success"] is True
    # Payload assertion: must use referenceImages[] for capability model
    call_payload = mock_predict.call_args[0][1]
    assert "referenceImages" in call_payload["instances"][0]
    refs = call_payload["instances"][0]["referenceImages"]
    assert refs[0]["referenceType"] == "REFERENCE_TYPE_RAW"
    assert refs[0]["referenceImage"]["bytesBase64Encoded"] == FAKE_B64
    assert call_payload["parameters"]["editMode"] == "EDIT_MODE_DEFAULT"


@pytest.mark.asyncio
@patch("vertex_ai_tools._imagen_predict")
async def test_edit_image_capability_with_mask(mock_predict, tmp_path):
    mock_predict.return_value = {"predictions": [{"bytesBase64Encoded": FAKE_B64}]}
    base = tmp_path / "base.png"; base.write_bytes(FAKE_PNG)
    mask = tmp_path / "mask.png"; mask.write_bytes(b"\x89PNG\r\n\x1a\nmaskbytes")

    result = await edit_image(
        prompt="add a hat", base_image_path=str(base),
        mask_image_path=str(mask), edit_mode="EDIT_MODE_INPAINT_INSERTION",
        model_name="imagen-3.0-capability-001",
    )
    assert result["success"] is True
    refs = mock_predict.call_args[0][1]["instances"][0]["referenceImages"]
    assert len(refs) == 2
    assert refs[1]["referenceType"] == "REFERENCE_TYPE_MASK"
    assert refs[1]["maskImageConfig"]["maskMode"] == "MASK_MODE_USER_PROVIDED"


@pytest.mark.asyncio
async def test_edit_image_mask_required_validation(tmp_path):
    base = tmp_path / "base.png"; base.write_bytes(FAKE_PNG)
    result = await edit_image(
        prompt="remove a thing", base_image_path=str(base),
        edit_mode="EDIT_MODE_INPAINT_REMOVAL",
        model_name="imagen-3.0-capability-001",
    )
    assert result["success"] is False
    assert result["error"]["code"] == "VALIDATION"
    assert "requires mask_image_path" in result["error"]["message"]


@pytest.mark.asyncio
async def test_edit_image_unsupported_mode(tmp_path):
    base = tmp_path / "base.png"; base.write_bytes(FAKE_PNG)
    result = await edit_image(
        prompt="x", base_image_path=str(base),
        edit_mode="EDIT_MODE_BOGUS",
        model_name="imagen-3.0-capability-001",
    )
    assert result["success"] is False
    assert result["error"]["code"] == "VALIDATION"


@pytest.mark.asyncio
async def test_edit_image_base_not_found():
    result = await edit_image(prompt="x", base_image_path="nonexistent.png")
    assert result["success"] is False
    assert result["error"]["code"] == "VALIDATION"
    assert "not found" in result["error"]["message"]


@pytest.mark.asyncio
@patch("vertex_ai_tools._imagen_predict")
async def test_edit_image_legacy_model_uses_image_field(mock_predict, tmp_path):
    """imagen-3.0-generate-002 must use legacy 'image' field, not referenceImages."""
    mock_predict.return_value = {"predictions": [{"bytesBase64Encoded": FAKE_B64}]}
    base = tmp_path / "base.png"; base.write_bytes(FAKE_PNG)

    result = await edit_image(
        prompt="x", base_image_path=str(base),
        model_name="imagen-3.0-generate-002",
    )
    assert result["success"] is True
    instance = mock_predict.call_args[0][1]["instances"][0]
    assert "image" in instance
    assert "referenceImages" not in instance


@pytest.mark.asyncio
async def test_edit_image_legacy_model_rejects_mask(tmp_path):
    base = tmp_path / "base.png"; base.write_bytes(FAKE_PNG)
    mask = tmp_path / "mask.png"; mask.write_bytes(FAKE_PNG)
    result = await edit_image(
        prompt="x", base_image_path=str(base), mask_image_path=str(mask),
        model_name="imagen-3.0-generate-002",
    )
    assert result["success"] is False
    assert result["error"]["code"] == "VALIDATION"
    assert "does not support mask" in result["error"]["message"]


# ---- transform_image (Gemini Flash Image) ----------------------------------

@pytest.mark.asyncio
@patch("vertex_ai_tools._gemini_generate_content")
async def test_transform_image_success(mock_gen, tmp_path):
    mock_gen.return_value = {
        "candidates": [{"content": {"parts": [{"inlineData": {"mimeType": "image/png", "data": FAKE_B64}}]}}]
    }
    base = tmp_path / "base.png"; base.write_bytes(FAKE_PNG)
    out = str(tmp_path / "transformed.png")

    result = await transform_image(
        prompt="oil painting style", base_image_path=str(base),
        output_path=out, return_base64=True,
    )
    assert result["success"] is True
    assert result["results"][0]["base64"] == FAKE_B64
    assert result["results"][0]["path"] == out
    # Verify payload: parts must include inlineData first, then text
    call_args = mock_gen.call_args
    contents = call_args[0][1]
    parts = contents[0]["parts"]
    assert parts[0]["inlineData"]["data"] == FAKE_B64
    assert parts[-1]["text"] == "oil painting style"


@pytest.mark.asyncio
@patch("vertex_ai_tools._gemini_generate_content")
async def test_transform_image_with_additional_refs(mock_gen, tmp_path):
    mock_gen.return_value = {
        "candidates": [{"content": {"parts": [{"inlineData": {"data": FAKE_B64}}]}}]
    }
    base = tmp_path / "base.png"; base.write_bytes(FAKE_PNG)
    ref = tmp_path / "ref.png"; ref.write_bytes(FAKE_PNG)

    result = await transform_image(
        prompt="match this style", base_image_path=str(base),
        additional_image_paths=[str(ref)],
    )
    assert result["success"] is True
    parts = mock_gen.call_args[0][1][0]["parts"]
    # base + extra + text = 3 parts
    assert len(parts) == 3
    assert "inlineData" in parts[1]


@pytest.mark.asyncio
async def test_transform_image_base_not_found():
    result = await transform_image(prompt="x", base_image_path="nope.png")
    assert result["success"] is False
    assert result["error"]["code"] == "VALIDATION"


@pytest.mark.asyncio
@patch("vertex_ai_tools._gemini_generate_content")
async def test_transform_image_no_image_in_response(mock_gen, tmp_path):
    mock_gen.return_value = {"candidates": [{"content": {"parts": [{"text": "sorry, refused"}]}}]}
    base = tmp_path / "base.png"; base.write_bytes(FAKE_PNG)
    result = await transform_image(prompt="x", base_image_path=str(base))
    assert result["success"] is False
    assert "no image" in result["error"]["message"].lower()


# ---- _handle_vertex_http_error ----------------------------------------------

def test_handle_http_error_404_includes_hint():
    err = _handle_vertex_http_error(
        _http_error(404, '{"error":{"message":"Publisher Model not found"}}'),
        "gemini-9.9-fake", ":generateContent", 0.4,
    )
    assert err["code"] == 404
    assert err["model"] == "gemini-9.9-fake"
    assert "not found" in err["hint"].lower()
    assert err["docs_url"].startswith("https://")


def test_handle_http_error_401_says_reauth():
    err = _handle_vertex_http_error(_http_error(401, "{}"), "imagen-4.0-generate-001", ":predict", 0.1)
    assert err["code"] == 401
    assert "application-default login" in err["hint"]


def test_handle_http_error_403_says_iam():
    err = _handle_vertex_http_error(_http_error(403, "{}"), "imagen-4.0-generate-001", ":predict", 0.1)
    assert err["code"] == 403
    assert "aiplatform" in err["hint"].lower()


def test_handle_http_error_429_includes_retry_after():
    err = _handle_vertex_http_error(_http_error(429, "{}", headers={"Retry-After": "30"}),
                                     "gemini-2.5-flash", ":generateContent", 0.1)
    assert err["code"] == 429
    assert "Retry after 30" in err["hint"]


def test_handle_http_error_500_retryable():
    err = _handle_vertex_http_error(_http_error(503, "{}"), "imagen-4.0-generate-001", ":predict", 0.1)
    assert err["code"] == 503
    assert "retry" in err["hint"].lower()


def test_validation_error_structure():
    err = _build_validation_error("bad input")
    assert err["code"] == "VALIDATION"
    assert err["message"] == "bad input"


# ---- analyze / upscale / remove_background (regression) --------------------

@pytest.mark.asyncio
@patch("vertex_ai_tools.GenerativeModel")
async def test_analyze_image_success(mock_gen_model, tmp_path):
    img_path = tmp_path / "test.jpg"; img_path.write_bytes(b"fake-data")
    mock_instance = MagicMock()
    mock_response = MagicMock(); mock_response.text = "This is a cat."
    mock_instance.generate_content_async = AsyncMock(return_value=mock_response)
    mock_gen_model.return_value = mock_instance

    result = await analyze_image(prompt="what is this?", image_path=str(img_path))
    assert result["success"] is True
    assert result["analysis"] == "This is a cat."


@pytest.mark.asyncio
async def test_analyze_image_file_not_found():
    result = await analyze_image(prompt="test", image_path="non_existent.jpg")
    assert result["success"] is False
    assert result["error"]["code"] == "VALIDATION"


@pytest.mark.asyncio
@patch("vertex_ai_tools._imagen_predict")
async def test_upscale_image_success(mock_predict, tmp_path):
    mock_predict.return_value = {"predictions": [{"bytesBase64Encoded": FAKE_B64}]}
    base = tmp_path / "low.png"; base.write_bytes(FAKE_PNG)
    result = await upscale_image(base_image_path=str(base),
                                  model_name="imagen-3.0-generate-002",
                                  return_base64=True)
    assert result["success"] is True


@pytest.mark.asyncio
@patch("vertex_ai_tools._imagen_predict")
async def test_remove_background_success(mock_predict, tmp_path):
    mock_predict.return_value = {"predictions": [{"bytesBase64Encoded": FAKE_B64}]}
    base = tmp_path / "subj.png"; base.write_bytes(FAKE_PNG)
    result = await remove_background(base_image_path=str(base),
                                      model_name="imagen-3.0-capability-001",
                                      return_base64=True)
    assert result["success"] is True


# ---- constants --------------------------------------------------------------

def test_supported_edit_modes_includes_defaults():
    assert "EDIT_MODE_DEFAULT" in SUPPORTED_EDIT_MODES
    assert "EDIT_MODE_INPAINT_INSERTION" in SUPPORTED_EDIT_MODES
    assert "EDIT_MODE_OUTPAINT" in SUPPORTED_EDIT_MODES


# ---- _validate_output_path --------------------------------------------------

def test_validate_output_path_accepts_absolute():
    import os
    abs_path = os.path.abspath("outputs/test.png")
    result = _validate_output_path(abs_path)
    assert result == abs_path


def test_validate_output_path_accepts_dotdot_in_filename():
    import os
    abs_path = os.path.abspath("outputs/foo..bar.png")
    result = _validate_output_path(abs_path)
    assert result == abs_path


def test_validate_output_path_rejects_relative():
    with pytest.raises(ValueError, match="absolute"):
        _validate_output_path("relative/path/file.png")


def test_validate_output_path_rejects_dotdot():
    with pytest.raises(ValueError, match=r"\.\."):
        _validate_output_path("C:/outputs/../secret/file.png")


def test_validate_output_path_rejects_dotdot_backslash():
    with pytest.raises(ValueError, match=r"\.\."):
        _validate_output_path("C:\\outputs\\..\\secret\\file.png")


# ---- generate_image expanded params (Task 3) --------------------------------

@pytest.mark.asyncio
async def test_generate_image_passes_seed_to_payload():
    captured = {}
    def fake_predict(model_name, payload):
        captured["payload"] = payload
        return {"predictions": [{"bytesBase64Encoded": "iVBORw0KGgo="}]}

    with patch("vertex_ai_tools._imagen_predict", side_effect=fake_predict):
        result = await vertex_ai_tools.generate_image(
            prompt="test",
            seed=42,
            add_watermark=False,
        )
    assert captured["payload"]["parameters"]["seed"] == 42
    assert captured["payload"]["parameters"]["addWatermark"] is False

@pytest.mark.asyncio
async def test_generate_image_passes_negative_prompt():
    captured = {}
    def fake_predict(model_name, payload):
        captured["payload"] = payload
        return {"predictions": [{"bytesBase64Encoded": "iVBORw0KGgo="}]}

    with patch("vertex_ai_tools._imagen_predict", side_effect=fake_predict):
        await vertex_ai_tools.generate_image(prompt="test", negative_prompt="blurry")
    assert captured["payload"]["parameters"]["negativePrompt"] == "blurry"

@pytest.mark.asyncio
async def test_generate_image_enhance_prompt_default_true():
    captured = {}
    def fake_predict(model_name, payload):
        captured["payload"] = payload
        return {"predictions": [{"bytesBase64Encoded": "iVBORw0KGgo="}]}

    with patch("vertex_ai_tools._imagen_predict", side_effect=fake_predict):
        await vertex_ai_tools.generate_image(prompt="test")
    assert captured["payload"]["parameters"]["enhancePrompt"] is True


# ---- gemini_generate_image (Task 4) ----------------------------------------

@pytest.mark.asyncio
async def test_gemini_generate_image_returns_success():
    fake_b64 = "iVBORw0KGgo="
    fake_response = {
        "candidates": [{
            "content": {
                "parts": [{"inlineData": {"data": fake_b64, "mimeType": "image/png"}}]
            }
        }]
    }
    captured = {}
    def fake_generate(model_name, contents, generation_config):
        captured["model"] = model_name
        captured["generation_config"] = generation_config
        return fake_response

    with patch("vertex_ai_tools._gemini_generate_content", side_effect=fake_generate):
        result = await vertex_ai_tools.gemini_generate_image(
            prompt="a red apple",
            model_name="gemini-2.5-flash-image",
            return_base64=True,
        )
    assert result["success"] is True
    assert "results" in result
    assert result["results"][0]["base64"] == fake_b64
    assert captured["generation_config"]["responseModalities"] == ["IMAGE", "TEXT"]
    assert captured["model"] == "gemini-2.5-flash-image"

@pytest.mark.asyncio
async def test_gemini_generate_image_no_image_returns_failure():
    fake_response = {"candidates": [{"content": {"parts": [{"text": "I cannot generate images"}]}}]}
    with patch("vertex_ai_tools._gemini_generate_content", return_value=fake_response):
        result = await vertex_ai_tools.gemini_generate_image(
            prompt="test",
            model_name="gemini-2.5-flash-image",
        )
    assert result["success"] is False
