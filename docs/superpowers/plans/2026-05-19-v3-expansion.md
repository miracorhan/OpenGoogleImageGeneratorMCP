# v3 Expansion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add 4 new MCP tools (embed, analyze_video, generate_speech, live_generate) and WebP/AVIF image output format support, using google-genai SDK as a second backend alongside Vertex AI.

**Architecture:** Vertex AI SDK keeps Imagen/Veo/Lyria. `google-genai` handles embedding, TTS speech, video analysis, and live streaming. A new `format_converter.py` handles client-side PNG→WebP/AVIF conversion.

**Tech Stack:** Python, google-genai>=1.0.0, Pillow (existing), FastMCP (existing), pytest-asyncio (existing)

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `format_converter.py` | Create | PNG→WebP/AVIF/JPEG conversion via Pillow |
| `genai_tools.py` | Create | embed, analyze_video, generate_speech, live_generate |
| `model_registry.py` | Modify (append) | EMBED_MODEL_*, SPEECH_MODELS, VIDEO_ANALYZE_MODELS, LIVE_MODELS |
| `discovery.py` | Modify | Add embedding/speech/live_text to get_recommended_models() |
| `config.py` | Modify (append) | GOOGLE_GENAI_API_KEY, GOOGLE_GENAI_BACKEND env vars |
| `requirements.txt` | Modify | Add google-genai>=1.0.0 |
| `vertex_ai_tools.py` | Modify | save_format param + format_converter call in 4 functions |
| `mcp_server.py` | Modify | 4 new tool functions + save_format/output_format params on existing |
| `tests/test_format_converter.py` | Create | format_converter unit tests |
| `tests/test_genai_tools.py` | Create | genai_tools unit tests (mocked) |

---

## Task 1: Write Design Spec and Commit

**Files:**
- Already written: `docs/superpowers/specs/2026-05-19-v3-expansion-design.md`
- This plan: `docs/superpowers/plans/2026-05-19-v3-expansion.md`

- [ ] **Step 1.1: Commit spec and plan**

```bash
git add docs/superpowers/specs/2026-05-19-v3-expansion-design.md
git add docs/superpowers/plans/2026-05-19-v3-expansion.md
git commit -m "docs: add v3 expansion design spec and implementation plan"
```

Expected: 1 commit, 2 new files.

---

## Task 2: Update Dependencies and Config

**Files:**
- Modify: `requirements.txt`
- Modify: `config.py`

- [ ] **Step 2.1: Write the failing test for new config vars**

Add to a new file `tests/test_config_genai.py`:

```python
import os
import importlib


def test_genai_api_key_reads_from_env(monkeypatch):
    monkeypatch.setenv("GOOGLE_GENAI_API_KEY", "test-key-abc")
    import config
    importlib.reload(config)
    assert config.GOOGLE_GENAI_API_KEY == "test-key-abc"


def test_genai_backend_defaults_to_vertex_ai(monkeypatch):
    monkeypatch.delenv("GOOGLE_GENAI_BACKEND", raising=False)
    import config
    importlib.reload(config)
    assert config.GOOGLE_GENAI_BACKEND == "vertex_ai"


def test_genai_backend_reads_gemini_api(monkeypatch):
    monkeypatch.setenv("GOOGLE_GENAI_BACKEND", "gemini_api")
    import config
    importlib.reload(config)
    assert config.GOOGLE_GENAI_BACKEND == "gemini_api"
```

- [ ] **Step 2.2: Run test to verify it fails**

```bash
cd C:\Users\admin\source\repos\OpenGoogleImageGeneratorMCP
python -m pytest tests/test_config_genai.py -v
```

Expected: FAIL — `module 'config' has no attribute 'GOOGLE_GENAI_API_KEY'`

- [ ] **Step 2.3: Add env vars to config.py**

Append these two lines at the end of the settings section in `config.py`, after the `os.makedirs(DEFAULT_OUTPUT_DIR, ...)` call and before `def check_for_updates()`:

```python
# Google GenAI SDK settings
GOOGLE_GENAI_API_KEY = os.environ.get("GOOGLE_GENAI_API_KEY")
GOOGLE_GENAI_BACKEND = os.environ.get("GOOGLE_GENAI_BACKEND", "vertex_ai")
```

- [ ] **Step 2.4: Update requirements.txt**

Replace the contents of `requirements.txt` with:

```
google-cloud-aiplatform
google-genai>=1.0.0
mcp
Pillow
python-dotenv
pydantic
```

- [ ] **Step 2.5: Run tests to verify they pass**

```bash
python -m pytest tests/test_config_genai.py -v
```

Expected: 3 PASSED

- [ ] **Step 2.6: Commit**

```bash
git add requirements.txt config.py tests/test_config_genai.py
git commit -m "feat: add GOOGLE_GENAI_API_KEY and GOOGLE_GENAI_BACKEND config, add google-genai dependency"
```

---

## Task 3: Create format_converter.py

**Files:**
- Create: `format_converter.py`
- Create: `tests/test_format_converter.py`

- [ ] **Step 3.1: Write the failing tests**

Create `tests/test_format_converter.py`:

```python
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
```

- [ ] **Step 3.2: Run tests to verify they fail**

```bash
python -m pytest tests/test_format_converter.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'format_converter'`

- [ ] **Step 3.3: Create format_converter.py**

```python
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
```

- [ ] **Step 3.4: Run tests to verify they pass**

```bash
python -m pytest tests/test_format_converter.py -v
```

Expected: 8 PASSED

- [ ] **Step 3.5: Verify existing tests still pass**

```bash
python -m pytest --tb=short -q
```

Expected: 80+ PASSED, 0 FAILED

- [ ] **Step 3.6: Commit**

```bash
git add format_converter.py tests/test_format_converter.py
git commit -m "feat: add format_converter.py with PNG/JPEG/WebP/AVIF output support"
```

---

## Task 4: Update model_registry.py and discovery.py

**Files:**
- Modify: `model_registry.py`
- Modify: `discovery.py`
- Modify: `tests/test_model_registry.py`

- [ ] **Step 4.1: Write failing tests for new registry constants**

Append to `tests/test_model_registry.py`:

```python
def test_embed_model_vertex_is_defined():
    from model_registry import EMBED_MODEL_VERTEX
    assert EMBED_MODEL_VERTEX == "text-embedding-004"


def test_embed_model_gemini_api_is_defined():
    from model_registry import EMBED_MODEL_GEMINI_API
    assert EMBED_MODEL_GEMINI_API == "gemini-embedding-2"


def test_speech_models_has_fast_and_quality():
    from model_registry import SPEECH_MODELS
    assert "fast" in SPEECH_MODELS
    assert "quality" in SPEECH_MODELS
    assert "tts" in SPEECH_MODELS["fast"]
    assert "tts" in SPEECH_MODELS["quality"]


def test_video_analyze_models_has_fast_and_quality():
    from model_registry import VIDEO_ANALYZE_MODELS
    assert "fast" in VIDEO_ANALYZE_MODELS
    assert "quality" in VIDEO_ANALYZE_MODELS


def test_live_models_has_fast_and_quality():
    from model_registry import LIVE_MODELS
    assert "fast" in LIVE_MODELS
    assert "quality" in LIVE_MODELS
```

- [ ] **Step 4.2: Run test to verify it fails**

```bash
python -m pytest tests/test_model_registry.py -v -k "embed or speech or video_analyze or live"
```

Expected: FAIL — `ImportError: cannot import name 'EMBED_MODEL_VERTEX'`

- [ ] **Step 4.3: Append constants to model_registry.py**

Add the following block at the end of `model_registry.py` (after the existing `resolve_model` function):

```python

# ---------------------------------------------------------------------------
# Google GenAI SDK model constants
# Not part of the 4-tier Vertex AI system — used by genai_tools.py
# ---------------------------------------------------------------------------

EMBED_MODEL_VERTEX = "text-embedding-004"
EMBED_MODEL_GEMINI_API = "gemini-embedding-2"

SPEECH_MODELS = {
    "fast": "gemini-2.5-flash-preview-tts",
    "quality": "gemini-2.5-pro-preview-tts",
}

VIDEO_ANALYZE_MODELS = {
    "fast": "gemini-2.5-flash",
    "quality": "gemini-3.1-pro",
}

LIVE_MODELS = {
    "fast": "gemini-2.5-flash",
    "quality": "gemini-3.1-pro",
}
```

- [ ] **Step 4.4: Update discovery.py — add 3 new categories**

In `discovery.py`, inside `get_recommended_models()`, add three new keys after `"video_generation"`:

```python
        "embedding": [
            "text-embedding-004",
            "gemini-embedding-2",
        ],
        "speech": [
            "gemini-2.5-flash-preview-tts",
            "gemini-2.5-pro-preview-tts",
        ],
        "live_text": [
            "gemini-2.5-flash",
            "gemini-3.1-pro",
        ],
```

- [ ] **Step 4.5: Run tests to verify they pass**

```bash
python -m pytest tests/test_model_registry.py tests/test_discovery.py -v
```

Expected: All PASSED (existing + 5 new)

- [ ] **Step 4.6: Commit**

```bash
git add model_registry.py discovery.py tests/test_model_registry.py
git commit -m "feat: add GenAI SDK model constants and embedding/speech/live catalog entries"
```

---

## Task 5: Add save_format to vertex_ai_tools.py

This adds a `save_format` parameter to `gemini_generate_image`, `transform_image`, `upscale_image`, and `edit_image`. When `save_format != "png"`, the image bytes are passed through `format_converter.save_with_format()` instead of `_save_image_bytes()`. Also extends `generate_image()` to handle WEBP/AVIF by requesting PNG from Imagen and converting.

**Files:**
- Modify: `vertex_ai_tools.py`

- [ ] **Step 5.1: Write failing tests**

Add to `tests/test_vertex_ai_tools.py`:

```python
# ---- format conversion in gemini_generate_image -------------------------

@pytest.mark.asyncio
@patch("vertex_ai_tools.GenerativeModel")
async def test_gemini_generate_image_webp_output(mock_gm, tmp_path):
    from io import BytesIO
    from PIL import Image
    img = Image.new("RGB", (4, 4), color=(10, 20, 30))
    buf = BytesIO()
    img.save(buf, format="PNG")
    png_bytes = buf.getvalue()
    b64 = base64.b64encode(png_bytes).decode()

    mock_part = MagicMock()
    mock_part.text = None
    mock_part.inline_data = MagicMock()
    mock_part.inline_data.mime_type = "image/png"
    mock_part.inline_data.data = png_bytes

    mock_resp = MagicMock()
    mock_resp.candidates = [MagicMock(content=MagicMock(parts=[mock_part]))]

    mock_model = MagicMock()
    mock_model.generate_content.return_value = mock_resp
    mock_gm.return_value = mock_model

    from vertex_ai_tools import gemini_generate_image
    out = str(tmp_path / "img.png")
    result = await gemini_generate_image(
        prompt="a cat", output_path=out, save_format="webp"
    )
    assert result["success"] is True
    assert result["results"][0]["path"].endswith(".webp")
```

- [ ] **Step 5.2: Run test to verify it fails**

```bash
python -m pytest tests/test_vertex_ai_tools.py::test_gemini_generate_image_webp_output -v
```

Expected: FAIL — `gemini_generate_image() got unexpected keyword argument 'save_format'`

- [ ] **Step 5.3: Update gemini_generate_image() signature and save logic**

In `vertex_ai_tools.py`, change `gemini_generate_image` signature from:
```python
async def gemini_generate_image(
    prompt: str,
    output_path: Optional[str] = None,
    model_name: str = "gemini-2.5-flash-image",
    return_base64: bool = False,
) -> Dict[str, Any]:
```
to:
```python
async def gemini_generate_image(
    prompt: str,
    output_path: Optional[str] = None,
    model_name: str = "gemini-2.5-flash-image",
    return_base64: bool = False,
    save_format: str = "png",
) -> Dict[str, Any]:
```

Then replace the save block (the `if output_path:` section that calls `_save_image_bytes`) with:

```python
        if output_path:
            if save_format != "png":
                from format_converter import save_with_format
                final_path, mime = save_with_format(image_bytes, output_path, save_format)
                res["path"] = final_path
                res["mime_type"] = mime
            else:
                _save_image_bytes(image_bytes, output_path)
                res["path"] = output_path
```

- [ ] **Step 5.4: Update transform_image() signature and save logic**

Change `transform_image` signature — add `save_format: str = "png"` parameter:

```python
async def transform_image(
    prompt: str,
    base_image_path: str,
    output_path: Optional[str] = None,
    additional_image_paths: Optional[List[str]] = None,
    model_name: str = "gemini-2.5-flash-image",
    return_base64: bool = False,
    save_format: str = "png",
) -> Dict[str, Any]:
```

Replace the `if output_path:` save block in `transform_image` with:

```python
        if output_path:
            if save_format != "png":
                from format_converter import save_with_format
                final_path, mime = save_with_format(image_bytes, output_path, save_format)
                res["path"] = final_path
                res["mime_type"] = mime
            else:
                _save_image_bytes(image_bytes, output_path)
                res["path"] = output_path
```

- [ ] **Step 5.5: Update upscale_image() signature and save logic**

Change `upscale_image` signature — add `save_format: str = "png"` parameter:

```python
async def upscale_image(
    base_image_path: str,
    output_path: Optional[str] = None,
    model_name: str = "imagen-3.0-generate-002",
    return_base64: bool = False,
    save_format: str = "png",
) -> Dict[str, Any]:
```

Replace the `if output_path:` save block in `upscale_image` with:

```python
        if output_path:
            if save_format != "png":
                from format_converter import save_with_format
                final_path, mime = save_with_format(image_bytes, output_path, save_format)
                res["path"] = final_path
                res["mime_type"] = mime
            else:
                _save_image_bytes(image_bytes, output_path)
                res["path"] = output_path
```

- [ ] **Step 5.6: Update edit_image() signature and save logic**

Change `edit_image` signature — add `save_format: str = "png"` parameter:

```python
async def edit_image(
    prompt: str,
    base_image_path: str,
    output_path: Optional[str] = None,
    mask_image_path: Optional[str] = None,
    edit_mode: str = "EDIT_MODE_DEFAULT",
    model_name: str = "imagen-3.0-capability-001",
    negative_prompt: Optional[str] = None,
    sample_count: int = 1,
    return_base64: bool = False,
    save_format: str = "png",
) -> Dict[str, Any]:
```

Replace the `if output_path:` save block in `edit_image` (the loop that calls `_save_image_bytes`) with:

```python
            if output_path:
                current_out = output_path
                if len(images) > 1:
                    base, ext = os.path.splitext(output_path)
                    current_out = f"{base}_{i}{ext}"
                if save_format != "png":
                    from format_converter import save_with_format
                    final_path, mime = save_with_format(image_bytes, current_out, save_format)
                    res["path"] = final_path
                    res["mime_type"] = mime
                else:
                    _save_image_bytes(image_bytes, current_out)
                    res["path"] = current_out
```

- [ ] **Step 5.7: Extend generate_image() for WEBP/AVIF**

In `generate_image()`, change the API mime type line from:
```python
                "mimeType": f"image/{output_format.lower()}",
```
to:
```python
                "mimeType": "image/png" if output_format.upper() in ("WEBP", "AVIF")
                             else f"image/{output_format.lower()}",
```

Then replace the `if output_path:` save block inside the `for i, image_bytes in enumerate(images):` loop with:

```python
            if output_path:
                current_out = output_path
                if number_of_images > 1:
                    base, ext = os.path.splitext(output_path)
                    current_out = f"{base}_{i}{ext}"
                if output_format.upper() in ("WEBP", "AVIF"):
                    from format_converter import save_with_format
                    final_path, mime = save_with_format(
                        image_bytes, current_out, output_format.lower()
                    )
                    res["path"] = final_path
                    res["mime_type"] = mime
                else:
                    _save_image_bytes(image_bytes, current_out)
                    res["path"] = current_out
```

- [ ] **Step 5.8: Run all tests to verify nothing broke**

```bash
python -m pytest --tb=short -q
```

Expected: 80+ PASSED, 0 FAILED (new test from 5.1 also passes)

- [ ] **Step 5.9: Commit**

```bash
git add vertex_ai_tools.py tests/test_vertex_ai_tools.py
git commit -m "feat: add save_format/WEBP/AVIF support to image generation tools"
```

---

## Task 6: Create genai_tools.py

**Files:**
- Create: `genai_tools.py`
- Create: `tests/test_genai_tools.py`

- [ ] **Step 6.1: Write the failing tests**

Create `tests/test_genai_tools.py`:

```python
import os
import pytest
from unittest.mock import MagicMock, patch

import genai_tools


@pytest.fixture(autouse=True)
def reset_genai_client():
    genai_tools._genai_client = None
    yield
    genai_tools._genai_client = None


# ---- _get_genai_client -------------------------------------------------

def test_get_client_raises_without_project(monkeypatch):
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    monkeypatch.setenv("GOOGLE_GENAI_BACKEND", "vertex_ai")
    import config
    config.PROJECT_ID = None
    config.GOOGLE_GENAI_BACKEND = "vertex_ai"
    with patch("genai_tools.google_genai") as mock_genai:
        mock_genai.Client.side_effect = Exception("no project")
        with pytest.raises(Exception):
            genai_tools._get_genai_client()


# ---- embed -------------------------------------------------------------

@pytest.mark.asyncio
@patch("genai_tools._get_genai_client")
async def test_embed_success(mock_get_client):
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client

    mock_embedding = MagicMock()
    mock_embedding.values = [0.1, 0.2, 0.3]
    mock_result = MagicMock()
    mock_result.embeddings = [mock_embedding]
    mock_client.models.embed_content.return_value = mock_result

    result = await genai_tools.embed("hello world")
    assert result["success"] is True
    assert result["embedding"] == [0.1, 0.2, 0.3]
    assert result["dimension"] == 3
    assert "model" in result
    assert result["input_length"] == 11


@pytest.mark.asyncio
@patch("genai_tools._get_genai_client")
async def test_embed_api_failure_returns_error(mock_get_client):
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client
    mock_client.models.embed_content.side_effect = RuntimeError("API down")

    result = await genai_tools.embed("hello")
    assert result["success"] is False
    assert result["error"]["code"] == "RuntimeError"


# ---- analyze_video -----------------------------------------------------

@pytest.mark.asyncio
async def test_analyze_video_missing_file():
    result = await genai_tools.analyze_video("/nonexistent/video.mp4", "describe")
    assert result["success"] is False
    assert result["error"]["code"] == "VALIDATION"


@pytest.mark.asyncio
async def test_analyze_video_file_too_large(tmp_path, monkeypatch):
    fake_video = tmp_path / "big.mp4"
    fake_video.write_bytes(b"\x00")
    monkeypatch.setattr(os.path, "getsize", lambda p: 21 * 1024 * 1024)
    result = await genai_tools.analyze_video(str(fake_video), "describe")
    assert result["success"] is False
    assert result["error"]["code"] == "FILE_TOO_LARGE"


@pytest.mark.asyncio
@patch("genai_tools._get_genai_client")
async def test_analyze_video_success(mock_get_client, tmp_path):
    fake_video = tmp_path / "clip.mp4"
    fake_video.write_bytes(b"\x00" * 50)

    mock_client = MagicMock()
    mock_get_client.return_value = mock_client
    mock_response = MagicMock()
    mock_response.text = "A dog runs across the field."
    mock_client.models.generate_content.return_value = mock_response

    result = await genai_tools.analyze_video(str(fake_video), "What happens?")
    assert result["success"] is True
    assert result["analysis"] == "A dog runs across the field."
    assert "model" in result


# ---- generate_speech ---------------------------------------------------

@pytest.mark.asyncio
async def test_generate_speech_invalid_voice():
    result = await genai_tools.generate_speech("Hello", voice="InvalidVoice")
    assert result["success"] is False
    assert result["error"]["code"] == "VALIDATION"
    assert "InvalidVoice" in result["error"]["message"]


@pytest.mark.asyncio
@patch("genai_tools._get_genai_client")
async def test_generate_speech_success(mock_get_client, tmp_path):
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client

    mock_part = MagicMock()
    mock_part.inline_data.data = b"RIFF....fake-wav-bytes"
    mock_candidate = MagicMock()
    mock_candidate.content.parts = [mock_part]
    mock_response = MagicMock()
    mock_response.candidates = [mock_candidate]
    mock_client.models.generate_content.return_value = mock_response

    out = str(tmp_path / "speech.wav")
    result = await genai_tools.generate_speech("Hello world", output_path=out)
    assert result["success"] is True
    assert result["audio_path"] == out
    assert result["format"] == "wav"
    assert os.path.exists(out)


# ---- live_generate -----------------------------------------------------

@pytest.mark.asyncio
@patch("genai_tools._get_genai_client")
async def test_live_generate_success(mock_get_client):
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client

    chunk1 = MagicMock()
    chunk1.text = "Hello "
    chunk2 = MagicMock()
    chunk2.text = "world"
    mock_client.models.generate_content_stream.return_value = iter([chunk1, chunk2])

    result = await genai_tools.live_generate("Say hello")
    assert result["success"] is True
    assert result["text"] == "Hello world"
    assert result["stream_chunks"] == 2


@pytest.mark.asyncio
@patch("genai_tools._get_genai_client")
async def test_live_generate_api_failure(mock_get_client):
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client
    mock_client.models.generate_content_stream.side_effect = RuntimeError("Model down")

    result = await genai_tools.live_generate("test prompt")
    assert result["success"] is False
    assert result["error"]["code"] == "RuntimeError"


def test_available_voices_constant():
    from genai_tools import AVAILABLE_VOICES
    assert "Kore" in AVAILABLE_VOICES
    assert "Aoede" in AVAILABLE_VOICES
    assert len(AVAILABLE_VOICES) == 5
```

- [ ] **Step 6.2: Run tests to verify they fail**

```bash
python -m pytest tests/test_genai_tools.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'genai_tools'`

- [ ] **Step 6.3: Create genai_tools.py**

```python
# Developer: Mirac Orhan (mirac.orhan@gmail.com)
# License: Open Source (MIT License)
import asyncio
import functools
import os
import time
from typing import Any, Dict, Optional

from config import logger, PROJECT_ID, LOCATION, GOOGLE_GENAI_API_KEY, GOOGLE_GENAI_BACKEND

try:
    import google.genai as google_genai
    from google.genai import types as genai_types
    _GENAI_AVAILABLE = True
except ImportError:
    google_genai = None  # type: ignore
    genai_types = None   # type: ignore
    _GENAI_AVAILABLE = False

AVAILABLE_VOICES = ("Aoede", "Charon", "Fenrir", "Kore", "Puck")

_genai_client = None


def _get_genai_client():
    global _genai_client
    if _genai_client is not None:
        return _genai_client
    if not _GENAI_AVAILABLE:
        raise RuntimeError(
            "google-genai package is required for this tool. "
            "Install with: pip install google-genai>=1.0.0"
        )
    if GOOGLE_GENAI_BACKEND == "gemini_api" and GOOGLE_GENAI_API_KEY:
        _genai_client = google_genai.Client(api_key=GOOGLE_GENAI_API_KEY)
        logger.info("[genai] Initialized with Gemini API key")
    else:
        if not PROJECT_ID:
            raise RuntimeError(
                "GOOGLE_CLOUD_PROJECT not set. "
                "Required for Vertex AI backend of google-genai SDK."
            )
        _genai_client = google_genai.Client(
            vertexai=True, project=PROJECT_ID, location=LOCATION
        )
        logger.info(f"[genai] Initialized Vertex AI backend (project={PROJECT_ID}, location={LOCATION})")
    return _genai_client


async def _to_thread(func, *args, timeout: float = 60.0, **kwargs):
    loop = asyncio.get_running_loop()
    future = loop.run_in_executor(None, functools.partial(func, *args, **kwargs))
    done, _ = await asyncio.wait({future}, timeout=timeout)
    if not done:
        raise asyncio.TimeoutError()
    return future.result()


async def embed(text: str, model: Optional[str] = None) -> Dict[str, Any]:
    """Embed text into a float vector using Gemini Embedding."""
    from model_registry import EMBED_MODEL_VERTEX, EMBED_MODEL_GEMINI_API
    t0 = time.time()
    client = _get_genai_client()
    model_name = model or (
        EMBED_MODEL_GEMINI_API if GOOGLE_GENAI_BACKEND == "gemini_api" else EMBED_MODEL_VERTEX
    )
    logger.info(f"[embed] START | model={model_name} | text_len={len(text)}")
    try:
        result = await _to_thread(
            client.models.embed_content,
            model=model_name,
            contents=text,
            timeout=30.0,
        )
        embedding = list(result.embeddings[0].values)
        logger.info(f"[embed] SUCCESS in {time.time()-t0:.1f}s | dim={len(embedding)}")
        return {
            "success": True,
            "embedding": embedding,
            "dimension": len(embedding),
            "model": model_name,
            "input_length": len(text),
        }
    except Exception as e:
        logger.error(f"[embed] FAIL | {type(e).__name__}: {e}")
        return {"success": False, "error": {"code": type(e).__name__, "message": str(e)}}


def _video_mime_type(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    return {
        ".mp4": "video/mp4",
        ".mov": "video/quicktime",
        ".avi": "video/avi",
        ".webm": "video/webm",
        ".mkv": "video/x-matroska",
    }.get(ext, "video/mp4")


async def analyze_video(
    video_path: str,
    prompt: str,
    model_tier: str = "fast",
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """Analyze a local video file using Gemini Vision. Max 20MB inline."""
    from model_registry import VIDEO_ANALYZE_MODELS
    t0 = time.time()
    if not os.path.exists(video_path):
        return {
            "success": False,
            "error": {"code": "VALIDATION", "message": f"Video not found: {video_path}"},
        }
    file_size = os.path.getsize(video_path)
    MAX_INLINE = 20 * 1024 * 1024
    if file_size > MAX_INLINE:
        return {
            "success": False,
            "error": {
                "code": "FILE_TOO_LARGE",
                "message": (
                    f"Video ({file_size // (1024*1024)}MB) exceeds 20MB inline limit. "
                    "Upload to GCS and pass a gs:// URI instead."
                ),
            },
        }
    client = _get_genai_client()
    model_name = model or VIDEO_ANALYZE_MODELS.get(model_tier, VIDEO_ANALYZE_MODELS["fast"])
    mime_type = _video_mime_type(video_path)
    logger.info(f"[analyze_video] START | model={model_name} | path={video_path} | size={file_size}")
    try:
        with open(video_path, "rb") as f:
            video_bytes = f.read()
        video_part = genai_types.Part(
            inline_data=genai_types.Blob(mime_type=mime_type, data=video_bytes)
        )

        def _do_request():
            return client.models.generate_content(
                model=model_name,
                contents=[video_part, prompt],
            )

        response = await _to_thread(_do_request, timeout=120.0)
        analysis = response.text or ""
        logger.info(f"[analyze_video] SUCCESS in {time.time()-t0:.1f}s | text_len={len(analysis)}")
        return {
            "success": True,
            "analysis": analysis,
            "model": model_name,
            "video_path": video_path,
            "duration_s": round(time.time() - t0, 2),
        }
    except Exception as e:
        logger.error(f"[analyze_video] FAIL | {type(e).__name__}: {e}")
        return {"success": False, "error": {"code": type(e).__name__, "message": str(e)}}


async def generate_speech(
    text: str,
    voice: str = "Kore",
    output_path: Optional[str] = None,
    model_tier: str = "fast",
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """Convert text to speech using Gemini TTS. Returns a WAV file."""
    from model_registry import SPEECH_MODELS
    from config import DEFAULT_OUTPUT_DIR
    t0 = time.time()
    if voice not in AVAILABLE_VOICES:
        return {
            "success": False,
            "error": {
                "code": "VALIDATION",
                "message": f"Invalid voice '{voice}'. Choose from: {', '.join(AVAILABLE_VOICES)}",
            },
        }
    client = _get_genai_client()
    model_name = model or SPEECH_MODELS.get(model_tier, SPEECH_MODELS["fast"])
    if output_path is None:
        ts = int(t0)
        output_path = os.path.join(DEFAULT_OUTPUT_DIR, f"speech_{ts}.wav")
    logger.info(f"[generate_speech] START | model={model_name} | voice={voice} | text_len={len(text)}")
    try:
        def _do_request():
            return client.models.generate_content(
                model=model_name,
                contents=text,
                config=genai_types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=genai_types.SpeechConfig(
                        voice_config=genai_types.VoiceConfig(
                            prebuilt_voice_config=genai_types.PrebuiltVoiceConfig(
                                voice_name=voice
                            )
                        )
                    ),
                ),
            )

        response = await _to_thread(_do_request, timeout=60.0)
        audio_data = response.candidates[0].content.parts[0].inline_data.data
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(audio_data)
        logger.info(f"[generate_speech] SUCCESS in {time.time()-t0:.1f}s | path={output_path}")
        return {
            "success": True,
            "audio_path": output_path,
            "format": "wav",
            "voice": voice,
            "model": model_name,
            "duration_s": round(time.time() - t0, 2),
        }
    except Exception as e:
        logger.error(f"[generate_speech] FAIL | {type(e).__name__}: {e}")
        return {"success": False, "error": {"code": type(e).__name__, "message": str(e)}}


async def live_generate(
    prompt: str,
    model_tier: str = "fast",
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate text using Gemini streaming. Full response is accumulated before return."""
    from model_registry import LIVE_MODELS
    t0 = time.time()
    client = _get_genai_client()
    model_name = model or LIVE_MODELS.get(model_tier, LIVE_MODELS["fast"])
    logger.info(f"[live_generate] START | model={model_name} | prompt_len={len(prompt)}")
    try:
        def _do_stream():
            accumulated = ""
            chunks = 0
            for chunk in client.models.generate_content_stream(
                model=model_name,
                contents=prompt,
            ):
                if chunk.text:
                    accumulated += chunk.text
                    chunks += 1
            return accumulated, chunks

        text, chunks = await _to_thread(_do_stream, timeout=60.0)
        logger.info(f"[live_generate] SUCCESS in {time.time()-t0:.1f}s | chunks={chunks} | text_len={len(text)}")
        return {
            "success": True,
            "text": text,
            "model": model_name,
            "stream_chunks": chunks,
            "duration_s": round(time.time() - t0, 2),
        }
    except Exception as e:
        logger.error(f"[live_generate] FAIL | {type(e).__name__}: {e}")
        return {"success": False, "error": {"code": type(e).__name__, "message": str(e)}}
```

- [ ] **Step 6.4: Run tests to verify they pass**

```bash
python -m pytest tests/test_genai_tools.py -v
```

Expected: 12+ PASSED, 0 FAILED

- [ ] **Step 6.5: Run full suite**

```bash
python -m pytest --tb=short -q
```

Expected: All PASSED, 0 FAILED

- [ ] **Step 6.6: Commit**

```bash
git add genai_tools.py tests/test_genai_tools.py
git commit -m "feat: add genai_tools.py with embed, analyze_video, generate_speech, live_generate"
```

---

## Task 7: Update mcp_server.py — Register 4 New Tools and Format Params

**Files:**
- Modify: `mcp_server.py`

- [ ] **Step 7.1: Add import for genai_tools at the top of mcp_server.py**

After the `from pipeline import run_pipeline` import line, add:

```python
from genai_tools import embed, analyze_video, generate_speech, live_generate, AVAILABLE_VOICES
```

- [ ] **Step 7.2: Add save_format to TransformImageParams**

In `mcp_server.py`, in the `TransformImageParams` class, add the field:

```python
    save_format: Literal["png", "jpeg", "webp", "avif"] = Field(
        "png",
        description="Post-processing output format. WebP and AVIF are converted client-side from PNG.",
    )
```

Then in `tool_transform_image`, add `save_format=params.save_format` to the `transform_image(...)` call.

- [ ] **Step 7.3: Add save_format to EditImageParams**

In `EditImageParams`, add:

```python
    save_format: Literal["png", "jpeg", "webp", "avif"] = Field(
        "png",
        description="Post-processing output format. WebP and AVIF are converted client-side.",
    )
```

Then in `tool_edit_image`, add `save_format=params.save_format` to the `edit_image(...)` call.

- [ ] **Step 7.4: Extend output_format in GenerateImageParams to include WEBP and AVIF**

Change the existing field from:
```python
    output_format: Literal["PNG", "JPEG"] = Field("PNG", description="Output format.")
```
to:
```python
    output_format: Literal["PNG", "JPEG", "WEBP", "AVIF"] = Field(
        "PNG",
        description=(
            "Output format. PNG and JPEG are processed natively by Imagen. "
            "WEBP and AVIF are converted client-side from PNG."
        ),
    )
```

- [ ] **Step 7.5: Add the 4 new tool definitions**

Add the following 4 tool definitions at the end of `mcp_server.py`, before the `if __name__ == "__main__":` block (or at end of file):

```python

# ---------------------------------------------------------------------------
# GenAI SDK Tools — Embedding, Video Analysis, Speech, Live Generation
# ---------------------------------------------------------------------------

class EmbedParams(BaseModel):
    text: str = Field(..., description="Text to embed into a float vector using Gemini Embedding.")
    model: Optional[str] = Field(
        None,
        description="Override embedding model. Default: text-embedding-004 (Vertex AI) or gemini-embedding-2 (Gemini API).",
    )


@mcp.tool()
async def tool_embed(params: EmbedParams) -> dict:
    """
    Embed text into a float vector using Gemini Embedding.
    Useful for semantic similarity search, clustering, RAG pipelines,
    and nearest-neighbour lookup. Returns the full embedding vector.
    """
    return await embed(text=params.text, model=params.model)


class AnalyzeVideoParams(BaseModel):
    video_path: str = Field(
        ...,
        description="Absolute path to the video file (MP4, MOV, AVI, WebM). Max 20MB for inline processing.",
    )
    prompt: str = Field(
        ...,
        description="Analysis instruction, e.g. 'Describe what happens in this video'.",
    )
    model_tier: Literal["fast", "quality"] = Field(
        "fast",
        description="fast → gemini-2.5-flash, quality → gemini-3.1-pro.",
    )
    model: Optional[str] = Field(None, description="Override model name.")


@mcp.tool()
async def tool_analyze_video(params: AnalyzeVideoParams) -> dict:
    """
    Analyze a local video file using Gemini Vision.
    Supports scene description, object detection, action recognition, and content
    summarization. Video must be under 20MB; larger files require GCS storage.
    """
    return await analyze_video(
        video_path=params.video_path,
        prompt=params.prompt,
        model_tier=params.model_tier,
        model=params.model,
    )


class GenerateSpeechParams(BaseModel):
    text: str = Field(..., description="Text to convert to speech.")
    voice: Literal["Aoede", "Charon", "Fenrir", "Kore", "Puck"] = Field(
        "Kore",
        description="Voice name. Options: Aoede, Charon, Fenrir, Kore, Puck.",
    )
    output_filename: Optional[str] = Field(
        None,
        description="Filename to save the WAV file as (e.g. speech.wav). Saved to outputs/ directory.",
    )
    output_path: Optional[str] = Field(
        None,
        description="Absolute output path for the WAV file. Takes priority over output_filename.",
    )
    model_tier: Literal["fast", "quality"] = Field(
        "fast",
        description="fast → gemini-2.5-flash-preview-tts, quality → gemini-2.5-pro-preview-tts.",
    )


@mcp.tool()
async def tool_generate_speech(params: GenerateSpeechParams) -> dict:
    """
    Convert text to speech using Gemini TTS. Returns a WAV audio file.
    Available voices: Aoede, Charon, Fenrir, Kore, Puck.
    """
    if params.output_path:
        try:
            final_path = _validate_output_path(params.output_path)
        except ValueError as e:
            return {"success": False, "error": {"code": "VALIDATION", "message": str(e)}}
    elif params.output_filename:
        final_path = os.path.join(DEFAULT_OUTPUT_DIR, params.output_filename)
    else:
        final_path = None  # genai_tools will auto-name with timestamp

    return await generate_speech(
        text=params.text,
        voice=params.voice,
        output_path=final_path,
        model_tier=params.model_tier,
    )


class LiveGenerateParams(BaseModel):
    prompt: str = Field(..., description="Text prompt to generate a response for.")
    model_tier: Literal["fast", "quality"] = Field(
        "fast",
        description="fast → gemini-2.5-flash (lower latency), quality → gemini-3.1-pro.",
    )
    model: Optional[str] = Field(None, description="Override model name.")


@mcp.tool()
async def tool_live_generate(params: LiveGenerateParams) -> dict:
    """
    Generate text with Gemini using HTTP streaming (lower latency than standard generation).
    The full response is accumulated server-side and returned as a complete result.
    Use model_tier='quality' for more detailed responses.
    """
    return await live_generate(
        prompt=params.prompt,
        model_tier=params.model_tier,
        model=params.model,
    )
```

- [ ] **Step 7.6: Run full test suite**

```bash
python -m pytest --tb=short -q
```

Expected: All PASSED, 0 FAILED

- [ ] **Step 7.7: Verify MCP server starts without errors**

```bash
python -c "import mcp_server; print('OK — server imports clean')"
```

Expected: `OK — server imports clean` (no traceback)

- [ ] **Step 7.8: Commit**

```bash
git add mcp_server.py
git commit -m "feat: register 4 new MCP tools and extend format params in mcp_server.py"
```

---

## Task 8: Final Integration Check

- [ ] **Step 8.1: Run the complete test suite**

```bash
python -m pytest -v --tb=short
```

Expected:
- All original 80 tests: PASSED
- `test_format_converter.py`: 8 PASSED
- `test_genai_tools.py`: 12+ PASSED
- `test_model_registry.py`: 5+ new PASSED
- `test_config_genai.py`: 3 PASSED
- **Total: ~108+ PASSED, 0 FAILED**

- [ ] **Step 8.2: Check tool count in mcp_server.py**

```bash
python -c "
import mcp_server
tools = [t for t in dir(mcp_server) if t.startswith('tool_')]
print(f'Tool count: {len(tools)}')
print('\n'.join(sorted(tools)))
"
```

Expected output includes all 21 tools:
```
Tool count: 21
tool_analyze_image
tool_analyze_video        ← NEW
tool_batch_generate
tool_edit_image
tool_embed                ← NEW
tool_extend_video
tool_generate_image
tool_generate_music
tool_generate_speech      ← NEW
tool_generate_video
tool_image_to_video
tool_list_available_models
tool_live_generate        ← NEW
tool_remove_background
tool_run_pipeline
tool_transform_image
tool_upload_file
tool_upscale_image
tool_video_object_edit
```

- [ ] **Step 8.3: Verify format_converter with a real PNG**

```bash
python -c "
from io import BytesIO
from PIL import Image
from format_converter import convert_image_bytes

img = Image.new('RGB', (8, 8), color=(255, 0, 0))
buf = BytesIO(); img.save(buf, 'PNG')
png = buf.getvalue()

webp, mime = convert_image_bytes(png, 'webp')
assert mime == 'image/webp'
assert len(webp) > 0

jpeg, mime2 = convert_image_bytes(png, 'jpeg')
assert mime2 == 'image/jpeg'

print('format_converter: PNG→WebP and PNG→JPEG OK')
"
```

Expected: `format_converter: PNG→WebP and PNG→JPEG OK`

- [ ] **Step 8.4: Final commit with version bump**

In `config.py`, update `__version__` from `"1.0.0"` to `"3.0.0"`.

```bash
git add config.py
git commit -m "feat: bump version to 3.0.0 — v3 feature expansion complete (embed, analyze_video, generate_speech, live_generate, WebP/AVIF format support)"
```

---

## Summary

After all tasks:

| What | Before | After |
|---|---|---|
| MCP tools | 17 | 21 |
| New SDKs | 1 (Vertex AI) | 2 (+google-genai) |
| Output formats | PNG, JPEG | PNG, JPEG, WebP, AVIF |
| New files | — | format_converter.py, genai_tools.py |
| Tests | ~80 | ~108+ |
| Version | 1.0.0 | 3.0.0 |
