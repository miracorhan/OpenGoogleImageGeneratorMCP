# Feature Expansion v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add model_tier abstraction, full output_path control, expanded Imagen parameters, Veo video tools, batch generation, pipeline chaining, file upload, and music generation to OpenGoogleImageGeneratorMCP.

**Architecture:** New `model_registry.py` resolves tier strings to (model_name, api_backend) tuples; new `pipeline.py` chains tool calls sequentially; all new async functions follow the existing `vertex_ai_tools.py` pattern (`VertexAPIError`, `_build_*_error`, `{"success": bool, ...}` return shape). Video and music tools are production-structured stubs matching the existing `generate_video` stub pattern—real SDK integration is a follow-up.

**Tech Stack:** Python 3.10+, FastMCP, Pydantic v2, asyncio, Vertex AI REST API (urllib-based, no extra deps required)

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `model_registry.py` | **Create** | tier → (model_name, api_backend) resolution |
| `pipeline.py` | **Create** | Sequential step-chaining engine |
| `vertex_ai_tools.py` | **Modify** | New core async functions + updated signatures |
| `mcp_server.py` | **Modify** | New Params classes + tool registrations |
| `tests/test_model_registry.py` | **Create** | Unit tests for tier resolution |
| `tests/test_pipeline.py` | **Create** | Unit tests for pipeline engine |
| `tests/test_vertex_ai_tools.py` | **Modify** | Tests for new functions |

---

## Task 1: model_registry.py

**Files:**
- Create: `model_registry.py`
- Create: `tests/test_model_registry.py`

- [ ] **Step 1.1: Write failing tests**

```python
# tests/test_model_registry.py
import pytest
from model_registry import resolve_model, VALID_TIERS, VALID_TOOL_TYPES

def test_generate_fast_returns_imagen():
    model, backend = resolve_model("fast", "generate")
    assert model == "imagen-4.0-fast-generate-001"
    assert backend == "imagen"

def test_generate_balanced_returns_gemini():
    model, backend = resolve_model("balanced", "generate")
    assert model == "gemini-2.5-flash-image"
    assert backend == "gemini"

def test_generate_quality_returns_imagen():
    model, backend = resolve_model("quality", "generate")
    assert model == "imagen-4.0-generate-001"
    assert backend == "imagen"

def test_generate_ultra_returns_imagen_ultra():
    model, backend = resolve_model("ultra", "generate")
    assert model == "imagen-4.0-ultra-generate-001"
    assert backend == "imagen"

def test_transform_fast():
    model, backend = resolve_model("fast", "transform")
    assert model == "gemini-2.5-flash-image"
    assert backend == "gemini"

def test_transform_quality():
    model, backend = resolve_model("quality", "transform")
    assert model == "gemini-2.5-pro-image"
    assert backend == "gemini"

def test_video_fast():
    model, backend = resolve_model("fast", "video")
    assert model == "veo-3.1-fast-generate-001"
    assert backend == "veo"

def test_video_quality():
    model, backend = resolve_model("quality", "video")
    assert model == "veo-3.1-generate-001"
    assert backend == "veo"

def test_unknown_tier_falls_back_to_fast():
    model, backend = resolve_model("nonexistent", "generate")
    assert model == "imagen-4.0-fast-generate-001"
    assert backend == "imagen"

def test_unknown_tool_type_raises():
    with pytest.raises(ValueError, match="Unknown tool_type"):
        resolve_model("fast", "unknown_tool")

def test_valid_tiers_constant():
    assert "fast" in VALID_TIERS
    assert "balanced" in VALID_TIERS
    assert "quality" in VALID_TIERS
    assert "ultra" in VALID_TIERS

def test_valid_tool_types_constant():
    assert "generate" in VALID_TOOL_TYPES
    assert "transform" in VALID_TOOL_TYPES
    assert "video" in VALID_TOOL_TYPES
```

- [ ] **Step 1.2: Run tests to verify they fail**

```
cd C:\Users\admin\source\repos\OpenGoogleImageGeneratorMCP
python -m pytest tests/test_model_registry.py -v
```

Expected: `ModuleNotFoundError: No module named 'model_registry'`

- [ ] **Step 1.3: Create model_registry.py**

```python
# model_registry.py
# Developer: Mirac Orhan (mirac.orhan@gmail.com)
# License: Open Source (MIT License)
from typing import Tuple

VALID_TIERS = ("fast", "balanced", "quality", "ultra")
VALID_TOOL_TYPES = ("generate", "transform", "video")

_TIER_MAP: dict = {
    "generate": {
        "fast":     ("imagen-4.0-fast-generate-001",  "imagen"),
        "balanced": ("gemini-2.5-flash-image",         "gemini"),
        "quality":  ("imagen-4.0-generate-001",        "imagen"),
        "ultra":    ("imagen-4.0-ultra-generate-001",  "imagen"),
    },
    "transform": {
        "fast":     ("gemini-2.5-flash-image", "gemini"),
        "balanced": ("gemini-2.5-flash-image", "gemini"),
        "quality":  ("gemini-2.5-pro-image",   "gemini"),
        "ultra":    ("gemini-2.5-pro-image",   "gemini"),
    },
    "video": {
        "fast":     ("veo-3.1-fast-generate-001", "veo"),
        "balanced": ("veo-3.1-fast-generate-001", "veo"),
        "quality":  ("veo-3.1-generate-001",      "veo"),
        "ultra":    ("veo-3.1-generate-001",      "veo"),
    },
}


def resolve_model(tier: str, tool_type: str) -> Tuple[str, str]:
    """Return (model_name, api_backend) for the given tier and tool_type.

    Falls back to 'fast' if tier is unrecognized.
    Raises ValueError for unknown tool_type.
    """
    if tool_type not in _TIER_MAP:
        raise ValueError(f"Unknown tool_type '{tool_type}'. Valid: {list(_TIER_MAP)}")
    tool_map = _TIER_MAP[tool_type]
    return tool_map.get(tier, tool_map["fast"])
```

- [ ] **Step 1.4: Run tests to verify they pass**

```
python -m pytest tests/test_model_registry.py -v
```

Expected: all 12 tests PASS

- [ ] **Step 1.5: Commit**

```
git add model_registry.py tests/test_model_registry.py
git commit -m "feat: add model_registry with tier-to-model resolution"
```

---

## Task 2: _validate_output_path helper

**Files:**
- Modify: `vertex_ai_tools.py` (add helper after `_mime_for_path`)
- Modify: `tests/test_vertex_ai_tools.py` (add validation tests)

- [ ] **Step 2.1: Write failing tests**

Add to `tests/test_vertex_ai_tools.py`:

```python
from vertex_ai_tools import _validate_output_path
import os

def test_validate_output_path_accepts_absolute():
    # On Windows use a drive-letter path; on Linux use /tmp
    abs_path = os.path.abspath("outputs/test.png")
    result = _validate_output_path(abs_path)
    assert result == abs_path

def test_validate_output_path_rejects_relative():
    with pytest.raises(ValueError, match="absolute"):
        _validate_output_path("relative/path/file.png")

def test_validate_output_path_rejects_dotdot():
    with pytest.raises(ValueError, match="absolute"):
        _validate_output_path("C:/outputs/../secret/file.png")
```

- [ ] **Step 2.2: Run tests to verify they fail**

```
python -m pytest tests/test_vertex_ai_tools.py::test_validate_output_path_accepts_absolute -v
```

Expected: `ImportError` or `AttributeError` — `_validate_output_path` not yet defined.

- [ ] **Step 2.3: Add helper to vertex_ai_tools.py**

Add this function right after `_mime_for_path` (line ~104 in `vertex_ai_tools.py`):

```python
def _validate_output_path(path: str) -> str:
    """Ensure output_path is absolute and contains no '..' components."""
    if not os.path.isabs(path) or ".." in path.replace("\\", "/").split("/"):
        raise ValueError(
            f"output_path must be an absolute path without '..' components. Got: {path!r}"
        )
    return os.path.abspath(path)
```

- [ ] **Step 2.4: Run tests**

```
python -m pytest tests/test_vertex_ai_tools.py -k "validate_output_path" -v
```

Expected: all 3 tests PASS

- [ ] **Step 2.5: Commit**

```
git add vertex_ai_tools.py tests/test_vertex_ai_tools.py
git commit -m "feat: add _validate_output_path security helper"
```

---

## Task 3: generate_image — expanded parameters

**Files:**
- Modify: `vertex_ai_tools.py` — update `generate_image` signature and payload
- Modify: `mcp_server.py` — update `GenerateImageParams`

- [ ] **Step 3.1: Write failing tests**

Add to `tests/test_vertex_ai_tools.py`:

```python
import pytest
from unittest.mock import patch, AsyncMock
import asyncio

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
```

- [ ] **Step 3.2: Run tests to verify they fail**

```
python -m pytest tests/test_vertex_ai_tools.py -k "seed or negative_prompt or enhance_prompt" -v
```

Expected: FAIL — signature mismatch

- [ ] **Step 3.3: Update generate_image signature in vertex_ai_tools.py**

Replace the existing `generate_image` function signature and payload block (lines 483–501):

```python
async def generate_image(
    prompt: str,
    output_path: Optional[str] = None,
    model_name: str = "imagen-4.0-fast-generate-001",
    number_of_images: int = 1,
    aspect_ratio: str = "1:1",
    return_base64: bool = False,
    negative_prompt: Optional[str] = None,
    seed: Optional[int] = None,
    enhance_prompt: bool = True,
    add_watermark: bool = True,
    safety_setting: str = "block_medium_and_above",
    person_generation: str = "allow_adult",
    output_format: str = "PNG",
    compression_quality: int = 85,
    storage_uri: Optional[str] = None,
) -> Dict[str, Any]:
    t0 = time.time()
    logger.info(
        f"[generate_image] START | model={model_name} | prompt='{prompt[:80]}' | "
        f"aspect={aspect_ratio} | n={number_of_images}"
    )

    if not _is_imagen_model(model_name):
        return {"success": False, "error": _build_validation_error(_unsupported_image_model_error(model_name))}

    try:
        parameters: Dict[str, Any] = {
            "sampleCount": number_of_images,
            "aspectRatio": aspect_ratio,
            "enhancePrompt": enhance_prompt,
            "addWatermark": add_watermark,
            "safetySetting": safety_setting,
            "personGeneration": person_generation,
            "outputOptions": {
                "mimeType": f"image/{output_format.lower()}",
                "compressionQuality": compression_quality,
            },
        }
        if negative_prompt:
            parameters["negativePrompt"] = negative_prompt
        if seed is not None:
            parameters["seed"] = seed
        if storage_uri:
            parameters["storageUri"] = storage_uri

        payload = {
            "instances": [{"prompt": prompt}],
            "parameters": parameters,
        }
        # rest of function body unchanged from here
```

Keep the rest of the function body (response handling, save logic, error handling) identical to the original.

- [ ] **Step 3.4: Update GenerateImageParams in mcp_server.py**

Replace the existing `GenerateImageParams` class:

```python
class GenerateImageParams(BaseModel):
    prompt: str = Field(..., description="The text description of the image to generate.")
    output_filename: Optional[str] = Field(None, description="Filename to save the image as (e.g. image.png). Required if output_path not given.")
    output_path: Optional[str] = Field(None, description="Absolute path for the output file (e.g. C:/outputs/image.png). Takes priority over output_filename.")
    model_name: str = Field(
        "imagen-4.0-fast-generate-001",
        description="Imagen model. GA: imagen-4.0-fast-generate-001 (fast, default), imagen-4.0-generate-001 (quality), imagen-4.0-ultra-generate-001 (ultra), imagen-3.0-generate-002 (stable).",
    )
    model_tier: Optional[str] = Field(None, description="Shorthand tier: fast / balanced / quality / ultra. Overrides model_name when set. balanced → gemini-2.5-flash-image.")
    number_of_images: int = Field(1, ge=1, le=4, description="Number of images to generate.")
    aspect_ratio: str = Field("1:1", description="Aspect ratio (e.g., 1:1, 16:9, 4:3, 9:16).")
    return_base64: bool = Field(False, description="Whether to return the base64 encoded image.")
    negative_prompt: Optional[str] = Field(None, description="Elements to exclude from the image.")
    seed: Optional[int] = Field(None, description="Seed for deterministic output. Requires add_watermark=False.")
    enhance_prompt: bool = Field(True, description="Use LLM-based prompt rewriting for better results.")
    add_watermark: bool = Field(True, description="Add SynthID digital watermark. Must be False when seed is set.")
    safety_setting: str = Field("block_medium_and_above", description="Safety filter: block_low_and_above / block_medium_and_above / block_only_high.")
    person_generation: str = Field("allow_adult", description="Person generation policy: allow_all / allow_adult / dont_allow.")
    output_format: str = Field("PNG", description="Output format: PNG or JPEG.")
    compression_quality: int = Field(85, ge=0, le=100, description="JPEG compression quality (0-100). Only applies when output_format=JPEG.")
    storage_uri: Optional[str] = Field(None, description="Cloud Storage destination (e.g. gs://bucket/path/). Image is written directly to GCS.")
```

- [ ] **Step 3.5: Update tool_generate_image in mcp_server.py**

Replace the existing `tool_generate_image` function:

```python
@mcp.tool()
async def tool_generate_image(params: GenerateImageParams) -> dict:
    """
    Generate an image from a text prompt using Vertex AI Imagen or Gemini.
    Use model_tier for simple model selection: fast/balanced/quality/ultra.
    balanced routes to gemini-2.5-flash-image (Gemini API path).
    """
    from model_registry import resolve_model

    # Resolve output path
    if params.output_path:
        from vertex_ai_tools import _validate_output_path
        final_path = _validate_output_path(params.output_path)
    elif params.output_filename:
        final_path = os.path.join(DEFAULT_OUTPUT_DIR, params.output_filename)
    else:
        return {"success": False, "error": "Provide output_filename or output_path."}

    # Resolve model
    resolved_model = params.model_name
    api_backend = "imagen"
    if params.model_tier:
        resolved_model, api_backend = resolve_model(params.model_tier, "generate")

    if api_backend == "gemini":
        return await gemini_generate_image(
            prompt=params.prompt,
            output_path=final_path,
            model_name=resolved_model,
            return_base64=params.return_base64,
        )

    return await generate_image(
        prompt=params.prompt,
        output_path=final_path,
        model_name=resolved_model,
        number_of_images=params.number_of_images,
        aspect_ratio=params.aspect_ratio,
        return_base64=params.return_base64,
        negative_prompt=params.negative_prompt,
        seed=params.seed,
        enhance_prompt=params.enhance_prompt,
        add_watermark=params.add_watermark,
        safety_setting=params.safety_setting,
        person_generation=params.person_generation,
        output_format=params.output_format,
        compression_quality=params.compression_quality,
        storage_uri=params.storage_uri,
    )
```

Add the import at the top of `mcp_server.py`:
```python
from vertex_ai_tools import (
    generate_image, edit_image, transform_image, analyze_image,
    upscale_image, remove_background, generate_video,
    gemini_generate_image,           # ← add this
    image_to_video, extend_video, video_object_edit,  # ← add later in Tasks 7-9
    upload_file, batch_generate, generate_music,       # ← add later in Tasks 10-13
    probe_available_models, get_cached_availability,
    _validate_output_path,
    SUPPORTED_EDIT_MODES,
)
```

Note: Update the import incrementally as each function is added.

- [ ] **Step 3.6: Run tests**

```
python -m pytest tests/test_vertex_ai_tools.py -k "seed or negative_prompt or enhance_prompt" -v
```

Expected: all 3 tests PASS

- [ ] **Step 3.7: Commit**

```
git add vertex_ai_tools.py mcp_server.py
git commit -m "feat: expand generate_image with seed, negative_prompt, enhance_prompt, safety params"
```

---

## Task 4: gemini_generate_image (model_tier=balanced path)

**Files:**
- Modify: `vertex_ai_tools.py` — add `gemini_generate_image` function after `generate_image`

- [ ] **Step 4.1: Write failing test**

Add to `tests/test_vertex_ai_tools.py`:

```python
@pytest.mark.asyncio
async def test_gemini_generate_image_returns_success():
    fake_response = {
        "candidates": [{
            "content": {
                "parts": [{"inlineData": {"data": "iVBORw0KGgo=", "mimeType": "image/png"}}]
            }
        }]
    }
    with patch("vertex_ai_tools._gemini_generate_content", return_value=fake_response):
        result = await vertex_ai_tools.gemini_generate_image(
            prompt="a red apple",
            model_name="gemini-2.5-flash-image",
        )
    assert result["success"] is True
    assert "results" in result

@pytest.mark.asyncio
async def test_gemini_generate_image_no_image_returns_failure():
    fake_response = {"candidates": [{"content": {"parts": [{"text": "I cannot generate images"}]}}]}
    with patch("vertex_ai_tools._gemini_generate_content", return_value=fake_response):
        result = await vertex_ai_tools.gemini_generate_image(
            prompt="test",
            model_name="gemini-2.5-flash-image",
        )
    assert result["success"] is False
```

- [ ] **Step 4.2: Run tests to verify they fail**

```
python -m pytest tests/test_vertex_ai_tools.py -k "gemini_generate_image" -v
```

Expected: `AttributeError` — function not defined

- [ ] **Step 4.3: Add gemini_generate_image to vertex_ai_tools.py**

Add this function immediately after the `generate_image` function:

```python
async def gemini_generate_image(
    prompt: str,
    output_path: Optional[str] = None,
    model_name: str = "gemini-2.5-flash-image",
    return_base64: bool = False,
) -> Dict[str, Any]:
    """Text-to-image generation via Gemini multimodal models (no input image required).
    Used when model_tier='balanced' or a gemini-*-image model is specified directly.
    """
    t0 = time.time()
    logger.info(f"[gemini_generate_image] START | model={model_name} | prompt='{prompt[:80]}'")

    try:
        contents = [{"role": "user", "parts": [{"text": prompt}]}]
        generation_config = {"responseModalities": ["IMAGE", "TEXT"]}

        response = await _to_thread(
            _gemini_generate_content, model_name, contents, generation_config, timeout=API_TIMEOUT
        )
        image_bytes = _extract_gemini_image_bytes(response)
        if not image_bytes:
            text_fallback = _extract_gemini_text(response)
            err_msg = (
                f"Model returned no image. Text: {text_fallback[:200]!r}"
                if text_fallback else
                f"Model returned no image. Response keys: {list(response.keys())}"
            )
            return {"success": False, "error": _build_unexpected_error(
                model_name, ":generateContent", ValueError(err_msg), time.time() - t0
            )}

        res: Dict[str, Any] = {}
        if return_base64:
            res["base64"] = _encode_base64(image_bytes)
            res["mime_type"] = "image/png"
        if output_path:
            _save_image_bytes(image_bytes, output_path)
            res["path"] = output_path

        logger.info(f"[gemini_generate_image] SUCCESS in {time.time()-t0:.1f}s")
        return {"success": True, "results": [res]}

    except VertexAPIError as e:
        return {"success": False, "error": e.error_dict}
    except asyncio.TimeoutError:
        return {"success": False, "error": _build_timeout_error(model_name, ":generateContent", time.time() - t0)}
    except Exception as e:
        return {"success": False, "error": _build_unexpected_error(model_name, ":generateContent", e, time.time() - t0)}
```

- [ ] **Step 4.4: Update mcp_server.py import**

In `mcp_server.py`, add `gemini_generate_image` to the import from `vertex_ai_tools`.

- [ ] **Step 4.5: Run tests**

```
python -m pytest tests/test_vertex_ai_tools.py -k "gemini_generate_image" -v
```

Expected: both tests PASS

- [ ] **Step 4.6: Commit**

```
git add vertex_ai_tools.py mcp_server.py
git commit -m "feat: add gemini_generate_image for model_tier=balanced routing"
```

---

## Task 5: model_tier in edit_image and transform_image

**Files:**
- Modify: `mcp_server.py` — update `EditImageParams`, `TransformImageParams`, their tool functions

- [ ] **Step 5.1: Update EditImageParams**

In `mcp_server.py`, add to the existing `EditImageParams` class:

```python
model_tier: Optional[str] = Field(None, description="Shorthand tier: fast / quality. Overrides model_name.")
output_path: Optional[str] = Field(None, description="Absolute path for the output file. Takes priority over output_filename.")
```

Change `output_filename` to:
```python
output_filename: Optional[str] = Field(None, description="Filename to save the edited image as. Required if output_path not given.")
```

- [ ] **Step 5.2: Update tool_edit_image**

Replace the existing `tool_edit_image` function body:

```python
@mcp.tool()
async def tool_edit_image(params: EditImageParams) -> dict:
    """
    Precision image editing via Imagen 3 Capability.
    For free-form natural-language transforms, use tool_transform_image instead.
    """
    from model_registry import resolve_model

    if params.output_path:
        final_path = _validate_output_path(params.output_path)
    elif params.output_filename:
        final_path = os.path.join(DEFAULT_OUTPUT_DIR, params.output_filename)
    else:
        return {"success": False, "error": "Provide output_filename or output_path."}

    resolved_model = params.model_name
    if params.model_tier:
        resolved_model, _ = resolve_model(params.model_tier, "transform")

    return await edit_image(
        prompt=params.prompt,
        base_image_path=params.base_image_path,
        output_path=final_path,
        mask_image_path=params.mask_image_path,
        edit_mode=params.edit_mode,
        model_name=resolved_model,
        negative_prompt=params.negative_prompt,
        sample_count=params.sample_count,
        return_base64=params.return_base64,
    )
```

- [ ] **Step 5.3: Update TransformImageParams**

Add to the existing `TransformImageParams` class:

```python
model_tier: Optional[str] = Field(None, description="Shorthand tier: fast / quality. Overrides model_name.")
output_path: Optional[str] = Field(None, description="Absolute path for the output file. Takes priority over output_filename.")
```

Change `output_filename` to:
```python
output_filename: Optional[str] = Field(None, description="Filename to save transformed image. Required if output_path not given.")
```

- [ ] **Step 5.4: Update tool_transform_image**

```python
@mcp.tool()
async def tool_transform_image(params: TransformImageParams) -> dict:
    """
    Free-form 'image + text -> image' transformation via Gemini multimodal models.
    """
    from model_registry import resolve_model

    if params.output_path:
        final_path = _validate_output_path(params.output_path)
    elif params.output_filename:
        final_path = os.path.join(DEFAULT_OUTPUT_DIR, params.output_filename)
    else:
        return {"success": False, "error": "Provide output_filename or output_path."}

    resolved_model = params.model_name
    if params.model_tier:
        resolved_model, _ = resolve_model(params.model_tier, "transform")

    return await transform_image(
        prompt=params.prompt,
        base_image_path=params.base_image_path,
        output_path=final_path,
        additional_image_paths=params.additional_image_paths,
        model_name=resolved_model,
        return_base64=params.return_base64,
    )
```

- [ ] **Step 5.5: Apply output_path to remaining existing tools**

Apply the same output_path pattern to `UpscaleImageParams`, `RemoveBackgroundParams`, `AnalyzeImageParams` (analyze has no output), and their tool functions. For each:

1. Change `output_filename` from required to `Optional[str]`
2. Add `output_path: Optional[str]`
3. In the tool function, resolve final_path with the same if/elif/else block

- [ ] **Step 5.6: Run existing tests**

```
python -m pytest tests/ -v
```

Expected: all previously passing tests still PASS

- [ ] **Step 5.7: Commit**

```
git add mcp_server.py
git commit -m "feat: add model_tier and output_path to edit, transform, upscale, remove_background tools"
```

---

## Task 6: generate_video expansion

**Files:**
- Modify: `vertex_ai_tools.py` — update `generate_video` signature
- Modify: `mcp_server.py` — update `GenerateVideoParams`, `tool_generate_video`

- [ ] **Step 6.1: Write failing test**

Add to `tests/test_vertex_ai_tools.py`:

```python
@pytest.mark.asyncio
async def test_generate_video_accepts_new_params(tmp_path):
    out = str(tmp_path / "video.mp4")
    result = await vertex_ai_tools.generate_video(
        prompt="a sunset",
        output_path=out,
        duration=8,
        resolution="1080p",
        aspect_ratio="16:9",
        audio_enabled=True,
    )
    assert result["success"] is True
    assert result["duration"] == 8
    assert result["resolution"] == "1080p"
```

- [ ] **Step 6.2: Run to verify failure**

```
python -m pytest tests/test_vertex_ai_tools.py -k "generate_video_accepts" -v
```

Expected: FAIL — unexpected keyword arguments

- [ ] **Step 6.3: Update generate_video in vertex_ai_tools.py**

Replace the existing `generate_video` function:

```python
async def generate_video(
    prompt: str,
    output_path: str,
    model_name: str = "veo-3.1-fast-generate-001",
    duration: int = 4,
    resolution: str = "1080p",
    aspect_ratio: str = "16:9",
    audio_enabled: bool = False,
) -> Dict[str, Any]:
    t0 = time.time()
    logger.info(f"[generate_video] START | model={model_name} | prompt='{prompt[:80]}'")
    try:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w") as f:
            f.write(f"Simulated video | prompt={prompt} | duration={duration}s | {resolution} | {aspect_ratio}")
        logger.info(f"[generate_video] Placeholder written to {output_path} in {time.time()-t0:.1f}s")
        return {
            "success": True,
            "path": output_path,
            "duration": duration,
            "resolution": resolution,
            "aspect_ratio": aspect_ratio,
            "audio_enabled": audio_enabled,
            "note": "Placeholder stub — real Veo SDK integration pending.",
        }
    except Exception as e:
        return {"success": False, "error": _build_unexpected_error(model_name, ":predictLongRunning", e, time.time() - t0)}
```

- [ ] **Step 6.4: Update GenerateVideoParams and tool_generate_video in mcp_server.py**

Replace `GenerateVideoParams`:

```python
class GenerateVideoParams(BaseModel):
    prompt: str = Field(..., description="The text description of the video to generate.")
    output_filename: Optional[str] = Field(None, description="Output filename (e.g. video.mp4). Required if output_path not given.")
    output_path: Optional[str] = Field(None, description="Absolute output file path. Takes priority over output_filename.")
    model_name: str = Field(
        "veo-3.1-fast-generate-001",
        description="Veo model. GA: veo-3.1-fast-generate-001 (default, low latency), veo-3.1-generate-001 (premium).",
    )
    model_tier: Optional[str] = Field(None, description="fast / quality. Overrides model_name.")
    duration: int = Field(4, description="Video duration in seconds: 4, 6, or 8.")
    resolution: str = Field("1080p", description="Output resolution: 720p / 1080p / 4k.")
    aspect_ratio: str = Field("16:9", description="Aspect ratio: 16:9 (landscape) or 9:16 (portrait).")
    audio_enabled: bool = Field(False, description="Enable audio generation (Veo 3+ only).")
```

Replace `tool_generate_video`:

```python
@mcp.tool()
async def tool_generate_video(params: GenerateVideoParams) -> dict:
    """Generate a video from a text prompt using Veo."""
    from model_registry import resolve_model

    if params.output_path:
        final_path = _validate_output_path(params.output_path)
    elif params.output_filename:
        final_path = os.path.join(DEFAULT_OUTPUT_DIR, params.output_filename)
    else:
        return {"success": False, "error": "Provide output_filename or output_path."}

    resolved_model = params.model_name
    if params.model_tier:
        resolved_model, _ = resolve_model(params.model_tier, "video")

    return await generate_video(
        prompt=params.prompt,
        output_path=final_path,
        model_name=resolved_model,
        duration=params.duration,
        resolution=params.resolution,
        aspect_ratio=params.aspect_ratio,
        audio_enabled=params.audio_enabled,
    )
```

- [ ] **Step 6.5: Run tests**

```
python -m pytest tests/test_vertex_ai_tools.py -k "generate_video" -v
```

Expected: PASS

- [ ] **Step 6.6: Commit**

```
git add vertex_ai_tools.py mcp_server.py
git commit -m "feat: expand generate_video with duration, resolution, aspect_ratio, audio_enabled, model_tier"
```

---

## Task 7: tool_image_to_video

**Files:**
- Modify: `vertex_ai_tools.py` — add `image_to_video` function
- Modify: `mcp_server.py` — add `ImageToVideoParams` and `tool_image_to_video`

- [ ] **Step 7.1: Write failing test**

Add to `tests/test_vertex_ai_tools.py`:

```python
@pytest.mark.asyncio
async def test_image_to_video_rejects_missing_frame(tmp_path):
    result = await vertex_ai_tools.image_to_video(
        first_frame_path="/nonexistent/frame.png",
        output_path=str(tmp_path / "out.mp4"),
        prompt="pan right slowly",
    )
    assert result["success"] is False
    assert "not found" in result["error"]["message"].lower()

@pytest.mark.asyncio
async def test_image_to_video_success(tmp_path):
    frame = tmp_path / "frame.png"
    frame.write_bytes(b"fake-png")
    out = str(tmp_path / "video.mp4")
    result = await vertex_ai_tools.image_to_video(
        first_frame_path=str(frame),
        output_path=out,
        prompt="zoom in",
        duration=6,
    )
    assert result["success"] is True
    assert result["path"] == out
```

- [ ] **Step 7.2: Run to verify failure**

```
python -m pytest tests/test_vertex_ai_tools.py -k "image_to_video" -v
```

Expected: `AttributeError` — not defined

- [ ] **Step 7.3: Add image_to_video to vertex_ai_tools.py**

Add after `generate_video`:

```python
async def image_to_video(
    first_frame_path: str,
    output_path: str,
    prompt: str = "",
    last_frame_path: Optional[str] = None,
    model_name: str = "veo-3.1-fast-generate-001",
    duration: int = 4,
    aspect_ratio: str = "16:9",
) -> Dict[str, Any]:
    """Generate a video using an image as the first frame (optionally last frame too).

    Stub — real Veo image-to-video SDK integration pending.
    """
    t0 = time.time()
    logger.info(f"[image_to_video] START | model={model_name} | first_frame={first_frame_path}")

    if not os.path.exists(first_frame_path):
        return {"success": False, "error": _build_validation_error(f"First frame not found: {first_frame_path}")}
    if last_frame_path and not os.path.exists(last_frame_path):
        return {"success": False, "error": _build_validation_error(f"Last frame not found: {last_frame_path}")}

    try:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        mode = "first+last" if last_frame_path else "first-frame"
        with open(output_path, "w") as f:
            f.write(f"Simulated video | mode={mode} | prompt={prompt} | duration={duration}s | {aspect_ratio}")
        logger.info(f"[image_to_video] Placeholder written to {output_path} in {time.time()-t0:.1f}s")
        return {
            "success": True,
            "path": output_path,
            "duration": duration,
            "aspect_ratio": aspect_ratio,
            "mode": mode,
            "note": "Placeholder stub — real Veo image-to-video SDK integration pending.",
        }
    except Exception as e:
        return {"success": False, "error": _build_unexpected_error(model_name, ":predictLongRunning", e, time.time() - t0)}
```

- [ ] **Step 7.4: Add ImageToVideoParams and tool_image_to_video to mcp_server.py**

```python
class ImageToVideoParams(BaseModel):
    first_frame_path: str = Field(..., description="Absolute path to the image used as the first video frame.")
    prompt: str = Field("", description="Motion or scene description to guide video generation.")
    output_filename: Optional[str] = Field(None, description="Output filename (e.g. video.mp4). Required if output_path not given.")
    output_path: Optional[str] = Field(None, description="Absolute output file path. Takes priority over output_filename.")
    last_frame_path: Optional[str] = Field(None, description="Optional image for the last frame (first+last frame mode).")
    model_name: str = Field("veo-3.1-fast-generate-001", description="Veo model to use.")
    model_tier: Optional[str] = Field(None, description="fast / quality. Overrides model_name.")
    duration: int = Field(4, description="Video duration in seconds: 4, 6, or 8.")
    aspect_ratio: str = Field("16:9", description="Aspect ratio: 16:9 or 9:16.")

@mcp.tool()
async def tool_image_to_video(params: ImageToVideoParams) -> dict:
    """Generate a video from a still image as the first frame using Veo."""
    from model_registry import resolve_model

    if params.output_path:
        final_path = _validate_output_path(params.output_path)
    elif params.output_filename:
        final_path = os.path.join(DEFAULT_OUTPUT_DIR, params.output_filename)
    else:
        return {"success": False, "error": "Provide output_filename or output_path."}

    resolved_model = params.model_name
    if params.model_tier:
        resolved_model, _ = resolve_model(params.model_tier, "video")

    return await image_to_video(
        first_frame_path=params.first_frame_path,
        output_path=final_path,
        prompt=params.prompt,
        last_frame_path=params.last_frame_path,
        model_name=resolved_model,
        duration=params.duration,
        aspect_ratio=params.aspect_ratio,
    )
```

- [ ] **Step 7.5: Run tests**

```
python -m pytest tests/test_vertex_ai_tools.py -k "image_to_video" -v
```

Expected: both tests PASS

- [ ] **Step 7.6: Commit**

```
git add vertex_ai_tools.py mcp_server.py
git commit -m "feat: add tool_image_to_video (image-to-video stub via Veo)"
```

---

## Task 8: tool_extend_video

**Files:**
- Modify: `vertex_ai_tools.py` — add `extend_video`
- Modify: `mcp_server.py` — add `ExtendVideoParams`, `tool_extend_video`

- [ ] **Step 8.1: Write failing tests**

```python
@pytest.mark.asyncio
async def test_extend_video_rejects_missing_video(tmp_path):
    result = await vertex_ai_tools.extend_video(
        video_path="/nonexistent/video.mp4",
        output_path=str(tmp_path / "out.mp4"),
    )
    assert result["success"] is False
    assert "not found" in result["error"]["message"].lower()

@pytest.mark.asyncio
async def test_extend_video_success(tmp_path):
    vid = tmp_path / "input.mp4"
    vid.write_text("fake video data")
    out = str(tmp_path / "extended.mp4")
    result = await vertex_ai_tools.extend_video(
        video_path=str(vid),
        output_path=out,
        prompt="continue slowly",
        extra_seconds=6,
    )
    assert result["success"] is True
    assert result["extra_seconds"] == 6
```

- [ ] **Step 8.2: Run to verify failure**

```
python -m pytest tests/test_vertex_ai_tools.py -k "extend_video" -v
```

Expected: `AttributeError`

- [ ] **Step 8.3: Add extend_video to vertex_ai_tools.py**

Add after `image_to_video`:

```python
async def extend_video(
    video_path: str,
    output_path: str,
    prompt: str = "",
    extra_seconds: int = 4,
    model_name: str = "veo-3.1-fast-generate-001",
) -> Dict[str, Any]:
    """Extend an existing video by extra_seconds seconds.

    Stub — real Veo video-extension SDK integration pending.
    """
    t0 = time.time()
    logger.info(f"[extend_video] START | model={model_name} | video={video_path} | extra={extra_seconds}s")

    if not os.path.exists(video_path):
        return {"success": False, "error": _build_validation_error(f"Video not found: {video_path}")}
    if extra_seconds not in (4, 6, 8):
        return {"success": False, "error": _build_validation_error(
            f"extra_seconds must be 4, 6, or 8. Got: {extra_seconds}"
        )}

    try:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w") as f:
            f.write(f"Simulated extended video | source={video_path} | extra={extra_seconds}s | prompt={prompt}")
        logger.info(f"[extend_video] Placeholder written to {output_path} in {time.time()-t0:.1f}s")
        return {
            "success": True,
            "path": output_path,
            "extra_seconds": extra_seconds,
            "note": "Placeholder stub — real Veo extend SDK integration pending.",
        }
    except Exception as e:
        return {"success": False, "error": _build_unexpected_error(model_name, ":predictLongRunning", e, time.time() - t0)}
```

- [ ] **Step 8.4: Add ExtendVideoParams and tool_extend_video to mcp_server.py**

```python
class ExtendVideoParams(BaseModel):
    video_path: str = Field(..., description="Absolute path to the source video to extend.")
    output_filename: Optional[str] = Field(None, description="Output filename. Required if output_path not given.")
    output_path: Optional[str] = Field(None, description="Absolute output file path.")
    prompt: str = Field("", description="Optional motion description to guide the extension.")
    extra_seconds: int = Field(4, description="Seconds to add: 4, 6, or 8.")
    model_name: str = Field("veo-3.1-fast-generate-001", description="Veo model to use.")
    model_tier: Optional[str] = Field(None, description="fast / quality. Overrides model_name.")

@mcp.tool()
async def tool_extend_video(params: ExtendVideoParams) -> dict:
    """Extend an existing video by generating additional seconds at the end."""
    from model_registry import resolve_model

    if params.output_path:
        final_path = _validate_output_path(params.output_path)
    elif params.output_filename:
        final_path = os.path.join(DEFAULT_OUTPUT_DIR, params.output_filename)
    else:
        return {"success": False, "error": "Provide output_filename or output_path."}

    resolved_model = params.model_name
    if params.model_tier:
        resolved_model, _ = resolve_model(params.model_tier, "video")

    return await extend_video(
        video_path=params.video_path,
        output_path=final_path,
        prompt=params.prompt,
        extra_seconds=params.extra_seconds,
        model_name=resolved_model,
    )
```

- [ ] **Step 8.5: Run tests**

```
python -m pytest tests/test_vertex_ai_tools.py -k "extend_video" -v
```

Expected: both PASS

- [ ] **Step 8.6: Commit**

```
git add vertex_ai_tools.py mcp_server.py
git commit -m "feat: add tool_extend_video (video extension stub via Veo)"
```

---

## Task 9: tool_video_object_edit

**Files:**
- Modify: `vertex_ai_tools.py` — add `video_object_edit`
- Modify: `mcp_server.py` — add `VideoObjectEditParams`, `tool_video_object_edit`

- [ ] **Step 9.1: Write failing tests**

```python
@pytest.mark.asyncio
async def test_video_object_edit_rejects_invalid_operation(tmp_path):
    vid = tmp_path / "v.mp4"
    vid.write_text("data")
    result = await vertex_ai_tools.video_object_edit(
        video_path=str(vid),
        operation="replace",
        prompt="replace car with bike",
        output_path=str(tmp_path / "out.mp4"),
    )
    assert result["success"] is False
    assert "operation" in result["error"]["message"].lower()

@pytest.mark.asyncio
async def test_video_object_edit_insert_success(tmp_path):
    vid = tmp_path / "v.mp4"
    vid.write_text("data")
    out = str(tmp_path / "out.mp4")
    result = await vertex_ai_tools.video_object_edit(
        video_path=str(vid),
        operation="insert",
        prompt="add a red balloon",
        output_path=out,
    )
    assert result["success"] is True
    assert result["operation"] == "insert"
```

- [ ] **Step 9.2: Run to verify failure**

```
python -m pytest tests/test_vertex_ai_tools.py -k "video_object_edit" -v
```

Expected: `AttributeError`

- [ ] **Step 9.3: Add video_object_edit to vertex_ai_tools.py**

Add after `extend_video`:

```python
async def video_object_edit(
    video_path: str,
    operation: str,
    prompt: str,
    output_path: str,
    model_name: str = "veo-3.1-fast-generate-001",
) -> Dict[str, Any]:
    """Insert or remove an object in a video.

    operation: 'insert' | 'remove'
    Stub — real Veo object-edit SDK integration pending.
    """
    t0 = time.time()
    logger.info(f"[video_object_edit] START | op={operation} | model={model_name} | video={video_path}")

    if operation not in ("insert", "remove"):
        return {"success": False, "error": _build_validation_error(
            f"operation must be 'insert' or 'remove'. Got: {operation!r}"
        )}
    if not os.path.exists(video_path):
        return {"success": False, "error": _build_validation_error(f"Video not found: {video_path}")}

    try:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w") as f:
            f.write(f"Simulated video object edit | op={operation} | prompt={prompt}")
        logger.info(f"[video_object_edit] Placeholder written to {output_path} in {time.time()-t0:.1f}s")
        return {
            "success": True,
            "path": output_path,
            "operation": operation,
            "note": "Placeholder stub — real Veo object-edit SDK integration pending.",
        }
    except Exception as e:
        return {"success": False, "error": _build_unexpected_error(model_name, ":predictLongRunning", e, time.time() - t0)}
```

- [ ] **Step 9.4: Add VideoObjectEditParams and tool_video_object_edit to mcp_server.py**

```python
class VideoObjectEditParams(BaseModel):
    video_path: str = Field(..., description="Absolute path to the source video.")
    operation: str = Field(..., description="'insert' to add an object, 'remove' to delete one.")
    prompt: str = Field(..., description="Description of the object to insert or remove.")
    output_filename: Optional[str] = Field(None, description="Output filename. Required if output_path not given.")
    output_path: Optional[str] = Field(None, description="Absolute output file path.")
    model_name: str = Field("veo-3.1-fast-generate-001", description="Veo model to use.")
    model_tier: Optional[str] = Field(None, description="fast / quality. Overrides model_name.")

@mcp.tool()
async def tool_video_object_edit(params: VideoObjectEditParams) -> dict:
    """Insert or remove an object in a video using Veo."""
    from model_registry import resolve_model

    if params.output_path:
        final_path = _validate_output_path(params.output_path)
    elif params.output_filename:
        final_path = os.path.join(DEFAULT_OUTPUT_DIR, params.output_filename)
    else:
        return {"success": False, "error": "Provide output_filename or output_path."}

    resolved_model = params.model_name
    if params.model_tier:
        resolved_model, _ = resolve_model(params.model_tier, "video")

    return await video_object_edit(
        video_path=params.video_path,
        operation=params.operation,
        prompt=params.prompt,
        output_path=final_path,
        model_name=resolved_model,
    )
```

- [ ] **Step 9.5: Run tests**

```
python -m pytest tests/test_vertex_ai_tools.py -k "video_object_edit" -v
```

Expected: both PASS

- [ ] **Step 9.6: Commit**

```
git add vertex_ai_tools.py mcp_server.py
git commit -m "feat: add tool_video_object_edit (insert/remove objects in video, Veo stub)"
```

---

## Task 10: tool_upload_file

**Files:**
- Modify: `vertex_ai_tools.py` — add `upload_file`
- Modify: `mcp_server.py` — add `UploadFileParams`, `tool_upload_file`

- [ ] **Step 10.1: Write failing tests**

```python
@pytest.mark.asyncio
async def test_upload_file_rejects_missing_file():
    result = await vertex_ai_tools.upload_file(file_path="/nonexistent/image.png")
    assert result["success"] is False
    assert "not found" in result["error"]["message"].lower()

@pytest.mark.asyncio
async def test_upload_file_returns_metadata(tmp_path):
    f = tmp_path / "test.png"
    f.write_bytes(b"\x89PNG\r\n")
    result = await vertex_ai_tools.upload_file(file_path=str(f))
    assert result["success"] is True
    assert result["file_uri"].startswith("file://") or result["file_uri"].startswith("/")
    assert result["size_bytes"] == 6
    assert result["mime_type"] == "image/png"

@pytest.mark.asyncio
async def test_upload_file_autodetects_jpeg_mime(tmp_path):
    f = tmp_path / "photo.jpg"
    f.write_bytes(b"\xff\xd8\xff")
    result = await vertex_ai_tools.upload_file(file_path=str(f))
    assert result["mime_type"] == "image/jpeg"
```

- [ ] **Step 10.2: Run to verify failure**

```
python -m pytest tests/test_vertex_ai_tools.py -k "upload_file" -v
```

Expected: `AttributeError`

- [ ] **Step 10.3: Add upload_file to vertex_ai_tools.py**

Add after `video_object_edit`:

```python
async def upload_file(
    file_path: str,
    mime_type: Optional[str] = None,
    display_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Register a local file as a reusable reference for other tool calls.

    Returns a file_uri that can be passed to transform_image as additional_image_paths.
    Local-reference implementation — for GCS upload, set storage_uri in generate_image instead.
    """
    t0 = time.time()
    logger.info(f"[upload_file] START | path={file_path}")

    if not os.path.exists(file_path):
        return {"success": False, "error": _build_validation_error(f"File not found: {file_path}")}

    abs_path = os.path.abspath(file_path)
    size_bytes = os.path.getsize(abs_path)
    detected_mime = mime_type or _mime_for_path(abs_path)
    name = display_name or os.path.basename(abs_path)
    file_uri = abs_path  # Local reference; use as additional_image_paths value

    logger.info(f"[upload_file] SUCCESS in {time.time()-t0:.1f}s | size={size_bytes} | mime={detected_mime}")
    return {
        "success": True,
        "file_uri": file_uri,
        "name": name,
        "mime_type": detected_mime,
        "size_bytes": size_bytes,
        "note": "Local file reference. Pass file_uri as additional_image_paths in tool_transform_image.",
    }
```

- [ ] **Step 10.4: Add UploadFileParams and tool_upload_file to mcp_server.py**

```python
class UploadFileParams(BaseModel):
    file_path: str = Field(..., description="Absolute path to the local file to register.")
    mime_type: Optional[str] = Field(None, description="MIME type (auto-detected from extension if omitted).")
    display_name: Optional[str] = Field(None, description="Human-readable name for this file reference.")

@mcp.tool()
async def tool_upload_file(params: UploadFileParams) -> dict:
    """Register a local file for use as a reference image in other tools (e.g. tool_transform_image).
    Returns a file_uri to pass as additional_image_paths."""
    return await upload_file(
        file_path=params.file_path,
        mime_type=params.mime_type,
        display_name=params.display_name,
    )
```

- [ ] **Step 10.5: Run tests**

```
python -m pytest tests/test_vertex_ai_tools.py -k "upload_file" -v
```

Expected: all 3 PASS

- [ ] **Step 10.6: Commit**

```
git add vertex_ai_tools.py mcp_server.py
git commit -m "feat: add tool_upload_file for local file registration"
```

---

## Task 11: tool_batch_generate

**Files:**
- Modify: `vertex_ai_tools.py` — add `batch_generate`
- Modify: `mcp_server.py` — add `BatchGenerateParams`, `tool_batch_generate`

- [ ] **Step 11.1: Write failing tests**

```python
@pytest.mark.asyncio
async def test_batch_generate_rejects_too_many_prompts():
    result = await vertex_ai_tools.batch_generate(
        prompts=["p"] * 11,
        output_prefix="batch",
        output_dir="/tmp",
    )
    assert result["success"] is False
    assert "10" in result["error"]["message"]

@pytest.mark.asyncio
async def test_batch_generate_returns_per_prompt_results(tmp_path):
    fake_response = {"predictions": [{"bytesBase64Encoded": "iVBORw0KGgo="}]}
    with patch("vertex_ai_tools._imagen_predict", return_value=fake_response):
        result = await vertex_ai_tools.batch_generate(
            prompts=["cat", "dog"],
            output_prefix="item",
            output_dir=str(tmp_path),
        )
    assert result["success"] is True
    assert len(result["results"]) == 2
    assert result["results"][0]["prompt"] == "cat"
    assert result["results"][1]["prompt"] == "dog"

@pytest.mark.asyncio
async def test_batch_generate_partial_failure(tmp_path):
    call_count = 0
    def fake_predict(model_name, payload):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise Exception("API error")
        return {"predictions": [{"bytesBase64Encoded": "iVBORw0KGgo="}]}

    with patch("vertex_ai_tools._imagen_predict", side_effect=fake_predict):
        result = await vertex_ai_tools.batch_generate(
            prompts=["ok", "fail", "ok2"],
            output_prefix="b",
            output_dir=str(tmp_path),
        )
    assert result["success"] is True  # overall success even if some fail
    assert result["results"][0]["success"] is True
    assert result["results"][1]["success"] is False
    assert result["results"][2]["success"] is True
```

- [ ] **Step 11.2: Run to verify failure**

```
python -m pytest tests/test_vertex_ai_tools.py -k "batch_generate" -v
```

Expected: `AttributeError`

- [ ] **Step 11.3: Add batch_generate to vertex_ai_tools.py**

Add after `upload_file`:

```python
async def batch_generate(
    prompts: List[str],
    output_prefix: str,
    output_dir: Optional[str] = None,
    model_name: str = "imagen-4.0-fast-generate-001",
    aspect_ratio: str = "1:1",
) -> Dict[str, Any]:
    """Generate images for multiple prompts in parallel (max 4 concurrent)."""
    t0 = time.time()
    logger.info(f"[batch_generate] START | n={len(prompts)} | model={model_name}")

    if len(prompts) > 10:
        return {"success": False, "error": _build_validation_error(
            f"batch_generate accepts at most 10 prompts. Got: {len(prompts)}"
        )}

    resolved_dir = output_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    os.makedirs(resolved_dir, exist_ok=True)

    semaphore = asyncio.Semaphore(4)

    async def _one(prompt: str, idx: int) -> Dict[str, Any]:
        async with semaphore:
            out = os.path.join(resolved_dir, f"{output_prefix}_{idx}.png")
            result = await generate_image(
                prompt=prompt,
                output_path=out,
                model_name=model_name,
                aspect_ratio=aspect_ratio,
            )
            return {"prompt": prompt, "index": idx, **result}

    tasks = [_one(p, i) for i, p in enumerate(prompts)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    processed = []
    for r in results:
        if isinstance(r, Exception):
            processed.append({"success": False, "error": {"message": str(r)}})
        else:
            processed.append(r)

    logger.info(f"[batch_generate] DONE in {time.time()-t0:.1f}s | n={len(processed)}")
    return {"success": True, "results": processed, "count": len(processed)}
```

- [ ] **Step 11.4: Add BatchGenerateParams and tool_batch_generate to mcp_server.py**

```python
class BatchGenerateParams(BaseModel):
    prompts: List[str] = Field(..., description="List of text prompts (max 10).")
    output_prefix: str = Field(..., description="Filename prefix. Files: <prefix>_0.png, <prefix>_1.png, ...")
    output_dir: Optional[str] = Field(None, description="Absolute directory path for output. Defaults to DEFAULT_OUTPUT_DIR.")
    model_name: str = Field("imagen-4.0-fast-generate-001", description="Imagen model to use for all prompts.")
    model_tier: Optional[str] = Field(None, description="fast / balanced / quality / ultra. Overrides model_name.")
    aspect_ratio: str = Field("1:1", description="Aspect ratio for all generated images.")

@mcp.tool()
async def tool_batch_generate(params: BatchGenerateParams) -> dict:
    """Generate images for multiple prompts in a single call (max 10, max 4 concurrent)."""
    from model_registry import resolve_model

    resolved_model = params.model_name
    if params.model_tier:
        resolved_model, _ = resolve_model(params.model_tier, "generate")

    output_dir = params.output_dir or DEFAULT_OUTPUT_DIR

    return await batch_generate(
        prompts=params.prompts,
        output_prefix=params.output_prefix,
        output_dir=output_dir,
        model_name=resolved_model,
        aspect_ratio=params.aspect_ratio,
    )
```

- [ ] **Step 11.5: Run tests**

```
python -m pytest tests/test_vertex_ai_tools.py -k "batch_generate" -v
```

Expected: all 3 PASS

- [ ] **Step 11.6: Commit**

```
git add vertex_ai_tools.py mcp_server.py
git commit -m "feat: add tool_batch_generate with parallel execution (max 4 concurrent)"
```

---

## Task 12: pipeline.py + tool_run_pipeline

**Files:**
- Create: `pipeline.py`
- Create: `tests/test_pipeline.py`
- Modify: `mcp_server.py` — add `PipelineStep`, `RunPipelineParams`, `tool_run_pipeline`

- [ ] **Step 12.1: Write failing tests**

```python
# tests/test_pipeline.py
import pytest
import os
from unittest.mock import AsyncMock, patch, MagicMock

@pytest.mark.asyncio
async def test_pipeline_generate_then_upscale(tmp_path):
    from pipeline import run_pipeline

    gen_result = {"success": True, "results": [{"path": str(tmp_path / "step_0.png")}]}
    ups_result = {"success": True, "results": [{"path": str(tmp_path / "step_1.png")}]}

    # Create the intermediate file so upscale can "find" it
    (tmp_path / "step_0.png").write_bytes(b"fake")

    steps = [
        {"tool": "generate", "params": {"prompt": "red apple", "model_tier": "fast"}},
        {"tool": "upscale", "params": {}},
    ]

    with patch("pipeline.generate_image", new=AsyncMock(return_value=gen_result)), \
         patch("pipeline.upscale_image", new=AsyncMock(return_value=ups_result)):
        result = await run_pipeline(steps=steps, output_path=str(tmp_path / "final.png"), work_dir=str(tmp_path))

    assert result["success"] is True
    assert len(result["steps"]) == 2
    assert result["steps"][0]["success"] is True

@pytest.mark.asyncio
async def test_pipeline_stops_on_failure(tmp_path):
    from pipeline import run_pipeline

    gen_result = {"success": False, "error": {"message": "quota exceeded"}}

    steps = [
        {"tool": "generate", "params": {"prompt": "cat"}},
        {"tool": "upscale", "params": {}},
    ]

    with patch("pipeline.generate_image", new=AsyncMock(return_value=gen_result)):
        result = await run_pipeline(steps=steps, output_path=str(tmp_path / "out.png"), work_dir=str(tmp_path))

    assert result["success"] is False
    assert len(result["steps"]) == 1
    assert "quota" in result["error"].lower()

@pytest.mark.asyncio
async def test_pipeline_rejects_unknown_tool(tmp_path):
    from pipeline import run_pipeline
    steps = [{"tool": "fly", "params": {}}]
    result = await run_pipeline(steps=steps, output_path=str(tmp_path / "out.png"), work_dir=str(tmp_path))
    assert result["success"] is False
    assert "unknown" in result["error"].lower()
```

- [ ] **Step 12.2: Run to verify failure**

```
python -m pytest tests/test_pipeline.py -v
```

Expected: `ModuleNotFoundError: No module named 'pipeline'`

- [ ] **Step 12.3: Create pipeline.py**

```python
# pipeline.py
# Developer: Mirac Orhan (mirac.orhan@gmail.com)
# License: Open Source (MIT License)
import asyncio
import os
import uuid
import shutil
from typing import List, Dict, Any, Optional

from vertex_ai_tools import (
    generate_image, edit_image, transform_image,
    upscale_image, remove_background,
)
from model_registry import resolve_model
from config import DEFAULT_OUTPUT_DIR, logger

_TOOL_REGISTRY = {
    "generate":          generate_image,
    "edit":              edit_image,
    "transform":         transform_image,
    "upscale":           upscale_image,
    "remove_background": remove_background,
}

_FIRST_STEP_TOOLS = {"generate"}  # tools that create images without a base_image_path

_TOOL_TYPE_FOR_TIER = {
    "generate":          "generate",
    "edit":              "transform",
    "transform":         "transform",
    "upscale":           "generate",
    "remove_background": "generate",
}


def _resolve_step_params(tool_name: str, step_params: dict) -> dict:
    """Resolve model_tier → model_name so vertex_ai_tools functions receive clean kwargs."""
    params = dict(step_params)
    if "model_tier" in params:
        tier = params.pop("model_tier")
        tool_type = _TOOL_TYPE_FOR_TIER.get(tool_name, "generate")
        model_name, _ = resolve_model(tier, tool_type)
        params.setdefault("model_name", model_name)
    return params


async def run_pipeline(
    steps: List[Dict[str, Any]],
    output_path: Optional[str] = None,
    work_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute steps sequentially; pipe each step's output into the next as base_image_path.

    Returns {"success": bool, "steps": [...], "final_path": str|None, "error": str|None}
    """
    if not steps:
        return {"success": False, "steps": [], "final_path": None, "error": "No steps provided."}

    run_id = uuid.uuid4().hex[:8]
    temp_dir = work_dir or os.path.join(DEFAULT_OUTPUT_DIR, f"pipeline_{run_id}")
    os.makedirs(temp_dir, exist_ok=True)
    cleanup_temp = work_dir is None  # Only clean up dirs we created

    step_results: List[Dict[str, Any]] = []
    current_image_path: Optional[str] = None

    try:
        for i, step in enumerate(steps):
            tool_name = step.get("tool", "")
            step_params = dict(step.get("params", {}))

            if tool_name not in _TOOL_REGISTRY:
                return {
                    "success": False,
                    "steps": step_results,
                    "final_path": None,
                    "error": f"Unknown tool '{tool_name}' at step {i}. Valid: {list(_TOOL_REGISTRY)}",
                }

            is_last_step = (i == len(steps) - 1)
            step_output = output_path if is_last_step and output_path else os.path.join(temp_dir, f"step_{i}.png")

            # Wire base_image_path from previous step (skip for first-step tools)
            if current_image_path and tool_name not in _FIRST_STEP_TOOLS:
                step_params.setdefault("base_image_path", current_image_path)

            step_params["output_path"] = step_output

            logger.info(f"[pipeline:{run_id}] step {i}/{len(steps)-1} tool={tool_name}")
            fn = _TOOL_REGISTRY[tool_name]
            resolved_params = _resolve_step_params(tool_name, step_params)
            result = await fn(**resolved_params)

            step_results.append({"step": i, "tool": tool_name, **result})

            if not result.get("success"):
                return {
                    "success": False,
                    "steps": step_results,
                    "final_path": None,
                    "error": f"Step {i} ({tool_name}) failed: {result.get('error', {}).get('message', 'unknown error')}",
                }

            # Extract output path for next step
            results_list = result.get("results", [])
            if results_list and results_list[0].get("path"):
                current_image_path = results_list[0]["path"]
            else:
                current_image_path = step_output

        return {
            "success": True,
            "steps": step_results,
            "final_path": current_image_path,
            "error": None,
        }

    finally:
        if cleanup_temp and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
```

- [ ] **Step 12.4: Add RunPipelineParams and tool_run_pipeline to mcp_server.py**

```python
from pipeline import run_pipeline  # add to imports at top of mcp_server.py

class PipelineStepModel(BaseModel):
    tool: str = Field(..., description="Tool name: generate / edit / transform / upscale / remove_background")
    params: dict = Field(default_factory=dict, description="Parameters for this tool step (excluding output path, which is managed by the pipeline).")

class RunPipelineParams(BaseModel):
    steps: List[PipelineStepModel] = Field(..., description="Ordered list of pipeline steps.")
    output_path: Optional[str] = Field(None, description="Absolute path for the final output image.")
    output_filename: Optional[str] = Field(None, description="Output filename saved to DEFAULT_OUTPUT_DIR. Used if output_path not given.")

@mcp.tool()
async def tool_run_pipeline(params: RunPipelineParams) -> dict:
    """Chain image processing steps sequentially. Each step's output becomes the next step's input.
    Supported tools: generate, edit, transform, upscale, remove_background.
    Example: generate → remove_background → upscale."""

    if params.output_path:
        final_path = _validate_output_path(params.output_path)
    elif params.output_filename:
        final_path = os.path.join(DEFAULT_OUTPUT_DIR, params.output_filename)
    else:
        final_path = None  # pipeline will use temp dir for intermediates

    steps_dicts = [{"tool": s.tool, "params": s.params} for s in params.steps]
    return await run_pipeline(steps=steps_dicts, output_path=final_path)
```

- [ ] **Step 12.5: Run tests**

```
python -m pytest tests/test_pipeline.py -v
```

Expected: all 3 PASS

- [ ] **Step 12.6: Run full test suite**

```
python -m pytest tests/ -v
```

Expected: all tests PASS

- [ ] **Step 12.7: Commit**

```
git add pipeline.py tests/test_pipeline.py mcp_server.py
git commit -m "feat: add pipeline.py engine and tool_run_pipeline for sequential image processing"
```

---

## Task 13: tool_generate_music

**Files:**
- Modify: `vertex_ai_tools.py` — add `generate_music`
- Modify: `mcp_server.py` — add `GenerateMusicParams`, `tool_generate_music`

- [ ] **Step 13.1: Write failing tests**

```python
@pytest.mark.asyncio
async def test_generate_music_rejects_invalid_model():
    result = await vertex_ai_tools.generate_music(
        prompt="upbeat jazz",
        output_path="/tmp/track.mp3",
        model_name="lyria-99",
    )
    assert result["success"] is False
    assert "lyria" in result["error"]["message"].lower()

@pytest.mark.asyncio
async def test_generate_music_success(tmp_path):
    out = str(tmp_path / "track.mp3")
    result = await vertex_ai_tools.generate_music(
        prompt="calm piano",
        output_path=out,
        model_name="lyria-2",
        duration=30,
    )
    assert result["success"] is True
    assert result["path"] == out
    assert result["duration"] == 30
```

- [ ] **Step 13.2: Run to verify failure**

```
python -m pytest tests/test_vertex_ai_tools.py -k "generate_music" -v
```

Expected: `AttributeError`

- [ ] **Step 13.3: Add generate_music to vertex_ai_tools.py**

Add at the end of `vertex_ai_tools.py`:

```python
_SUPPORTED_MUSIC_MODELS = ("lyria-2", "lyria-3")


async def generate_music(
    prompt: str,
    output_path: str,
    model_name: str = "lyria-2",
    duration: Optional[int] = None,
) -> Dict[str, Any]:
    """Generate music from a text prompt using Lyria.

    Stub — Lyria API availability is project-dependent.
    Run tool_list_available_models to check if Lyria is enabled in your project.
    """
    t0 = time.time()
    logger.info(f"[generate_music] START | model={model_name} | prompt='{prompt[:80]}'")

    if model_name not in _SUPPORTED_MUSIC_MODELS:
        return {"success": False, "error": _build_validation_error(
            f"Unsupported music model '{model_name}'. Use one of: {', '.join(_SUPPORTED_MUSIC_MODELS)}"
        )}

    try:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w") as f:
            f.write(f"Simulated music | model={model_name} | prompt={prompt} | duration={duration}s")
        logger.info(f"[generate_music] Placeholder written to {output_path} in {time.time()-t0:.1f}s")
        return {
            "success": True,
            "path": output_path,
            "model": model_name,
            "duration": duration,
            "note": "Placeholder stub — Lyria SDK integration pending. Check tool_list_available_models for Lyria availability.",
        }
    except Exception as e:
        return {"success": False, "error": _build_unexpected_error(model_name, ":predict", e, time.time() - t0)}
```

- [ ] **Step 13.4: Add GenerateMusicParams and tool_generate_music to mcp_server.py**

```python
class GenerateMusicParams(BaseModel):
    prompt: str = Field(..., description="Text description of the music to generate.")
    output_filename: Optional[str] = Field(None, description="Output filename (e.g. track.mp3). Required if output_path not given.")
    output_path: Optional[str] = Field(None, description="Absolute output file path.")
    model_name: str = Field("lyria-2", description="Lyria model: lyria-2 (default) or lyria-3.")
    duration: Optional[int] = Field(None, description="Desired music duration in seconds (optional).")

@mcp.tool()
async def tool_generate_music(params: GenerateMusicParams) -> dict:
    """Generate music from a text prompt using Lyria. Check tool_list_available_models to verify Lyria is enabled in your project."""

    if params.output_path:
        final_path = _validate_output_path(params.output_path)
    elif params.output_filename:
        final_path = os.path.join(DEFAULT_OUTPUT_DIR, params.output_filename)
    else:
        return {"success": False, "error": "Provide output_filename or output_path."}

    return await generate_music(
        prompt=params.prompt,
        output_path=final_path,
        model_name=params.model_name,
        duration=params.duration,
    )
```

- [ ] **Step 13.5: Run tests**

```
python -m pytest tests/test_vertex_ai_tools.py -k "generate_music" -v
```

Expected: both PASS

- [ ] **Step 13.6: Run full test suite**

```
python -m pytest tests/ -v
```

Expected: all tests PASS

- [ ] **Step 13.7: Final commit**

```
git add vertex_ai_tools.py mcp_server.py
git commit -m "feat: add tool_generate_music with Lyria 2/3 stub"
```

---

## Final Verification

- [ ] Run the full test suite one last time:

```
python -m pytest tests/ -v --tb=short
```

- [ ] Verify MCP server starts without errors:

```
python mcp_server.py
```

Expected: Server starts, logs show all tools registered, no import errors.

- [ ] Verify tool count in startup log shows 14 tools (original 7 + 7 new).
