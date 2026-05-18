import pytest
import os
from unittest.mock import AsyncMock, patch

@pytest.mark.asyncio
async def test_pipeline_generate_then_upscale(tmp_path):
    from pipeline import run_pipeline

    gen_result = {"success": True, "results": [{"path": str(tmp_path / "step_0.png")}]}
    ups_result = {"success": True, "results": [{"path": str(tmp_path / "step_1.png")}]}

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


def test_resolve_step_params_edit_tier_returns_imagen_model():
    """edit steps with model_tier must resolve to an Imagen model, not Gemini."""
    from pipeline import _resolve_step_params
    result = _resolve_step_params("edit", {"prompt": "add hat", "model_tier": "fast"})
    assert "model_tier" not in result
    assert "model_name" in result
    assert result["model_name"].startswith("imagen"), (
        f"Expected Imagen model for edit tier, got {result['model_name']!r}"
    )


def test_resolve_step_params_edit_quality_tier_returns_imagen_model():
    from pipeline import _resolve_step_params
    result = _resolve_step_params("edit", {"prompt": "x", "model_tier": "quality"})
    assert result["model_name"].startswith("imagen")
