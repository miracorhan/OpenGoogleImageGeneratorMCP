import pytest
from unittest.mock import patch
from pydantic import ValidationError
from mcp_server import (
    tool_generate_image, tool_edit_image, tool_transform_image,
    tool_list_available_models, tool_run_pipeline, tool_batch_generate,
    GenerateImageParams, EditImageParams, TransformImageParams,
    BatchGenerateParams, RunPipelineParams, PipelineStepModel,
)


@pytest.mark.asyncio
@patch("mcp_server.gemini_generate_image")
async def test_tool_generate_image(mock_gemini_generate):
    mock_gemini_generate.return_value = {"success": True, "results": [{"path": "outputs/test.png"}]}
    params = GenerateImageParams(prompt="a cat", output_filename="test.png", aspect_ratio="16:9")
    result = await tool_generate_image(params)
    assert result["success"] is True
    assert result["results"][0]["path"] == "outputs/test.png"
    kwargs = mock_gemini_generate.call_args.kwargs
    assert kwargs["model_name"] == "gemini-3.1-flash-lite-image"
    assert kwargs["aspect_ratio"] == "16:9"


@pytest.mark.asyncio
@patch("mcp_server.generate_image")
async def test_tool_generate_image_explicit_imagen_model(mock_generate_image):
    mock_generate_image.return_value = {"success": True, "results": [{"path": "outputs/test.png"}]}
    params = GenerateImageParams(prompt="a cat", output_filename="test.png",
                                 model_name="imagen-4.0-fast-generate-001")
    result = await tool_generate_image(params)
    assert result["success"] is True
    assert mock_generate_image.call_args.kwargs["model_name"] == "imagen-4.0-fast-generate-001"


@pytest.mark.asyncio
@patch("mcp_server.edit_image")
async def test_tool_edit_image_passes_all_params(mock_edit_image):
    mock_edit_image.return_value = {"success": True, "results": [{"path": "outputs/x.png"}]}
    params = EditImageParams(
        prompt="add a hat",
        base_image_path="base.png",
        output_filename="x.png",
        mask_image_path="mask.png",
        edit_mode="EDIT_MODE_INPAINT_INSERTION",
        negative_prompt="ugly",
        sample_count=2,
    )
    result = await tool_edit_image(params)
    assert result["success"] is True
    kwargs = mock_edit_image.call_args.kwargs
    assert kwargs["prompt"] == "add a hat"
    assert kwargs["mask_image_path"] == "mask.png"
    assert kwargs["edit_mode"] == "EDIT_MODE_INPAINT_INSERTION"
    assert kwargs["negative_prompt"] == "ugly"
    assert kwargs["sample_count"] == 2
    assert kwargs["model_name"] == "gemini-3.1-flash-image"


@pytest.mark.asyncio
@patch("mcp_server.transform_image")
async def test_tool_transform_image_passes_all_params(mock_transform):
    mock_transform.return_value = {"success": True, "results": [{"path": "outputs/y.png"}]}
    params = TransformImageParams(
        prompt="oil painting",
        base_image_path="base.png",
        output_filename="y.png",
        additional_image_paths=["ref1.png", "ref2.png"],
    )
    result = await tool_transform_image(params)
    assert result["success"] is True
    kwargs = mock_transform.call_args.kwargs
    assert kwargs["prompt"] == "oil painting"
    assert kwargs["additional_image_paths"] == ["ref1.png", "ref2.png"]
    assert kwargs["model_name"] == "gemini-3.1-flash-image"


@pytest.mark.asyncio
@patch("mcp_server.probe_available_models")
async def test_tool_list_available_models_returns_live_scan(mock_probe):
    mock_probe.return_value = {
        "image_generation": ["imagen-4.0-fast-generate-001"],
        "image_transformation": ["gemini-2.5-flash-image"],
        "text": ["gemini-2.5-flash"],
    }
    result = await tool_list_available_models()
    assert "available" in result
    assert result["available"]["image_transformation"] == ["gemini-2.5-flash-image"]
    assert "recommended" in result
    assert "project" in result
    mock_probe.assert_called_once()


def test_edit_image_params_defaults():
    p = EditImageParams(prompt="x", base_image_path="b.png", output_filename="o.png")
    assert p.edit_mode == "EDIT_MODE_DEFAULT"
    assert p.model_name == "gemini-3.1-flash-image"
    assert p.mask_image_path is None
    assert p.sample_count == 1


def test_transform_image_params_defaults():
    p = TransformImageParams(prompt="x", base_image_path="b.png", output_filename="o.png")
    assert p.model_name == "gemini-3.1-flash-image"
    assert p.additional_image_paths is None


def test_batch_generate_params_accepts_all_tiers():
    for tier in ("fast", "balanced", "quality", "ultra"):
        assert BatchGenerateParams(prompts=["a"], output_prefix="p", model_tier=tier).model_tier == tier
    with pytest.raises(ValidationError):
        BatchGenerateParams(prompts=["a"], output_prefix="p", model_tier="mega")


@pytest.mark.asyncio
async def test_tool_run_pipeline_requires_output():
    params = RunPipelineParams(
        steps=[PipelineStepModel(tool="generate", params={"prompt": "cat"})],
    )
    result = await tool_run_pipeline(params)
    assert result["success"] is False
    assert result["error"]["code"] == "VALIDATION"
    assert "output" in result["error"]["message"].lower()


@pytest.mark.asyncio
@patch("mcp_server.edit_image")
async def test_tool_edit_image_model_tier(mock_edit_image):
    mock_edit_image.return_value = {"success": True, "results": []}
    await tool_edit_image(EditImageParams(prompt="x", base_image_path="b.png", output_filename="o.png",
                                          model_tier="ultra"))
    assert mock_edit_image.call_args.kwargs["model_name"] == "gemini-3-pro-image"


@pytest.mark.asyncio
async def test_tool_upscale_image_passes_target_resolution_and_tier():
    from mcp_server import tool_upscale_image, UpscaleImageParams
    with patch("mcp_server.upscale_image") as mock_upscale:
        mock_upscale.return_value = {"success": True, "results": []}
        await tool_upscale_image(UpscaleImageParams(base_image_path="b.png", output_filename="o.png",
                                                    target_resolution="4K", model_tier="quality"))
    kwargs = mock_upscale.call_args.kwargs
    assert kwargs["target_resolution"] == "4K"
    assert kwargs["model_name"] == "gemini-nano-banana-2.1"


@pytest.mark.asyncio
async def test_tool_remove_background_defaults():
    from mcp_server import tool_remove_background, RemoveBackgroundParams
    with patch("mcp_server.remove_background") as mock_rb:
        mock_rb.return_value = {"success": True, "results": []}
        await tool_remove_background(RemoveBackgroundParams(base_image_path="b.png", output_filename="o.png"))
    kwargs = mock_rb.call_args.kwargs
    assert kwargs["model_name"] == "gemini-3.1-flash-image"
    assert kwargs["key_color"] == "green"
