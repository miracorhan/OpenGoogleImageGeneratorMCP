import pytest
from unittest.mock import patch
from mcp_server import (
    tool_generate_image, tool_edit_image, tool_transform_image,
    tool_list_available_models,
    GenerateImageParams, EditImageParams, TransformImageParams,
)


@pytest.mark.asyncio
@patch("mcp_server.generate_image")
async def test_tool_generate_image(mock_generate_image):
    mock_generate_image.return_value = {"success": True, "results": [{"path": "outputs/test.png"}]}
    params = GenerateImageParams(prompt="a cat", output_filename="test.png")
    result = await tool_generate_image(params)
    assert result["success"] is True
    assert result["results"][0]["path"] == "outputs/test.png"
    mock_generate_image.assert_called_once()


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
    assert kwargs["model_name"] == "imagen-3.0-capability-001"


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
    assert kwargs["model_name"] == "gemini-2.5-flash-image"


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
    assert p.model_name == "imagen-3.0-capability-001"
    assert p.mask_image_path is None
    assert p.sample_count == 1


def test_transform_image_params_defaults():
    p = TransformImageParams(prompt="x", base_image_path="b.png", output_filename="o.png")
    assert p.model_name == "gemini-2.5-flash-image"
    assert p.additional_image_paths is None
