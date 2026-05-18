import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from mcp_server import tool_generate_image, tool_list_available_models, GenerateImageParams

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
@patch("mcp_server.list_vertex_publisher_models")
@patch("mcp_server.get_recommended_models")
async def test_tool_list_available_models(mock_get_recommended, mock_list_publisher):
    mock_get_recommended.return_value = {"recommended": "models"}
    mock_list_publisher.return_value = [{"name": "model1"}]
    
    # Reset cache for test
    import mcp_server
    mcp_server._publisher_models_cache = []
    
    with patch("mcp_server.PROJECT_ID", "test-project"):
        result = await tool_list_available_models()
    
    assert "recommended" in result
    assert "all_publishers" in result
    assert result["all_publishers"] == [{"name": "model1"}]
