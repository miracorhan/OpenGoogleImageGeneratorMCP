"""Tests for check_for_updates() return value and tool_list_available_models update field."""
import pytest
from unittest.mock import patch, MagicMock
from config import check_for_updates, __version__, GITHUB_REPO
from mcp_server import tool_list_available_models


# --- check_for_updates() ---

def _mock_github_response(tag_name: str):
    resp = MagicMock()
    resp.read.return_value = f'{{"tag_name": "{tag_name}"}}'.encode()
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


def test_check_for_updates_returns_dict():
    result = check_for_updates()
    assert isinstance(result, dict), "check_for_updates() must return a dict"


def test_check_for_updates_has_required_keys():
    with patch("urllib.request.urlopen", return_value=_mock_github_response("v3.0.0")):
        result = check_for_updates()
    for key in ("up_to_date", "current", "latest", "url"):
        assert key in result, f"Missing key '{key}' in check_for_updates() result"


def test_check_for_updates_current_version_matches():
    with patch("urllib.request.urlopen", return_value=_mock_github_response("v3.0.0")):
        result = check_for_updates()
    assert result["current"] == __version__


def test_check_for_updates_up_to_date_when_same_version():
    with patch("urllib.request.urlopen", return_value=_mock_github_response(f"v{__version__}")):
        result = check_for_updates()
    assert result["up_to_date"] is True
    assert result["latest"] == __version__


def test_check_for_updates_not_up_to_date_when_newer_available():
    with patch("urllib.request.urlopen", return_value=_mock_github_response("v99.0.0")):
        result = check_for_updates()
    assert result["up_to_date"] is False
    assert result["latest"] == "99.0.0"


def test_check_for_updates_returns_dict_on_network_error():
    with patch("urllib.request.urlopen", side_effect=Exception("network down")):
        result = check_for_updates()
    assert isinstance(result, dict)
    assert "up_to_date" in result


def test_check_for_updates_url_contains_repo():
    with patch("urllib.request.urlopen", return_value=_mock_github_response("v3.0.0")):
        result = check_for_updates()
    assert GITHUB_REPO in result["url"]


# --- tool_list_available_models() includes update info ---

@pytest.mark.asyncio
@patch("mcp_server.probe_available_models")
@patch("mcp_server.check_for_updates")
async def test_tool_list_available_models_includes_update_key(mock_update, mock_probe):
    mock_probe.return_value = {"image_generation": []}
    mock_update.return_value = {
        "up_to_date": True,
        "current": "3.0.0",
        "latest": "3.0.0",
        "url": "https://github.com/miracorhan/OpenGoogleImageGeneratorMCP/releases/latest",
    }
    result = await tool_list_available_models()
    assert "update" in result, "tool_list_available_models must include 'update' key"


@pytest.mark.asyncio
@patch("mcp_server.probe_available_models")
@patch("mcp_server.check_for_updates")
async def test_tool_list_available_models_update_fields(mock_update, mock_probe):
    mock_probe.return_value = {"image_generation": []}
    mock_update.return_value = {
        "up_to_date": False,
        "current": "1.0.0",
        "latest": "3.0.0",
        "url": "https://github.com/miracorhan/OpenGoogleImageGeneratorMCP/releases/latest",
    }
    result = await tool_list_available_models()
    assert result["update"]["up_to_date"] is False
    assert result["update"]["latest"] == "3.0.0"
