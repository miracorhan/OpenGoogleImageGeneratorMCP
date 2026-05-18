import os
import pytest
from unittest.mock import MagicMock, patch

import genai_tools


@pytest.fixture(autouse=True)
def reset_genai_client():
    genai_tools._genai_client = None
    yield
    genai_tools._genai_client = None


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
