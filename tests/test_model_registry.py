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
