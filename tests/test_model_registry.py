import pytest
from model_registry import resolve_model, VALID_TIERS, VALID_TOOL_TYPES

def test_generate_fast_returns_gemini_lite_image():
    model, backend = resolve_model("fast", "generate")
    assert model == "gemini-3.1-flash-lite-image"
    assert backend == "gemini"

def test_generate_balanced_returns_gemini():
    model, backend = resolve_model("balanced", "generate")
    assert model == "gemini-3.1-flash-image"
    assert backend == "gemini"

def test_generate_quality_returns_nano_banana_2_1():
    model, backend = resolve_model("quality", "generate")
    assert model == "gemini-nano-banana-2.1"
    assert backend == "gemini"

def test_generate_ultra_returns_pro_image():
    model, backend = resolve_model("ultra", "generate")
    assert model == "gemini-3-pro-image"
    assert backend == "gemini"

def test_no_imagen_in_generate_tiers():
    """Imagen 4 is shut down; generate tiers must not route to it."""
    for tier in VALID_TIERS:
        model, _ = resolve_model(tier, "generate")
        assert not model.startswith("imagen"), (tier, model)

def test_transform_fast():
    model, backend = resolve_model("fast", "transform")
    assert model == "gemini-3.1-flash-lite-image"
    assert backend == "gemini"

def test_transform_quality():
    model, backend = resolve_model("quality", "transform")
    assert model == "gemini-nano-banana-2.1"
    assert backend == "gemini"

def test_transform_ultra():
    model, backend = resolve_model("ultra", "transform")
    assert model == "gemini-3-pro-image"
    assert backend == "gemini"

def test_no_imagen_in_any_tier():
    """Imagen returns 404 on Vertex AI in every region; no tier may route to it."""
    for tool_type in VALID_TOOL_TYPES:
        for tier in VALID_TIERS:
            model, backend = resolve_model(tier, tool_type)
            assert not model.startswith("imagen"), (tool_type, tier, model)
            assert backend != "imagen"

def test_edit_tiers_use_gemini_images():
    assert resolve_model("fast", "edit") == ("gemini-3.1-flash-lite-image", "gemini")
    assert resolve_model("ultra", "edit") == ("gemini-3-pro-image", "gemini")

def test_upscale_tiers_support_high_res():
    """gemini-3.1-flash-lite-image is 1K-only and must never be an upscale model."""
    for tier in VALID_TIERS:
        model, _ = resolve_model(tier, "upscale")
        assert "lite" not in model and not model.startswith("gemini-2.5"), (tier, model)

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
    assert model == "gemini-3.1-flash-lite-image"
    assert backend == "gemini"

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
    assert "edit" in VALID_TOOL_TYPES
    assert "video" in VALID_TOOL_TYPES


def test_embed_model_vertex_is_defined():
    from model_registry import EMBED_MODEL_VERTEX
    assert EMBED_MODEL_VERTEX == "gemini-embedding-001"


def test_embed_model_gemini_api_is_defined():
    from model_registry import EMBED_MODEL_GEMINI_API
    assert EMBED_MODEL_GEMINI_API == "gemini-embedding-2"


def test_speech_models_has_fast_and_quality():
    from model_registry import SPEECH_MODELS
    assert "fast" in SPEECH_MODELS
    assert "quality" in SPEECH_MODELS
    assert SPEECH_MODELS["fast"] == "gemini-3.8-flash-lite-tts"
    assert SPEECH_MODELS["quality"] == "gemini-3.8-flash-tts"


def test_embed_model_gemini_api_v1_is_defined():
    from model_registry import EMBED_MODEL_GEMINI_API_V1
    assert EMBED_MODEL_GEMINI_API_V1 == "gemini-embedding-001"


def test_video_analyze_models_has_all_tiers():
    from model_registry import VIDEO_ANALYZE_MODELS
    assert VIDEO_ANALYZE_MODELS["fast"] == "gemini-3.5-flash-lite"
    assert VIDEO_ANALYZE_MODELS["balanced"] == "gemini-3.8-flash"
    assert VIDEO_ANALYZE_MODELS["quality"] == "gemini-4-argon"


def test_live_models_has_all_tiers():
    from model_registry import LIVE_MODELS
    assert LIVE_MODELS["fast"] == "gemini-3.5-flash-lite"
    assert LIVE_MODELS["balanced"] == "gemini-3.8-flash"
    assert LIVE_MODELS["quality"] == "gemini-4-argon"


def test_live_models_are_not_bidi_only():
    """live_generate uses generate_content_stream; Live API (bidi-only) models 404 there."""
    from model_registry import LIVE_MODELS
    for model in LIVE_MODELS.values():
        assert "live" not in model, model


def test_no_retiring_gemini_2_5_text_defaults():
    """gemini-2.5-pro/flash/flash-lite retire on Vertex AI on 2026-10-20."""
    from model_registry import VIDEO_ANALYZE_MODELS, LIVE_MODELS, DEFAULT_VISION_MODEL
    for model in [*VIDEO_ANALYZE_MODELS.values(), *LIVE_MODELS.values(), DEFAULT_VISION_MODEL]:
        assert not model.startswith("gemini-2.5"), model


def test_music_models_include_lyria_3_5():
    from model_registry import MUSIC_MODELS
    assert MUSIC_MODELS["lyria-3.5"] == "lyria-3.5"
