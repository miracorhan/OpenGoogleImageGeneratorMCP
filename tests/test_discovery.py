from discovery import list_vertex_publisher_models, get_recommended_models


def test_get_recommended_models_has_all_categories():
    models = get_recommended_models()
    for cat in ("text", "vision", "image_generation", "image_transformation", "video_generation"):
        assert cat in models, f"missing category: {cat}"
        assert isinstance(models[cat], list) and models[cat], f"empty: {cat}"


def test_get_recommended_models_includes_known_ids():
    models = get_recommended_models()
    assert "gemini-4-argon" in models["text"]
    assert "gemini-3.8-flash" in models["text"]
    assert "gemini-3.5-flash" in models["text"]
    assert "gemini-3.1-flash-lite" in models["text"]
    assert "gemini-3.1-pro-preview" in models["text"]
    assert "gemini-3.1-flash-lite-image" in models["image_generation"]
    assert "gemini-3-pro-image" in models["image_generation"]
    assert "gemini-nano-banana-2.1" in models["image_transformation"]
    assert "gemini-3.1-flash-image" in models["image_transformation"]
    assert "gemini-3-pro-image" in models["image_transformation"]
    assert "veo-3.1-fast-generate-001" in models["video_generation"]
    assert "gemini-3.8-flash-tts" in models["speech"]
    assert "gemini-embedding-001" in models["embedding"]
    assert "gemini-3.8-live" in models["live_audio"]
    assert "lyria-3.5" in models["music"]


def test_recommended_models_exclude_retired():
    models = get_recommended_models()
    flat = {m for ms in models.values() for m in ms}
    for retired in ("veo-2.0-generate-001", "veo-3.0-generate-001", "imagen-3.0-generate-002",
                    "imagen-3.0-capability-001", "imagen-4.0-fast-generate-001",
                    "text-embedding-004", "gemini-2.5-pro", "gemini-2.5-flash"):
        assert retired not in flat, retired


def test_list_vertex_publisher_models_returns_empty():
    # The Vertex AI publishers list REST endpoint does not exist; the function
    # is a documented stub that returns [].
    assert list_vertex_publisher_models("test-project") == []
