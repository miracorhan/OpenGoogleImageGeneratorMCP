from discovery import list_vertex_publisher_models, get_recommended_models


def test_get_recommended_models_has_all_categories():
    models = get_recommended_models()
    for cat in ("text", "vision", "image_generation", "image_transformation", "video_generation"):
        assert cat in models, f"missing category: {cat}"
        assert isinstance(models[cat], list) and models[cat], f"empty: {cat}"


def test_get_recommended_models_includes_known_ids():
    models = get_recommended_models()
    assert "gemini-2.5-flash" in models["text"]
    assert "gemini-2.5-pro" in models["text"]
    assert "imagen-4.0-fast-generate-001" in models["image_generation"]
    assert "imagen-3.0-generate-002" in models["image_generation"]
    assert "imagen-3.0-capability-001" in models["image_generation"]
    assert "gemini-2.5-flash-image" in models["image_transformation"]
    assert "veo-3.1-fast-generate-001" in models["video_generation"]


def test_list_vertex_publisher_models_returns_empty():
    # The Vertex AI publishers list REST endpoint does not exist; the function
    # is a documented stub that returns [].
    assert list_vertex_publisher_models("test-project") == []
