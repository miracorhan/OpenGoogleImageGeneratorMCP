from discovery import list_vertex_publisher_models, get_recommended_models


def test_get_recommended_models():
    models = get_recommended_models()
    assert "text" in models
    assert "vision" in models
    assert "image_generation" in models
    assert "video_generation" in models
    assert "gemini-2.5-flash" in models["text"]
    assert "gemini-2.5-pro" in models["text"]
    assert "imagen-4.0-fast-generate-001" in models["image_generation"]
    assert "imagen-3.0-generate-002" in models["image_generation"]
    assert "veo-3.1-fast-generate-001" in models["video_generation"]


def test_list_vertex_publisher_models_returns_empty():
    # The Vertex AI publishers list REST endpoint does not exist; the function
    # is now a documented stub that returns [].
    assert list_vertex_publisher_models("test-project") == []
