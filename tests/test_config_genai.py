import os
import importlib


def test_genai_api_key_reads_from_env(monkeypatch):
    monkeypatch.setenv("GOOGLE_GENAI_API_KEY", "test-key-abc")
    import config
    importlib.reload(config)
    assert config.GOOGLE_GENAI_API_KEY == "test-key-abc"


def test_genai_backend_defaults_to_vertex_ai(monkeypatch):
    monkeypatch.delenv("GOOGLE_GENAI_BACKEND", raising=False)
    import config
    importlib.reload(config)
    assert config.GOOGLE_GENAI_BACKEND == "vertex_ai"


def test_genai_backend_reads_gemini_api(monkeypatch):
    monkeypatch.setenv("GOOGLE_GENAI_BACKEND", "gemini_api")
    import config
    importlib.reload(config)
    assert config.GOOGLE_GENAI_BACKEND == "gemini_api"
