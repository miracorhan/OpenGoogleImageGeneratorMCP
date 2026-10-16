import pytest


@pytest.fixture(autouse=True)
def _block_real_vertex_auth(monkeypatch):
    """Tests must never reach Google: with valid local ADC an unmocked call would
    really generate (and bill) images. Fail fast instead of fetching a token."""
    import vertex_ai_tools

    def _no_token():
        raise RuntimeError("Real Vertex AI access is blocked in tests — mock the API call.")

    monkeypatch.setattr(vertex_ai_tools, "_get_access_token", _no_token)
