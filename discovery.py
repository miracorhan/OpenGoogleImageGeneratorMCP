# Developer: Mirac Orhan (mirac.orhan@gmail.com)
# License: Open Source (MIT License) - Free for everyone to use, modify, and distribute.

"""Static recommended-model catalog.

For the live-probed list of what is actually reachable in the current
project/location, use vertex_ai_tools.probe_available_models() or the MCP
tool tool_list_available_models. That probe runs a minimal request against
every candidate and returns only the models that respond.
"""


def list_vertex_publisher_models(project_id: str, location: str = "us-central1"):
    """
    Vertex AI publishers/google/models list endpoint does not exist as a public
    REST list API. Returns an empty list; callers should rely on
    get_recommended_models() or vertex_ai_tools.probe_available_models() instead.
    """
    return []


def get_recommended_models():
    """Static recommended list. Some entries may 404 in a given project/location
    until enabled — use tool_list_available_models for the live-verified set."""
    return {
        "text": [
            "gemini-3.5-flash",
            "gemini-3.1-pro",
            "gemini-3-flash",
            "gemini-3.1-flash-lite",
            "gemini-2.5-pro",
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite",
        ],
        "vision": [
            "gemini-3.5-flash",
            "gemini-3.1-pro",
            "gemini-2.5-pro",
            "gemini-2.5-flash",
        ],
        "image_generation": [
            "imagen-4.0-fast-generate-001",
            "imagen-4.0-generate-001",
            "imagen-3.0-generate-002",
            "imagen-3.0-capability-001",
        ],
        "image_transformation": [
            # 'image + text -> image' via Gemini multimodal :generateContent
            "gemini-2.5-flash-image",
            "gemini-3.1-flash-image",
            "gemini-3-pro-image",
        ],
        "video_generation": [
            "veo-3.1-generate-001",
            "veo-3.1-fast-generate-001",
            "veo-3.1-lite-generate-001",
            "veo-3.0-generate-001",
            "veo-3.0-fast-generate-001",
            "veo-2.0-generate-001",
        ],
        "embedding": [
            "text-embedding-004",
            "gemini-embedding-2",
        ],
        "speech": [
            "gemini-2.5-flash-preview-tts",
            "gemini-2.5-pro-preview-tts",
        ],
        "live_text": [
            "gemini-3.5-flash",
            "gemini-3.1-pro",
            "gemini-2.5-flash-live-api",
            "gemini-2.5-flash",
        ],
    }
