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
    # Verified against both catalogs on 2026-10-16. gemini-2.5-pro/flash/flash-lite
    # retire on Vertex AI on 2026-10-20; Imagen 4 was shut down on the Gemini API
    # on 2026-08-17 and every Imagen model 404s on Vertex AI; Veo 2.0/3.0 retired 2026-06-30.
    return {
        "text": [
            "gemini-4-argon",
            "gemini-3.8-flash",
            "gemini-3.5-flash",
            "gemini-3.5-flash-lite",
            "gemini-3.1-flash-lite",
            "gemini-3.1-pro-preview",
        ],
        "vision": [
            "gemini-4-argon",
            "gemini-3.8-flash",
            "gemini-3.5-flash",
            "gemini-3.5-flash-lite",
        ],
        "image_generation": [
            "gemini-3.1-flash-lite-image",
            "gemini-3.1-flash-image",
            "gemini-nano-banana-2.1",
            "gemini-3-pro-image",
        ],
        "image_transformation": [
            "gemini-3.1-flash-lite-image",
            "gemini-3.1-flash-image",
            "gemini-nano-banana-2.1",
            "gemini-3-pro-image",
            "gemini-2.5-flash-image",
        ],
        "video_generation": [
            "veo-3.1-generate-001",
            "veo-3.1-fast-generate-001",
            "veo-3.1-lite-generate-001",
            "gemini-omni-1.1-flash-preview",
        ],
        "embedding": [
            "gemini-embedding-2",
            "gemini-embedding-001",
        ],
        "speech": [
            "gemini-3.8-flash-tts",
            "gemini-3.8-flash-lite-tts",
        ],
        "live_text": [
            "gemini-4-argon",
            "gemini-3.8-flash",
            "gemini-3.5-flash-lite",
        ],
        "live_audio": [
            "gemini-3.8-live",
            "gemini-3.8-live-extended-thinking",
        ],
        "music": [
            "lyria-3.5",
            "lyria-3-pro-preview",
            "lyria-3-clip-preview",
        ],
    }
