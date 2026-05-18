def list_vertex_publisher_models(project_id: str, location: str = "us-central1"):
    """
    Vertex AI publishers/google/models list endpoint does not exist as a public REST list API.
    Returns an empty list; callers should rely on get_recommended_models() instead.
    """
    return []


def get_recommended_models():
    """
    Verified working model IDs in this project (live-probed against Vertex AI).
    """
    return {
        "text": [
            "gemini-2.5-flash",
            "gemini-2.5-pro",
            "gemini-2.5-flash-lite",
        ],
        "vision": [
            "gemini-2.5-flash",
            "gemini-2.5-pro",
        ],
        "image_generation": [
            "imagen-4.0-fast-generate-001",
            "imagen-4.0-generate-001",
            "imagen-3.0-generate-002",
        ],
        "video_generation": [
            "veo-3.1-fast-generate-001",
            "veo-3.1-generate-001",
            "veo-3.0-generate-001",
            "veo-2.0-generate-001",
        ],
    }
