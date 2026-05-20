# Developer: Mirac Orhan (mirac.orhan@gmail.com)
# License: Open Source (MIT License)
from typing import Dict, Tuple

VALID_TIERS = ("fast", "balanced", "quality", "ultra")
VALID_TOOL_TYPES = ("generate", "transform", "video")

_TIER_MAP: Dict[str, Dict[str, Tuple[str, str]]] = {
    "generate": {
        "fast":     ("imagen-4.0-fast-generate-001",  "imagen"),
        "balanced": ("gemini-2.5-flash-image",         "gemini"),
        "quality":  ("imagen-4.0-generate-001",        "imagen"),
        "ultra":    ("imagen-4.0-ultra-generate-001",  "imagen"),
    },
    "transform": {
        "fast":     ("gemini-2.5-flash-image", "gemini"),
        "balanced": ("gemini-2.5-flash-image", "gemini"),
        "quality":  ("gemini-2.5-pro-image",   "gemini"),
        "ultra":    ("gemini-2.5-pro-image",   "gemini"),
    },
    "video": {
        "fast":     ("veo-3.1-fast-generate-001", "veo"),
        "balanced": ("veo-3.1-fast-generate-001", "veo"),
        "quality":  ("veo-3.1-generate-001",      "veo"),
        "ultra":    ("veo-3.1-generate-001",      "veo"),
    },
}

assert set(VALID_TOOL_TYPES) == set(_TIER_MAP), "VALID_TOOL_TYPES out of sync with _TIER_MAP"
assert all(set(VALID_TIERS) == set(tier_map) for tier_map in _TIER_MAP.values()), \
    "VALID_TIERS out of sync with _TIER_MAP"


def resolve_model(tier: str, tool_type: str) -> Tuple[str, str]:
    """Return (model_name, api_backend) for the given tier and tool_type.

    Falls back to 'fast' if tier is unrecognized.
    Raises ValueError for unknown tool_type.
    """
    if tool_type not in _TIER_MAP:
        raise ValueError(f"Unknown tool_type '{tool_type}'. Valid: {list(_TIER_MAP)}")
    tool_map = _TIER_MAP[tool_type]
    return tool_map.get(tier, tool_map["fast"])


# ---------------------------------------------------------------------------
# Google GenAI SDK model constants
# Not part of the 4-tier Vertex AI system — used by genai_tools.py
# ---------------------------------------------------------------------------

EMBED_MODEL_VERTEX = "text-embedding-004"
EMBED_MODEL_GEMINI_API = "gemini-embedding-2"

SPEECH_MODELS = {
    "fast": "gemini-2.5-flash-preview-tts",
    "quality": "gemini-2.5-pro-preview-tts",
}

VIDEO_ANALYZE_MODELS = {
    "fast": "gemini-3.5-flash",
    "quality": "gemini-2.5-pro",
}

LIVE_MODELS = {
    "fast": "gemini-3.5-flash",
    "quality": "gemini-2.5-pro",
}
