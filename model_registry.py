# Developer: Mirac Orhan (mirac.orhan@gmail.com)
# License: Open Source (MIT License)
from typing import Dict, Tuple

VALID_TIERS = ("fast", "balanced", "quality", "ultra")
VALID_TOOL_TYPES = ("generate", "transform", "edit", "upscale", "video")

# Imagen was shut down on the Gemini API (2026-08-17) and returns 404 on Vertex AI
# in every region, so every image tier routes to the Gemini (Nano Banana) family.
# Legacy imagen-* IDs are still accepted when passed explicitly as model_name.
_TIER_MAP: Dict[str, Dict[str, Tuple[str, str]]] = {
    "generate": {
        "fast":     ("gemini-3.1-flash-lite-image", "gemini"),
        "balanced": ("gemini-3.1-flash-image",      "gemini"),
        "quality":  ("gemini-nano-banana-2.1",      "gemini"),
        "ultra":    ("gemini-3-pro-image",          "gemini"),
    },
    "transform": {
        "fast":     ("gemini-3.1-flash-lite-image", "gemini"),
        "balanced": ("gemini-3.1-flash-image",      "gemini"),
        "quality":  ("gemini-nano-banana-2.1",      "gemini"),
        "ultra":    ("gemini-3-pro-image",          "gemini"),
    },
    # Mask-based edit and background removal (mask passed as a reference image).
    "edit": {
        "fast":     ("gemini-3.1-flash-lite-image", "gemini"),
        "balanced": ("gemini-3.1-flash-image",      "gemini"),
        "quality":  ("gemini-nano-banana-2.1",      "gemini"),
        "ultra":    ("gemini-3-pro-image",          "gemini"),
    },
    # Needs 2K/4K output, which gemini-3.1-flash-lite-image (1K only) lacks.
    "upscale": {
        "fast":     ("gemini-3.1-flash-image",      "gemini"),
        "balanced": ("gemini-3.1-flash-image",      "gemini"),
        "quality":  ("gemini-nano-banana-2.1",      "gemini"),
        "ultra":    ("gemini-3-pro-image",          "gemini"),
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
# Default text/vision model for tool_analyze_image (Vertex AI).
# gemini-2.5-* retire on Vertex AI on 2026-10-20.
# ---------------------------------------------------------------------------

DEFAULT_VISION_MODEL = "gemini-3.8-flash"

# ---------------------------------------------------------------------------
# Google GenAI SDK model constants
# Not part of the 4-tier Vertex AI system — used by genai_tools.py
# ---------------------------------------------------------------------------

EMBED_MODEL_VERTEX = "gemini-embedding-001"
EMBED_MODEL_GEMINI_API = "gemini-embedding-2"
EMBED_MODEL_GEMINI_API_V1 = "gemini-embedding-001"

SPEECH_MODELS = {
    "fast":    "gemini-3.8-flash-lite-tts",
    "quality": "gemini-3.8-flash-tts",
}

VIDEO_ANALYZE_MODELS = {
    "fast":     "gemini-3.5-flash-lite",
    "balanced": "gemini-3.8-flash",
    "quality":  "gemini-4-argon",
}

# tool_live_generate streams via generateContent, so every entry must support
# generateContent — Live API models (gemini-3.8-live, *-live-preview) are
# bidiGenerateContent-only and cannot be used here.
LIVE_MODELS = {
    "fast":     "gemini-3.5-flash-lite",
    "balanced": "gemini-3.8-flash",
    "quality":  "gemini-4-argon",
}

MUSIC_MODELS = {
    "lyria-3.5":    "lyria-3.5",
    "lyria-3-pro":  "lyria-3-pro-preview",
    "lyria-3-clip": "lyria-3-clip-preview",
    "lyria-2":      "lyria-3-clip-preview",  # legacy alias
}
