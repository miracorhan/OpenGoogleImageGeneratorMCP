# Developer: Mirac Orhan (mirac.orhan@gmail.com)
# License: Open Source (MIT License)
from typing import Tuple

VALID_TIERS = ("fast", "balanced", "quality", "ultra")
VALID_TOOL_TYPES = ("generate", "transform", "video")

_TIER_MAP: dict = {
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


def resolve_model(tier: str, tool_type: str) -> Tuple[str, str]:
    """Return (model_name, api_backend) for the given tier and tool_type.

    Falls back to 'fast' if tier is unrecognized.
    Raises ValueError for unknown tool_type.
    """
    if tool_type not in _TIER_MAP:
        raise ValueError(f"Unknown tool_type '{tool_type}'. Valid: {list(_TIER_MAP)}")
    tool_map = _TIER_MAP[tool_type]
    return tool_map.get(tier, tool_map["fast"])
