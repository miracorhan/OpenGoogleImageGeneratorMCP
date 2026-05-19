# Developer: Mirac Orhan (mirac.orhan@gmail.com)
# License: Open Source (MIT License) - Free for everyone to use, modify, and distribute.

import os
import sys
import logging
from dotenv import load_dotenv

__version__ = "3.0.0"
GITHUB_REPO = "miracorhan/OpenGoogleImageGeneratorMCP"

# Load environment variables from .env file
load_dotenv()

# Configure logging
_log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(_log_dir, exist_ok=True)
_log_file = os.path.join(_log_dir, "vertex_ai_mcp.log")

_fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
_stderr_handler = logging.StreamHandler(sys.stderr)
_stderr_handler.setFormatter(_fmt)
_file_handler = logging.FileHandler(_log_file, encoding="utf-8")
_file_handler.setFormatter(_fmt)

logging.basicConfig(level=logging.INFO, handlers=[_stderr_handler, _file_handler], force=True)
logger = logging.getLogger("VertexAI-MCP")

# Core Settings
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT")
LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
DEFAULT_OUTPUT_DIR = os.environ.get("DEFAULT_OUTPUT_DIR", "./outputs")

# Authentication Settings
GOOGLE_ACCESS_TOKEN = os.environ.get("GOOGLE_ACCESS_TOKEN")
IMPERSONATE_SERVICE_ACCOUNT = os.environ.get("IMPERSONATE_SERVICE_ACCOUNT")

# Ensure output directory exists
os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)

# Google GenAI SDK settings
GOOGLE_GENAI_API_KEY = os.environ.get("GOOGLE_GENAI_API_KEY")
GOOGLE_GENAI_BACKEND = os.environ.get("GOOGLE_GENAI_BACKEND", "vertex_ai")


def check_for_updates() -> dict:
    """Query GitHub releases API and return update status dict.

    Returns:
        {"up_to_date": bool, "current": str, "latest": str, "url": str}
    """
    import urllib.request
    import json

    releases_url = f"https://github.com/{GITHUB_REPO}/releases/latest"
    result = {"up_to_date": True, "current": __version__, "latest": __version__, "url": releases_url}
    api_url = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"
    try:
        req = urllib.request.Request(
            api_url, headers={"User-Agent": f"OpenGoogleImageGeneratorMCP/{__version__}"}
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
        latest_tag = data.get("tag_name", "").lstrip("v")
        if not latest_tag:
            return result
        current = tuple(int(x) for x in __version__.split("."))
        latest = tuple(int(x) for x in latest_tag.split("."))
        result["latest"] = latest_tag
        if latest > current:
            result["up_to_date"] = False
            logger.warning(
                f"Update available: v{__version__} → v{latest_tag}. "
                f"Run 'git pull' to update. {releases_url}"
            )
        else:
            logger.info(f"OpenGoogleImageGeneratorMCP v{__version__} is up to date.")
    except Exception:
        pass
    return result
