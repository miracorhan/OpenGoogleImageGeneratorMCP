import os
import sys
import logging
from dotenv import load_dotenv

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
