import os
import pytest
from config import PROJECT_ID, LOCATION, DEFAULT_OUTPUT_DIR

def test_config_defaults():
    # Since we can't easily change env vars before config is imported in this test session
    # (as it might have already been imported), we just check if they are present or have defaults.
    assert LOCATION in ["us-central1", os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")]
    assert DEFAULT_OUTPUT_DIR in ["./outputs", os.environ.get("DEFAULT_OUTPUT_DIR", "./outputs")]

def test_output_dir_exists():
    assert os.path.exists(DEFAULT_OUTPUT_DIR)
