"""Tests for update.py auto-update script."""
import subprocess
import sys
import pytest
from unittest.mock import patch, MagicMock, call
import update as upd


def test_run_git_pull_succeeds(tmp_path):
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="Already up to date.", stderr="")
        result = upd.git_pull(cwd=str(tmp_path))
    assert result["success"] is True
    assert "stdout" in result


def test_run_git_pull_fails_returns_error(tmp_path):
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="not a git repo")
        result = upd.git_pull(cwd=str(tmp_path))
    assert result["success"] is False
    assert "not a git repo" in result["error"]


def test_run_pip_install_succeeds(tmp_path):
    req = tmp_path / "requirements.txt"
    req.write_text("mcp\n")
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0, stdout="Successfully installed", stderr="")
        result = upd.pip_install(requirements=str(req))
    assert result["success"] is True


def test_run_pip_install_fails_returns_error(tmp_path):
    req = tmp_path / "requirements.txt"
    req.write_text("nonexistent-pkg-xyz==0.0.0\n")
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="No matching distribution")
        result = upd.pip_install(requirements=str(req))
    assert result["success"] is False
    assert "error" in result


def test_full_update_calls_pull_then_pip(tmp_path):
    with patch("update.git_pull", return_value={"success": True, "stdout": "Already up to date."}) as mock_pull, \
         patch("update.pip_install", return_value={"success": True, "stdout": "ok"}) as mock_pip:
        result = upd.run_update(repo_dir=str(tmp_path))
    mock_pull.assert_called_once()
    mock_pip.assert_called_once()
    assert result["git"]["success"] is True
    assert result["pip"]["success"] is True


def test_full_update_skips_pip_if_pull_fails(tmp_path):
    with patch("update.git_pull", return_value={"success": False, "error": "fatal"}) as mock_pull, \
         patch("update.pip_install") as mock_pip:
        result = upd.run_update(repo_dir=str(tmp_path))
    mock_pip.assert_not_called()
    assert result["git"]["success"] is False
