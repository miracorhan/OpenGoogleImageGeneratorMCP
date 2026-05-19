#!/usr/bin/env python3
"""Auto-update script: git pull + pip install -r requirements.txt."""
import os
import subprocess
import sys


def git_pull(cwd: str = None) -> dict:
    if cwd is None:
        cwd = os.path.dirname(os.path.abspath(__file__))
    result = subprocess.run(
        ["git", "pull"],
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return {"success": True, "stdout": result.stdout.strip()}
    return {"success": False, "error": result.stderr.strip() or result.stdout.strip()}


def pip_install(requirements: str = None) -> dict:
    if requirements is None:
        requirements = os.path.join(os.path.dirname(os.path.abspath(__file__)), "requirements.txt")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", requirements],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return {"success": True, "stdout": result.stdout.strip()}
    return {"success": False, "error": result.stderr.strip() or result.stdout.strip()}


def run_update(repo_dir: str = None) -> dict:
    if repo_dir is None:
        repo_dir = os.path.dirname(os.path.abspath(__file__))

    print("Pulling latest changes from GitHub...")
    git_result = git_pull(cwd=repo_dir)
    if not git_result["success"]:
        print(f"  git pull failed: {git_result['error']}")
        return {"git": git_result}

    print(f"  {git_result['stdout']}")

    req_path = os.path.join(repo_dir, "requirements.txt")
    print("Installing/updating dependencies...")
    pip_result = pip_install(requirements=req_path)
    if pip_result["success"]:
        print("  Dependencies up to date.")
    else:
        print(f"  pip install failed: {pip_result['error']}")

    return {"git": git_result, "pip": pip_result}


if __name__ == "__main__":
    result = run_update()
    if result.get("git", {}).get("success") and result.get("pip", {}).get("success"):
        print("\nUpdate complete. Restart the MCP server to apply changes.")
        sys.exit(0)
    else:
        print("\nUpdate failed. See errors above.")
        sys.exit(1)
