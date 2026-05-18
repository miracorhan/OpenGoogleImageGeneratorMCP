# Developer: Mirac Orhan (mirac.orhan@gmail.com)
# License: Open Source (MIT License)
import asyncio
import os
import sys
import uuid
import shutil
from typing import List, Dict, Any, Optional

from vertex_ai_tools import (
    generate_image, edit_image, transform_image,
    upscale_image, remove_background,
)
from model_registry import resolve_model
from config import DEFAULT_OUTPUT_DIR, logger

# Mapping of tool name → attribute name in this module.
# Looked up dynamically at call time so test patches (pipeline.generate_image etc.) are honoured.
_TOOL_ATTR_MAP = {
    "generate":          "generate_image",
    "edit":              "edit_image",
    "transform":         "transform_image",
    "upscale":           "upscale_image",
    "remove_background": "remove_background",
}

_FIRST_STEP_TOOLS = {"generate"}  # tools that create images without a base_image_path

_TOOL_TYPE_FOR_TIER = {
    "generate":          "generate",
    "edit":              "transform",
    "transform":         "transform",
    "upscale":           "generate",
    "remove_background": "generate",
}


def _get_tool_fn(tool_name: str):
    """Return the current function bound to tool_name in this module's namespace.

    Using sys.modules[__name__] ensures that unittest.mock.patch('pipeline.<fn>')
    patches are visible here rather than using stale references captured at import
    time in a dict.
    """
    attr = _TOOL_ATTR_MAP[tool_name]
    return getattr(sys.modules[__name__], attr)


def _resolve_step_params(tool_name: str, step_params: dict) -> dict:
    """Resolve model_tier → model_name so vertex_ai_tools functions receive clean kwargs."""
    params = dict(step_params)
    if "model_tier" in params:
        tier = params.pop("model_tier")
        tool_type = _TOOL_TYPE_FOR_TIER.get(tool_name, "generate")
        model_name, _ = resolve_model(tier, tool_type)
        params.setdefault("model_name", model_name)
    return params


async def run_pipeline(
    steps: List[Dict[str, Any]],
    output_path: Optional[str] = None,
    work_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute steps sequentially; pipe each step's output into the next as base_image_path.

    Returns {"success": bool, "steps": [...], "final_path": str|None, "error": str|None}
    """
    if not steps:
        return {"success": False, "steps": [], "final_path": None, "error": "No steps provided."}

    run_id = uuid.uuid4().hex[:8]
    temp_dir = work_dir or os.path.join(DEFAULT_OUTPUT_DIR, f"pipeline_{run_id}")
    os.makedirs(temp_dir, exist_ok=True)
    cleanup_temp = work_dir is None  # Only clean up dirs we created

    step_results: List[Dict[str, Any]] = []
    current_image_path: Optional[str] = None

    try:
        for i, step in enumerate(steps):
            tool_name = step.get("tool", "")
            step_params = dict(step.get("params", {}))

            if tool_name not in _TOOL_ATTR_MAP:
                return {
                    "success": False,
                    "steps": step_results,
                    "final_path": None,
                    "error": f"Unknown tool '{tool_name}' at step {i}. Valid: {list(_TOOL_ATTR_MAP)}",
                }

            is_last_step = (i == len(steps) - 1)
            step_output = output_path if is_last_step and output_path else os.path.join(temp_dir, f"step_{i}.png")

            # Wire base_image_path from previous step (skip for first-step tools)
            if current_image_path and tool_name not in _FIRST_STEP_TOOLS:
                step_params.setdefault("base_image_path", current_image_path)

            step_params["output_path"] = step_output

            logger.info(f"[pipeline:{run_id}] step {i}/{len(steps)-1} tool={tool_name}")
            fn = _get_tool_fn(tool_name)
            resolved_params = _resolve_step_params(tool_name, step_params)
            result = await fn(**resolved_params)

            step_results.append({"step": i, "tool": tool_name, **result})

            if not result.get("success"):
                return {
                    "success": False,
                    "steps": step_results,
                    "final_path": None,
                    "error": f"Step {i} ({tool_name}) failed: {result.get('error', {}).get('message', 'unknown error')}",
                }

            # Extract output path for next step
            results_list = result.get("results", [])
            if results_list and results_list[0].get("path"):
                current_image_path = results_list[0]["path"]
            else:
                current_image_path = step_output

        return {
            "success": True,
            "steps": step_results,
            "final_path": current_image_path,
            "error": None,
        }

    finally:
        if cleanup_temp and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
