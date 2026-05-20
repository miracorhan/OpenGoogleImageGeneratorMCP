# Developer: Mirac Orhan (mirac.orhan@gmail.com)
# License: Open Source (MIT License) - Free for everyone to use, modify, and distribute.

import asyncio
import os
from typing import Literal, Optional, List
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

import google.auth
from google.oauth2 import credentials as oauth2_credentials
from google.auth.impersonated_credentials import Credentials as ImpersonatedCredentials

import vertexai
from vertex_ai_tools import (
    generate_image, edit_image, transform_image, analyze_image,
    upscale_image, remove_background, generate_video, image_to_video,
    extend_video, video_object_edit, upload_file,
    batch_generate, generate_music,
    gemini_generate_image,
    probe_available_models, get_cached_availability,
    _validate_output_path,
    SUPPORTED_EDIT_MODES,
)
from pipeline import run_pipeline
from config import PROJECT_ID, LOCATION, DEFAULT_OUTPUT_DIR, GOOGLE_ACCESS_TOKEN, IMPERSONATE_SERVICE_ACCOUNT, logger, check_for_updates, __version__
from discovery import get_recommended_models

# Initialize the MCP Server
mcp = FastMCP("OpenGoogleImageGenerator")

# Initialize Vertex AI
if PROJECT_ID:
    creds = None
    if GOOGLE_ACCESS_TOKEN:
        creds = oauth2_credentials.Credentials(token=GOOGLE_ACCESS_TOKEN)
        logger.info("Using OAuth 2.0 Access Token for authentication.")
    elif IMPERSONATE_SERVICE_ACCOUNT:
        source_credentials, _ = google.auth.default()
        creds = ImpersonatedCredentials(
            source_credentials=source_credentials,
            target_principal=IMPERSONATE_SERVICE_ACCOUNT,
            target_scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
        logger.info(f"Using Service Account Impersonation for {IMPERSONATE_SERVICE_ACCOUNT}.")

    vertexai.init(project=PROJECT_ID, location=LOCATION, credentials=creds)
    logger.info(f"Vertex AI initialized for project {PROJECT_ID} in {LOCATION}")
else:
    logger.warning("GOOGLE_CLOUD_PROJECT not set. Vertex AI calls may fail.")

@mcp.tool()
async def tool_list_available_models(force_refresh: bool = False) -> dict:
    """
    Live-probe every candidate Vertex AI publisher model in this project/location
    and return only those that respond (HTTP 200 or 400 = reachable; 404 = not
    found and excluded). Results are cached for the server process lifetime;
    pass force_refresh=true to rescan.

    Returns:
      {
        "available": { "image_generation": [...], "image_transformation": [...],
                        "text": [...], "vision": [...] },
        "recommended": <static list, may include models that 404 in this project>,
        "checked_at": <ISO timestamp>,
        "project": ..., "location": ...
      }
    """
    import time as _t
    t0 = _t.time()
    available = await asyncio.to_thread(probe_available_models, force_refresh)
    cached = get_cached_availability()
    update_info = await asyncio.to_thread(check_for_updates)
    response: dict = {
        "available": available,
        "recommended": get_recommended_models(),
        "checked_at": cached.get("checked_at"),
        "project": PROJECT_ID,
        "location": LOCATION,
        "duration_s": round(_t.time() - t0, 2),
        "update": update_info,
    }
    if cached.get("auth_error"):
        response["warning"] = cached["auth_error"]
    return response

class GenerateImageParams(BaseModel):
    prompt: str = Field(..., description="The text description of the image to generate.")
    output_filename: Optional[str] = Field(None, description="Filename to save the image as (e.g. image.png). Required if output_path not given.")
    output_path: Optional[str] = Field(None, description="Absolute path for the output file (e.g. C:/outputs/image.png). Takes priority over output_filename.")
    model_name: str = Field(
        "imagen-4.0-fast-generate-001",
        description="Imagen model. GA: imagen-4.0-fast-generate-001 (fast, default), imagen-4.0-generate-001 (quality), imagen-4.0-ultra-generate-001 (ultra), imagen-3.0-generate-002 (stable).",
    )
    model_tier: Optional[Literal["fast", "balanced", "quality", "ultra"]] = Field(None, description="Shorthand tier: fast / balanced / quality / ultra. Overrides model_name when set. balanced routes to gemini-2.5-flash-image.")
    number_of_images: int = Field(1, ge=1, le=4, description="Number of images to generate.")
    aspect_ratio: str = Field("1:1", description="Aspect ratio (e.g., 1:1, 16:9, 4:3, 9:16).")
    return_base64: bool = Field(False, description="Whether to return the base64 encoded image.")
    negative_prompt: Optional[str] = Field(None, description="Elements to exclude from the image.")
    seed: Optional[int] = Field(None, description="Seed for deterministic output. Requires add_watermark=False.")
    enhance_prompt: bool = Field(True, description="Use LLM-based prompt rewriting for better results.")
    add_watermark: bool = Field(True, description="Add SynthID digital watermark. Must be False when seed is set.")
    safety_setting: Literal["block_low_and_above", "block_medium_and_above", "block_only_high"] = Field("block_medium_and_above", description="Safety filter level.")
    person_generation: Literal["allow_all", "allow_adult", "dont_allow"] = Field("allow_adult", description="Person generation policy.")
    output_format: Literal["PNG", "JPEG", "WEBP", "AVIF"] = Field("PNG", description="Output format: PNG (default), JPEG, WEBP, or AVIF. WEBP/AVIF are converted client-side from PNG.")
    compression_quality: int = Field(85, ge=0, le=100, description="JPEG compression quality (0-100). Only applies when output_format=JPEG.")
    storage_uri: Optional[str] = Field(None, description="Cloud Storage destination (e.g. gs://bucket/path/). Image is written directly to GCS.")

@mcp.tool()
async def tool_generate_image(params: GenerateImageParams) -> dict:
    """
    Generate an image from a text prompt using Vertex AI Imagen or Gemini.
    Use model_tier for simple model selection: fast/balanced/quality/ultra.
    balanced routes to gemini-2.5-flash-image (Gemini API path).
    """
    from model_registry import resolve_model

    if params.output_path:
        try:
            final_path = _validate_output_path(params.output_path)
        except ValueError as e:
            return {"success": False, "error": {"code": "VALIDATION", "message": str(e)}}
    elif params.output_filename:
        final_path = os.path.join(DEFAULT_OUTPUT_DIR, params.output_filename)
    else:
        return {"success": False, "error": {"code": "VALIDATION", "message": "Provide output_filename or output_path."}}

    resolved_model = params.model_name
    api_backend = "imagen"
    if params.model_tier:
        resolved_model, api_backend = resolve_model(params.model_tier, "generate")

    if api_backend == "gemini":
        return await gemini_generate_image(
            prompt=params.prompt,
            output_path=final_path,
            model_name=resolved_model,
            return_base64=params.return_base64,
            save_format=params.output_format.lower(),
        )

    return await generate_image(
        prompt=params.prompt,
        output_path=final_path,
        model_name=resolved_model,
        number_of_images=params.number_of_images,
        aspect_ratio=params.aspect_ratio,
        return_base64=params.return_base64,
        negative_prompt=params.negative_prompt,
        seed=params.seed,
        enhance_prompt=params.enhance_prompt,
        add_watermark=params.add_watermark,
        safety_setting=params.safety_setting,
        person_generation=params.person_generation,
        output_format=params.output_format,
        compression_quality=params.compression_quality,
        storage_uri=params.storage_uri,
    )

class EditImageParams(BaseModel):
    prompt: str = Field(..., description="Text description of the edits to apply.")
    base_image_path: str = Field(..., description="Absolute or relative path to the source image.")
    output_filename: Optional[str] = Field(None, description="Name of the file to save the edited image as.")
    output_path: Optional[str] = Field(None, description="Absolute path for the output file. Takes priority over output_filename.")
    mask_image_path: Optional[str] = Field(
        None,
        description=(
            "Optional path to a mask image (PNG; white = edit region, black = preserve). "
            "Required when edit_mode is EDIT_MODE_INPAINT_INSERTION, "
            "EDIT_MODE_INPAINT_REMOVAL, or EDIT_MODE_OUTPAINT."
        ),
    )
    edit_mode: str = Field(
        "EDIT_MODE_DEFAULT",
        description=(
            "Edit mode: EDIT_MODE_DEFAULT (mask-free, prompt-driven full image edit), "
            "EDIT_MODE_INPAINT_INSERTION (mask required), EDIT_MODE_INPAINT_REMOVAL (mask required), "
            "EDIT_MODE_OUTPAINT (mask required), EDIT_MODE_BGSWAP, EDIT_MODE_PRODUCT_IMAGE."
        ),
    )
    model_name: str = Field(
        "imagen-3.0-capability-001",
        description=(
            "Image edit model. Default: imagen-3.0-capability-001 (full mask + edit-mode support). "
            "Fallback: imagen-3.0-generate-002 (legacy schema, no mask, EDIT_MODE_DEFAULT only)."
        ),
    )
    negative_prompt: Optional[str] = Field(None, description="Optional negative prompt.")
    sample_count: int = Field(1, ge=1, le=4, description="Number of edited samples to generate.")
    return_base64: bool = Field(False, description="Whether to return the base64-encoded image.")
    save_format: Literal["png", "jpeg", "webp", "avif"] = Field("png", description="Output image format: png (default), jpeg, webp, or avif.")

@mcp.tool()
async def tool_edit_image(params: EditImageParams) -> dict:
    """
    Precision image editing via Imagen 3 Capability — mask-based inpaint/outpaint,
    background swap, product image, or mask-free prompt-driven edit.
    For free-form natural-language transforms (style transfer, scene rewriting),
    use tool_transform_image instead.
    """
    if params.output_path:
        try:
            final_path = _validate_output_path(params.output_path)
        except ValueError as e:
            return {"success": False, "error": {"code": "VALIDATION", "message": str(e)}}
    elif params.output_filename:
        final_path = os.path.join(DEFAULT_OUTPUT_DIR, params.output_filename)
    else:
        return {"success": False, "error": {"code": "VALIDATION", "message": "Provide output_filename or output_path."}}

    return await edit_image(
        prompt=params.prompt,
        base_image_path=params.base_image_path,
        output_path=final_path,
        mask_image_path=params.mask_image_path,
        edit_mode=params.edit_mode,
        model_name=params.model_name,
        negative_prompt=params.negative_prompt,
        sample_count=params.sample_count,
        return_base64=params.return_base64,
        save_format=params.save_format,
    )


class TransformImageParams(BaseModel):
    prompt: str = Field(..., description="Natural-language transformation instruction.")
    base_image_path: str = Field(..., description="Path to the primary input image.")
    output_filename: Optional[str] = Field(None, description="Name of the file to save the transformed image as.")
    output_path: Optional[str] = Field(None, description="Absolute path for the output file. Takes priority over output_filename.")
    model_tier: Optional[Literal["fast", "balanced", "quality", "ultra"]] = Field(None, description="Shorthand tier: fast / balanced / quality / ultra. Overrides model_name. fast/balanced → gemini-2.5-flash-image, quality/ultra → gemini-2.5-pro-image.")
    additional_image_paths: Optional[List[str]] = Field(
        None,
        description="Optional list of additional reference image paths (e.g. style refs).",
    )
    model_name: str = Field(
        "gemini-2.5-flash-image",
        description=(
            "Vertex AI Gemini image model. Verified available: gemini-2.5-flash-image. "
            "Newer variants (gemini-3.1-flash-image, gemini-3-pro-image) may exist in other "
            "projects/regions — check tool_list_available_models for the live list."
        ),
    )
    return_base64: bool = Field(False, description="Whether to return the base64-encoded image.")
    save_format: Literal["png", "jpeg", "webp", "avif"] = Field("png", description="Output image format: png (default), jpeg, webp, or avif.")

@mcp.tool()
async def tool_transform_image(params: TransformImageParams) -> dict:
    """
    Free-form 'image + text -> image' transformation via Gemini multimodal models.
    Use for style transfer, scene rewriting, or any natural-language image edit
    that does not require pixel-precise masking.
    """
    from model_registry import resolve_model

    if params.output_path:
        try:
            final_path = _validate_output_path(params.output_path)
        except ValueError as e:
            return {"success": False, "error": {"code": "VALIDATION", "message": str(e)}}
    elif params.output_filename:
        final_path = os.path.join(DEFAULT_OUTPUT_DIR, params.output_filename)
    else:
        return {"success": False, "error": {"code": "VALIDATION", "message": "Provide output_filename or output_path."}}

    resolved_model = params.model_name
    if params.model_tier:
        resolved_model, _ = resolve_model(params.model_tier, "transform")

    return await transform_image(
        prompt=params.prompt,
        base_image_path=params.base_image_path,
        output_path=final_path,
        additional_image_paths=params.additional_image_paths,
        model_name=resolved_model,
        return_base64=params.return_base64,
        save_format=params.save_format,
    )

class AnalyzeImageParams(BaseModel):
    prompt: str = Field(..., description="The question or instruction about the image.")
    image_path: str = Field(..., description="The path to the image to analyze.")
    model_name: str = Field(
        "gemini-2.5-flash",
        description=(
            "Vision model to use. GA: gemini-2.5-flash (fast, default), "
            "gemini-2.5-pro (highest reasoning), gemini-2.5-flash-lite (cheapest)."
        ),
    )
    thinking_level: str = Field("HIGH", description="Thinking/reasoning level (MINIMAL, LOW, MEDIUM, HIGH).")
    media_resolution: str = Field("MEDIUM", description="Media resolution for vision tasks (LOW, MEDIUM, HIGH, ULTRA_HIGH).")

@mcp.tool()
async def tool_analyze_image(params: AnalyzeImageParams) -> dict:
    """
    Analyze an image using a multimodal prompt with Vertex AI.
    """
    return await analyze_image(
        prompt=params.prompt,
        image_path=params.image_path,
        model_name=params.model_name,
        thinking_level=params.thinking_level,
        media_resolution=params.media_resolution
    )

class UpscaleImageParams(BaseModel):
    base_image_path: str = Field(..., description="The path to the image to upscale.")
    output_filename: Optional[str] = Field(None, description="The name of the file to save the upscaled image as.")
    output_path: Optional[str] = Field(None, description="Absolute path for the output file. Takes priority over output_filename.")
    model_name: str = Field(
        "imagen-3.0-generate-002",
        description="Model to use. GA: imagen-3.0-generate-002 (default). Other Imagen variants supported.",
    )
    return_base64: bool = Field(False, description="Whether to return the base64 encoded image.")
    save_format: Literal["png", "jpeg", "webp", "avif"] = Field("png", description="Output image format: png (default), jpeg, webp, or avif.")

@mcp.tool()
async def tool_upscale_image(params: UpscaleImageParams) -> dict:
    """
    Upscale a low resolution image.
    """
    if params.output_path:
        try:
            final_path = _validate_output_path(params.output_path)
        except ValueError as e:
            return {"success": False, "error": {"code": "VALIDATION", "message": str(e)}}
    elif params.output_filename:
        final_path = os.path.join(DEFAULT_OUTPUT_DIR, params.output_filename)
    else:
        return {"success": False, "error": {"code": "VALIDATION", "message": "Provide output_filename or output_path."}}

    return await upscale_image(
        base_image_path=params.base_image_path,
        output_path=final_path,
        model_name=params.model_name,
        return_base64=params.return_base64,
        save_format=params.save_format,
    )

class RemoveBackgroundParams(BaseModel):
    base_image_path: str = Field(..., description="The path to the image to process.")
    output_filename: Optional[str] = Field(None, description="The name of the file to save the background-removed image as.")
    output_path: Optional[str] = Field(None, description="Absolute path for the output file. Takes priority over output_filename.")
    model_name: str = Field(
        "imagen-3.0-capability-001",
        description="Model to use. imagen-3.0-capability-001 (default, required for BGSWAP). Do not use imagen-3.0-generate-002 for background removal.",
    )
    return_base64: bool = Field(False, description="Whether to return the base64 encoded image.")

@mcp.tool()
async def tool_remove_background(params: RemoveBackgroundParams) -> dict:
    """
    Remove background from an image.
    """
    if params.output_path:
        try:
            final_path = _validate_output_path(params.output_path)
        except ValueError as e:
            return {"success": False, "error": {"code": "VALIDATION", "message": str(e)}}
    elif params.output_filename:
        final_path = os.path.join(DEFAULT_OUTPUT_DIR, params.output_filename)
    else:
        return {"success": False, "error": {"code": "VALIDATION", "message": "Provide output_filename or output_path."}}

    return await remove_background(
        base_image_path=params.base_image_path,
        output_path=final_path,
        model_name=params.model_name,
        return_base64=params.return_base64
    )

class GenerateVideoParams(BaseModel):
    prompt: str = Field(..., description="The text description of the video to generate.")
    output_filename: Optional[str] = Field(None, description="Output filename (e.g. video.mp4). Required if output_path not given.")
    output_path: Optional[str] = Field(None, description="Absolute output file path. Takes priority over output_filename.")
    model_name: str = Field(
        "veo-3.1-fast-generate-001",
        description="Veo model. GA: veo-3.1-fast-generate-001 (default, low latency), veo-3.1-generate-001 (premium).",
    )
    model_tier: Optional[Literal["fast", "quality"]] = Field(None, description="fast / quality. Overrides model_name.")
    duration: Literal[4, 6, 8] = Field(4, description="Video duration in seconds.")
    resolution: Literal["720p", "1080p", "4k"] = Field("1080p", description="Output resolution.")
    aspect_ratio: Literal["16:9", "9:16"] = Field("16:9", description="Aspect ratio: 16:9 (landscape) or 9:16 (portrait).")
    audio_enabled: bool = Field(False, description="Enable audio generation (Veo 3+ only).")

@mcp.tool()
async def tool_generate_video(params: GenerateVideoParams) -> dict:
    """Generate a video from a text prompt using Veo."""
    from model_registry import resolve_model

    if params.output_path:
        try:
            final_path = _validate_output_path(params.output_path)
        except ValueError as e:
            return {"success": False, "error": {"code": "VALIDATION", "message": str(e)}}
    elif params.output_filename:
        final_path = os.path.join(DEFAULT_OUTPUT_DIR, params.output_filename)
    else:
        return {"success": False, "error": {"code": "VALIDATION", "message": "Provide output_filename or output_path."}}

    resolved_model = params.model_name
    if params.model_tier:
        resolved_model, _ = resolve_model(params.model_tier, "video")

    return await generate_video(
        prompt=params.prompt,
        output_path=final_path,
        model_name=resolved_model,
        duration=params.duration,
        resolution=params.resolution,
        aspect_ratio=params.aspect_ratio,
        audio_enabled=params.audio_enabled,
    )


class ImageToVideoParams(BaseModel):
    first_frame_path: str = Field(..., description="Absolute path to the image used as the first video frame.")
    prompt: str = Field("", description="Motion or scene description to guide video generation.")
    output_filename: Optional[str] = Field(None, description="Output filename (e.g. video.mp4). Required if output_path not given.")
    output_path: Optional[str] = Field(None, description="Absolute output file path. Takes priority over output_filename.")
    last_frame_path: Optional[str] = Field(None, description="Optional image for the last frame (first+last frame mode).")
    model_name: str = Field("veo-3.1-fast-generate-001", description="Veo model to use.")
    model_tier: Optional[Literal["fast", "quality"]] = Field(None, description="fast / quality. Overrides model_name.")
    duration: Literal[4, 6, 8] = Field(4, description="Video duration in seconds.")
    aspect_ratio: Literal["16:9", "9:16"] = Field("16:9", description="Aspect ratio: 16:9 or 9:16.")

@mcp.tool()
async def tool_image_to_video(params: ImageToVideoParams) -> dict:
    """Generate a video from a still image as the first frame using Veo."""
    from model_registry import resolve_model

    if params.output_path:
        try:
            final_path = _validate_output_path(params.output_path)
        except ValueError as e:
            return {"success": False, "error": {"code": "VALIDATION", "message": str(e)}}
    elif params.output_filename:
        final_path = os.path.join(DEFAULT_OUTPUT_DIR, params.output_filename)
    else:
        return {"success": False, "error": {"code": "VALIDATION", "message": "Provide output_filename or output_path."}}

    resolved_model = params.model_name
    if params.model_tier:
        resolved_model, _ = resolve_model(params.model_tier, "video")

    return await image_to_video(
        first_frame_path=params.first_frame_path,
        output_path=final_path,
        prompt=params.prompt,
        last_frame_path=params.last_frame_path,
        model_name=resolved_model,
        duration=params.duration,
        aspect_ratio=params.aspect_ratio,
    )

class ExtendVideoParams(BaseModel):
    video_path: str = Field(..., description="Absolute path to the source video to extend.")
    output_filename: Optional[str] = Field(None, description="Output filename. Required if output_path not given.")
    output_path: Optional[str] = Field(None, description="Absolute output file path.")
    prompt: str = Field("", description="Optional motion description to guide the extension.")
    extra_seconds: Literal[4, 6, 8] = Field(4, description="Seconds to add: 4, 6, or 8.")
    model_name: str = Field("veo-3.1-fast-generate-001", description="Veo model to use.")
    model_tier: Optional[Literal["fast", "quality"]] = Field(None, description="fast / quality. Overrides model_name.")

@mcp.tool()
async def tool_extend_video(params: ExtendVideoParams) -> dict:
    """Extend an existing video by generating additional seconds at the end."""
    from model_registry import resolve_model

    if params.output_path:
        try:
            final_path = _validate_output_path(params.output_path)
        except ValueError as e:
            return {"success": False, "error": {"code": "VALIDATION", "message": str(e)}}
    elif params.output_filename:
        final_path = os.path.join(DEFAULT_OUTPUT_DIR, params.output_filename)
    else:
        return {"success": False, "error": {"code": "VALIDATION", "message": "Provide output_filename or output_path."}}

    resolved_model = params.model_name
    if params.model_tier:
        resolved_model, _ = resolve_model(params.model_tier, "video")

    return await extend_video(
        video_path=params.video_path,
        output_path=final_path,
        prompt=params.prompt,
        extra_seconds=params.extra_seconds,
        model_name=resolved_model,
    )


class VideoObjectEditParams(BaseModel):
    video_path: str = Field(..., description="Absolute path to the source video.")
    operation: Literal["insert", "remove"] = Field(..., description="'insert' to add an object, 'remove' to delete one.")
    prompt: str = Field(..., description="Description of the object to insert or remove.")
    output_filename: Optional[str] = Field(None, description="Output filename. Required if output_path not given.")
    output_path: Optional[str] = Field(None, description="Absolute output file path.")
    model_name: str = Field("veo-3.1-fast-generate-001", description="Veo model to use.")
    model_tier: Optional[Literal["fast", "quality"]] = Field(None, description="fast / quality. Overrides model_name.")

@mcp.tool()
async def tool_video_object_edit(params: VideoObjectEditParams) -> dict:
    """Insert or remove an object in a video using Veo."""
    from model_registry import resolve_model

    if params.output_path:
        try:
            final_path = _validate_output_path(params.output_path)
        except ValueError as e:
            return {"success": False, "error": {"code": "VALIDATION", "message": str(e)}}
    elif params.output_filename:
        final_path = os.path.join(DEFAULT_OUTPUT_DIR, params.output_filename)
    else:
        return {"success": False, "error": {"code": "VALIDATION", "message": "Provide output_filename or output_path."}}

    resolved_model = params.model_name
    if params.model_tier:
        resolved_model, _ = resolve_model(params.model_tier, "video")

    return await video_object_edit(
        video_path=params.video_path,
        operation=params.operation,
        prompt=params.prompt,
        output_path=final_path,
        model_name=resolved_model,
    )


class UploadFileParams(BaseModel):
    file_path: str = Field(..., description="Absolute path to the local file to register.")
    mime_type: Optional[str] = Field(None, description="MIME type (auto-detected from extension if omitted).")
    display_name: Optional[str] = Field(None, description="Human-readable name for this file reference.")

@mcp.tool()
async def tool_upload_file(params: UploadFileParams) -> dict:
    """Register a local file for use as a reference image in other tools (e.g. tool_transform_image).
    Returns a file_uri to pass as additional_image_paths."""
    return await upload_file(
        file_path=params.file_path,
        mime_type=params.mime_type,
        display_name=params.display_name,
    )


class BatchGenerateParams(BaseModel):
    prompts: List[str] = Field(..., description="List of text prompts (max 10).")
    output_prefix: str = Field(..., description="Filename prefix. Files: <prefix>_0.png, <prefix>_1.png, ...")
    output_dir: Optional[str] = Field(None, description="Absolute directory path for output. Defaults to DEFAULT_OUTPUT_DIR.")
    model_name: str = Field("imagen-4.0-fast-generate-001", description="Imagen model to use for all prompts.")
    model_tier: Optional[Literal["fast", "quality", "ultra"]] = Field(None, description="fast / quality / ultra. Overrides model_name. ('balanced' is not supported for batch — it routes to Gemini which is Imagen-incompatible.)")
    aspect_ratio: str = Field("1:1", description="Aspect ratio for all generated images.")

@mcp.tool()
async def tool_batch_generate(params: BatchGenerateParams) -> dict:
    """Generate images for multiple prompts in a single call (max 10, max 4 concurrent)."""
    from model_registry import resolve_model

    resolved_model = params.model_name
    if params.model_tier:
        resolved_model, _ = resolve_model(params.model_tier, "generate")

    output_dir = params.output_dir or DEFAULT_OUTPUT_DIR

    return await batch_generate(
        prompts=params.prompts,
        output_prefix=params.output_prefix,
        output_dir=output_dir,
        model_name=resolved_model,
        aspect_ratio=params.aspect_ratio,
    )


class PipelineStepModel(BaseModel):
    tool: str = Field(..., description="Tool name: generate / edit / transform / upscale / remove_background")
    params: dict = Field(default_factory=dict, description="Parameters for this tool step (excluding output path, which is managed by the pipeline).")

class RunPipelineParams(BaseModel):
    steps: List[PipelineStepModel] = Field(..., description="Ordered list of pipeline steps.")
    output_path: Optional[str] = Field(None, description="Absolute path for the final output image.")
    output_filename: Optional[str] = Field(None, description="Output filename saved to DEFAULT_OUTPUT_DIR. Used if output_path not given.")

@mcp.tool()
async def tool_run_pipeline(params: RunPipelineParams) -> dict:
    """Chain image processing steps sequentially. Each step's output becomes the next step's input.
    Supported tools: generate, edit, transform, upscale, remove_background.
    Example: generate → remove_background → upscale."""

    if params.output_path:
        try:
            final_path = _validate_output_path(params.output_path)
        except ValueError as e:
            return {"success": False, "error": {"code": "VALIDATION", "message": str(e)}}
    elif params.output_filename:
        final_path = os.path.join(DEFAULT_OUTPUT_DIR, params.output_filename)
    else:
        return {"success": False, "error": {"code": "VALIDATION", "message": "Provide output_filename or output_path."}}

    steps_dicts = [{"tool": s.tool, "params": s.params} for s in params.steps]
    return await run_pipeline(steps=steps_dicts, output_path=final_path)


class GenerateMusicParams(BaseModel):
    prompt: str = Field(..., description="Text description of the music to generate.")
    output_filename: Optional[str] = Field(None, description="Output filename (e.g. track.mp3). Required if output_path not given.")
    output_path: Optional[str] = Field(None, description="Absolute output file path.")
    model_name: Literal["lyria-2", "lyria-3-pro", "lyria-3-clip"] = Field("lyria-2", description="Lyria model version. lyria-3-pro: highest quality; lyria-3-clip: short clips.")
    duration: Optional[int] = Field(None, description="Desired music duration in seconds (optional).")

@mcp.tool()
async def tool_generate_music(params: GenerateMusicParams) -> dict:
    """Generate music from a text prompt using Lyria. Check tool_list_available_models to verify Lyria is enabled in your project."""

    if params.output_path:
        try:
            final_path = _validate_output_path(params.output_path)
        except ValueError as e:
            return {"success": False, "error": {"code": "VALIDATION", "message": str(e)}}
    elif params.output_filename:
        final_path = os.path.join(DEFAULT_OUTPUT_DIR, params.output_filename)
    else:
        return {"success": False, "error": {"code": "VALIDATION", "message": "Provide output_filename or output_path."}}

    return await generate_music(
        prompt=params.prompt,
        output_path=final_path,
        model_name=params.model_name,
        duration=params.duration,
    )


class EmbedParams(BaseModel):
    text: str = Field(..., description="Text to embed into a float vector.")
    model: Optional[str] = Field(None, description="Override embedding model. Defaults to text-embedding-004 (Vertex AI) or gemini-embedding-2 (Gemini API).")

@mcp.tool()
async def tool_embed(params: EmbedParams) -> dict:
    """Generate a float embedding vector for the given text using Gemini Embedding."""
    from genai_tools import embed
    return await embed(text=params.text, model=params.model)


class AnalyzeVideoParams(BaseModel):
    video_path: str = Field(..., description="Absolute path to the local video file (max 20MB). Supported: mp4, mov, avi, webm, mkv.")
    prompt: str = Field(..., description="The question or analysis instruction about the video.")
    model_tier: Literal["fast", "quality"] = Field("fast", description="fast → gemini-3.5-flash, quality → gemini-3.1-pro.")
    model: Optional[str] = Field(None, description="Override the model name directly.")

@mcp.tool()
async def tool_analyze_video(params: AnalyzeVideoParams) -> dict:
    """Analyze a local video file using Gemini Vision. Supports mp4, mov, avi, webm, mkv. Max 20MB inline."""
    from genai_tools import analyze_video
    return await analyze_video(
        video_path=params.video_path,
        prompt=params.prompt,
        model_tier=params.model_tier,
        model=params.model,
    )


class GenerateSpeechParams(BaseModel):
    text: str = Field(..., description="Text to convert to speech.")
    voice: Literal["Aoede", "Charon", "Fenrir", "Kore", "Puck"] = Field("Kore", description="Voice to use for speech synthesis.")
    output_filename: Optional[str] = Field(None, description="Output filename (e.g. speech.wav). Saved to DEFAULT_OUTPUT_DIR.")
    output_path: Optional[str] = Field(None, description="Absolute output file path. Takes priority over output_filename.")
    model_tier: Literal["fast", "quality"] = Field("fast", description="fast → gemini-2.5-flash-preview-tts, quality → gemini-2.5-pro-preview-tts.")
    model: Optional[str] = Field(None, description="Override the model name directly.")

@mcp.tool()
async def tool_generate_speech(params: GenerateSpeechParams) -> dict:
    """Convert text to speech using Gemini TTS. Returns a WAV file path in outputs/."""
    from genai_tools import generate_speech
    output_path = params.output_path
    if not output_path and params.output_filename:
        output_path = os.path.join(DEFAULT_OUTPUT_DIR, params.output_filename)
    return await generate_speech(
        text=params.text,
        voice=params.voice,
        output_path=output_path,
        model_tier=params.model_tier,
        model=params.model,
    )


class LiveGenerateParams(BaseModel):
    prompt: str = Field(..., description="The text prompt to send to Gemini streaming.")
    model_tier: Literal["fast", "quality"] = Field("fast", description="fast → gemini-3.5-flash, quality → gemini-3.1-pro.")
    model: Optional[str] = Field(None, description="Override the model name directly.")

@mcp.tool()
async def tool_live_generate(params: LiveGenerateParams) -> dict:
    """Generate text using Gemini streaming. The full streamed response is accumulated and returned."""
    from genai_tools import live_generate
    return await live_generate(
        prompt=params.prompt,
        model_tier=params.model_tier,
        model=params.model,
    )


@mcp.resource("local://outputs/{filename}")
def get_generated_image(filename: str) -> bytes:
    """
    Serve generated images as resources directly to the MCP client.
    """
    file_path = os.path.join(DEFAULT_OUTPUT_DIR, filename)
    try:
        with open(file_path, "rb") as f:
            return f.read()
    except FileNotFoundError:
        raise ValueError(f"Image resource not found: {filename}")

# --- MCP Prompts ---

@mcp.prompt()
def character_design(character_type: str, setting: str) -> str:
    """
    Template for creating a detailed character design image prompt following Gemini 3 guidelines.
    """
    return f"First, verify if you can visualize a {character_type} in a {setting} setting. " \
           f"If yes, create a highly detailed concept art of this {character_type} in the {setting} setting. " \
           "Focus on intricate clothing details, realistic lighting, and cinematic composition. " \
           "Style: Digital Art, 8k resolution, trending on ArtStation. " \
           "Constraints: Do not include text, do not use a cartoonish style, and ensure the aspect ratio is 16:9."

@mcp.prompt()
def logo_concept(company_name: str, industry: str, style_keywords: str) -> str:
    """
    Template for designing a modern logo following Gemini 3 guidelines.
    """
    return f"First, consider the core visual elements of the {industry} industry. " \
           f"Based on that, design a modern, minimalist vector logo for a company named '{company_name}'. " \
           f"Incorporate these style keywords: {style_keywords}. " \
           "The logo should be flat, easily scalable, with a clean white background. " \
           "Constraints: Ensure there is no gradient, do not use more than 3 colors, and keep the typography highly legible."

@mcp.prompt()
def UI_UX_mockup(app_purpose: str, color_palette: str) -> str:
    """
    Template for generating UI/UX mockups following Gemini 3 guidelines.
    """
    return f"First, identify the primary user flow for an app designed for {app_purpose}. " \
           f"Then, generate a high-fidelity UI/UX screen mockup for this mobile app. " \
           f"Use a color palette based on {color_palette}. " \
           "Include modern UI elements like glassmorphism, rounded corners, soft shadows, and clean typography. " \
           "Constraints: Do not include low-contrast elements, ensure the layout is mobile-first, and do not use generic placeholder text like 'Lorem Ipsum'."

def main():
    logger.info(f"Starting OpenGoogleImageGeneratorMCP v{__version__}...")
    check_for_updates()
    mcp.run()


if __name__ == "__main__":
    main()
