# Developer: Mirac Orhan (mirac.orhan@gmail.com)
# License: Open Source (MIT License) - Free for everyone to use, modify, and distribute.

import asyncio
import os
import base64
import json
import shutil
import subprocess
import time
import urllib.parse
import urllib.request
import urllib.error
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List, Tuple

from vertexai.generative_models import GenerativeModel, Part, GenerationConfig
from config import logger, PROJECT_ID, LOCATION

API_TIMEOUT = 90.0
HTTP_TIMEOUT = 90.0
TOKEN_TIMEOUT = 30.0

_GCLOUD_WINDOWS_FALLBACK = r"C:\Program Files (x86)\Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd"

_LOG_FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "vertex_ai_mcp.log")
_DOCS_URL = "https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/google-models"

SUPPORTED_IMAGE_MODELS = (
    "imagen-4.0-fast-generate-001",
    "imagen-4.0-generate-001",
    "imagen-3.0-generate-002",
    "imagen-3.0-capability-001",
)

SUPPORTED_EDIT_MODES = (
    "EDIT_MODE_DEFAULT",
    "EDIT_MODE_INPAINT_INSERTION",
    "EDIT_MODE_INPAINT_REMOVAL",
    "EDIT_MODE_OUTPAINT",
    "EDIT_MODE_BGSWAP",
    "EDIT_MODE_PRODUCT_IMAGE",
)

MASK_REQUIRED_MODES = (
    "EDIT_MODE_INPAINT_INSERTION",
    "EDIT_MODE_INPAINT_REMOVAL",
    "EDIT_MODE_OUTPAINT",
)

# Candidate models to probe (publisher_id, endpoint, category)
_PROBE_CANDIDATES: Tuple[Tuple[str, str, str], ...] = (
    ("imagen-4.0-fast-generate-001",   ":predict",         "image_generation"),
    ("imagen-4.0-generate-001",        ":predict",         "image_generation"),
    ("imagen-3.0-generate-002",        ":predict",         "image_generation"),
    ("imagen-3.0-capability-001",      ":predict",         "image_generation"),
    ("gemini-3.1-flash-image",         ":generateContent", "image_transformation"),
    ("gemini-3-pro-image",             ":generateContent", "image_transformation"),
    ("gemini-3-flash",                 ":generateContent", "image_transformation"),
    ("gemini-2.5-flash-image",         ":generateContent", "image_transformation"),
    ("gemini-2.5-flash-image-preview", ":generateContent", "image_transformation"),
    ("gemini-3.1-pro",                 ":generateContent", "text"),
    ("gemini-3-pro",                   ":generateContent", "text"),
    ("gemini-2.5-pro",                 ":generateContent", "text"),
    ("gemini-2.5-flash",               ":generateContent", "text"),
    ("gemini-3.1-flash-lite",          ":generateContent", "text"),
    ("gemini-2.5-flash-lite",          ":generateContent", "text"),
)

# Modules-scope cache: {category: [available model names]}
_AVAILABLE_MODELS_CACHE: Dict[str, List[str]] = {}
_AVAILABLE_MODELS_CACHE_AT: Optional[str] = None

_gemini_model_cache: Dict[str, Any] = {}
_cached_token: Optional[str] = None
_cached_token_expiry: float = 0.0  # epoch seconds


def _is_imagen_model(model_name: str) -> bool:
    return model_name.startswith("imagen") or model_name.startswith("imagegeneration")


def _save_image_bytes(image_bytes: bytes, output_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(image_bytes)


def _encode_base64(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode("utf-8")


def _read_image_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _mime_for_path(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".png":
        return "image/png"
    if ext == ".webp":
        return "image/webp"
    return "image/jpeg"


def _validate_output_path(path: str) -> str:
    """Ensure output_path is absolute and contains no '..' components."""
    if not os.path.isabs(path):
        raise ValueError(f"output_path must be an absolute path. Got: {path!r}")
    if ".." in path.replace("\\", "/").split("/"):
        raise ValueError(f"output_path must not contain '..' components. Got: {path!r}")
    return os.path.abspath(path)


async def _to_thread(func, *args, timeout: float = API_TIMEOUT, **kwargs):
    return await asyncio.wait_for(asyncio.to_thread(func, *args, **kwargs), timeout=timeout)


def _resolve_gcloud_path() -> str:
    """Locate gcloud executable. shutil.which on PATH first, then Windows default install."""
    for candidate in ("gcloud.cmd", "gcloud"):
        found = shutil.which(candidate)
        if found:
            return found
    if os.name == "nt" and os.path.exists(_GCLOUD_WINDOWS_FALLBACK):
        return _GCLOUD_WINDOWS_FALLBACK
    raise RuntimeError(
        "gcloud not found on PATH and Windows fallback path does not exist. "
        "Install Google Cloud SDK or add it to PATH."
    )


def _try_adc_refresh_token() -> Optional[str]:
    """Try direct refresh-token grant from ADC JSON. Returns None if unavailable
    or if account requires gcloud-mediated reauth (invalid_rapt)."""
    adc_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if not adc_path:
        appdata = os.environ.get("APPDATA")
        if appdata:
            adc_path = os.path.join(appdata, "gcloud", "application_default_credentials.json")
    if not adc_path or not os.path.exists(adc_path):
        return None
    try:
        with open(adc_path, "r", encoding="utf-8") as f:
            adc = json.load(f)
        if adc.get("type") != "authorized_user":
            return None
        body = urllib.parse.urlencode({
            "client_id": adc["client_id"],
            "client_secret": adc["client_secret"],
            "refresh_token": adc["refresh_token"],
            "grant_type": "refresh_token",
        }).encode("utf-8")
        req = urllib.request.Request("https://oauth2.googleapis.com/token", data=body, method="POST")
        req.add_header("Content-Type", "application/x-www-form-urlencoded")
        with urllib.request.urlopen(req, timeout=15) as r:
            payload = json.loads(r.read().decode("utf-8"))
        return payload.get("access_token")
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        logger.info(f"[auth] ADC refresh grant failed ({e.code}): {detail[:120]} - falling back to gcloud")
        return None
    except Exception as e:
        logger.info(f"[auth] ADC refresh skipped ({type(e).__name__}: {e}) - falling back to gcloud")
        return None


def _get_access_token() -> str:
    """Return a fresh access token, cached for ~50 minutes.

    Prefers a direct ADC refresh-token grant (no subprocess). Falls back to
    invoking gcloud — with explicit gcloud.cmd path, stdin=DEVNULL (critical:
    in MCP subprocess context, inheriting the JSON-RPC stdin causes gcloud's
    child python to hang), and no shell."""
    global _cached_token, _cached_token_expiry
    if _cached_token and time.time() < _cached_token_expiry:
        return _cached_token
    t = time.time()

    token = _try_adc_refresh_token()
    if token:
        _cached_token = token
        _cached_token_expiry = time.time() + 50 * 60
        logger.info(f"[auth] got token via ADC refresh in {time.time()-t:.1f}s (len={len(token)})")
        return token

    gcloud_path = _resolve_gcloud_path()
    logger.info(f"[auth] invoking '{gcloud_path} auth print-access-token' ...")
    proc = subprocess.run(
        [gcloud_path, "auth", "print-access-token"],
        capture_output=True,
        text=True,
        timeout=TOKEN_TIMEOUT,
        stdin=subprocess.DEVNULL,
        shell=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"gcloud auth print-access-token failed (rc={proc.returncode}): {proc.stderr.strip()}")
    token = proc.stdout.strip()
    if not token:
        raise RuntimeError("gcloud returned empty token")
    _cached_token = token
    _cached_token_expiry = time.time() + 50 * 60
    logger.info(f"[auth] got token via gcloud in {time.time()-t:.1f}s (len={len(token)})")
    return token


# ---------------------------------------------------------------------------
# Error handling — central helper used by all REST calls
# ---------------------------------------------------------------------------

class VertexAPIError(Exception):
    """Carries the structured error dict so async wrappers can re-raise and
    callers can build their `{success: False, error: {...}}` response."""

    def __init__(self, error_dict: Dict[str, Any]):
        super().__init__(error_dict.get("message", "Vertex API error"))
        self.error_dict = error_dict


def _alternatives_for(category: str, excluding: str, limit: int = 3) -> List[str]:
    """Suggest available models from the live cache, excluding the failing one."""
    avail = _AVAILABLE_MODELS_CACHE.get(category, [])
    return [m for m in avail if m != excluding][:limit]


def _hint_for(code: int, model_name: str, endpoint: str, detail: str, retry_after: Optional[str]) -> str:
    if code == 400:
        return f"Check parameters; Vertex returned: {detail[:200]}"
    if code == 401:
        return "Auth token rejected. Run `gcloud auth application-default login` and retry."
    if code == 403:
        return (
            f"Project '{PROJECT_ID}' lacks 'aiplatform.endpoints.predict' on '{model_name}'. "
            f"Grant role 'roles/aiplatform.user' and ensure Vertex AI API is enabled."
        )
    if code == 404:
        category = (
            "image_transformation" if endpoint == ":generateContent" and "image" in model_name
            else "image_generation" if endpoint == ":predict"
            else "text" if endpoint == ":generateContent"
            else ""
        )
        alts = _alternatives_for(category, excluding=model_name) if category else []
        alt_str = f" Try: {', '.join(alts)}." if alts else " Run tool_list_available_models for a live list."
        return f"Model '{model_name}' not found in project '{PROJECT_ID}' / location '{LOCATION}'.{alt_str}"
    if code == 429:
        ra = f" Retry after {retry_after}s." if retry_after else ""
        return f"Quota exceeded for {model_name} in {LOCATION}.{ra} Or request a quota increase."
    if code in (500, 502, 503, 504):
        return f"Transient Vertex AI error ({code}). Safe to retry once."
    return f"Unhandled Vertex error ({code})."


def _handle_vertex_http_error(
    e: urllib.error.HTTPError,
    model_name: str,
    endpoint: str,
    duration_s: float,
) -> Dict[str, Any]:
    """Build a structured error dict from a urllib HTTPError."""
    try:
        raw = e.read().decode("utf-8", errors="replace")
    except Exception:
        raw = ""
    try:
        parsed = json.loads(raw)
        msg = (parsed.get("error", {}) or {}).get("message", raw[:500])
    except Exception:
        msg = raw[:500] if raw else f"HTTP {e.code}"
    retry_after = e.headers.get("Retry-After") if hasattr(e, "headers") and e.headers else None

    error_dict = {
        "code": e.code,
        "model": model_name,
        "endpoint": endpoint,
        "message": msg,
        "hint": _hint_for(e.code, model_name, endpoint, msg, retry_after),
        "docs_url": _DOCS_URL,
        "log_path": _LOG_FILE_PATH,
        "duration_s": round(duration_s, 2),
    }
    logger.error(
        f"[vertex] model={model_name} endpoint={endpoint} status={e.code} "
        f"dur={duration_s:.2f}s detail={msg[:200]!r}"
    )
    return error_dict


def _build_timeout_error(model_name: str, endpoint: str, duration_s: float) -> Dict[str, Any]:
    err = {
        "code": "TIMEOUT",
        "model": model_name,
        "endpoint": endpoint,
        "message": f"Request to {model_name} exceeded {duration_s:.0f}s timeout.",
        "hint": "Try a faster variant (e.g. an -fast- model) or reduce sampleCount.",
        "docs_url": _DOCS_URL,
        "log_path": _LOG_FILE_PATH,
        "duration_s": round(duration_s, 2),
    }
    logger.error(f"[vertex] model={model_name} endpoint={endpoint} status=TIMEOUT dur={duration_s:.2f}s")
    return err


def _build_validation_error(message: str) -> Dict[str, Any]:
    err = {
        "code": "VALIDATION",
        "message": message,
        "hint": "Fix the parameters and retry. No HTTP call was made.",
        "log_path": _LOG_FILE_PATH,
    }
    logger.warning(f"[validation] {message}")
    return err


def _build_unexpected_error(model_name: str, endpoint: str, exc: Exception, duration_s: float) -> Dict[str, Any]:
    err = {
        "code": type(exc).__name__,
        "model": model_name,
        "endpoint": endpoint,
        "message": str(exc),
        "hint": "Unexpected error. See log file for full trace.",
        "log_path": _LOG_FILE_PATH,
        "duration_s": round(duration_s, 2),
    }
    logger.exception(f"[vertex] unexpected error model={model_name} endpoint={endpoint} dur={duration_s:.2f}s")
    return err


# ---------------------------------------------------------------------------
# REST helpers — :predict (Imagen) and :generateContent (Gemini)
# ---------------------------------------------------------------------------

def _vertex_post(url: str, payload: dict, model_name: str, endpoint_tag: str) -> dict:
    """Generic sync POST. Raises VertexAPIError on HTTP error with structured info."""
    t0 = time.time()
    token = _get_access_token()
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Content-Type", "application/json")
    logger.info(f"[vertex] POST model={model_name} endpoint={endpoint_tag} body_size={len(body)}")
    t1 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
            raw = resp.read()
            dur = time.time() - t1
            logger.info(
                f"[vertex] model={model_name} endpoint={endpoint_tag} status=200 "
                f"dur={dur:.2f}s response_size={len(raw)}"
            )
            return json.loads(raw.decode("utf-8"))
    except urllib.error.HTTPError as e:
        dur = time.time() - t1
        err = _handle_vertex_http_error(e, model_name, endpoint_tag, dur)
        raise VertexAPIError(err) from e


def _imagen_predict(model_name: str, payload: dict) -> dict:
    """Synchronous REST call to publisher model :predict endpoint."""
    url = (
        f"https://{LOCATION}-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}"
        f"/locations/{LOCATION}/publishers/google/models/{model_name}:predict"
    )
    return _vertex_post(url, payload, model_name, ":predict")


def _gemini_generate_content(model_name: str, contents: list, generation_config: Optional[dict] = None) -> dict:
    """Synchronous REST call to publisher model :generateContent endpoint."""
    url = (
        f"https://{LOCATION}-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}"
        f"/locations/{LOCATION}/publishers/google/models/{model_name}:generateContent"
    )
    payload: Dict[str, Any] = {"contents": contents}
    if generation_config:
        payload["generationConfig"] = generation_config
    return _vertex_post(url, payload, model_name, ":generateContent")


async def _get_gemini_model(model_name: str):
    if model_name in _gemini_model_cache:
        return _gemini_model_cache[model_name]
    model = await _to_thread(GenerativeModel, model_name, timeout=API_TIMEOUT)
    _gemini_model_cache[model_name] = model
    return model


def _unsupported_image_model_error(model_name: str) -> str:
    return (
        f"Model '{model_name}' is not a supported image-generation model. "
        f"Use one of: {', '.join(SUPPORTED_IMAGE_MODELS)}."
    )


def _extract_image_bytes_list(response: dict) -> list:
    preds = response.get("predictions") or []
    out = []
    for p in preds:
        b64 = p.get("bytesBase64Encoded")
        if b64:
            out.append(base64.b64decode(b64))
    return out


def _extract_gemini_image_bytes(response: dict) -> Optional[bytes]:
    """Walk a :generateContent response and return the first inline image."""
    for cand in response.get("candidates", []) or []:
        for part in (cand.get("content", {}) or {}).get("parts", []) or []:
            inline = part.get("inlineData") or part.get("inline_data")
            if inline and inline.get("data"):
                return base64.b64decode(inline["data"])
    return None


def _extract_gemini_text(response: dict) -> Optional[str]:
    parts_text: List[str] = []
    for cand in response.get("candidates", []) or []:
        for part in (cand.get("content", {}) or {}).get("parts", []) or []:
            if part.get("text"):
                parts_text.append(part["text"])
    return "\n".join(parts_text) if parts_text else None


# ---------------------------------------------------------------------------
# Live availability probe (used by tool_list_available_models)
# ---------------------------------------------------------------------------

def _probe_one(model_name: str, endpoint: str) -> bool:
    """Returns True if model responds 200 or 400 (= reachable). False on 404."""
    url = (
        f"https://{LOCATION}-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}"
        f"/locations/{LOCATION}/publishers/google/models/{model_name}{endpoint}"
    )
    body = b'{"instances":[{}],"parameters":{}}' if endpoint == ":predict" \
        else b'{"contents":[{"role":"user","parts":[{"text":"hi"}]}]}'
    try:
        token = _get_access_token()
    except Exception as e:
        logger.warning(f"[probe] cannot get token: {e}")
        return False
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Content-Type", "application/json")
    try:
        # 20s: some Gemini endpoints take 15-20s even for a trivial "hi" prompt
        urllib.request.urlopen(req, timeout=20)
        return True
    except urllib.error.HTTPError as e:
        return e.code == 400  # 400 = model exists but our payload was minimal
    except Exception:
        return False


def probe_available_models(force: bool = False) -> Dict[str, List[str]]:
    """Probe all candidate models, return {category: [available]}.
    Caches the result for the process lifetime."""
    global _AVAILABLE_MODELS_CACHE, _AVAILABLE_MODELS_CACHE_AT
    if _AVAILABLE_MODELS_CACHE and not force:
        return _AVAILABLE_MODELS_CACHE
    t0 = time.time()
    logger.info(f"[probe] starting model availability scan ({len(_PROBE_CANDIDATES)} candidates) ...")
    result: Dict[str, List[str]] = {}
    for model, endpoint, category in _PROBE_CANDIDATES:
        ok = _probe_one(model, endpoint)
        logger.info(f"[probe] {model} ({category}) -> {'AVAILABLE' if ok else 'NOT FOUND'}")
        if ok:
            result.setdefault(category, []).append(model)
    # Add convenience aliases: text-capable Gemini models are also vision-capable
    if "text" in result:
        result["vision"] = [m for m in result["text"] if m.startswith("gemini-")]
    # Imagen capability model can also act as image_generation
    if "image_generation" in result and "image_transformation" not in result:
        result.setdefault("image_transformation", [])
    _AVAILABLE_MODELS_CACHE = result
    _AVAILABLE_MODELS_CACHE_AT = datetime.now(timezone.utc).isoformat()
    logger.info(f"[probe] scan finished in {time.time()-t0:.1f}s: { {k: len(v) for k, v in result.items()} }")
    return result


def get_cached_availability() -> Dict[str, Any]:
    return {
        "available": dict(_AVAILABLE_MODELS_CACHE),
        "checked_at": _AVAILABLE_MODELS_CACHE_AT,
    }


# ---------------------------------------------------------------------------
# Public async tool functions
# ---------------------------------------------------------------------------

async def generate_image(
    prompt: str,
    output_path: Optional[str] = None,
    model_name: str = "imagen-4.0-fast-generate-001",
    number_of_images: int = 1,
    aspect_ratio: str = "1:1",
    return_base64: bool = False,
) -> Dict[str, Any]:
    t0 = time.time()
    logger.info(f"[generate_image] START | model={model_name} | prompt='{prompt[:80]}...' | aspect={aspect_ratio} | n={number_of_images}")

    if not _is_imagen_model(model_name):
        return {"success": False, "error": _build_validation_error(_unsupported_image_model_error(model_name))}

    try:
        payload = {
            "instances": [{"prompt": prompt}],
            "parameters": {"sampleCount": number_of_images, "aspectRatio": aspect_ratio},
        }
        response = await _to_thread(_imagen_predict, model_name, payload, timeout=API_TIMEOUT)
        images = _extract_image_bytes_list(response)
        if not images:
            return {"success": False, "error": _build_unexpected_error(
                model_name, ":predict",
                ValueError(f"Model returned no images. Response keys: {list(response.keys())}"),
                time.time() - t0,
            )}

        results = []
        for i, image_bytes in enumerate(images):
            res: Dict[str, Any] = {}
            if return_base64:
                res["base64"] = _encode_base64(image_bytes)
                res["mime_type"] = "image/png"
            if output_path:
                current_out = output_path
                if number_of_images > 1:
                    base, ext = os.path.splitext(output_path)
                    current_out = f"{base}_{i}{ext}"
                _save_image_bytes(image_bytes, current_out)
                res["path"] = current_out
            results.append(res)
        logger.info(f"[generate_image] SUCCESS in {time.time()-t0:.1f}s")
        return {"success": True, "results": results}

    except VertexAPIError as e:
        return {"success": False, "error": e.error_dict}
    except asyncio.TimeoutError:
        return {"success": False, "error": _build_timeout_error(model_name, ":predict", time.time() - t0)}
    except Exception as e:
        return {"success": False, "error": _build_unexpected_error(model_name, ":predict", e, time.time() - t0)}


async def edit_image(
    prompt: str,
    base_image_path: str,
    output_path: Optional[str] = None,
    mask_image_path: Optional[str] = None,
    edit_mode: str = "EDIT_MODE_DEFAULT",
    model_name: str = "imagen-3.0-capability-001",
    negative_prompt: Optional[str] = None,
    sample_count: int = 1,
    return_base64: bool = False,
) -> Dict[str, Any]:
    """Precision image editing via Imagen 3 Capability.

    Supported edit_mode values:
      EDIT_MODE_DEFAULT          — mask-free, prompt-driven full image edit
      EDIT_MODE_INPAINT_INSERTION — mask required: add object into masked region
      EDIT_MODE_INPAINT_REMOVAL   — mask required: remove content in masked region
      EDIT_MODE_OUTPAINT          — mask required: extend image beyond original
      EDIT_MODE_BGSWAP            — swap the background
      EDIT_MODE_PRODUCT_IMAGE     — product reference styling
    """
    t0 = time.time()
    logger.info(
        f"[edit_image] START | model={model_name} | mode={edit_mode} | "
        f"base={base_image_path} | mask={mask_image_path}"
    )

    # ---- Validation (no HTTP if invalid) ----
    if not _is_imagen_model(model_name):
        return {"success": False, "error": _build_validation_error(_unsupported_image_model_error(model_name))}
    if edit_mode not in SUPPORTED_EDIT_MODES:
        return {"success": False, "error": _build_validation_error(
            f"edit_mode '{edit_mode}' is not supported. Use one of: {', '.join(SUPPORTED_EDIT_MODES)}."
        )}
    if edit_mode in MASK_REQUIRED_MODES and not mask_image_path:
        return {"success": False, "error": _build_validation_error(
            f"edit_mode '{edit_mode}' requires mask_image_path."
        )}
    if not os.path.exists(base_image_path):
        return {"success": False, "error": _build_validation_error(
            f"Base image not found: {base_image_path}"
        )}
    if mask_image_path and not os.path.exists(mask_image_path):
        return {"success": False, "error": _build_validation_error(
            f"Mask image not found: {mask_image_path}"
        )}
    if not (1 <= sample_count <= 4):
        return {"success": False, "error": _build_validation_error(
            f"sample_count must be in [1, 4], got {sample_count}."
        )}

    try:
        base_b64 = _read_image_b64(base_image_path)

        # Capability model uses referenceImages[]; legacy generate-002 uses image{}
        use_reference_schema = "capability" in model_name

        if use_reference_schema:
            reference_images: List[Dict[str, Any]] = [{
                "referenceType": "REFERENCE_TYPE_RAW",
                "referenceId": 1,
                "referenceImage": {"bytesBase64Encoded": base_b64},
            }]
            if mask_image_path:
                mask_b64 = _read_image_b64(mask_image_path)
                reference_images.append({
                    "referenceType": "REFERENCE_TYPE_MASK",
                    "referenceId": 2,
                    "referenceImage": {"bytesBase64Encoded": mask_b64},
                    "maskImageConfig": {"maskMode": "MASK_MODE_USER_PROVIDED"},
                })
            instance: Dict[str, Any] = {"prompt": prompt, "referenceImages": reference_images}
        else:
            # Legacy schema fallback (imagen-3.0-generate-002, no capability model features)
            if mask_image_path:
                return {"success": False, "error": _build_validation_error(
                    f"Model '{model_name}' does not support mask_image_path. "
                    f"Use 'imagen-3.0-capability-001' for mask-based edits."
                )}
            instance = {"prompt": prompt, "image": {"bytesBase64Encoded": base_b64}}

        parameters: Dict[str, Any] = {"sampleCount": sample_count, "editMode": edit_mode}
        if negative_prompt:
            parameters["negativePrompt"] = negative_prompt

        payload = {"instances": [instance], "parameters": parameters}

        response = await _to_thread(_imagen_predict, model_name, payload, timeout=API_TIMEOUT)
        images = _extract_image_bytes_list(response)
        if not images:
            return {"success": False, "error": _build_unexpected_error(
                model_name, ":predict",
                ValueError(f"Model returned no edited images. Response keys: {list(response.keys())}"),
                time.time() - t0,
            )}

        results = []
        for i, image_bytes in enumerate(images):
            res: Dict[str, Any] = {}
            if return_base64:
                res["base64"] = _encode_base64(image_bytes)
                res["mime_type"] = "image/png"
            if output_path:
                current_out = output_path
                if len(images) > 1:
                    base, ext = os.path.splitext(output_path)
                    current_out = f"{base}_{i}{ext}"
                _save_image_bytes(image_bytes, current_out)
                res["path"] = current_out
            results.append(res)
        logger.info(f"[edit_image] SUCCESS in {time.time()-t0:.1f}s (n={len(images)})")
        return {"success": True, "results": results}

    except VertexAPIError as e:
        return {"success": False, "error": e.error_dict}
    except asyncio.TimeoutError:
        return {"success": False, "error": _build_timeout_error(model_name, ":predict", time.time() - t0)}
    except Exception as e:
        return {"success": False, "error": _build_unexpected_error(model_name, ":predict", e, time.time() - t0)}


async def transform_image(
    prompt: str,
    base_image_path: str,
    output_path: Optional[str] = None,
    additional_image_paths: Optional[List[str]] = None,
    model_name: str = "gemini-2.5-flash-image",
    return_base64: bool = False,
) -> Dict[str, Any]:
    """Free-form natural-language image transformation via Gemini multimodal models.

    Accepts a base image plus optional additional reference images and a prompt.
    Returns an image generated by the model. Use this for style transfers,
    scene rewriting, or any 'image + text -> image' task that does not require
    pixel-precise masking. For precision edits, use edit_image with Imagen
    Capability instead.
    """
    t0 = time.time()
    logger.info(
        f"[transform_image] START | model={model_name} | base={base_image_path} | "
        f"extras={additional_image_paths or []}"
    )

    # ---- Validation ----
    if not os.path.exists(base_image_path):
        return {"success": False, "error": _build_validation_error(
            f"Base image not found: {base_image_path}"
        )}
    if additional_image_paths:
        for p in additional_image_paths:
            if not os.path.exists(p):
                return {"success": False, "error": _build_validation_error(
                    f"Additional image not found: {p}"
                )}

    try:
        parts: List[Dict[str, Any]] = []
        parts.append({"inlineData": {"mimeType": _mime_for_path(base_image_path),
                                     "data": _read_image_b64(base_image_path)}})
        if additional_image_paths:
            for p in additional_image_paths:
                parts.append({"inlineData": {"mimeType": _mime_for_path(p),
                                             "data": _read_image_b64(p)}})
        parts.append({"text": prompt})

        contents = [{"role": "user", "parts": parts}]
        generation_config = {"responseModalities": ["IMAGE", "TEXT"]}

        response = await _to_thread(
            _gemini_generate_content, model_name, contents, generation_config, timeout=API_TIMEOUT
        )
        image_bytes = _extract_gemini_image_bytes(response)
        if not image_bytes:
            text_fallback = _extract_gemini_text(response)
            err_msg = (
                f"Model returned no image. Text fallback: {text_fallback[:200]!r}"
                if text_fallback else
                f"Model returned no image. Response keys: {list(response.keys())}"
            )
            return {"success": False, "error": _build_unexpected_error(
                model_name, ":generateContent", ValueError(err_msg), time.time() - t0,
            )}

        res: Dict[str, Any] = {}
        if return_base64:
            res["base64"] = _encode_base64(image_bytes)
            res["mime_type"] = "image/png"
        if output_path:
            _save_image_bytes(image_bytes, output_path)
            res["path"] = output_path
        logger.info(f"[transform_image] SUCCESS in {time.time()-t0:.1f}s")
        return {"success": True, "results": [res]}

    except VertexAPIError as e:
        return {"success": False, "error": e.error_dict}
    except asyncio.TimeoutError:
        return {"success": False, "error": _build_timeout_error(model_name, ":generateContent", time.time() - t0)}
    except Exception as e:
        return {"success": False, "error": _build_unexpected_error(model_name, ":generateContent", e, time.time() - t0)}


async def analyze_image(
    prompt: str,
    image_path: str,
    model_name: str = "gemini-2.5-flash",
    safety_settings: Optional[Dict] = None,
    thinking_level: str = "HIGH",
    media_resolution: str = "MEDIUM",
) -> Dict[str, Any]:
    t0 = time.time()
    logger.info(f"[analyze_image] START | model={model_name} | image={image_path}")
    try:
        if not os.path.exists(image_path):
            return {"success": False, "error": _build_validation_error(f"Image not found: {image_path}")}

        with open(image_path, "rb") as f:
            image_data = f.read()
        mime = _mime_for_path(image_path)

        model = await _get_gemini_model(model_name)
        image_part = Part.from_data(mime_type=mime, data=image_data)

        response = await asyncio.wait_for(
            model.generate_content_async(
                [image_part, prompt],
                safety_settings=safety_settings,
                generation_config=GenerationConfig(temperature=1.0),
            ),
            timeout=API_TIMEOUT,
        )
        logger.info(f"[analyze_image] SUCCESS in {time.time()-t0:.1f}s")
        return {"success": True, "analysis": response.text}

    except asyncio.TimeoutError:
        return {"success": False, "error": _build_timeout_error(model_name, ":generateContent", time.time() - t0)}
    except Exception as e:
        return {"success": False, "error": _build_unexpected_error(model_name, ":generateContent", e, time.time() - t0)}


async def upscale_image(
    base_image_path: str,
    output_path: Optional[str] = None,
    model_name: str = "imagen-3.0-generate-002",
    return_base64: bool = False,
) -> Dict[str, Any]:
    t0 = time.time()
    logger.info(f"[upscale_image] START | model={model_name} | base={base_image_path}")

    if not _is_imagen_model(model_name):
        return {"success": False, "error": _build_validation_error(_unsupported_image_model_error(model_name))}
    if not os.path.exists(base_image_path):
        return {"success": False, "error": _build_validation_error(f"Image not found: {base_image_path}")}

    try:
        base_b64 = _read_image_b64(base_image_path)
        payload = {
            "instances": [{"image": {"bytesBase64Encoded": base_b64}}],
            "parameters": {"sampleCount": 1, "mode": "upscale", "upscaleConfig": {"upscaleFactor": "x2"}},
        }
        response = await _to_thread(_imagen_predict, model_name, payload, timeout=API_TIMEOUT)
        images = _extract_image_bytes_list(response)
        if not images:
            return {"success": False, "error": _build_unexpected_error(
                model_name, ":predict",
                ValueError(f"Model returned no upscaled image. Response keys: {list(response.keys())}"),
                time.time() - t0,
            )}
        image_bytes = images[0]

        res: Dict[str, Any] = {}
        if return_base64:
            res["base64"] = _encode_base64(image_bytes)
            res["mime_type"] = "image/png"
        if output_path:
            _save_image_bytes(image_bytes, output_path)
            res["path"] = output_path
        logger.info(f"[upscale_image] SUCCESS in {time.time()-t0:.1f}s")
        return {"success": True, "results": [res]}

    except VertexAPIError as e:
        return {"success": False, "error": e.error_dict}
    except asyncio.TimeoutError:
        return {"success": False, "error": _build_timeout_error(model_name, ":predict", time.time() - t0)}
    except Exception as e:
        return {"success": False, "error": _build_unexpected_error(model_name, ":predict", e, time.time() - t0)}


async def remove_background(
    base_image_path: str,
    output_path: Optional[str] = None,
    model_name: str = "imagen-3.0-capability-001",
    return_base64: bool = False,
) -> Dict[str, Any]:
    t0 = time.time()
    logger.info(f"[remove_background] START | model={model_name} | base={base_image_path}")

    if not _is_imagen_model(model_name):
        return {"success": False, "error": _build_validation_error(_unsupported_image_model_error(model_name))}
    if not os.path.exists(base_image_path):
        return {"success": False, "error": _build_validation_error(f"Image not found: {base_image_path}")}

    try:
        base_b64 = _read_image_b64(base_image_path)
        payload = {
            "instances": [{
                "prompt": "",
                "referenceImages": [{
                    "referenceType": "REFERENCE_TYPE_RAW",
                    "referenceId": 1,
                    "referenceImage": {"bytesBase64Encoded": base_b64},
                }],
            }],
            "parameters": {"sampleCount": 1, "editMode": "EDIT_MODE_BGSWAP"},
        }
        response = await _to_thread(_imagen_predict, model_name, payload, timeout=API_TIMEOUT)
        images = _extract_image_bytes_list(response)
        if not images:
            return {"success": False, "error": _build_unexpected_error(
                model_name, ":predict",
                ValueError(f"Model returned no images. Response keys: {list(response.keys())}"),
                time.time() - t0,
            )}
        image_bytes = images[0]

        res: Dict[str, Any] = {}
        if return_base64:
            res["base64"] = _encode_base64(image_bytes)
            res["mime_type"] = "image/png"
        if output_path:
            _save_image_bytes(image_bytes, output_path)
            res["path"] = output_path
        logger.info(f"[remove_background] SUCCESS in {time.time()-t0:.1f}s")
        return {"success": True, "results": [res]}

    except VertexAPIError as e:
        return {"success": False, "error": e.error_dict}
    except asyncio.TimeoutError:
        return {"success": False, "error": _build_timeout_error(model_name, ":predict", time.time() - t0)}
    except Exception as e:
        return {"success": False, "error": _build_unexpected_error(model_name, ":predict", e, time.time() - t0)}


async def generate_video(
    prompt: str,
    output_path: str,
    model_name: str = "veo-3.1-fast-generate-001",
) -> Dict[str, Any]:
    t0 = time.time()
    logger.info(f"[generate_video] START | model={model_name} | prompt='{prompt[:80]}...'")
    try:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w") as f:
            f.write("Simulated video data for: " + prompt)
        logger.info(f"[generate_video] Placeholder written to {output_path} in {time.time()-t0:.1f}s")
        return {
            "success": True,
            "path": output_path,
            "note": "Placeholder stub - real VideoGenerationModel SDK integration pending.",
        }
    except Exception as e:
        return {"success": False, "error": _build_unexpected_error(model_name, ":predictLongRunning", e, time.time() - t0)}
