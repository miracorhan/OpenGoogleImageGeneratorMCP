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
from typing import Optional, Dict, Any

from vertexai.generative_models import GenerativeModel, Part, GenerationConfig
from config import logger, PROJECT_ID, LOCATION

API_TIMEOUT = 90.0
HTTP_TIMEOUT = 90.0
TOKEN_TIMEOUT = 30.0

_GCLOUD_WINDOWS_FALLBACK = r"C:\Program Files (x86)\Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd"

SUPPORTED_IMAGE_MODELS = (
    "imagen-4.0-fast-generate-001",
    "imagen-4.0-generate-001",
    "imagen-3.0-generate-002",
)

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


def _imagen_predict(model_name: str, payload: dict) -> dict:
    """Synchronous REST call to publisher model :predict endpoint."""
    t0 = time.time()
    logger.info(f"[predict] getting access token ...")
    token = _get_access_token()
    logger.info(f"[predict] got token in {time.time()-t0:.1f}s")
    url = (
        f"https://{LOCATION}-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}"
        f"/locations/{LOCATION}/publishers/google/models/{model_name}:predict"
    )
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Content-Type", "application/json")
    logger.info(f"[predict] POST {url} | body_size={len(body)}")
    t1 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
            raw = resp.read()
            logger.info(f"[predict] urlopen returned in {time.time()-t1:.1f}s | response_size={len(raw)}")
            return json.loads(raw.decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        logger.error(f"[predict] HTTPError {e.code} after {time.time()-t1:.1f}s: {detail[:200]}")
        raise RuntimeError(f"HTTP {e.code}: {detail[:500]}") from e


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
        msg = _unsupported_image_model_error(model_name)
        logger.error(f"[generate_image] {msg}")
        return {"success": False, "error": msg}

    try:
        payload = {
            "instances": [{"prompt": prompt}],
            "parameters": {
                "sampleCount": number_of_images,
                "aspectRatio": aspect_ratio,
            },
        }
        logger.info(f"[generate_image] Calling :predict REST endpoint ...")
        response = await _to_thread(_imagen_predict, model_name, payload, timeout=API_TIMEOUT)
        images = _extract_image_bytes_list(response)
        logger.info(f"[generate_image] Got {len(images)} image(s) in {time.time()-t0:.1f}s")

        if not images:
            raise ValueError(f"Model returned no images. Response keys: {list(response.keys())}")

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
                logger.info(f"[generate_image] Saved image {i} -> {current_out}")
            results.append(res)

        logger.info(f"[generate_image] SUCCESS in {time.time()-t0:.1f}s")
        return {"success": True, "results": results}

    except asyncio.TimeoutError:
        elapsed = time.time() - t0
        logger.error(f"[generate_image] TIMEOUT after {elapsed:.1f}s - model={model_name}")
        return {"success": False, "error": f"Request timed out after {elapsed:.0f}s. Model '{model_name}' may be slow or unavailable."}
    except Exception as e:
        elapsed = time.time() - t0
        logger.error(f"[generate_image] ERROR after {elapsed:.1f}s - {type(e).__name__}: {e}")
        return {"success": False, "error": f"{type(e).__name__}: {e}"}


async def edit_image(
    prompt: str,
    base_image_path: str,
    output_path: Optional[str] = None,
    model_name: str = "imagen-3.0-generate-002",
    return_base64: bool = False,
) -> Dict[str, Any]:
    t0 = time.time()
    logger.info(f"[edit_image] START | model={model_name} | base={base_image_path}")

    if not _is_imagen_model(model_name):
        msg = _unsupported_image_model_error(model_name)
        logger.error(f"[edit_image] {msg}")
        return {"success": False, "error": msg}

    try:
        if not os.path.exists(base_image_path):
            raise FileNotFoundError(f"Base image not found: {base_image_path}")

        with open(base_image_path, "rb") as f:
            base_b64 = base64.b64encode(f.read()).decode("utf-8")

        payload = {
            "instances": [{
                "prompt": prompt,
                "image": {"bytesBase64Encoded": base_b64},
            }],
            "parameters": {"sampleCount": 1, "editMode": "EDIT_MODE_INPAINT_INSERTION"},
        }
        logger.info(f"[edit_image] Calling :predict REST endpoint ...")
        response = await _to_thread(_imagen_predict, model_name, payload, timeout=API_TIMEOUT)
        images = _extract_image_bytes_list(response)
        if not images:
            raise ValueError(f"Model returned no edited images. Response keys: {list(response.keys())}")
        image_bytes = images[0]

        res: Dict[str, Any] = {}
        if return_base64:
            res["base64"] = _encode_base64(image_bytes)
            res["mime_type"] = "image/png"
        if output_path:
            _save_image_bytes(image_bytes, output_path)
            res["path"] = output_path
        logger.info(f"[edit_image] SUCCESS in {time.time()-t0:.1f}s")
        return {"success": True, "results": [res]}

    except asyncio.TimeoutError:
        elapsed = time.time() - t0
        logger.error(f"[edit_image] TIMEOUT after {elapsed:.1f}s")
        return {"success": False, "error": f"Request timed out after {elapsed:.0f}s."}
    except Exception as e:
        logger.error(f"[edit_image] ERROR after {time.time()-t0:.1f}s - {type(e).__name__}: {e}")
        return {"success": False, "error": f"{type(e).__name__}: {e}"}


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
            raise FileNotFoundError(f"Image not found: {image_path}")

        with open(image_path, "rb") as f:
            image_data = f.read()

        ext = os.path.splitext(image_path)[1].lower()
        mime = "image/png" if ext == ".png" else ("image/webp" if ext == ".webp" else "image/jpeg")

        model = await _get_gemini_model(model_name)
        image_part = Part.from_data(mime_type=mime, data=image_data)
        logger.info(f"[analyze_image] Calling generate_content_async ...")

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
        elapsed = time.time() - t0
        logger.error(f"[analyze_image] TIMEOUT after {elapsed:.1f}s")
        return {"success": False, "error": f"Request timed out after {elapsed:.0f}s."}
    except Exception as e:
        logger.error(f"[analyze_image] ERROR after {time.time()-t0:.1f}s - {type(e).__name__}: {e}")
        return {"success": False, "error": f"{type(e).__name__}: {e}"}


async def upscale_image(
    base_image_path: str,
    output_path: Optional[str] = None,
    model_name: str = "imagen-3.0-generate-002",
    return_base64: bool = False,
) -> Dict[str, Any]:
    t0 = time.time()
    logger.info(f"[upscale_image] START | model={model_name} | base={base_image_path}")

    if not _is_imagen_model(model_name):
        msg = _unsupported_image_model_error(model_name)
        logger.error(f"[upscale_image] {msg}")
        return {"success": False, "error": msg}

    try:
        if not os.path.exists(base_image_path):
            raise FileNotFoundError(f"Image not found: {base_image_path}")

        with open(base_image_path, "rb") as f:
            base_b64 = base64.b64encode(f.read()).decode("utf-8")

        payload = {
            "instances": [{"image": {"bytesBase64Encoded": base_b64}}],
            "parameters": {"sampleCount": 1, "mode": "upscale", "upscaleConfig": {"upscaleFactor": "x2"}},
        }
        logger.info(f"[upscale_image] Calling :predict REST endpoint ...")
        response = await _to_thread(_imagen_predict, model_name, payload, timeout=API_TIMEOUT)
        images = _extract_image_bytes_list(response)
        if not images:
            raise ValueError(f"Model returned no upscaled image. Response keys: {list(response.keys())}")
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

    except asyncio.TimeoutError:
        elapsed = time.time() - t0
        logger.error(f"[upscale_image] TIMEOUT after {elapsed:.1f}s")
        return {"success": False, "error": f"Request timed out after {elapsed:.0f}s."}
    except Exception as e:
        logger.error(f"[upscale_image] ERROR after {time.time()-t0:.1f}s - {type(e).__name__}: {e}")
        return {"success": False, "error": f"{type(e).__name__}: {e}"}


async def remove_background(
    base_image_path: str,
    output_path: Optional[str] = None,
    model_name: str = "imagen-3.0-generate-002",
    return_base64: bool = False,
) -> Dict[str, Any]:
    t0 = time.time()
    logger.info(f"[remove_background] START | model={model_name} | base={base_image_path}")

    if not _is_imagen_model(model_name):
        msg = _unsupported_image_model_error(model_name)
        logger.error(f"[remove_background] {msg}")
        return {"success": False, "error": msg}

    try:
        if not os.path.exists(base_image_path):
            raise FileNotFoundError(f"Image not found: {base_image_path}")

        with open(base_image_path, "rb") as f:
            base_b64 = base64.b64encode(f.read()).decode("utf-8")

        payload = {
            "instances": [{
                "prompt": "",
                "image": {"bytesBase64Encoded": base_b64},
            }],
            "parameters": {"sampleCount": 1, "editMode": "EDIT_MODE_BGSWAP"},
        }
        logger.info(f"[remove_background] Calling :predict REST endpoint ...")
        response = await _to_thread(_imagen_predict, model_name, payload, timeout=API_TIMEOUT)
        images = _extract_image_bytes_list(response)
        if not images:
            raise ValueError(f"Model returned no images. Response keys: {list(response.keys())}")
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

    except asyncio.TimeoutError:
        elapsed = time.time() - t0
        logger.error(f"[remove_background] TIMEOUT after {elapsed:.1f}s")
        return {"success": False, "error": f"Request timed out after {elapsed:.0f}s."}
    except Exception as e:
        logger.error(f"[remove_background] ERROR after {time.time()-t0:.1f}s - {type(e).__name__}: {e}")
        return {"success": False, "error": f"{type(e).__name__}: {e}"}


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
        logger.error(f"[generate_video] ERROR - {type(e).__name__}: {e}")
        return {"success": False, "error": f"{type(e).__name__}: {e}"}
