# Developer: Mirac Orhan (mirac.orhan@gmail.com)
# License: Open Source (MIT License)
import asyncio
import functools
import json
import os
import struct
import threading
import time
from typing import Any, Dict, Optional

from config import logger, PROJECT_ID, LOCATION, GOOGLE_GENAI_API_KEY, GOOGLE_GENAI_BACKEND

try:
    import google.genai as google_genai
    from google.genai import types as genai_types
    _GENAI_AVAILABLE = True
except ImportError:
    google_genai = None  # type: ignore
    genai_types = None   # type: ignore
    _GENAI_AVAILABLE = False

AVAILABLE_VOICES = ("Aoede", "Charon", "Fenrir", "Kore", "Puck")

_genai_client = None
_genai_client_lock = threading.Lock()

_DEPENDENCY_ERROR = {
    "success": False,
    "error": {
        "code": "DEPENDENCY_MISSING",
        "message": "google-genai package is required. Install with: pip install google-genai>=1.0.0",
    },
}


def _build_adc_credentials():
    """Read ADC JSON and build Credentials with refresh_token, bypassing RAPT reauth."""
    adc_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if not adc_path:
        appdata = os.environ.get("APPDATA")
        if appdata:
            adc_path = os.path.join(appdata, "gcloud", "application_default_credentials.json")
    if not adc_path or not os.path.exists(adc_path):
        return None
    try:
        with open(adc_path, encoding="utf-8") as f:
            adc = json.load(f)
        if adc.get("type") != "authorized_user":
            return None
        from google.oauth2.credentials import Credentials
        return Credentials(
            token=None,
            refresh_token=adc["refresh_token"],
            client_id=adc["client_id"],
            client_secret=adc["client_secret"],
            token_uri="https://oauth2.googleapis.com/token",
        )
    except Exception as e:
        logger.info(f"[genai] ADC credential build skipped ({type(e).__name__}: {e})")
        return None


def _wrap_pcm_as_wav(pcm: bytes, sample_rate: int = 24000, channels: int = 1, bits: int = 16) -> bytes:
    """Prepend a RIFF/WAV header to raw Linear16 PCM bytes."""
    data_size = len(pcm)
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 36 + data_size, b"WAVE",
        b"fmt ", 16, 1, channels, sample_rate,
        sample_rate * channels * bits // 8,
        channels * bits // 8, bits,
        b"data", data_size,
    )
    return header + pcm


def _reset_genai_client():
    """Reset cached client so next call rebuilds it (e.g. after auth error)."""
    global _genai_client
    with _genai_client_lock:
        _genai_client = None


def _get_genai_client():
    global _genai_client
    if _genai_client is not None:
        return _genai_client
    with _genai_client_lock:
        if _genai_client is not None:
            return _genai_client
        if not _GENAI_AVAILABLE:
            raise RuntimeError(
                "google-genai package is required for this tool. "
                "Install with: pip install google-genai>=1.0.0"
            )
        if GOOGLE_GENAI_BACKEND == "gemini_api" and GOOGLE_GENAI_API_KEY:
            _genai_client = google_genai.Client(api_key=GOOGLE_GENAI_API_KEY)
            logger.info("[genai] Initialized with Gemini API key")
        else:
            if not PROJECT_ID:
                raise RuntimeError(
                    "GOOGLE_CLOUD_PROJECT not set. "
                    "Required for Vertex AI backend of google-genai SDK."
                )
            creds = _build_adc_credentials()
            kwargs = {"vertexai": True, "project": PROJECT_ID, "location": LOCATION}
            if creds is not None:
                kwargs["credentials"] = creds
                logger.info(f"[genai] Initialized Vertex AI backend with ADC credentials (project={PROJECT_ID})")
            else:
                logger.info(f"[genai] Initialized Vertex AI backend with default ADC (project={PROJECT_ID})")
            _genai_client = google_genai.Client(**kwargs)
        return _genai_client


async def _to_thread(func, *args, timeout: float = 60.0, **kwargs):
    loop = asyncio.get_running_loop()
    future = loop.run_in_executor(None, functools.partial(func, *args, **kwargs))
    done, _ = await asyncio.wait({future}, timeout=timeout)
    if not done:
        raise asyncio.TimeoutError()
    return future.result()


async def embed(text: str, model: Optional[str] = None) -> Dict[str, Any]:
    """Embed text into a float vector using Gemini Embedding."""
    if not _GENAI_AVAILABLE:
        return _DEPENDENCY_ERROR
    from model_registry import EMBED_MODEL_VERTEX, EMBED_MODEL_GEMINI_API
    t0 = time.time()
    model_name = model or (
        EMBED_MODEL_GEMINI_API if GOOGLE_GENAI_BACKEND == "gemini_api" else EMBED_MODEL_VERTEX
    )
    logger.info(f"[embed] START | model={model_name} | text_len={len(text)}")
    try:
        def _do_request():
            client = _get_genai_client()
            return client.models.embed_content(model=model_name, contents=text)

        result = await _to_thread(_do_request, timeout=30.0)
        embedding = list(result.embeddings[0].values)
        logger.info(f"[embed] SUCCESS in {time.time()-t0:.1f}s | dim={len(embedding)}")
        return {
            "success": True,
            "embedding": embedding,
            "dimension": len(embedding),
            "model": model_name,
            "input_length": len(text),
            "duration_s": round(time.time() - t0, 2),
        }
    except Exception as e:
        if "RefreshError" in type(e).__name__ or "RefreshError" in type(e).__qualname__:
            _reset_genai_client()
        logger.error(f"[embed] FAIL | {type(e).__name__}: {e}")
        return {"success": False, "error": {"code": type(e).__name__, "message": str(e)}}


def _video_mime_type(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    return {
        ".mp4": "video/mp4",
        ".mov": "video/quicktime",
        ".avi": "video/avi",
        ".webm": "video/webm",
        ".mkv": "video/x-matroska",
    }.get(ext, "video/mp4")


async def analyze_video(
    video_path: str,
    prompt: str,
    model_tier: str = "fast",
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """Analyze a local video file using Gemini Vision. Max 20MB inline."""
    if not _GENAI_AVAILABLE:
        return _DEPENDENCY_ERROR
    from model_registry import VIDEO_ANALYZE_MODELS
    t0 = time.time()
    if not os.path.exists(video_path):
        return {
            "success": False,
            "error": {"code": "VALIDATION", "message": f"Video not found: {video_path}"},
        }
    file_size = os.path.getsize(video_path)
    MAX_INLINE = 20 * 1024 * 1024
    if file_size > MAX_INLINE:
        return {
            "success": False,
            "error": {
                "code": "FILE_TOO_LARGE",
                "message": (
                    f"Video ({file_size // (1024*1024)}MB) exceeds 20MB inline limit. "
                    "Upload to GCS and pass a gs:// URI instead."
                ),
            },
        }
    model_name = model or VIDEO_ANALYZE_MODELS.get(model_tier, VIDEO_ANALYZE_MODELS["fast"])
    mime_type = _video_mime_type(video_path)
    logger.info(f"[analyze_video] START | model={model_name} | path={video_path} | size={file_size}")
    try:
        def _do_request():
            client = _get_genai_client()
            with open(video_path, "rb") as f:
                video_bytes = f.read()
            video_part = genai_types.Part(
                inline_data=genai_types.Blob(mime_type=mime_type, data=video_bytes)
            )
            return client.models.generate_content(
                model=model_name,
                contents=[video_part, prompt],
            )

        response = await _to_thread(_do_request, timeout=120.0)
        analysis = response.text or ""
        logger.info(f"[analyze_video] SUCCESS in {time.time()-t0:.1f}s | text_len={len(analysis)}")
        return {
            "success": True,
            "analysis": analysis,
            "model": model_name,
            "video_path": video_path,
            "duration_s": round(time.time() - t0, 2),
        }
    except Exception as e:
        if "RefreshError" in type(e).__name__ or "RefreshError" in type(e).__qualname__:
            _reset_genai_client()
        logger.error(f"[analyze_video] FAIL | {type(e).__name__}: {e}")
        return {"success": False, "error": {"code": type(e).__name__, "message": str(e)}}


async def generate_speech(
    text: str,
    voice: str = "Kore",
    output_path: Optional[str] = None,
    model_tier: str = "fast",
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """Convert text to speech using Gemini TTS. Returns a WAV file."""
    if not _GENAI_AVAILABLE:
        return _DEPENDENCY_ERROR
    from model_registry import SPEECH_MODELS
    from config import DEFAULT_OUTPUT_DIR
    t0 = time.time()
    if voice not in AVAILABLE_VOICES:
        return {
            "success": False,
            "error": {
                "code": "VALIDATION",
                "message": f"Invalid voice '{voice}'. Choose from: {', '.join(AVAILABLE_VOICES)}",
            },
        }
    model_name = model or SPEECH_MODELS.get(model_tier, SPEECH_MODELS["fast"])
    if output_path is None:
        ts = int(t0)
        output_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), DEFAULT_OUTPUT_DIR, f"speech_{ts}.wav"
        )
    logger.info(f"[generate_speech] START | model={model_name} | voice={voice} | text_len={len(text)}")
    try:
        def _do_request():
            client = _get_genai_client()
            return client.models.generate_content(
                model=model_name,
                contents=text,
                config=genai_types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=genai_types.SpeechConfig(
                        voice_config=genai_types.VoiceConfig(
                            prebuilt_voice_config=genai_types.PrebuiltVoiceConfig(
                                voice_name=voice
                            )
                        )
                    ),
                ),
            )

        response = await _to_thread(_do_request, timeout=60.0)
        raw_pcm = response.candidates[0].content.parts[0].inline_data.data
        wav_bytes = _wrap_pcm_as_wav(raw_pcm)
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(wav_bytes)
        logger.info(f"[generate_speech] SUCCESS in {time.time()-t0:.1f}s | path={output_path}")
        return {
            "success": True,
            "audio_path": output_path,
            "format": "wav",
            "voice": voice,
            "model": model_name,
            "duration_s": round(time.time() - t0, 2),
        }
    except Exception as e:
        if "RefreshError" in type(e).__name__ or "RefreshError" in type(e).__qualname__:
            _reset_genai_client()
        logger.error(f"[generate_speech] FAIL | {type(e).__name__}: {e}")
        return {"success": False, "error": {"code": type(e).__name__, "message": str(e)}}


async def live_generate(
    prompt: str,
    model_tier: str = "fast",
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate text using Gemini streaming. Full response is accumulated before return."""
    if not _GENAI_AVAILABLE:
        return _DEPENDENCY_ERROR
    from model_registry import LIVE_MODELS
    t0 = time.time()
    model_name = model or LIVE_MODELS.get(model_tier, LIVE_MODELS["fast"])
    logger.info(f"[live_generate] START | model={model_name} | prompt_len={len(prompt)}")
    try:
        def _do_stream():
            client = _get_genai_client()
            accumulated = ""
            chunks = 0
            for chunk in client.models.generate_content_stream(
                model=model_name,
                contents=prompt,
            ):
                if chunk.text:
                    accumulated += chunk.text
                    chunks += 1
            return accumulated, chunks

        text, chunks = await _to_thread(_do_stream, timeout=60.0)
        logger.info(f"[live_generate] SUCCESS in {time.time()-t0:.1f}s | chunks={chunks} | text_len={len(text)}")
        return {
            "success": True,
            "text": text,
            "model": model_name,
            "stream_chunks": chunks,
            "duration_s": round(time.time() - t0, 2),
        }
    except Exception as e:
        if "RefreshError" in type(e).__name__ or "RefreshError" in type(e).__qualname__:
            _reset_genai_client()
        logger.error(f"[live_generate] FAIL | {type(e).__name__}: {e}")
        return {"success": False, "error": {"code": type(e).__name__, "message": str(e)}}
