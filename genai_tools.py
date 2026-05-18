# Developer: Mirac Orhan (mirac.orhan@gmail.com)
# License: Open Source (MIT License)
import asyncio
import functools
import os
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


def _get_genai_client():
    global _genai_client
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
        _genai_client = google_genai.Client(
            vertexai=True, project=PROJECT_ID, location=LOCATION
        )
        logger.info(f"[genai] Initialized Vertex AI backend (project={PROJECT_ID}, location={LOCATION})")
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
    from model_registry import EMBED_MODEL_VERTEX, EMBED_MODEL_GEMINI_API
    t0 = time.time()
    client = _get_genai_client()
    model_name = model or (
        EMBED_MODEL_GEMINI_API if GOOGLE_GENAI_BACKEND == "gemini_api" else EMBED_MODEL_VERTEX
    )
    logger.info(f"[embed] START | model={model_name} | text_len={len(text)}")
    try:
        result = await _to_thread(
            client.models.embed_content,
            model=model_name,
            contents=text,
            timeout=30.0,
        )
        embedding = list(result.embeddings[0].values)
        logger.info(f"[embed] SUCCESS in {time.time()-t0:.1f}s | dim={len(embedding)}")
        return {
            "success": True,
            "embedding": embedding,
            "dimension": len(embedding),
            "model": model_name,
            "input_length": len(text),
        }
    except Exception as e:
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
    client = _get_genai_client()
    model_name = model or VIDEO_ANALYZE_MODELS.get(model_tier, VIDEO_ANALYZE_MODELS["fast"])
    mime_type = _video_mime_type(video_path)
    logger.info(f"[analyze_video] START | model={model_name} | path={video_path} | size={file_size}")
    try:
        with open(video_path, "rb") as f:
            video_bytes = f.read()

        def _do_request():
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
    client = _get_genai_client()
    model_name = model or SPEECH_MODELS.get(model_tier, SPEECH_MODELS["fast"])
    if output_path is None:
        ts = int(t0)
        output_path = os.path.join(DEFAULT_OUTPUT_DIR, f"speech_{ts}.wav")
    logger.info(f"[generate_speech] START | model={model_name} | voice={voice} | text_len={len(text)}")
    try:
        def _do_request():
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
        audio_data = response.candidates[0].content.parts[0].inline_data.data
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(audio_data)
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
        logger.error(f"[generate_speech] FAIL | {type(e).__name__}: {e}")
        return {"success": False, "error": {"code": type(e).__name__, "message": str(e)}}


async def live_generate(
    prompt: str,
    model_tier: str = "fast",
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate text using Gemini streaming. Full response is accumulated before return."""
    from model_registry import LIVE_MODELS
    t0 = time.time()
    client = _get_genai_client()
    model_name = model or LIVE_MODELS.get(model_tier, LIVE_MODELS["fast"])
    logger.info(f"[live_generate] START | model={model_name} | prompt_len={len(prompt)}")
    try:
        def _do_stream():
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
        logger.error(f"[live_generate] FAIL | {type(e).__name__}: {e}")
        return {"success": False, "error": {"code": type(e).__name__, "message": str(e)}}
