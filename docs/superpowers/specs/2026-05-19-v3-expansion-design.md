# OpenGoogleImageGeneratorMCP v3 Expansion — Design Spec

**Date:** 2026-05-19  
**Status:** Approved

## Context

Google deprecated Vertex AI documentation in favour of the Gemini Enterprise Agent Platform. This spec formalises adding four new capability areas to the MCP — text embedding, video analysis, speech synthesis, and live streaming text generation — alongside multi-format image output (WebP/AVIF). The existing Vertex AI SDK stays for Imagen/Veo/Lyria; the new `google-genai` SDK handles the four new capabilities.

---

## Architecture

```
SDK Responsibility Split
├── google-cloud-aiplatform  →  Imagen (image gen/edit/transform/upscale/bg-remove)
│                               Veo   (video gen/extend/edit)
│                               Lyria (music gen)
└── google-genai             →  Embedding, Video Analysis, Speech TTS, Live Streaming
```

---

## New Files

| File | Purpose |
|---|---|
| `format_converter.py` | Client-side image format conversion (PNG→WebP/AVIF/JPEG) |
| `genai_tools.py` | Four new async tool functions using google-genai SDK |

## Modified Files

| File | Change |
|---|---|
| `vertex_ai_tools.py` | `save_format` param on `gemini_generate_image`, `transform_image`, `upscale_image`, `edit_image`; WebP/AVIF path in `generate_image` |
| `model_registry.py` | Add `EMBED_MODEL_*`, `SPEECH_MODELS`, `VIDEO_ANALYZE_MODELS`, `LIVE_MODELS` constants |
| `discovery.py` | Add `embedding`, `speech`, `live_text` categories to `get_recommended_models()` |
| `config.py` | Add `GOOGLE_GENAI_API_KEY`, `GOOGLE_GENAI_BACKEND` env vars |
| `requirements.txt` | Add `google-genai>=1.0.0` |
| `mcp_server.py` | 4 new `@mcp.tool()` functions; `save_format`/`output_format` extended on existing tools |

---

## New MCP Tools (17 → 21)

### `tool_embed`
- SDK: google-genai  
- Models: `text-embedding-004` (Vertex) / `gemini-embedding-2` (Gemini API)
- Input: `text: str`
- Output: `{embedding: [float...], dimension: int, model: str}`

### `tool_analyze_video`
- SDK: google-genai
- Models: `gemini-2.5-flash` (fast) / `gemini-3.1-pro` (quality)
- Input: `video_path: str, prompt: str, model_tier: str`
- Output: `{analysis: str, model: str, video_path: str}`
- Constraint: Inline only, max 20MB

### `tool_generate_speech`
- SDK: google-genai
- Models: `gemini-2.5-flash-preview-tts` (fast) / `gemini-2.5-pro-preview-tts` (quality)
- Voices: Aoede, Charon, Fenrir, Kore, Puck
- Input: `text: str, voice: str, output_path: str, model_tier: str`
- Output: `{audio_path: str, format: "wav", voice: str, model: str}`

### `tool_live_generate`
- SDK: google-genai (streaming)
- Models: `gemini-2.5-flash` (fast) / `gemini-3.1-pro` (quality)
- Input: `prompt: str, model_tier: str`
- Output: `{text: str, model: str, stream_chunks: int}`
- Note: Full response accumulated before return (MCP protocol limitation)

---

## Format Conversion

Extended `output_format` / `save_format` on image tools:
- `PNG`, `JPEG`: handled natively by Imagen API (existing behaviour)
- `WEBP`, `AVIF`: PNG requested from API, converted client-side via Pillow
- AVIF requires `pillow-avif-plugin` (optional; documented in README)

---

## Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `GOOGLE_GENAI_API_KEY` | — | Gemini API key (optional; uses Vertex ADC if absent) |
| `GOOGLE_GENAI_BACKEND` | `vertex_ai` | `vertex_ai` or `gemini_api` |

---

## Testing

- Existing 80 tests: must all pass unchanged
- New: `tests/test_format_converter.py` (~8 tests)
- New: `tests/test_genai_tools.py` (~12 tests, all mocked)
- Updated: `tests/test_model_registry.py` (assertions for new constants)
