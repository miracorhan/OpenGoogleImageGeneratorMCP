# Open Google Image Generator MCP

This project is a Model Context Protocol (MCP) server that exposes Google Cloud Vertex AI and Google GenAI SDK capabilities—Imagen, Gemini, Veo, Lyria, and Chirp models—to MCP-compatible clients. Built with the `FastMCP` framework.

> **Current version: 3.0.0** — Full GenAI SDK integration (embed, speech, video analysis, live generation), WebP/AVIF format support, multi-tier model selection, parallel batch generation, sequential pipeline engine, and comprehensive video tools.

## Features & Tools

### Image Tools

| Tool | Description | Backend |
|---|---|---|
| `tool_generate_image` | Text-to-image generation. Supports aspect ratio, negative prompt, seed, watermark, GCS output, and WebP/AVIF output | Imagen 4 (`imagen-4.0-fast-generate-001`) |
| `tool_edit_image` | Mask-based inpaint/outpaint, background swap, product image, and prompt-driven edit. See *Edit modes* below | Imagen 3 Capability (`imagen-3.0-capability-001`) |
| `tool_transform_image` | Free-form `image + text → image` transformation: style transfer, scene rewriting, multi-reference composition | Gemini multimodal (`gemini-2.5-flash-image`) |
| `tool_analyze_image` | Multimodal image understanding and Q&A. Supports `thinking_level` (MINIMAL/LOW/MEDIUM/HIGH) and `media_resolution` (LOW/MEDIUM/HIGH/ULTRA_HIGH) | Gemini Vision (`gemini-2.5-flash`) |
| `tool_upscale_image` | Upscale low-resolution images | Imagen |
| `tool_remove_background` | Remove background via `EDIT_MODE_BGSWAP` | Imagen |
| `tool_batch_generate` | Parallel batch text-to-image generation (up to 10 prompts, max 4 concurrent). `balanced` tier not supported for batch | Imagen |
| `tool_run_pipeline` | Sequential multi-step image processing pipeline (generate → edit → transform → …) | Mixed |

### Video Tools

| Tool | Description | Backend |
|---|---|---|
| `tool_generate_video` | Text-to-video generation. Supports `audio_enabled` for Veo 3+ | Veo (`veo-3.1-fast-generate-001`) |
| `tool_image_to_video` | Animate a still image into video. Supports optional `last_frame_path` for first+last frame mode | Veo |
| `tool_extend_video` | Extend an existing video clip by 4, 6, or 8 seconds | Veo |
| `tool_video_object_edit` | Insert or remove an object in a video via `operation` (`insert`/`remove`) and `prompt` | Veo |
| `tool_analyze_video` | Video understanding and Q&A (max 20MB; mp4, mov, avi, webm, mkv) | Gemini GenAI SDK |

### Audio Tools

| Tool | Description | Backend |
|---|---|---|
| `tool_generate_speech` | Text-to-speech with voice selection (Aoede, Charon, Fenrir, Kore, Puck). Supports `model_tier` (fast/quality). Outputs WAV | Gemini TTS (`gemini-2.5-flash-preview-tts` / `gemini-2.5-pro-preview-tts`) |
| `tool_generate_music` | Music generation from a text prompt | Lyria 2 / Lyria 3 (GenAI SDK) |

### GenAI SDK Tools

| Tool | Description | Backend |
|---|---|---|
| `tool_embed` | Text embeddings as float vectors | Gemini Embedding (`text-embedding-004` on Vertex AI, `gemini-embedding-2` on Gemini API) |
| `tool_live_generate` | Streaming text generation — response is accumulated and returned in full | Gemini Live (`gemini-2.5-flash` / `gemini-3.1-pro`) |

### Utility Tools

| Tool | Description |
|---|---|
| `tool_list_available_models` | Live-probes every candidate model in the configured project/location and returns only those that respond (200/400 = reachable, 404 = excluded). Cached for the server process lifetime; pass `force_refresh=true` to rescan. Also reports available update versions. |
| `tool_upload_file` | Register a local file for use as a reference image in subsequent tool calls (e.g. `tool_transform_image`). Returns a `file_uri`. |

---

### Edit modes (`tool_edit_image`)

| `edit_mode` | What it does | Mask required? |
|---|---|---|
| `EDIT_MODE_DEFAULT` *(default)* | Prompt-driven full-image edit, no mask | No |
| `EDIT_MODE_INPAINT_INSERTION` | Add an object into the masked region | **Yes** |
| `EDIT_MODE_INPAINT_REMOVAL` | Remove content in the masked region | **Yes** |
| `EDIT_MODE_OUTPAINT` | Extend the image beyond its original bounds | **Yes** |
| `EDIT_MODE_BGSWAP` | Swap the background | No |
| `EDIT_MODE_PRODUCT_IMAGE` | Product reference styling | No |

Use `imagen-3.0-capability-001` (default) for all of the above. The legacy `imagen-3.0-generate-002` only supports `EDIT_MODE_DEFAULT` and does not accept a mask.

### When to use which "image + text → image" tool

| Need | Use |
|---|---|
| Mask-based inpaint/outpaint/BG-swap with pixel precision | **`tool_edit_image`** (Imagen Capability) |
| "Make it look like X" / style transfer / scene rewriting / multi-reference compositions | **`tool_transform_image`** (Gemini multimodal) |

### Model tiers

Most tools accept a `model_tier` parameter:

| Tier | Description |
|---|---|
| `fast` *(default)* | Lowest latency, lowest cost |
| `balanced` | Quality / speed trade-off; routes to Gemini for image generation. **Not supported** for `tool_batch_generate` |
| `quality` | Higher quality, moderate latency |
| `ultra` | Maximum quality (Imagen 4 Ultra / Veo quality models) |

#### Model resolution by tier and tool

| Tier | `tool_generate_image` | `tool_transform_image` | `tool_generate_video` |
|---|---|---|---|
| `fast` | `imagen-4.0-fast-generate-001` | `gemini-2.5-flash-image` | `veo-3.1-fast-generate-001` |
| `balanced` | `gemini-2.5-flash-image` | `gemini-2.5-flash-image` | `veo-3.1-fast-generate-001` |
| `quality` | `imagen-4.0-generate-001` | `gemini-2.5-pro-image` | `veo-3.1-generate-001` |
| `ultra` | `imagen-4.0-ultra-generate-001` | `gemini-2.5-pro-image` | `veo-3.1-generate-001` |

### Output formats

`tool_generate_image`, `tool_edit_image`, `tool_transform_image`, and `tool_upscale_image` accept a `save_format` / `output_format` parameter:

| Format | Notes |
|---|---|
| `PNG` *(default)* | Lossless |
| `JPEG` | Smaller files, lossy. `compression_quality` (0-100, default 85) applies only to JPEG |
| `WEBP` | Modern lossless/lossy, wide browser support |
| `AVIF` | Best compression, requires `Pillow>=10` |

---

### Error handling

All tools return a uniform error shape:

```json
{
  "success": false,
  "error": {
    "code": 404,
    "model": "gemini-9.9-nonexistent",
    "endpoint": ":generateContent",
    "message": "Publisher Model `...` is not found.",
    "hint": "Model '...' not found in project '...' / location '...'. Try: gemini-2.5-flash-image.",
    "docs_url": "https://docs.cloud.google.com/...",
    "log_path": ".../logs/vertex_ai_mcp.log",
    "duration_s": 0.42
  }
}
```

| HTTP code | What you'll see in `error.hint` |
|---|---|
| 400 | Vertex's parameter-validation message verbatim |
| 401 | "Run `gcloud auth application-default login` and retry." |
| 403 | IAM role hint (`roles/aiplatform.user`) + Vertex AI API enablement check |
| 404 | Live alternatives from the probe cache (`tool_list_available_models`) |
| 429 | `Retry after N` (from `Retry-After` header) + quota-increase pointer |
| 500/502/503/504 | "Safe to retry once" |
| `TIMEOUT` | After 90s — suggests a `-fast-` variant |
| `VALIDATION` | Client-side validation failure (mask missing, file not found, etc.); **no HTTP call is made** |

Full request/response logs are written to `logs/vertex_ai_mcp.log`.

### Resources & Prompts

- **Local Resources (`local://outputs/{filename}`)**: Generated and processed media files are exposed as MCP resources for seamless display in MCP clients (Claude Desktop, Cursor, etc.).
- **Pre-built Prompts**: Includes specialized prompt templates for `character_design`, `logo_concept`, and `UI_UX_mockup`.

---

## Prerequisites & Resources

1. **Python** 3.9 or newer
2. **Google Cloud Account** with an active project
3. **Vertex AI API** enabled in your project
4. **Google Cloud CLI (`gcloud`)** installed and configured

For GenAI SDK tools (`tool_embed`, `tool_analyze_video`, `tool_generate_speech`, `tool_live_generate`, `tool_generate_music`), you additionally need either:
- A **Gemini API key** (`GOOGLE_GENAI_API_KEY`), or
- Vertex AI ADC credentials with `GOOGLE_GENAI_BACKEND=vertexai`

---

## Installation & Setup

### Option A: Install from PyPI

```bash
pip install open-google-image-generator-mcp
```

### Option B: Clone the Repository

```bash
git clone https://github.com/miracorhan/OpenGoogleImageGeneratorMCP.git
cd OpenGoogleImageGeneratorMCP
pip install -r requirements.txt
```

### Authentication (Critical Step)

The server uses Google Cloud Application Default Credentials (ADC):

```bash
gcloud auth application-default login
```

*This opens a browser for login. Use an account with access to your Google Cloud project.*

### Environment Configuration

Create a `.env` file in the project root:

```env
# Required
GOOGLE_CLOUD_PROJECT=your-google-cloud-project-id
GOOGLE_CLOUD_LOCATION=us-central1

# Output directory for generated media
DEFAULT_OUTPUT_DIR=./outputs

# --- GenAI SDK (for embed, speech, live, music, video-analysis tools) ---
# Option A: Gemini API key (free tier available)
GOOGLE_GENAI_API_KEY=AIza...

# Option B: Use Vertex AI backend (uses ADC above, no separate key needed)
GOOGLE_GENAI_BACKEND=vertexai

# --- Advanced Vertex AI Authentication (Optional) ---
# Direct OAuth 2.0 Access Token
# GOOGLE_ACCESS_TOKEN=ya29.a0AfB_by...

# Service Account Impersonation
# IMPERSONATE_SERVICE_ACCOUNT=your-service-account@your-project.iam.gserviceaccount.com
```

---

## Usage

### Running as a Standalone Script

```bash
python mcp_server.py
```

### Integrating with MCP Clients

**For Claude Desktop (`claude_desktop_config.json`):**

```json
{
  "mcpServers": {
    "OpenGoogleImageGenerator": {
      "command": "python",
      "args": ["/absolute/path/to/OpenGoogleImageGeneratorMCP/mcp_server.py"],
      "env": {
        "GOOGLE_CLOUD_PROJECT": "your-google-cloud-project-id",
        "GOOGLE_CLOUD_LOCATION": "us-central1",
        "GOOGLE_GENAI_API_KEY": "AIza..."
      }
    }
  }
}
```

*Replace `/absolute/path/to/your/...` with the actual path, and use the correct Python executable if using a virtual environment.*

### Example prompts

- *"Generate an image of a futuristic city at sunset."*
- *"Edit this banner — add a glowing cyan halo around the logo."* (`tool_edit_image`, `EDIT_MODE_DEFAULT`)
- *"Transform this photo into a hand-drawn pencil sketch."* (`tool_transform_image`)
- *"Remove the background from the image I just generated."*
- *"Analyze this image and tell me what objects are present."*
- *"Generate 8 product shots in parallel with different backgrounds."* (`tool_batch_generate`)
- *"Run a pipeline: generate → remove background → upscale."* (`tool_run_pipeline`)
- *"Convert this text to speech using the Kore voice."* (`tool_generate_speech`)
- *"Generate a 30-second ambient music track."* (`tool_generate_music`)
- *"Embed this sentence for semantic search."* (`tool_embed`)
- *"Animate this product photo into a 5-second video."* (`tool_image_to_video`)
- *"Generate a video of a sunset with audio."* (`tool_generate_video`, `audio_enabled=true`)

---

## Author & License

- **Developer:** Mirac Orhan (<mirac.orhan@gmail.com>)
- **License:** [MIT License](LICENSE) (Open Source — Free for everyone to use, modify, and distribute)
