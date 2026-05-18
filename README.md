# GoogleImageGenerator MCP

This project is a Model Context Protocol (MCP) server that exposes Google Cloud Vertex AI capabilities—specifically Imagen 3 and Gemini Vision models—to MCP-compatible clients. It is built using the `FastMCP` framework.

## Features & Tools

The server provides a comprehensive suite of MCP tools for interacting with Vertex AI:

- **`tool_list_available_models`**: Live-probes every candidate publisher model in the configured project/location and returns only the ones that actually respond (200/400 = reachable, 404 = excluded). Cached for the server process lifetime; pass `force_refresh=true` to rescan.
- **`tool_generate_image`**: Text-to-image generation via Imagen (default: `imagen-4.0-fast-generate-001`).
- **`tool_edit_image`**: Precision image editing via **Imagen 3 Capability** (`imagen-3.0-capability-001`). Supports mask-based inpaint/outpaint, background swap, product image, and mask-free prompt-driven edit. See *Edit modes* below.
- **`tool_transform_image`**: *(new)* Free-form `image + text → image` transformation via **Gemini multimodal** (`gemini-2.5-flash-image`). Use for style transfer, scene rewriting, or any natural-language image edit that doesn't need pixel-precise masking. Accepts optional additional reference images.
- **`tool_analyze_image`**: Multimodal image analysis via Gemini Vision (default: `gemini-2.5-flash`).
- **`tool_upscale_image`**: Upscales low-resolution images via Imagen.
- **`tool_remove_background`**: Removes background via Imagen `EDIT_MODE_BGSWAP`.
- **`tool_generate_video`**: Currently a forward-compatible stub for Veo 3.1.

### Edit modes (`tool_edit_image`)

| `edit_mode` | What it does | Mask required? |
|---|---|---|
| `EDIT_MODE_DEFAULT` *(default)* | Prompt-driven full-image edit, no mask | No |
| `EDIT_MODE_INPAINT_INSERTION` | Add an object into the masked region | **Yes** |
| `EDIT_MODE_INPAINT_REMOVAL` | Remove content in the masked region | **Yes** |
| `EDIT_MODE_OUTPAINT` | Extend the image beyond its original bounds | **Yes** |
| `EDIT_MODE_BGSWAP` | Swap the background | No |
| `EDIT_MODE_PRODUCT_IMAGE` | Product reference styling | No |

Use `imagen-3.0-capability-001` (default) for all of the above. The legacy `imagen-3.0-generate-002` model only supports `EDIT_MODE_DEFAULT` and does not accept a mask.

### When to use which "image + text → image" tool

| Need | Use |
|---|---|
| Mask-based inpaint/outpaint/BG-swap with pixel precision | **`tool_edit_image`** (Imagen Capability) |
| "Make it look like X" / style transfer / scene rewriting / multi-reference compositions | **`tool_transform_image`** (Gemini multimodal) |

### Error handling

All tools return a uniform error shape so MCP clients and direct Python callers see the same diagnostics:

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

Full request/response logs are written to `logs/vertex_ai_mcp.log` (also surfaced in `error.log_path`).

### Resources & Prompts

- **Local Resources (`local://outputs/{filename}`)**: The server directly exposes generated and processed media files as MCP resources, allowing seamless display within your MCP client (like Claude Desktop or Cursor).
- **Pre-built Prompts**: Includes specialized prompt templates for `character_design`, `logo_concept`, and `UI_UX_mockup` to help you get the best results following Gemini 3 prompting guidelines.

## Prerequisites & Resources

Before you begin, ensure you have the following resources and permissions set up:

1. **Python**: Python 3.9 or newer installed on your machine.
2. **Google Cloud Account**: An active Google Cloud account and project.
3. **Vertex AI API**: The Vertex AI API must be enabled in your Google Cloud Project.
4. **Google Cloud CLI (`gcloud`)**: Installed and configured for authentication.

## Installation & Setup

### 1. Clone the Repository
Navigate to the project directory in your terminal:
```bash
cd GoogleImageGeneratorMCP
```

### 2. Install Dependencies
Install the required Python packages using `pip`:
```bash
pip install -r requirements.txt
```

### 3. Authentication (Critical Step)
The server uses Google Cloud Application Default Credentials (ADC). You must authenticate your local environment using the `gcloud` CLI:
```bash
gcloud auth application-default login
```
*This command will open a browser window for you to log in to your Google account. Ensure you log in with an account that has access to your Google Cloud Project.*

### 4. Environment Configuration
Create a `.env` file in the root of the project directory. This file configures the server with your specific Google Cloud details:

```env
# Your Google Cloud Project ID (Required)
GOOGLE_CLOUD_PROJECT=your-google-cloud-project-id

# The Google Cloud region to use (e.g., us-central1, europe-west4)
GOOGLE_CLOUD_LOCATION=us-central1

# Directory where generated images/videos will be saved locally
DEFAULT_OUTPUT_DIR=./outputs

# --- Advanced Authentication Options (Optional) ---
# If you want to bypass Application Default Credentials, you can use one of these:

# 1. Direct OAuth 2.0 Access Token: 
# Useful if an upstream app manages tokens and passes them down.
# GOOGLE_ACCESS_TOKEN=ya29.a0AfB_by...

# 2. Service Account Impersonation:
# Useful for high-security environments where the default account assumes the role of a service account.
# IMPERSONATE_SERVICE_ACCOUNT=your-service-account@your-project.iam.gserviceaccount.com
```

## Usage

### Running as a Standalone Script
You can start the MCP server manually to verify it works without errors:
```bash
python mcp_server.py
```

### Integrating with MCP Clients

To use this server, you need to configure your MCP client (such as Claude Desktop or Cursor) to launch this script.

**For Claude Desktop (example `claude_desktop_config.json`):**
```json
{
  "mcpServers": {
    "GoogleImageGenerator": {
      "command": "python",
      "args": [
        "/absolute/path/to/your/GoogleImageGeneratorMCP/mcp_server.py"
      ],
      "env": {
        "GOOGLE_CLOUD_PROJECT": "your-google-cloud-project-id",
        "GOOGLE_CLOUD_LOCATION": "us-central1"
      }
    }
  }
}
```
*Note: Make sure to replace `/absolute/path/to/your/...` with the actual path to the script, and configure the path to your python executable if you are using a virtual environment.*

Once configured and the client is restarted, you can ask your AI assistant tasks like:
- *"Generate an image of a futuristic city at sunset."*
- *"Edit this banner — add a glowing cyan halo around the logo."* (uses `tool_edit_image`, `EDIT_MODE_DEFAULT`)
- *"Transform this photo into a hand-drawn pencil sketch."* (uses `tool_transform_image`)
- *"Remove the background from the image I just generated."*
- *"Analyze this image and tell me what objects are present."*
