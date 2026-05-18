# Google Image Generator MCP (Vertex AI 2026 Server)

This project is a Model Context Protocol (MCP) server that exposes Google Cloud Vertex AI capabilities—specifically Imagen 3 and Gemini Vision models—to MCP-compatible clients. It is built using the `FastMCP` framework.

## Features & Tools

The server provides a comprehensive suite of MCP tools for interacting with Vertex AI:

- **`tool_list_available_models`**: Dynamically fetches and lists recommended and published models available on Vertex AI.
- **`tool_generate_image`**: Generates high-quality images from text prompts using Imagen models (e.g., gemini-3.1-flash-image).
- **`tool_edit_image`**: Edits existing images based on text instructions.
- **`tool_analyze_image`**: Performs multimodal image analysis and reasoning using Gemini Vision models (e.g., gemini-3.1-pro).
- **`tool_upscale_image`**: Upscales low-resolution images to higher qualities.
- **`tool_remove_background`**: Isolates subjects by removing the background from an image.
- **`tool_generate_video`**: Generates videos from text prompts (Currently acts as a forward-compatible stub for models like Veo 3.1).

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
    "vertex-ai-images": {
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
- *"Remove the background from the image I just generated."*
- *"Analyze this image and tell me what objects are present."*
