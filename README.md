# Llama Knowledge Base AI Agent

![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

An end-to-end **AI Agent** built with **Flask**, **Groq-hosted LLaMA models**, and **semantic search**, designed to answer questions **strictly from a user-provided Microsoft Word (.docx) knowledge base**.

The agent supports:
- Text and audio prompts
- Semantic retrieval over a Word-document KB
- Voice transcription using **Groq’s hosted Whisper API**
- Persistent, versioned KB storage
- A modern web chat UI
- Embedding as a bottom-right widget in any web portal
- **OpenAPI (Swagger) documentation** for all endpoints

---

## Project Overview

**Key features**
- Upload a Word document as a Knowledge Base (KB)
- Semantic search using SentenceTransformers
- Grounded answers using Groq-hosted LLaMA models
- Voice input (browser, `.wav`, `.m4a`)
- KB persistence and version switching
- Flask-based web UI
- Embeddable bottom-right chat widget
- OpenAPI / Swagger documentation
- MIT licensed (dual attribution)

---

## Clone the Repository

```bash
git clone https://github.com/gaswani/llama-kb-agent.git
cd llama-kb-agent
```

---

## Python & Virtual Environment Setup

### Required Python Version
```
Python 3.11.9
```

### macOS (zsh / bash)

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

### Windows (PowerShell)

```powershell
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
```

---

## Install Requirements

```bash
pip install -r requirements.txt
```

---

## Environment Variables (.env)

Create a file named `.env` in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

> Never commit `.env` to version control.

---

## How to Obtain a Groq API Key

1. Visit: https://console.groq.com
2. Sign up / log in
3. Navigate to **API Keys**
4. Create a new API key

### macOS (optional manual export)
```bash
export GROQ_API_KEY=your_key_here
```

### Windows (optional manual set)
```powershell
setx GROQ_API_KEY "your_key_here"
```

> Using `.env` is recommended instead of manual exports.

---

## Run the Application

```bash
python app.py
```

Open:
```
http://127.0.0.1:8000/
```

---

## API Documentation (OpenAPI / Swagger)

This project exposes a fully documented **OpenAPI specification** for all endpoints,
including KB management, text chat, audio chat, and health checks.

### Swagger UI

Once the server is running, open:

```
http://127.0.0.1:8000/docs
```

You can:
- Explore all endpoints interactively
- Inspect request / response schemas
- Execute live test calls

### OpenAPI JSON

```
http://127.0.0.1:8000/openapi.json
```

This can be imported into:
- Postman
- Insomnia
- Swagger Editor
- API gateways

---

## OpenAPI Screenshots (for Proposals & Documentation)

Below are example screenshots you may include in proposals or technical documentation:

- **Swagger UI – Endpoint Overview**
  - Shows all available routes grouped by function
- **Swagger UI – Chat Endpoint**
  - Demonstrates request schema for text and audio
- **Swagger UI – KB Management**
  - Upload, list, load, clear, and reload KBs

> Tip: Capture screenshots from `/docs` in your deployed environment and save them
> under `docs/screenshots/` for reuse in presentations and reports.

---

## Upload Knowledge Base (KB)

```bash
curl -X POST http://localhost:8000/upload_kb \
  -F "file=@your_document.docx"
```

---

## Test Queries with curl

### Text Query

```bash
curl -X POST http://localhost:8000/chat_text \
  -H "Content-Type: application/json" \
  -d '{"message":"Hello, who are you?"}'
```

### Audio Query (.wav)

```bash
curl -X POST http://localhost:8000/chat \
  -F "audio=@question.wav"
```

### Audio Query (.m4a – iPhone)

```bash
curl -X POST http://localhost:8000/chat \
  -F "audio=@question.m4a"
```

---

## Audio Transcription (Why Groq Whisper?)

This project **does NOT use local `openai-whisper`**.

Instead, it uses **Groq-hosted Whisper (`whisper-large-v3-turbo`)**.

### Benefits
- No local compilation issues (Numba, LLVM, FFmpeg)
- Faster inference
- Lower operational complexity
- Works seamlessly with `.wav`, `.m4a`, `.mp3`

---

## Embedding the Agent in a Web Portal

### Recommended: Floating Bottom-Right Widget

A ready-made example is provided in this project as:

```
iframe_widget.html
```

#### What to change

Replace this:

```html
src="https://YOUR_AGENT_DOMAIN_OR_IP/"
```

With your deployed agent URL, for example:

```
http://127.0.0.1:8000/        (local testing)
https://agent.yourcompany.com (production)
```

Paste the widget snippet into your portal HTML just before `</body>`.

---

## For Integrators (Portal & App Teams)

This agent is designed to integrate cleanly into existing systems.

### Integration options
- **Iframe embed** (recommended for fastest rollout)
- **Direct API calls** using `/chat_text` and `/chat`
- **Audio-first workflows** (mobile & call-center use cases)

### Recommended flow
1. User submits text or voice input
2. Portal sends request to the agent API
3. Agent performs semantic retrieval over the active KB
4. LLaMA model generates a grounded response
5. Response is rendered back to the user

### Notes for enterprise deployments
- Protect admin endpoints (`/upload_kb`, `/kb/*`)
- Restrict `/docs` and `/openapi.json` if needed
- Deploy behind HTTPS and a reverse proxy
- Log usage for auditability

---

## License

This project is licensed under the **MIT License**.

© 2025 Gideon Aswani / Pathways Technologies Ltd  
See the `LICENSE` file for details.

---

## Third-Party Model Notice

This project uses **LLaMA-family models served via the Groq API**.
No model weights are distributed.

Usage is subject to:
- Groq API Terms of Service
- Meta LLaMA acceptable use policies

See the `NOTICE` file for details.
