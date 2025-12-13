# Llama Knowledge Base AI Agent (Hybrid Search)

![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

An end-to-end **AI Agent** built with **Flask**, **Groq-hosted LLaMA models**, and **Hybrid Retrieval (Semantic + Keyword)**, designed to answer questions **strictly from a user-provided Microsoft Word (.docx) knowledge base**.

This README reflects the **current architecture** of the agent, including hybrid search, KB versioning, table-aware ingestion, and operational best practices.

---

## Project Overview

**Key capabilities**
- Upload Microsoft Word documents as a **versioned Knowledge Base (KB)**
- **Hybrid retrieval**:
  - Semantic search (SentenceTransformers)
  - Keyword search (TF-IDF)
  - Weighted fusion of both
- Grounded responses using **Groq-hosted LLaMA models**
- Text and audio queries (`.wav`, `.m4a`, browser audio)
- Audio transcription using **Groq-hosted Whisper**
- Persistent KB storage (disk-backed, reloadable)
- Flask-based web UI
- Embeddable bottom-right chat widget
- Full **OpenAPI (Swagger)** documentation

---

## Retrieval Architecture (Important)

This agent uses **Hybrid Search**, not pure semantic search.

### How Hybrid Search Works
1. User query is embedded (semantic similarity)
2. Query is also processed using keyword matching (TF-IDF)
3. Scores are combined:

```
final_score = α * semantic_score + (1 - α) * keyword_score
```

Where:
- `α` (default 0.7) favors semantic meaning
- Keyword search ensures exact matches (names, regions, IDs, table rows)

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

### macOS

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

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Environment Variables (.env)

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

> `.env` is automatically loaded at startup  
> Never commit `.env` to Git

---

## Run the Application

```bash
python app.py
```

Open in browser:

```
http://127.0.0.1:8000/
```

> Run `python app.py` **only once**.  
> Use separate terminals for `curl` requests.

---

## OpenAPI / Swagger Documentation

### Swagger UI
```
http://127.0.0.1:8000/docs
```

### OpenAPI JSON
```
http://127.0.0.1:8000/openapi.json
```

---

## Upload Knowledge Base (KB)

```bash
curl -X POST http://127.0.0.1:8000/upload_kb \
  -F "file=@your_document.docx"
```

---

## KB Reloading (Hybrid Search)

```bash
curl -X POST http://127.0.0.1:8000/kb/reload
```

---

## Test Queries

```bash
curl -X POST http://127.0.0.1:8000/chat_text \
  -H "Content-Type: application/json" \
  -d '{"message":"List Digital Hubs in the Coast region"}'
```

---

## License

MIT License  
© 2025 Gideon Aswani / Pathways Technologies Ltd
