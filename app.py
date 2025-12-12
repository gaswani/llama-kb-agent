import os
import io
import json
import uuid
import tempfile
from datetime import datetime
from typing import List, Tuple, Optional

import numpy as np
from flask import Flask, request, jsonify, render_template, Response
from flask_cors import CORS

import docx  # python-docx
from sentence_transformers import SentenceTransformer
from groq import Groq
from sklearn.metrics.pairwise import cosine_similarity

# Optional .env support
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


# ---------------- Configuration ----------------

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("Please set GROQ_API_KEY in your environment (or .env).")

LLAMA_MODEL = os.getenv("LLAMA_MODEL", "llama-3.3-70b-versatile")

TOP_K = int(os.getenv("TOP_K", "5"))
SIM_THRESHOLD = float(os.getenv("SIM_THRESHOLD", "0.40"))

# KB storage (disk)
KB_ROOT = os.getenv("KB_ROOT", os.path.join(os.path.dirname(__file__), "kb_store"))
os.makedirs(KB_ROOT, exist_ok=True)

CURRENT_KB_FILE = os.path.join(KB_ROOT, "current_kb.json")  # stores {"kb_id": "..."} so we can auto-load


# ---------------- Initialize clients/models ----------------

llama_client = Groq(api_key=GROQ_API_KEY)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# In-memory active KB
kb_id_active: Optional[str] = None
kb_chunks: List[str] = []
kb_embeddings: Optional[np.ndarray] = None  # shape: (n_chunks, dim)


# ---------------- Helpers: KB parsing, chunking, embeddings ----------------

def load_docx_text(file_bytes: bytes) -> str:
    file_stream = io.BytesIO(file_bytes)
    doc = docx.Document(file_stream)
    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    return "\n\n".join(paragraphs)


def chunk_text(text: str, max_chars: int = 900) -> List[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: List[str] = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) + 2 <= max_chars:
            current = f"{current}\n\n{para}".strip() if current else para
        else:
            if current:
                chunks.append(current)
            current = para

    if current:
        chunks.append(current)
    return chunks


def embed_texts(texts: List[str]) -> np.ndarray:
    if not texts:
        return np.empty((0, 0))
    return embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)


def semantic_search(query: str, top_k: int = TOP_K) -> Tuple[List[str], List[float]]:
    global kb_chunks, kb_embeddings
    if kb_embeddings is None or kb_embeddings.shape[0] == 0:
        raise ValueError("Knowledge base is empty. Upload or load a KB first.")

    q_emb = embed_texts([query])[0].reshape(1, -1)
    sims = cosine_similarity(q_emb, kb_embeddings)[0]

    top_indices = np.argsort(-sims)[:top_k]
    top_chunks = [kb_chunks[i] for i in top_indices]
    top_scores = [float(sims[i]) for i in top_indices]
    return top_chunks, top_scores


# ---------------- Helpers: LLM prompting ----------------

def build_system_prompt() -> str:
    return (
        "You are a helpful, professional AI assistant that answers questions about a single document.\n"
        "You MUST only use the provided DOCUMENT EXCERPTS as your source of truth.\n"
        "- If the answer cannot be derived directly and clearly from the excerpts, you MUST say:\n"
        "  \"I'm sorry, I couldn't find that in the knowledge base.\"\n"
        "- Do NOT invent new facts or rely on outside knowledge.\n"
        "- If the information is ambiguous or incomplete, explain that clearly.\n"
        "- Be concise, well-structured, and easy to read.\n"
    )


def build_user_prompt(question: str, context_chunks: List[str]) -> str:
    context_text = "\n\n---\n\n".join(context_chunks)
    return (
        f"Question:\n{question}\n\n"
        f"DOCUMENT EXCERPTS (knowledge base):\n{context_text}\n\n"
        "Answer the question using ONLY the information in these excerpts."
    )


def call_llama(question: str, context_chunks: List[str]) -> str:
    completion = llama_client.chat.completions.create(
        model=LLAMA_MODEL,
        messages=[
            {"role": "system", "content": build_system_prompt()},
            {"role": "user", "content": build_user_prompt(question, context_chunks)},
        ],
        temperature=0.1,
        max_tokens=700,
    )
    return completion.choices[0].message.content.strip()


# ---------------- Helpers: audio transcription via Groq Whisper ----------------

def transcribe_audio_to_text(file_bytes: bytes, ext: str = "m4a") -> str:
    with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=True) as tmp:
        tmp.write(file_bytes)
        tmp.flush()

        with open(tmp.name, "rb") as f:
            transcription = llama_client.audio.transcriptions.create(
                file=f,
                model="whisper-large-v3-turbo",
                response_format="json",
                temperature=0.0,
            )
    text = getattr(transcription, "text", "") or ""
    return text.strip()


# ---------------- Helpers: small-talk / agent behavior ----------------

def handle_small_talk(query: str) -> Optional[str]:
    q = (query or "").lower().strip()
    q_no_punct = "".join(ch for ch in q if ch.isalnum() or ch.isspace())

    greetings = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]
    if any(q_no_punct.startswith(g) for g in greetings):
        return (
            "Hello! 👋 I'm your document assistant. "
            "I answer questions *strictly* using the Word document knowledge base you've uploaded. "
            "What would you like to know?"
        )

    if any(p in q_no_punct for p in [
        "who are you", "what can you do", "what do you do", "who is this", "what is your role"
    ]):
        return (
            "I'm an AI assistant that answers only from the uploaded document. "
            "Ask me about definitions, sections, policies, procedures, or anything in that document."
        )

    if "thank" in q_no_punct:
        return "You're welcome! 😊 Ask me anything else from the document."

    if any(p in q_no_punct for p in ["bye", "goodbye", "see you", "see ya"]):
        return "Goodbye! 👋 Come back anytime you need help with the document."

    return None


# ---------------- KB persistence helpers ----------------

def kb_dir(kb_id: str) -> str:
    return os.path.join(KB_ROOT, kb_id)


def list_kbs() -> List[str]:
    items = []
    for name in os.listdir(KB_ROOT):
        p = os.path.join(KB_ROOT, name)
        if os.path.isdir(p) and os.path.exists(os.path.join(p, "meta.json")):
            items.append(name)

    def sort_key(k):
        try:
            with open(os.path.join(kb_dir(k), "meta.json"), "r", encoding="utf-8") as f:
                return json.load(f).get("created_at", "")
        except Exception:
            return ""
    return sorted(items, key=sort_key, reverse=True)


def save_current_kb_id(kb_id: str) -> None:
    with open(CURRENT_KB_FILE, "w", encoding="utf-8") as f:
        json.dump({"kb_id": kb_id}, f)


def load_current_kb_id() -> Optional[str]:
    if not os.path.exists(CURRENT_KB_FILE):
        return None
    try:
        with open(CURRENT_KB_FILE, "r", encoding="utf-8") as f:
            return (json.load(f) or {}).get("kb_id")
    except Exception:
        return None


def persist_kb(kb_id: str, source_filename: str, source_bytes: bytes, chunks: List[str], embeddings: np.ndarray) -> None:
    d = kb_dir(kb_id)
    os.makedirs(d, exist_ok=True)

    with open(os.path.join(d, "source.docx"), "wb") as f:
        f.write(source_bytes)

    with open(os.path.join(d, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    np.save(os.path.join(d, "embeddings.npy"), embeddings)

    meta = {
        "kb_id": kb_id,
        "source_filename": source_filename,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "num_chunks": len(chunks),
        "embedding_dim": int(embeddings.shape[1]) if embeddings.ndim == 2 and embeddings.shape[0] else 0,
        "llama_model": LLAMA_MODEL,
        "top_k": TOP_K,
        "sim_threshold": SIM_THRESHOLD,
    }
    with open(os.path.join(d, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def load_kb_from_disk(kb_id: str) -> None:
    global kb_id_active, kb_chunks, kb_embeddings
    d = kb_dir(kb_id)
    chunks_path = os.path.join(d, "chunks.json")
    emb_path = os.path.join(d, "embeddings.npy")
    meta_path = os.path.join(d, "meta.json")

    if not (os.path.exists(chunks_path) and os.path.exists(emb_path) and os.path.exists(meta_path)):
        raise ValueError(f"KB '{kb_id}' is missing required files on disk.")

    with open(chunks_path, "r", encoding="utf-8") as f:
        kb_chunks = json.load(f)

    kb_embeddings = np.load(emb_path)
    kb_id_active = kb_id
    save_current_kb_id(kb_id)


def clear_kb_memory() -> None:
    global kb_id_active, kb_chunks, kb_embeddings
    kb_id_active = None
    kb_chunks = []
    kb_embeddings = None


def auto_load_kb_on_startup() -> None:
    candidate = load_current_kb_id()
    if candidate and os.path.isdir(kb_dir(candidate)):
        try:
            load_kb_from_disk(candidate)
            return
        except Exception:
            pass

    versions = list_kbs()
    if versions:
        load_kb_from_disk(versions[0])


# ---------------- Flask app ----------------

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)


@app.route("/", methods=["GET"])
def ui():
    return render_template("index.html", active_kb=kb_id_active)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "active_kb": kb_id_active})


# -------- KB management endpoints --------

@app.route("/kb/list", methods=["GET"])
def kb_list():
    return jsonify({
        "available_kbs": list_kbs(),
        "current_kb": kb_id_active
    })


@app.route("/kb/load", methods=["POST"])
def kb_load():
    data = request.get_json(silent=True) or {}
    kb_id = (data.get("kb_id") or "").strip()
    if not kb_id:
        return jsonify({"error": "kb_id is required"}), 400

    try:
        load_kb_from_disk(kb_id)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    return jsonify({"message": "KB loaded", "current_kb": kb_id_active})


@app.route("/kb/clear", methods=["POST"])
def kb_clear():
    clear_kb_memory()
    return jsonify({"message": "KB cleared from memory", "current_kb": kb_id_active})


@app.route("/kb/reload", methods=["POST"])
def kb_reload():
    if not kb_id_active:
        try:
            auto_load_kb_on_startup()
        except Exception as e:
            return jsonify({"error": str(e)}), 400
        return jsonify({"message": "KB loaded", "current_kb": kb_id_active})

    try:
        load_kb_from_disk(kb_id_active)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    return jsonify({"message": "KB reloaded", "current_kb": kb_id_active})


@app.route("/upload_kb", methods=["POST"])
def upload_kb():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if not file.filename.lower().endswith(".docx"):
        return jsonify({"error": "Only .docx files are supported"}), 400

    file_bytes = file.read()
    full_text = load_docx_text(file_bytes)
    chunks = chunk_text(full_text, max_chars=900)
    if not chunks:
        return jsonify({"error": "No text found in uploaded document"}), 400

    embeddings = embed_texts(chunks)

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    kb_id = f"kb_{ts}_{uuid.uuid4().hex[:8]}"

    persist_kb(kb_id, file.filename, file_bytes, chunks, embeddings)
    load_kb_from_disk(kb_id)

    return jsonify({
        "message": "Knowledge base uploaded, versioned, and activated.",
        "kb_id": kb_id_active,
        "num_chunks": len(kb_chunks),
    })


# -------- Chat endpoints --------

@app.route("/chat_text", methods=["POST"])
def chat_text():
    data = request.get_json(silent=True) or {}
    message = (data.get("message") or "").strip()
    if not message:
        return jsonify({"error": "Message cannot be empty."}), 400

    small = handle_small_talk(message)
    if small is not None:
        return jsonify({
            "answer": small,
            "query": message,
            "used_audio": False,
            "active_kb": kb_id_active,
            "top_scores": [],
            "top_snippets": [],
        })

    try:
        context_chunks, scores = semantic_search(message, top_k=TOP_K)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    if not scores or scores[0] < SIM_THRESHOLD:
        return jsonify({
            "answer": "I'm sorry, I couldn't find that in the knowledge base.",
            "query": message,
            "used_audio": False,
            "active_kb": kb_id_active,
            "top_scores": scores,
            "top_snippets": context_chunks,
        })

    answer = call_llama(message, context_chunks)
    return jsonify({
        "answer": answer,
        "query": message,
        "used_audio": False,
        "active_kb": kb_id_active,
        "top_scores": scores,
        "top_snippets": context_chunks,
    })


@app.route("/chat", methods=["POST"])
def chat():
    used_audio = False
    query = ""

    audio_file = request.files.get("audio")
    if audio_file:
        file_bytes = audio_file.read()
        ext = audio_file.filename.split(".")[-1].lower() if "." in audio_file.filename else "m4a"
        query = transcribe_audio_to_text(file_bytes, ext=ext)
        used_audio = True
    else:
        query = request.form.get("message", "").strip()

    if not query:
        return jsonify({"error": "Please provide either text or audio."}), 400

    small = handle_small_talk(query)
    if small is not None:
        return jsonify({
            "answer": small,
            "query": query,
            "used_audio": used_audio,
            "active_kb": kb_id_active,
            "top_scores": [],
            "top_snippets": [],
        })

    try:
        context_chunks, scores = semantic_search(query, top_k=TOP_K)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    if not scores or scores[0] < SIM_THRESHOLD:
        return jsonify({
            "answer": "I'm sorry, I couldn't find that in the knowledge base.",
            "query": query,
            "used_audio": used_audio,
            "active_kb": kb_id_active,
            "top_scores": scores,
            "top_snippets": context_chunks,
        })

    answer = call_llama(query, context_chunks)
    return jsonify({
        "answer": answer,
        "query": query,
        "used_audio": used_audio,
        "active_kb": kb_id_active,
        "top_scores": scores,
        "top_snippets": context_chunks,
    })
# -------- OpenAPI / Swagger UI endpoints --------

OPENAPI_SPEC = {
  "openapi": "3.0.3",
  "info": {
    "title": "Llama KB Agent API",
    "version": "1.0.0",
    "description": "Flask-based AI agent that answers strictly from an uploaded Word (.docx) knowledge base."
  },
  "servers": [
    {"url": "http://127.0.0.1:8000", "description": "Local dev"},
  ],
  "paths": {
    "/": {
      "get": {
        "summary": "Web UI",
        "description": "Serves the chat UI.",
        "responses": {"200": {"description": "HTML page"}}
      }
    },
    "/health": {
      "get": {
        "summary": "Health check",
        "responses": {
          "200": {
            "description": "OK",
            "content": {"application/json": {"schema": {"$ref": "#/components/schemas/Health"}}}
          }
        }
      }
    },

    "/upload_kb": {
      "post": {
        "summary": "Upload Knowledge Base (.docx)",
        "description": "Uploads a Word document, chunks it, embeds it, saves it as a new version, and activates it.",
        "requestBody": {
          "required": True,
          "content": {
            "multipart/form-data": {
              "schema": {
                "type": "object",
                "properties": {
                  "file": {"type": "string", "format": "binary"}
                },
                "required": ["file"]
              }
            }
          }
        },
        "responses": {
          "200": {"description": "Uploaded & activated", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/UploadKBResponse"}}}},
          "400": {"description": "Bad request", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/Error"}}}},
        }
      }
    },

    "/kb/list": {
      "get": {
        "summary": "List KB versions",
        "responses": {
          "200": {"description": "KB list", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/KBListResponse"}}}}
        }
      }
    },
    "/kb/load": {
      "post": {
        "summary": "Load a KB version",
        "requestBody": {
          "required": True,
          "content": {"application/json": {"schema": {"$ref": "#/components/schemas/KBLoadRequest"}}}
        },
        "responses": {
          "200": {"description": "Loaded", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/KBLoadResponse"}}}},
          "400": {"description": "Bad request", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/Error"}}}},
        }
      }
    },
    "/kb/clear": {
      "post": {
        "summary": "Clear KB from memory",
        "responses": {
          "200": {"description": "Cleared", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/KBLoadResponse"}}}}
        }
      }
    },
    "/kb/reload": {
      "post": {
        "summary": "Reload active KB from disk",
        "responses": {
          "200": {"description": "Reloaded", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/KBLoadResponse"}}}},
          "400": {"description": "Bad request", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/Error"}}}},
        }
      }
    },

    "/chat_text": {
      "post": {
        "summary": "Chat (text JSON)",
        "requestBody": {
          "required": True,
          "content": {"application/json": {"schema": {"$ref": "#/components/schemas/ChatTextRequest"}}}
        },
        "responses": {
          "200": {"description": "Answer", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/ChatResponse"}}}},
          "400": {"description": "Bad request", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/Error"}}}},
        }
      }
    },

    "/chat": {
      "post": {
        "summary": "Chat (multipart: audio or text form)",
        "description": "Send either an audio file (recommended) or a form field `message`.",
        "requestBody": {
          "required": True,
          "content": {
            "multipart/form-data": {
              "schema": {
                "type": "object",
                "properties": {
                  "audio": {"type": "string", "format": "binary", "description": "Audio file (.wav, .m4a, etc.)"},
                  "message": {"type": "string", "description": "Optional if audio is provided"}
                }
              }
            }
          }
        },
        "responses": {
          "200": {"description": "Answer", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/ChatResponse"}}}},
          "400": {"description": "Bad request", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/Error"}}}},
        }
      }
    },
  },
  "components": {
    "schemas": {
      "Error": {
        "type": "object",
        "properties": {"error": {"type": "string"}},
        "required": ["error"]
      },
      "Health": {
        "type": "object",
        "properties": {
          "status": {"type": "string"},
          "active_kb": {"type": ["string", "null"]}
        },
        "required": ["status", "active_kb"]
      },
      "UploadKBResponse": {
        "type": "object",
        "properties": {
          "message": {"type": "string"},
          "kb_id": {"type": ["string", "null"]},
          "num_chunks": {"type": "integer"}
        },
        "required": ["message", "kb_id", "num_chunks"]
      },
      "KBListResponse": {
        "type": "object",
        "properties": {
          "available_kbs": {"type": "array", "items": {"type": "string"}},
          "current_kb": {"type": ["string", "null"]}
        },
        "required": ["available_kbs", "current_kb"]
      },
      "KBLoadRequest": {
        "type": "object",
        "properties": {"kb_id": {"type": "string"}},
        "required": ["kb_id"]
      },
      "KBLoadResponse": {
        "type": "object",
        "properties": {
          "message": {"type": "string"},
          "current_kb": {"type": ["string", "null"]}
        },
        "required": ["message", "current_kb"]
      },
      "ChatTextRequest": {
        "type": "object",
        "properties": {"message": {"type": "string"}},
        "required": ["message"]
      },
      "ChatResponse": {
        "type": "object",
        "properties": {
          "answer": {"type": "string"},
          "query": {"type": "string"},
          "used_audio": {"type": "boolean"},
          "active_kb": {"type": ["string", "null"]},
          "top_scores": {"type": "array", "items": {"type": "number"}},
          "top_snippets": {"type": "array", "items": {"type": "string"}}
        },
        "required": ["answer", "query", "used_audio", "active_kb", "top_scores", "top_snippets"]
      }
    }
  }
}


@app.route("/openapi.json", methods=["GET"])
def openapi_json():
    # You can optionally update servers dynamically if behind a reverse proxy.
    return jsonify(OPENAPI_SPEC)


@app.route("/docs", methods=["GET"])
def swagger_ui():
    # Swagger UI via CDN (no extra Python deps)
    html = f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>API Docs - Llama KB Agent</title>
  <link rel="stylesheet" href="https://unpkg.com/swagger-ui-dist@5/swagger-ui.css">
  <style>
    body {{ margin:0; }}
  </style>
</head>
<body>
  <div id="swagger-ui"></div>
  <script src="https://unpkg.com/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
  <script>
    window.onload = () => {{
      SwaggerUIBundle({{
        url: "/openapi.json",
        dom_id: "#swagger-ui",
        deepLinking: true,
      }});
    }};
  </script>
</body>
</html>
"""
    return Response(html, mimetype="text/html")


# Auto-load KB when server starts
try:
    auto_load_kb_on_startup()
except Exception:
    pass


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
