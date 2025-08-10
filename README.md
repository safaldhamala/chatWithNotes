## Chat with Your Notes (CLI)

Turn a folder of `.txt` notes into a fully local chatbot you can chat with. This project builds a FAISS vector index over your notes using local embeddings from Ollama, then answers questions with a local chat model using retrieval-augmented generation (RAG).

### What you get
- **Local-first**: Runs entirely on your machine (no cloud calls)
- **Bring your own notes**: Point it at any folder of `.txt` files
- **Fast retrieval**: FAISS index + cosine similarity
- **Simple CLI**: Ask questions interactively or one-off

## Prerequisites
- **Python** 3.9+
- **pip**
- **Ollama** installed and running
  - Install from the official site: [Ollama website](https://ollama.com)
  - After installing, ensure `ollama` works in your terminal: `ollama --version`

## Install dependencies
```bash
cd chatWithNotes
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Prepare your notes
- Put your `.txt` files under `chatWithNotes/Notes_txt` (default). The tool scans this directory recursively.
- Prefer plain text. If you have PDFs/Markdown, convert them to `.txt` first.
- You can use a different folder by passing `--notes_dir` when indexing (see below).

## Get the models (free, open-source)
This project uses two models via Ollama:
- **Embedding model** (for indexing/search): default `nomic-embed-text`
- **Chat model** (for answering): default `gpt-oss:20b` (you can switch to something lighter like `llama3:8b`)

Pull them locally:
```bash
ollama pull nomic-embed-text
ollama pull llama3:8b  # example lightweight chat model; you can use another installed model, I used gpt-oss::20b
# If using gpt-oss::20b
# ollama pull gpt-oss:20b
```

See what you have installed:
```bash
ollama list
```

## 1) Build the FAISS index over your notes
By default, the script reads notes from `chatWithNotes/Notes_txt` and writes the index to `chatWithNotes/vector_index`.

```bash
python index.py \
  --notes_dir Notes_txt \
  --index_dir vector_index \
  --embedding_model nomic-embed-text \
  --chunk_size_words 300 \
  --overlap_words 60
```

Outputs:
- `index.faiss` and `metadata.json` saved under `--index_dir`.

Notes:
- Use absolute paths to avoid path issues.
- You can change chunking parameters for larger/smaller context windows.

## 2) Chat with your indexed notes (CLI)
Interactive chat:
```bash
python chat.py \
  --index_dir vector_index \
  --chat_model llama3:8b
```

One-off question (non-interactive):
```bash
python chat.py \
  --index_dir vector_index \
  --chat_model llama3:8b \
  "What reminders have I put in my notes?"
```

While in interactive mode, you can use:
- `:q` — quit
- `:k N` — set top_k retrieved chunks (e.g., `:k 5`)
- `:model TAG` — switch chat model at runtime (e.g., `:model llama3:8b`)
- `:reload` — reload the index from disk
- `:clear` — clear brief chat history

## Customizing defaults
If you prefer to change the code defaults instead of passing flags:
- Edit `DEFAULT_NOTES_DIR`, `DEFAULT_INDEX_DIR`, and `DEFAULT_EMBEDDING_MODEL` in `chatWithNotes/index.py`.
- Edit `DEFAULT_INDEX_DIR` and `DEFAULT_CHAT_MODEL` in `chatWithNotes/chat.py`.

## Troubleshooting
- **FAISS index not found**: Make sure you ran the indexing step and the `--index_dir` matches where `index.faiss` and `metadata.json` were saved.
- **Model not found / not pulled**: Run `ollama pull <model>` (e.g., `ollama pull nomic-embed-text`, `ollama pull llama3:8b`). Use `ollama list` to verify.
- **Slow/Resource-heavy**: Try a smaller chat model (e.g., `llama3:8b`) or reduce `--top_k`.
- **No results from the index**: Ensure you actually have `.txt` files and adjust chunking if needed.

## How it works (brief)
- `index.py` splits your notes into overlapping word chunks, embeds each with the embedding model, normalizes to unit length, and stores them in a FAISS Inner Product index. Metadata (source path, chunk indices, etc.) is saved to `metadata.json`.
- `chat.py` retrieves the most relevant chunks for your question, assembles a context block, and prompts a local chat model. Answers include inline citations like `[SOURCE: path#chunk]` to show provenance.

You're all set—drop in your `.txt` notes, index them, and start chatting locally!


