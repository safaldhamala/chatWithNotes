import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import faiss
import ollama


DEFAULT_INDEX_DIR = "/home/reza/Desktop/notes/chatWithNotes/vector_index"
DEFAULT_CHAT_MODEL = "gpt-oss:20b"  # adjust to your local tag, e.g., `ollama list`


def load_index(index_dir: str) -> Tuple[faiss.Index, Dict]:
    index_path = Path(index_dir) / "index.faiss"
    meta_path = Path(index_dir) / "metadata.json"

    if not index_path.exists():
        raise FileNotFoundError(f"FAISS index not found at: {index_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata JSON not found at: {meta_path}")

    index = faiss.read_index(str(index_path))
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    return index, metadata


def embed_query(text: str, embedding_model: str) -> np.ndarray:
    result = ollama.embeddings(model=embedding_model, prompt=text)
    vec = np.array(result["embedding"], dtype=np.float32)
    norm = np.linalg.norm(vec)
    if norm == 0.0:
        return vec
    return vec / norm


def search(
    index: faiss.Index,
    metadata: Dict,
    query: str,
    top_k: int,
) -> List[Tuple[float, Dict]]:
    embedding_model = metadata["embedding_model"]
    q = embed_query(query, embedding_model)
    if q.ndim == 1:
        q = q.reshape(1, -1)
    scores, indices = index.search(q.astype(np.float32), top_k)
    scores = scores[0].tolist()
    indices = indices[0].tolist()

    chunks: List[Dict] = metadata["chunks"]
    results: List[Tuple[float, Dict]] = []
    for s, i in zip(scores, indices):
        if i < 0 or i >= len(chunks):
            continue
        results.append((float(s), chunks[i]))
    return results


def build_context_block(results: List[Tuple[float, Dict]], max_chars: int = 8000) -> str:
    assembled: List[str] = []
    total = 0
    for score, chunk in results:
        source = chunk["source_path"]
        idx = chunk["chunk_index"]
        text = chunk["text"].strip()
        block = f"[SOURCE: {source}#{idx} | score={score:.3f}]\n{text}\n"
        if total + len(block) > max_chars:
            break
        assembled.append(block)
        total += len(block)
    return "\n---\n".join(assembled)


def chat_with_context(
    query: str,
    context_block: str,
    chat_model: str,
) -> str:
    system = (
        "You are a helpful assistant. Answer the user's question using ONLY the provided context. "
        "Cite sources inline like [SOURCE: path#chunk]. If the answer is not in the context, say you don't know."
    )
    user = (
        f"Question: {query}\n\n"
        f"Context:\n{context_block}\n\n"
        "Instructions: Use the context verbatim when relevant. Provide concise, accurate answers with citations."
    )

    resp = ollama.chat(
        model=chat_model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return resp.get("message", {}).get("content", "")


def interactive_loop(
    index: faiss.Index,
    metadata: Dict,
    chat_model: str,
    top_k: int,
    max_context_chars: int,
) -> None:
    print("Interactive mode. Type your question and press Enter.")
    print("Commands: :q to quit, :k N to set top_k, :model TAG to change model, :reload to reload index, :clear to reset history.")

    current_model = chat_model
    current_top_k = top_k
    history: List[Dict[str, str]] = []  # optional chat history, kept short

    while True:
        try:
            query = input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not query:
            continue

        # Commands
        if query == ":q":
            print("Bye.")
            break
        if query.startswith(":k "):
            try:
                current_top_k = max(1, int(query.split(" ", 1)[1]))
                print(f"top_k set to {current_top_k}")
            except ValueError:
                print("Usage: :k 5")
            continue
        if query.startswith(":model "):
            current_model = query.split(" ", 1)[1].strip()
            print(f"chat model set to {current_model}")
            continue
        if query == ":clear":
            history.clear()
            print("history cleared")
            continue
        if query == ":reload":
            # Re-load index and metadata from disk
            new_index, new_meta = load_index(metadata.get("index_dir", DEFAULT_INDEX_DIR))
            index = new_index
            metadata = new_meta
            print("index reloaded")
            continue

        # RAG retrieval
        results = search(index, metadata, query, current_top_k)
        if not results:
            print("No results from the index.")
            continue

        context_block = build_context_block(results, max_chars=max_context_chars)

        # Maintain a short rolling history of last 4 turns
        if len(history) > 8:
            history = history[-8:]

        system = (
            "You are a helpful assistant. Answer the user's question using ONLY the provided context. "
            "Cite sources inline like [SOURCE: path#chunk]. If the answer is not in the context, say you don't know."
        )

        messages = [{"role": "system", "content": system}] + history + [
            {
                "role": "user",
                "content": (
                    f"Question: {query}\n\nContext:\n{context_block}\n\n"
                    "Instructions: Use the context verbatim when relevant. Provide concise, accurate answers with citations."
                ),
            }
        ]

        resp = ollama.chat(model=current_model, messages=messages)
        answer = resp.get("message", {}).get("content", "")
        print("\nAssistant>\n" + answer)

        # Update history with compact entries
        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": answer})

        print("\nRetrieved:")
        for score, chunk in results:
            print(f"- {chunk['source_path']}#{chunk['chunk_index']} (score={score:.3f})")


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG chat over your FAISS index using an Ollama local model.")
    parser.add_argument(
        "--index_dir",
        type=str,
        default=DEFAULT_INDEX_DIR,
        help="Directory containing index.faiss and metadata.json",
    )
    parser.add_argument(
        "--chat_model",
        type=str,
        default=DEFAULT_CHAT_MODEL,
        help="Ollama chat model tag (e.g., gpt-oss:20b)",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of chunks to retrieve",
    )
    parser.add_argument(
        "--max_context_chars",
        type=int,
        default=8000,
        help="Max characters of retrieved context to include in the prompt",
    )
    parser.add_argument("query", type=str, nargs="?", help="Optional one-off question; if omitted, starts interactive mode")

    args = parser.parse_args()

    index, metadata = load_index(args.index_dir)
    # store index_dir into metadata for :reload
    metadata["index_dir"] = args.index_dir

    if args.query:
        results = search(index, metadata, args.query, args.top_k)
        if not results:
            print("No results from the index.")
            return
        context_block = build_context_block(results, max_chars=args.max_context_chars)
        answer = chat_with_context(args.query, context_block, args.chat_model)
        print("\n=== Answer ===\n")
        print(answer)
        print("\n=== Retrieved Chunks ===\n")
        for score, chunk in results:
            print(f"- {chunk['source_path']}#{chunk['chunk_index']} (score={score:.3f})")
    else:
        interactive_loop(index, metadata, args.chat_model, args.top_k, args.max_context_chars)


if __name__ == "__main__":
    main()


