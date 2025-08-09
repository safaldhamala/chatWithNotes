import argparse
import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import faiss
import ollama
from tqdm import tqdm


DEFAULT_NOTES_DIR = "/home/reza/Desktop/notes/Notes_txt"
DEFAULT_INDEX_DIR = "/home/reza/Desktop/notes/chatWithNotes/vector_index"
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"  # run: `ollama pull nomic-embed-text`


@dataclass
class ChunkMetadata:
    source_path: str
    chunk_index: int
    start_word_index: int
    end_word_index: int
    text: str


def find_text_files(root_dir: str) -> List[Path]:
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"Notes directory does not exist: {root_dir}")
    return sorted([p for p in root.rglob("*.txt") if p.is_file()])


def chunk_text_by_words(text: str, chunk_size_words: int, overlap_words: int) -> List[Tuple[str, int, int]]:
    words = text.split()
    if not words:
        return []

    if chunk_size_words <= 0:
        raise ValueError("chunk_size_words must be > 0")
    if overlap_words < 0:
        raise ValueError("overlap_words must be >= 0")
    if overlap_words >= chunk_size_words:
        raise ValueError("overlap_words must be smaller than chunk_size_words")

    step = chunk_size_words - overlap_words
    chunks: List[Tuple[str, int, int]] = []
    for start in range(0, len(words), step):
        end = min(start + chunk_size_words, len(words))
        chunk_words = words[start:end]
        if not chunk_words:
            break
        chunk_text = " ".join(chunk_words)
        chunks.append((chunk_text, start, end))
        if end == len(words):
            break
    return chunks


def embed_text_ollama(text: str, model_name: str) -> np.ndarray:
    # Ollama embeddings API returns { 'embedding': [floats...] }
    result = ollama.embeddings(model=model_name, prompt=text)
    vector = np.array(result["embedding"], dtype=np.float32)
    return vector


def build_faiss_index(
    notes_dir: str,
    index_dir: str,
    embedding_model: str,
    chunk_size_words: int,
    overlap_words: int,
) -> None:
    os.makedirs(index_dir, exist_ok=True)

    text_files = find_text_files(notes_dir)
    if not text_files:
        raise RuntimeError(f"No .txt files found under: {notes_dir}")

    # First pass: prepare chunks and count
    all_chunks: List[ChunkMetadata] = []
    for file_path in text_files:
        content = Path(file_path).read_text(encoding="utf-8", errors="ignore")
        chunks = chunk_text_by_words(content, chunk_size_words, overlap_words)
        for idx, (chunk_text, start_idx, end_idx) in enumerate(chunks):
            all_chunks.append(
                ChunkMetadata(
                    source_path=str(file_path),
                    chunk_index=idx,
                    start_word_index=start_idx,
                    end_word_index=end_idx,
                    text=chunk_text,
                )
            )

    if not all_chunks:
        raise RuntimeError("No chunks produced. Check chunking parameters.")

    # Discover embedding dimension using the first chunk
    sample_vector = embed_text_ollama(all_chunks[0].text, embedding_model)
    dimension = int(sample_vector.shape[0])

    # Cosine similarity: normalize vectors and use Inner Product index
    index = faiss.IndexFlatIP(dimension)

    vectors_to_add: List[np.ndarray] = []
    for meta in tqdm(all_chunks, desc="Embedding chunks", unit="chunk"):
        vec = embed_text_ollama(meta.text, embedding_model)
        # Normalize to unit length for cosine similarity with IP
        norm = np.linalg.norm(vec)
        if norm == 0.0:
            # Skip degenerate vectors
            continue
        vec = vec / norm
        vectors_to_add.append(vec)

    if not vectors_to_add:
        raise RuntimeError("No valid embeddings produced.")

    matrix = np.vstack(vectors_to_add).astype(np.float32)
    index.add(matrix)

    # Persist FAISS index and metadata
    faiss_path = str(Path(index_dir) / "index.faiss")
    meta_path = str(Path(index_dir) / "metadata.json")

    faiss.write_index(index, faiss_path)

    # Align metadata count with vectors actually added (in case any were skipped)
    if index.ntotal != len(all_chunks):
        # If some vectors were skipped due to zero norm, trim metadata accordingly
        # by matching the count of added vectors
        trimmed_chunks = all_chunks[: index.ntotal]
    else:
        trimmed_chunks = all_chunks

    payload: Dict = {
        "embedding_model": embedding_model,
        "dimension": dimension,
        "notes_dir": str(Path(notes_dir).resolve()),
        "total_vectors": index.ntotal,
        "chunks": [asdict(c) for c in trimmed_chunks],
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"Saved FAISS index -> {faiss_path}")
    print(f"Saved metadata -> {meta_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a FAISS index from local .txt files using Ollama embeddings.")
    parser.add_argument(
        "--notes_dir",
        type=str,
        default=DEFAULT_NOTES_DIR,
        help="Directory containing .txt files (scanned recursively)",
    )
    parser.add_argument(
        "--index_dir",
        type=str,
        default=DEFAULT_INDEX_DIR,
        help="Directory where the FAISS index and metadata are saved",
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        default=DEFAULT_EMBEDDING_MODEL,
        help="Ollama embedding model (e.g., nomic-embed-text, mxbai-embed-large, bge-m3)",
    )
    parser.add_argument(
        "--chunk_size_words",
        type=int,
        default=300,
        help="Number of words per chunk",
    )
    parser.add_argument(
        "--overlap_words",
        type=int,
        default=60,
        help="Number of overlapping words between consecutive chunks",
    )

    args = parser.parse_args()

    print(
        f"Indexing from '{args.notes_dir}' -> '{args.index_dir}' using embeddings '{args.embedding_model}'\n"
        f"Chunking: {args.chunk_size_words} words with {args.overlap_words} overlap\n"
        "Ensure the embedding model is available locally: e.g., `ollama pull nomic-embed-text`"
    )

    build_faiss_index(
        notes_dir=args.notes_dir,
        index_dir=args.index_dir,
        embedding_model=args.embedding_model,
        chunk_size_words=args.chunk_size_words,
        overlap_words=args.overlap_words,
    )


if __name__ == "__main__":
    main()


