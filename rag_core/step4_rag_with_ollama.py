import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any

import numpy as np
import lancedb
import pandas as pd
import requests
from sentence_transformers import SentenceTransformer


# ---------- CONFIG ----------

BASE_DIR = Path(__file__).resolve().parents[1]
DB_DIR = BASE_DIR / "lancedb_data"
DATA_DIR = BASE_DIR / "data"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TABLE_NAME = "doc_chunks_v4"

# Chunking
CHUNK_SIZE_WORDS = 80
CHUNK_OVERLAP_WORDS = 20

# Ollama
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

# Confidence thresholds (cosine similarity)
MIN_BEST_SCORE = 0.4      # below this → reject
MIN_AVG_SCORE = 0.30      # optional extra check
USE_OLLAMA = True

# ---------- DOMAIN MODELS ----------

@dataclass
class Document:
    doc_id: str
    path: Path
    doc_type: str
    version: int
    created_at: datetime


# ---------- CHUNKING / INDEXING ----------

def word_chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    words = text.split()
    if not words:
        return []

    chunks: List[str] = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        if not chunk_words:
            break
        chunks.append(" ".join(chunk_words))

        step = chunk_size - overlap
        if step <= 0:
            step = chunk_size
        start += step

    return chunks


def read_document_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Document not found at: {path}")
    return path.read_text(encoding="utf-8")


def get_embedding_model() -> SentenceTransformer:
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def compute_embeddings(model: SentenceTransformer, texts: List[str]):
    print(f"Computing embeddings for {len(texts)} chunks...")
    return model.encode(texts, convert_to_numpy=True, show_progress_bar=True)


def connect_lancedb(db_dir: Path):
    os.makedirs(db_dir, exist_ok=True)
    print(f"Connecting to LanceDB at: {db_dir}")
    return lancedb.connect(str(db_dir))


def build_document_from_path(path: Path, version: int = 1) -> Document:
    doc_id = f"{path.stem}_v{version}"

    lower_name = path.name.lower()
    if "policy" in lower_name:
        doc_type = "policy"
    elif "tutorial" in lower_name:
        doc_type = "tutorial"
    else:
        doc_type = "general"

    return Document(
        doc_id=doc_id,
        path=path,
        doc_type=doc_type,
        version=version,
        created_at=datetime.utcnow(),
    )


def index_documents_in_folder(
    data_dir: Path,
    model: SentenceTransformer,
    db,
    table_name: str = TABLE_NAME,
):
    txt_files = sorted(data_dir.glob("*.txt"))
    if not txt_files:
        raise RuntimeError(f"No .txt files found in {data_dir}")

    all_records: List[Dict] = []

    for path in txt_files:
        print(f"\nIndexing document: {path.name}")
        doc = build_document_from_path(path)

        full_text = read_document_text(doc.path)
        chunks = word_chunk_text(
            full_text,
            chunk_size=CHUNK_SIZE_WORDS,
            overlap=CHUNK_OVERLAP_WORDS,
        )
        print(f"  -> produced {len(chunks)} chunks")

        embeddings = compute_embeddings(model, chunks)

        for i, (chunk_text, emb) in enumerate(zip(chunks, embeddings)):
            all_records.append(
                {
                    "doc_id": doc.doc_id,
                    "chunk_id": i,
                    "doc_type": doc.doc_type,
                    "source": str(doc.path),
                    "version": doc.version,
                    "created_at": doc.created_at.isoformat(),
                    "text": chunk_text,
                    "embedding": emb.tolist(),
                }
            )

    if table_name in db.table_names():
        print(f"Table '{table_name}' exists. Dropping for a clean run...")
        db.drop_table(table_name)

    print(f"Creating table '{table_name}' with {len(all_records)} chunks...")
    table = db.create_table(table_name, data=all_records)
    return table


# ---------- RETRIEVAL WITH SCORES ----------

def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    denom = (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
    if denom == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / denom)


def retrieve_relevant_chunks(
    table,
    query_text: str,
    model: SentenceTransformer,
    top_k: int = 5,
    where: Optional[str] = None,
) -> pd.DataFrame:
    """
    Vector search + cosine similarity scores.
    """
    print(f"\n[RETRIEVE] Query: {query_text!r}")
    if where:
        print(f"[RETRIEVE] Filter: {where}")

    query_emb = model.encode([query_text], convert_to_numpy=True)[0]

    search_builder = table.search(query_emb)
    if where:
        search_builder = search_builder.where(where)

    # Also fetch embeddings so we can compute explicit scores
    df: pd.DataFrame = (
        search_builder
        .select(["doc_id", "chunk_id", "doc_type", "source", "version", "created_at", "text", "embedding"])
        .limit(top_k)
        .to_pandas()
    )

    if df.empty:
        return df

    # Compute cosine similarity scores manually
    scores: List[float] = []
    for emb_list in df["embedding"]:
        emb_vec = np.array(emb_list, dtype=np.float32)
        score = cosine_similarity(query_emb, emb_vec)
        scores.append(score)

    df["score"] = scores

    print("\n[RETRIEVE] Scores for top chunks:")
    for idx, row in df.iterrows():
        print(
            f"  rank={idx + 1}, doc_id={row['doc_id']}, "
            f"chunk_id={row['chunk_id']}, score={row['score']:.3f}"
        )

    return df


def build_context_from_chunks(df: pd.DataFrame) -> str:
    if df.empty:
        return ""

    lines: List[str] = []
    for idx, row in df.iterrows():
        rank = idx + 1
        tag = f"[{rank}] ({row['doc_id']}#chunk{row['chunk_id']} score={row['score']:.3f})"
        lines.append(f"{tag} {row['text']}")

    return "\n\n".join(lines)


def extract_sources(df: pd.DataFrame) -> List[Dict[str, Any]]:
    sources: List[Dict[str, Any]] = []
    for idx, row in df.iterrows():
        sources.append(
            {
                "rank": idx + 1,
                "doc_id": row["doc_id"],
                "chunk_id": int(row["chunk_id"]),
                "doc_type": row["doc_type"],
                "source": row["source"],
                "version": int(row["version"]),
                "created_at": row["created_at"],
                "score": float(row["score"]),
                "snippet": row["text"],
            }
        )
    return sources


# ---------- LLM LAYER: OLLAMA + FALLBACK ----------

def call_ollama_chat(query: str, context: str) -> str:
    """
    Call Ollama's /api/chat endpoint with a RAG-style prompt.
    """
    if not context:
        return (
            "I could not find any relevant information in the knowledge base "
            "to answer your question confidently."
        )

    prompt = (
        "You are a helpful assistant answering questions using ONLY the provided context.\n"
        "If the context is insufficient or does not contain the answer, explicitly say\n"
        "'I do not have enough information in the documents to answer this.'\n\n"
        "Context:\n"
        f"{context}\n\n"
        "Question:\n"
        f"{query}\n\n"
        "Answer based strictly on the context above. Do not invent facts."
    )

    url = f"{OLLAMA_BASE_URL}/api/chat"
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "stream": False,
    }

    resp = requests.post(url, json=payload, timeout=120)
    if resp.status_code != 200:
        print("\n[LLM] Ollama HTTP error:")
        print("Status code:", resp.status_code)
        try:
            print("Response body:", resp.text)
        except Exception:
            pass
        resp.raise_for_status()

    data = resp.json()
    return data["message"]["content"]


def dummy_llm_generate_answer(query: str, context: str) -> str:
    if not context:
        return (
            "I could not find any relevant information in the knowledge base "
            "to answer your question confidently."
        )

    lines = context.splitlines()
    bullet_points: List[str] = []

    for line in lines:
        if ") " in line:
            _, text_part = line.split(") ", maxsplit=1)
        else:
            text_part = line
        bullet_points.append(text_part)

    bullet_points = bullet_points[:3]
    bullets_str = "\n- ".join(bullet_points)

    answer = (
        f"Question: {query}\n\n"
        "Based on the retrieved documents, here are the key points:\n"
        f"- {bullets_str}\n\n"
        "This answer is synthesized from the closest matching chunks in the knowledge base."
    )

    return answer


def generate_answer_with_fallback(query: str, context: str) -> str:
    """
    If USE_OLLAMA is True, try Ollama then fall back.
    Otherwise just use dummy LLM.
    """
    if not USE_OLLAMA:
        print("\n[LLM] Using dummy LLM (Ollama disabled).")
        return dummy_llm_generate_answer(query, context)

    try:
        print("\n[LLM] Calling Ollama...")
        return call_ollama_chat(query, context)
    except Exception as e:
        print(f"[LLM] Ollama call failed: {e}")
        print("[LLM] Falling back to dummy LLM.")
        return dummy_llm_generate_answer(query, context)



# ---------- CONFIDENCE-BASED REJECTION ----------

def should_reject_answer(df: pd.DataFrame) -> bool:
    """
    Decide whether to answer or refuse based on similarity scores.
    """
    if df.empty:
        return True

    best_score = float(df["score"].max())
    avg_score = float(df["score"].mean())

    print(f"\n[CONFIDENCE] best_score={best_score:.3f}, avg_score={avg_score:.3f}")

    if best_score < MIN_BEST_SCORE:
        return True
    if avg_score < MIN_AVG_SCORE:
        return True

    return False


# ---------- FULL RAG PIPELINE ----------

def rag_answer(
    query_text: str,
    model: SentenceTransformer,
    table,
    top_k: int = 5,
    where: Optional[str] = None,
) -> Dict[str, Any]:
    df = retrieve_relevant_chunks(table, query_text, model, top_k=top_k, where=where)

    if should_reject_answer(df):
        return {
            "query": query_text,
            "answer": (
                "I do not have enough high-confidence information in the indexed documents "
                "to answer this question. Please refine your question or add more relevant documents."
            ),
            "context": "",
            "sources": [],
            "raw_chunks_df": df,
            "rejected": True,
        }

    context = build_context_from_chunks(df)
    sources = extract_sources(df)
    answer = generate_answer_with_fallback(query_text, context)

    return {
        "query": query_text,
        "answer": answer,
        "context": context,
        "sources": sources,
        "raw_chunks_df": df,
        "rejected": False,
    }


# ---------- MAIN DEMO ----------

def main():
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"DATA_DIR does not exist: {DATA_DIR}")

    print(f"Using data directory: {DATA_DIR}")

    model = get_embedding_model()
    db = connect_lancedb(DB_DIR)

    table = index_documents_in_folder(DATA_DIR, model, db, table_name=TABLE_NAME)

    query = "What is a RAG system and what are we building in this project?"
    result = rag_answer(query, model, table, top_k=5)

    print("\n================ RAG ANSWER (OLLAMA) ================")
    print(result["answer"])

    print("\n================ METADATA ============================")
    print(f"Rejected: {result['rejected']}")

    print("\n================ SOURCES =============================")
    for src in result["sources"]:
        print(
            f"[{src['rank']}] doc_id={src['doc_id']} "
            f"chunk_id={src['chunk_id']} "
            f"doc_type={src['doc_type']} "
            f"score={src['score']:.3f} "
            f"source={src['source']}"
        )
        print(f"   snippet: {src['snippet']}")


if __name__ == "__main__":
    main()
