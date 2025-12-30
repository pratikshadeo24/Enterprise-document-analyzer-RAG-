import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any

import numpy as np
import lancedb
import pandas as pd
from sentence_transformers import SentenceTransformer


# ---------- CONFIG ----------

BASE_DIR = Path(__file__).resolve().parents[1]
DB_DIR = BASE_DIR / "lancedb_data"
DATA_DIR = BASE_DIR / "data"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TABLE_NAME = "doc_chunks_versions"

CHUNK_SIZE_WORDS = 80
CHUNK_OVERLAP_WORDS = 20

USE_OLLAMA = False  # keep this off for now, dummy only


# ---------- DOMAIN MODELS ----------

@dataclass
class Document:
    doc_key: str          # logical identifier, e.g., "sample_docs"
    doc_version: int      # 1, 2, 3, ...
    path: Path
    doc_type: str
    created_at: datetime


# ---------- BASIC HELPERS ----------

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


# ---------- INDEXING WITH VERSIONS ----------

def build_document_from_path(path: Path, version: int) -> Document:
    """
    doc_key: path.stem (same logical doc for all versions of that file)
    doc_version: integer (1, 2, 3, ...)
    """
    doc_key = path.stem

    lower_name = path.name.lower()
    if "policy" in lower_name:
        doc_type = "policy"
    elif "tutorial" in lower_name:
        doc_type = "tutorial"
    else:
        doc_type = "general"

    return Document(
        doc_key=doc_key,
        doc_version=version,
        path=path,
        doc_type=doc_type,
        created_at=datetime.utcnow(),
    )


def index_documents_version(
    data_dir: Path,
    model: SentenceTransformer,
    db,
    table_name: str,
    version: int,
):
    """
    Index all .txt docs as a specific version into the same table.
    If table doesn't exist, create it.
    If it exists, append to it.
    """
    txt_files = sorted(data_dir.glob("*.txt"))
    if not txt_files:
        raise RuntimeError(f"No .txt files found in {data_dir}")

    all_records: List[Dict] = []

    for path in txt_files:
        print(f"\nIndexing document: {path.name} as version {version}")
        doc = build_document_from_path(path, version=version)

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
                    "doc_key": doc.doc_key,
                    "doc_version": doc.doc_version,
                    "chunk_id": i,
                    "doc_type": doc.doc_type,
                    "source": str(doc.path),
                    "created_at": doc.created_at.isoformat(),
                    "text": chunk_text,
                    "embedding": emb.tolist(),
                }
            )

    if table_name in db.table_names():
        print(f"Table '{table_name}' exists. Appending new version records...")
        table = db.open_table(table_name)
        table.add(all_records)
    else:
        print(f"Creating table '{table_name}' with initial records...")
        table = db.create_table(table_name, data=all_records)

    return table


# ---------- RETRIEVAL WITH VERSIONING ----------

def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    denom = (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
    if denom == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / denom)


def retrieve_relevant_chunks(
    table,
    query_text: str,
    model: SentenceTransformer,
    top_k: int = 8,
    where: Optional[str] = None,
) -> pd.DataFrame:
    """
    Retrieve chunks + embeddings + scores. No version filtering yet.
    """
    print(f"\n[RETRIEVE] Query: {query_text!r}")
    if where:
        print(f"[RETRIEVE] Filter: {where}")

    query_emb = model.encode([query_text], convert_to_numpy=True)[0]

    search_builder = table.search(query_emb)
    if where:
        search_builder = search_builder.where(where)

    df: pd.DataFrame = (
        search_builder
        .select(["doc_key", "doc_version", "chunk_id", "doc_type", "source", "created_at", "text", "embedding"])
        .limit(top_k)
        .to_pandas()
    )

    if df.empty:
        return df

    scores: List[float] = []
    for emb_list in df["embedding"]:
        emb_vec = np.array(emb_list, dtype=np.float32)
        scores.append(cosine_similarity(query_emb, emb_vec))

    df["score"] = scores

    print("\n[RETRIEVE] Raw results (before version filtering):")
    for idx, row in df.iterrows():
        print(
            f"  rank={idx + 1}, doc_key={row['doc_key']}, "
            f"version={row['doc_version']}, chunk_id={row['chunk_id']}, "
            f"score={row['score']:.3f}"
        )

    return df


def filter_latest_versions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only rows where doc_version is the max version per doc_key.
    """
    if df.empty:
        return df

    max_versions = df.groupby("doc_key")["doc_version"].transform("max")
    latest_df = df[df["doc_version"] == max_versions].copy()

    print("\n[VERSIONING] After keeping only latest versions per doc_key:")
    for idx, row in latest_df.iterrows():
        print(
            f"  rank={idx + 1}, doc_key={row['doc_key']}, "
            f"version={row['doc_version']}, chunk_id={row['chunk_id']}, "
            f"score={row['score']:.3f}"
        )

    return latest_df


def build_context_from_chunks(df: pd.DataFrame) -> str:
    if df.empty:
        return ""

    lines: List[str] = []
    for idx, row in df.iterrows():
        rank = idx + 1
        tag = f"[{rank}] ({row['doc_key']}@v{row['doc_version']}#chunk{row['chunk_id']} score={row['score']:.3f})"
        lines.append(f"{tag} {row['text']}")

    return "\n\n".join(lines)


def extract_sources(df: pd.DataFrame) -> List[Dict[str, Any]]:
    sources: List[Dict[str, Any]] = []
    for idx, row in df.iterrows():
        sources.append(
            {
                "rank": idx + 1,
                "doc_key": row["doc_key"],
                "doc_version": int(row["doc_version"]),
                "chunk_id": int(row["chunk_id"]),
                "doc_type": row["doc_type"],
                "source": row["source"],
                "score": float(row["score"]),
                "snippet": row["text"],
            }
        )
    return sources


# ---------- LLM (DUMMY ONLY) ----------

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

    bullet_points = bullet_points[:4]
    bullets_str = "\n- ".join(bullet_points)

    answer = (
        f"Question: {query}\n\n"
        "Based on the latest document versions in the knowledge base, key points are:\n"
        f"- {bullets_str}\n\n"
        "This answer is synthesized from the top-matching chunks."
    )
    return answer


# ---------- FULL RAG WITH VERSION CONTROL ----------

def rag_answer(
    query_text: str,
    model: SentenceTransformer,
    table,
    top_k: int = 8,
    where: Optional[str] = None,
    latest_only: bool = True,
) -> Dict[str, Any]:
    """
    RAG pipeline with 'latest_only' switch.
    """
    df = retrieve_relevant_chunks(table, query_text, model, top_k=top_k, where=where)

    if latest_only:
        df = filter_latest_versions(df)

    context = build_context_from_chunks(df)
    sources = extract_sources(df)
    answer = dummy_llm_generate_answer(query_text, context)

    return {
        "query": query_text,
        "answer": answer,
        "context": context,
        "sources": sources,
        "raw_chunks_df": df,
        "latest_only": latest_only,
    }


# ---------- MAIN DEMO ----------

def main():
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"DATA_DIR does not exist: {DATA_DIR}")

    print(f"Using data directory: {DATA_DIR}")

    model = get_embedding_model()
    db = connect_lancedb(DB_DIR)

    # Simulate two indexing runs (v1 and v2) of the same docs
    # In real life: you would run this script later with a new 'version' value.
    table = index_documents_version(DATA_DIR, model, db, TABLE_NAME, version=1)
    table = index_documents_version(DATA_DIR, model, db, TABLE_NAME, version=2)

    query = "What is a RAG system and what are we building in this project?"

    print("\n\n===== RAG ANSWER (latest_only = False, all versions) =====")
    result_all = rag_answer(query, model, table, top_k=8, latest_only=False)
    print(result_all["answer"])

    print("\n\n===== RAG ANSWER (latest_only = True, only newest versions) =====")
    result_latest = rag_answer(query, model, table, top_k=8, latest_only=True)
    print(result_latest["answer"])

    print("\n\n===== SOURCES (latest_only = True) =====")
    for src in result_latest["sources"]:
        print(
            f"[{src['rank']}] doc_key={src['doc_key']} "
            f"v={src['doc_version']} chunk={src['chunk_id']} "
            f"score={src['score']:.3f} source={src['source']}"
        )
        print(f"   snippet: {src['snippet']}")


if __name__ == "__main__":
    main()
