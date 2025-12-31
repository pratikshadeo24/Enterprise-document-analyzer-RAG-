import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any

import lancedb
import pandas as pd
from sentence_transformers import SentenceTransformer


# ---------- CONFIG ----------

BASE_DIR = Path(__file__).resolve().parents[1]
DB_DIR = BASE_DIR / "lancedb_data"
DATA_DIR = BASE_DIR / "data"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TABLE_NAME = "doc_chunks_v3"

# Chunking hyperparameters (still tunable later)
CHUNK_SIZE_WORDS = 80
CHUNK_OVERLAP_WORDS = 20


# ---------- DOMAIN MODELS ----------

@dataclass
class Document:
    doc_id: str
    path: Path
    doc_type: str
    version: int
    created_at: datetime


# ---------- CHUNKING / INDEXING HELPERS ----------

def word_chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Split text into overlapping word-based chunks.

    chunk_size: approx number of words per chunk
    overlap   : how many words to repeat from previous chunk
    """
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
            # safety: ensure we don't end up in infinite loop
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
    """Infer doc_id + doc_type from file name."""
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
    """
    Index all .txt files into a LanceDB table with metadata + embeddings.
    """
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

    # (Re)create table
    if table_name in db.table_names():
        print(f"Table '{table_name}' exists. Dropping for a clean run...")
        db.drop_table(table_name)

    print(f"Creating table '{table_name}' with {len(all_records)} chunks...")
    table = db.create_table(table_name, data=all_records)
    return table


# ---------- RETRIEVAL LAYER ----------

def retrieve_relevant_chunks(
    table,
    query_text: str,
    model: SentenceTransformer,
    top_k: int = 5,
    where: Optional[str] = None,
) -> pd.DataFrame:
    """
    Vector search with optional metadata filter.
    Returns a pandas DataFrame of top_k chunks.
    """
    print(f"\n[RETRIEVE] Query: {query_text!r}")
    if where:
        print(f"[RETRIEVE] Filter: {where}")

    query_emb = model.encode([query_text], convert_to_numpy=True)[0]

    search_builder = table.search(query_emb)
    if where:
        search_builder = search_builder.where(where)

    result_df: pd.DataFrame = (
        search_builder
        .select(["doc_id", "chunk_id", "doc_type", "source", "version", "created_at", "text"])
        .limit(top_k)
        .to_pandas()
    )

    return result_df


def build_context_from_chunks(df: pd.DataFrame) -> str:
    """
    Build a context string to feed into an LLM.
    Example format:

    [1] (doc_id#chunk_id) text...
    [2] (doc_id#chunk_id) text...
    """
    if df.empty:
        return ""

    lines: List[str] = []
    for idx, row in df.iterrows():
        rank = idx + 1
        tag = f"[{rank}] ({row['doc_id']}#chunk{row['chunk_id']})"
        lines.append(f"{tag} {row['text']}")

    return "\n\n".join(lines)


def extract_sources(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Build a lightweight list of sources for citations.
    """
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
                "snippet": row["text"],
            }
        )
    return sources


# ---------- LLM LAYER (DUMMY FOR NOW) ----------

def dummy_llm_generate_answer(query: str, context: str) -> str:
    """
    Placeholder 'LLM' that simulates answering using the retrieved context.
    It doesn't call any external API (completely free).

    Later you can replace this with:
      - Ollama (local LLM)
      - OpenAI ChatCompletion
      - AWS Bedrock, etc.
    """
    if not context:
        return (
            "I could not find any relevant information in the knowledge base "
            "to answer your question confidently."
        )

    # Very naive heuristic answer: echo main ideas from context.
    # This is just to prove the pipeline wiring.
    lines = context.splitlines()

    # Extract only the text part (after the tag)
    bullet_points: List[str] = []
    for line in lines:
        # Format: [rank] (doc_id#chunkX) text...
        # Split once on ') ' to get the text.
        if ") " in line:
            _, text_part = line.split(") ", maxsplit=1)
        else:
            text_part = line
        bullet_points.append(text_part)

    # Keep only first few points so answer isn't huge
    bullet_points = bullet_points[:3]

    bullets_str = "\n- ".join(bullet_points)

    answer = (
        f"Question: {query}\n\n"
        "Based on the retrieved documents, here are the key points:\n"
        f"- {bullets_str}\n\n"
        "This answer is synthesized from the closest matching chunks in the knowledge base."
    )

    return answer


# ---------- FULL RAG PIPELINE ----------

def rag_answer(
    query_text: str,
    model: SentenceTransformer,
    table,
    top_k: int = 5,
    where: Optional[str] = None,
) -> Dict[str, Any]:
    """
    End-to-end RAG pipeline:
      1. Retrieve top_k relevant chunks (with optional filter)
      2. Build a context string
      3. Call LLM (dummy for now)
      4. Return structured response with answer + sources
    """
    df = retrieve_relevant_chunks(table, query_text, model, top_k=top_k, where=where)

    context = build_context_from_chunks(df)
    sources = extract_sources(df)
    answer = dummy_llm_generate_answer(query_text, context)

    return {
        "query": query_text,
        "answer": answer,
        "context": context,
        "sources": sources,
        "raw_chunks_df": df,  # useful for debugging; you can drop this in production
    }


# ---------- MAIN (DEMO) ----------

def main():
    # 1. Ensure data dir exists
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"DATA_DIR does not exist: {DATA_DIR}")

    print(f"Using data directory: {DATA_DIR}")

    # 2. Load model
    model = get_embedding_model()

    # 3. Connect DB
    db = connect_lancedb(DB_DIR)

    # 4. Index all .txt docs into doc_chunks_v3 table
    table = index_documents_in_folder(DATA_DIR, model, db, table_name=TABLE_NAME)

    # 5. Run RAG pipeline for a test query
    query = "What is a RAG system and what are we building in this project?"
    result = rag_answer(query, model, table, top_k=5)

    print("\n================ RAG ANSWER ================")
    print(result["answer"])

    print("\n================ SOURCES ===================")
    for src in result["sources"]:
        print(
            f"[{src['rank']}] doc_id={src['doc_id']} "
            f"chunk_id={src['chunk_id']} "
            f"doc_type={src['doc_type']} "
            f"source={src['source']}"
        )
        print(f"   snippet: {src['snippet']}")


if __name__ == "__main__":
    main()
