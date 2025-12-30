import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

import lancedb
import pandas as pd
from sentence_transformers import SentenceTransformer


# ---------- CONFIG ----------

BASE_DIR = Path(__file__).resolve().parents[1]
DB_DIR = BASE_DIR / "lancedb_data"
DATA_DIR = BASE_DIR / "data"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TABLE_NAME = "doc_chunks_v2"

# Chunking hyperparameters
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


# ---------- CHUNKING LOGIC ----------

def word_chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Split text into overlapping word-based chunks.

    Example with chunk_size=5, overlap=2:
    words: w1 w2 w3 w4 w5 w6 w7 w8 ...
    chunk 0: w1 w2 w3 w4 w5
    chunk 1: w4 w5 w6 w7 w8
    etc.
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
        # move start forward, but keep some overlap
        start += (chunk_size - overlap)
        if start < 0:  # safety
            break

    return chunks


# ---------- IO / MODEL / DB HELPERS ----------

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


def create_or_replace_table(db, table_name: str, records: List[Dict]):
    """
    Create a LanceDB table with records (drop if exists).
    """
    if table_name in db.table_names():
        print(f"Table '{table_name}' exists. Dropping for a clean run...")
        db.drop_table(table_name)

    print(f"Creating table '{table_name}' with {len(records)} chunks...")
    table = db.create_table(table_name, data=records)
    return table


# ---------- INDEXING PIPELINE ----------

def build_document_from_path(path: Path, version: int = 1) -> Document:
    """
    Build a Document object for a given file path.
    You can customize doc_type based on file name, folder, etc.
    """
    # simple rule: use file stem as doc_id
    doc_id = f"{path.stem}_v{version}"

    # simple rule: tutorial vs policy vs general based on filename keywords
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
) -> "lancedb.table.LanceTable":
    """
    Index all .txt files in data_dir into one LanceDB table
    using our word-based chunking.
    """
    txt_files = sorted(data_dir.glob("*.txt"))
    if not txt_files:
        raise RuntimeError(f"No .txt files found in {data_dir}")

    all_records: List[Dict] = []

    for path in txt_files:
        print(f"\nIndexing document: {path.name}")
        doc = build_document_from_path(path)

        full_text = read_document_text(path)
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

    # Now write everything to one table
    table = create_or_replace_table(db, table_name, all_records)
    return table


# ---------- SEARCH PIPELINE ----------

def similarity_search(
    table,
    query_text: str,
    model: SentenceTransformer,
    top_k: int = 5,
    where: Optional[str] = None,
):
    """
    Run similarity search with optional metadata filter.

    Example:
      where="doc_type = 'policy'"
    """
    print(f"\n=== Query: {query_text!r} ===")
    if where:
        print(f"  with filter: {where}")

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

    if result_df.empty:
        print("No results found.")
        return

    for idx, row in result_df.iterrows():
        print(f"\nRank {idx + 1}:")
        print(f"doc_id    : {row['doc_id']}")
        print(f"chunk_id  : {row['chunk_id']}")
        print(f"doc_type  : {row['doc_type']}")
        print(f"version   : {row['version']}")
        print(f"created_at: {row['created_at']}")
        print(f"source    : {row['source']}")
        print(f"text      : {row['text']}")


# ---------- MAIN ----------

def main():
    # 1. Make sure data dir exists and has at least 1 file.
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"DATA_DIR does not exist: {DATA_DIR}")

    print(f"Using data directory: {DATA_DIR}")

    # 2. Load embedding model
    model = get_embedding_model()

    # 3. Connect to LanceDB
    db = connect_lancedb(DB_DIR)

    # 4. Index all .txt docs in data dir
    table = index_documents_in_folder(DATA_DIR, model, db, table_name=TABLE_NAME)

    # 5. Run a generic query (no filter)
    similarity_search(
        table,
        query_text="What is a RAG system?",
        model=model,
        top_k=3,
    )

    # 6. Run a filtered query (by doc_type, if any such docs exist)
    similarity_search(
        table,
        query_text="What is a RAG system?",
        model=model,
        top_k=3,
        where="doc_type = 'policy'",
    )


if __name__ == "__main__":
    main()
