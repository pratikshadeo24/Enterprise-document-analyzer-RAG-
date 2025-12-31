import os
from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass
from datetime import datetime
import lancedb
import pandas as pd
from sentence_transformers import SentenceTransformer


# ---------- CONFIG ----------

BASE_DIR = Path(__file__).resolve().parents[1]
DB_DIR = BASE_DIR / "lancedb_data"      # folder where LanceDB files will live
DATA_FILE = BASE_DIR / "data" / "sample_docs.txt"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TABLE_NAME = "doc_chunks"


@dataclass
class Document:
    doc_id: str
    path: Path
    doc_type: str
    version: int
    created_at: datetime

# ---------- HELPER FUNCTIONS ----------

def read_document_text(path: Path) -> str:
    """Read entire file as a single text blob."""
    if not path.exists():
        raise FileNotFoundError(f"Document not found at: {path}")
    return path.read_text(encoding="utf-8")


def simple_line_chunk(text: str) -> List[str]:
    """
    Very simple 'chunker' for now:
    - Split on lines
    - Strip whitespace
    - Drop empty lines

    Later we can replace this with smarter sentence/paragraph + overlap chunking.
    """
    lines = [ln.strip() for ln in text.splitlines()]
    return [ln for ln in lines if ln]


def get_embedding_model() -> SentenceTransformer:
    """Loads the sentence-transformers model (cached locally)."""
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return model


def compute_embeddings(model: SentenceTransformer, texts: List[str]):
    """Compute embeddings for a list of texts."""
    print(f"Computing embeddings for {len(texts)} texts...")
    # returns a numpy array of shape (n_texts, dim)
    embs = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    return embs


def connect_lancedb(db_dir: Path):
    """Connect to a LanceDB database in the given directory."""
    os.makedirs(db_dir, exist_ok=True)
    print(f"Connecting to LanceDB at: {db_dir}")
    db = lancedb.connect(str(db_dir))
    return db


def create_or_replace_table(db, table_name: str, records: List[Dict]):
    """
    Create a LanceDB table (or replace if exists) with the given records.
    Each record is a dict with keys: 'text', 'embedding'.
    """
    # If a table exists from previous runs, drop it for now (fresh start).
    if table_name in db.table_names():
        print(f"Table '{table_name}' exists. Dropping and recreating...")
        db.drop_table(table_name)

    print(f"Creating new table '{table_name}' with {len(records)} records...")
    table = db.create_table(table_name, data=records)
    return table

def index_single_document(
    doc: Document,
    model: SentenceTransformer,
    db,
):
    """
    Read one document, split into chunks, embed, and write to LanceDB.

    Returns the LanceDB table handle.
    """
    text = read_document_text(doc.path)
    chunks = simple_line_chunk(text)
    embeddings = compute_embeddings(model, chunks)

    records: List[Dict] = []
    for i, (chunk_text, emb) in enumerate(zip(chunks, embeddings)):
        records.append(
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

    table = create_or_replace_table(db, TABLE_NAME, records)
    return table


def run_similarity_search_with_metadata(
    table,
    query_text: str,
    model: SentenceTransformer,
    top_k: int = 3,
):
    """Search table and print results with metadata."""
    print(f"\n=== Similarity search for query: {query_text!r} ===")
    query_emb = model.encode([query_text], convert_to_numpy=True)[0]

    result_df: pd.DataFrame = (
        table.search(query_emb)
        .select(["doc_id", "chunk_id", "doc_type", "source", "version", "created_at", "text"])
        .limit(top_k)
        .to_pandas()
    )

    for idx, row in result_df.iterrows():
        print(f"\nRank {idx + 1}:")
        print(f"doc_id    : {row['doc_id']}")
        print(f"chunk_id  : {row['chunk_id']}")
        print(f"doc_type  : {row['doc_type']}")
        print(f"version   : {row['version']}")
        print(f"created_at: {row['created_at']}")
        print(f"source    : {row['source']}")
        print(f"text      : {row['text']}")


# ---------- MAIN FLOW ----------

def main():
    # 1. Build a Document object for our sample file
    doc = Document(
        doc_id="sample_docs_v1",
        path=DATA_FILE,
        doc_type="tutorial",
        version=1,
        created_at=datetime.utcnow(),
    )

    # 2. Load model
    model = get_embedding_model()

    # 3. Connect to LanceDB
    db = connect_lancedb(DB_DIR)

    # 4. Index this single document as chunks
    table = index_single_document(doc, model, db)

    # 5. Run a test query with metadata displayed
    run_similarity_search_with_metadata(
        table,
        query_text="What is a RAG system?",
        model=model,
        top_k=3,
    )


if __name__ == "__main__":
    main()
