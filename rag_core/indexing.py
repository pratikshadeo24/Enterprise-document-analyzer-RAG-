from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import lancedb
from sentence_transformers import SentenceTransformer

from .models import Document
from .chunking import word_chunk_text
from .embeddings import compute_embeddings
from .config import CHUNK_SIZE_WORDS, CHUNK_OVERLAP_WORDS
from pypdf import PdfReader 
from docx import Document as DocxDocument     # <-- NEW

# ---- NEW: supported extensions & helpers ----

SUPPORTED_EXTENSIONS = (".txt", ".md", ".pdf", ".docx")


def iter_source_files(data_dir: Path):
    """
    Yield all files in data_dir with supported extensions (.txt, .md, ...).
    """
    for ext in SUPPORTED_EXTENSIONS:
        for path in data_dir.glob(f"*{ext}"):
            yield path


def load_pdf_pages(path: Path) -> List[Tuple[int, str]]:
    """
    Return list of (page_number, text) for the PDF.
    page_number is 1-based.
    """
    reader = PdfReader(str(path))
    pages: List[Tuple[int, str]] = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text = text.strip()
        if not text:
            continue
        pages.append((i + 1, text))
    return pages


def load_docx_text(path: Path) -> str:
    """
    Load text from a DOCX file by concatenating all paragraph texts.
    No page numbers are available at this level.
    """
    doc = DocxDocument(str(path))
    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    return "\n".join(paragraphs)


def get_page_fragments(path: Path) -> List[Tuple[Optional[int], str]]:
    """
    Normalize content from different file types into a list of (page_number, text).

    - For PDFs: list of real pages.
    - For txt/md: single 'page' with page_number=None.
    """
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return load_pdf_pages(path)

    if suffix in (".txt", ".md"):
        text = path.read_text(encoding="utf-8")
        return [(None, text)]

    if suffix == ".docx":
        text = load_docx_text(path)
        return [(None, text)]

    raise ValueError(f"Unsupported file extension: {suffix} for {path}")
    


def _infer_doc_type(path: Path) -> str:
    lower_name = path.name.lower()
    if "policy" in lower_name:
        return "policy"
    if "tutorial" in lower_name:
        return "tutorial"
    return "general"


def build_document_from_path(path: Path, version: int) -> Document:
    doc_key = path.stem
    doc_type = _infer_doc_type(path)
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
    db: lancedb.DBConnection,
    table_name: str,
    version: int,
):
    """
    Index all supported docs (.txt, .md) as a specific version into the same table.
    If table doesn't exist, create it. Otherwise append.
    """
    files = sorted(iter_source_files(data_dir))
    if not files:
        raise RuntimeError(f"No supported files ({SUPPORTED_EXTENSIONS}) found in {data_dir}")

    all_records: List[Dict] = []

    for path in files:
        print(f"\nIndexing document: {path.name} as version {version}")
        doc = build_document_from_path(path, version=version)

        # NEW: iterate page fragments (per-page for PDF, single for txt/md)
        page_fragments = get_page_fragments(doc.path)
        print(f"  -> found {len(page_fragments)} page fragments")

        for page_number, page_text in page_fragments:
            chunks = word_chunk_text(
                page_text,
                chunk_size=CHUNK_SIZE_WORDS,
                overlap=CHUNK_OVERLAP_WORDS,
            )
            print(f"    page={page_number} -> produced {len(chunks)} chunks")

            if not chunks:
                continue

            embeddings = compute_embeddings(model, chunks)

            for i, (chunk_text, emb) in enumerate(zip(chunks, embeddings)):
                all_records.append(
                    {
                        "doc_key": doc.doc_key,
                        "doc_version": doc.doc_version,
                        "page_number": page_number,          # <-- NEW
                        "chunk_id": i,
                        "doc_type": doc.doc_type,
                        "source": str(doc.path),
                        "created_at": doc.created_at.isoformat(),
                        "text": chunk_text,
                        "embedding": emb.tolist(),
                    }
                )

    if table_name in db.table_names():
        print(f"Table '{table_name}' exists. Appending records...")
        db.drop_table(table_name)

    print(f"Creating table '{table_name}' with initial records...")
    table = db.create_table(table_name, data=all_records)

    return table
