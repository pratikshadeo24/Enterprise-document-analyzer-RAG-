from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class Document:
    doc_key: str          # logical id, e.g., "sample_docs"
    doc_version: int      # 1,2,3...
    path: Path
    doc_type: str
    created_at: datetime
