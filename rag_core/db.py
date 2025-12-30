import os
from pathlib import Path
import lancedb


def connect_lancedb(db_dir: Path):
    os.makedirs(db_dir, exist_ok=True)
    print(f"Connecting to LanceDB at: {db_dir}")
    return lancedb.connect(str(db_dir))
