from rag_core.config import EMBEDDING_MODEL_NAME
from rag_core.embeddings import get_embedding_model
from rag_core.indexing import index_paths_incremental


import argparse
from pathlib import Path
from typing import List

from rag_core.embeddings import get_embedding_model
from rag_core.indexing import index_paths_incremental


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Incrementally index document paths into LanceDB."
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="Paths to document files to index (txt, md, pdf, docx).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    raw_paths: List[str] = args.paths
    paths = [Path(p).resolve() for p in raw_paths]

    print("[INDEX_DOCS] Loading embedding model...")
    model = get_embedding_model(EMBEDDING_MODEL_NAME)

    print("[INDEX_DOCS] Incrementally indexing provided paths...")
    summaries = index_paths_incremental(paths, model=model)

    print("\n[Indexing Summary]")
    for s in summaries:
        print(
            f"- doc_key={s['doc_key']} v={s['doc_version']} "
            f"doc_type={s['doc_type']} chunks={s['num_chunks']} source={s['source']}"
        )


if __name__ == "__main__":
    main()
