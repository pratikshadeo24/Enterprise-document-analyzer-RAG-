from rag_core.config import DATA_DIR, DB_DIR, TABLE_NAME, EMBEDDING_MODEL_NAME
from rag_core.db import connect_lancedb
from rag_core.embeddings import get_embedding_model
from rag_core.indexing import index_documents_version


def main():
    print(f"DATA_DIR: {DATA_DIR}")
    model = get_embedding_model(EMBEDDING_MODEL_NAME)
    db = connect_lancedb(DB_DIR)

    # For now we just index as version=1 (or bump this manually later)
    table = index_documents_version(DATA_DIR, model, db, TABLE_NAME, version=1)
    print(f"Indexing completed. Table name: {TABLE_NAME}, rows: {table.count_rows()}")


if __name__ == "__main__":
    main()
