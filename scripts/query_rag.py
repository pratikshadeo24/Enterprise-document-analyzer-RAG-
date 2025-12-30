from rag_core.config import DB_DIR, TABLE_NAME, EMBEDDING_MODEL_NAME
from rag_core.db import connect_lancedb
from rag_core.embeddings import get_embedding_model
from rag_core.pipeline import rag_answer


def main():
    query = "What is a RAG system and what are we building in this project?"

    model = get_embedding_model(EMBEDDING_MODEL_NAME)
    db = connect_lancedb(DB_DIR)
    table = db.open_table(TABLE_NAME)

    result = rag_answer(query, model, table, top_k=8, latest_only=True)

    print("\n================ RAG ANSWER ================")
    print(result["answer"])

    print("\n================ SOURCES ===================")
    for src in result["sources"]:
        print(
        f"[{src['rank']}] doc_key={src['doc_key']} "
        f"v={src['doc_version']} page={src['page_number']} chunk={src['chunk_id']} "
        f"score={src['score']:.3f} source={src['source']}"
        )
        print(f"   snippet: {src['snippet']}")


if __name__ == "__main__":
    main()
