from rag_core.config import DB_DIR, TABLE_NAME, EMBEDDING_MODEL_NAME
from rag_core.pipeline import rag_answer


def main():
    query = "What is a RAG system and what are we building in this project?"

    result = rag_answer(query, top_k=8, latest_only=True)

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
