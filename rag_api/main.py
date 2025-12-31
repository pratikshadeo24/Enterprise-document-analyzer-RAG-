from pathlib import Path
from typing import Dict, Any, List

from fastapi import FastAPI, UploadFile, File, HTTPException

from rag_api.schemas import QueryRequest, QueryResponse, Source, IngestResponse

from rag_core.config import DATA_DIR, EMBEDDING_MODEL_NAME
from rag_core.embeddings import get_embedding_model
from rag_core.indexing import index_paths_incremental
from rag_core.pipeline import rag_answer
from functools import lru_cache
from sentence_transformers import SentenceTransformer


app = FastAPI(
    title="Enterprise RAG API",
    version="1.0.0",
    description="RAG service using LanceDB + semantic chunking + reranking + Ollama",
)

@lru_cache(maxsize=1)
def get_api_embedding_model() -> SentenceTransformer:
    """
    Cached embedding model instance for API usage.
    Reuses the same model as scripts/index_docs.py.
    """
    print("[API] Loading embedding model for uploads...")
    return get_embedding_model(EMBEDDING_MODEL_NAME)

@app.get("/healthz")
def healthcheck() -> Dict[str, str]:
    """
    Simple health check endpoint.
    """
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query_rag_endpoint(request: QueryRequest) -> QueryResponse:
    """
    Main RAG query endpoint.
    - Accepts a user query + retrieval options.
    - Calls rag_answer(...) from rag_core.pipeline.
    - Returns a structured response with answer, confidence, and sources.
    """

    # Call your core pipeline.
    # Adjust argument names if your rag_answer signature differs.
    result: Dict[str, Any] = rag_answer(
        query_text=request.query,
        top_k=request.top_k,
        latest_only=request.latest_only,
    )

    # Convert raw sources (list[dict]) to Pydantic Source models
    raw_sources: List[Dict[str, Any]] = result.get("sources", [])
    sources: List[Source] = []

    for s in raw_sources:
        sources.append(
            Source(
                rank=s["rank"],
                doc_key=s["doc_key"],
                doc_version=s["doc_version"],
                page_number=s.get("page_number"),
                chunk_id=s["chunk_id"],
                doc_type=s["doc_type"],
                source=s["source"],
                score=s["score"],
                snippet=s["snippet"],
            )
        )

    response = QueryResponse(
        query=result.get("query", request.query),
        answer=result.get("answer", ""),
        rejected=bool(result.get("rejected", False)),
        top_k=request.top_k,
        latest_only=request.latest_only,
        sources=sources,
    )

    return response


@app.post("/documents/upload", response_model=List[IngestResponse])
async def upload_documents(files: List[UploadFile] = File(...)) -> List[IngestResponse]:
    """
    Upload one or more documents, save them to DATA_DIR, and index them incrementally.

    - Supports txt, md, pdf, docx (same as your offline indexer).
    - Uses the same index_paths_incremental(...) as scripts/index_docs.py.
    - Returns per-document ingestion summaries.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    data_dir = Path(DATA_DIR)
    data_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: List[Path] = []

    for upload in files:
        # You might want to do simple validation on extension here if desired.
        dest = data_dir / upload.filename
        contents = await upload.read()
        dest.write_bytes(contents)
        print(f"[API] Saved uploaded file to: {dest}")
        saved_paths.append(dest)

    model = get_api_embedding_model()

    print("[API] Indexing uploaded files incrementally...")
    summaries = index_paths_incremental(saved_paths, model=model)

    return [
        IngestResponse(
            doc_key=s["doc_key"],
            doc_version=s["doc_version"],
            doc_type=s["doc_type"],
            num_chunks=s["num_chunks"],
            source=s["source"],
        )
        for s in summaries
    ]
