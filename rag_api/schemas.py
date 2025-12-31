from typing import List, Optional
from pydantic import BaseModel, Field


class Source(BaseModel):
    rank: int = Field(..., description="Final rank after retrieval/reranking")
    doc_key: str
    doc_version: int
    chunk_id: int
    doc_type: str
    source: str = Field(..., description="Path or identifier of the source document")
    score: float = Field(..., description="Original vector similarity score")
    page_number: Optional[int] = None
    snippet: str = Field(..., description="Chunk text snippet used as context")


class QueryRequest(BaseModel):
    query: str = Field(..., description="User question to answer from the KB")
    top_k: int = Field(
        8,
        ge=1,
        le=50,
        description="How many chunks to retrieve from LanceDB",
    )
    latest_only: bool = Field(
        True,
        description="If true, use only latest doc_version per doc_key",
    )


class QueryResponse(BaseModel):
    query: str
    answer: str
    rejected: bool = Field(
        False,
        description="True if the system abstained due to low confidence or guardrails",
    )
    # confidence: Optional[float] = Field(
    #     None,
    #     description="Combined confidence score used by abstention logic, if available",
    # )
    top_k: int
    latest_only: bool
    sources: List[Source]
    
class IngestResponse(BaseModel):
    doc_key: str = Field(..., description="Logical document key (usually filename without extension)")
    doc_version: int = Field(..., description="Version assigned for this ingestion of the document")
    doc_type: str = Field(..., description="Document type/category")
    num_chunks: int = Field(..., description="Number of chunks indexed into LanceDB")
    source: str = Field(..., description="Path or identifier of the source file")