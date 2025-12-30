from typing import Optional, Dict, Any

from sentence_transformers import SentenceTransformer
import pandas as pd
import lancedb

from .retrieval import (
    retrieve_relevant_chunks,
    filter_latest_versions,
    build_context_from_chunks,
    extract_sources,
)
from .llm import generate_answer
from .config import (
    CONF_TOPK_AVG,
    CONF_W_TOP1,
    CONF_W_AVG_TOPK,
    CONF_W_MARGIN,
    MIN_TOP1_SCORE,
    MIN_CONFIDENCE,
)


def _should_reject(df: pd.DataFrame) -> bool:
    """
    Decide whether to abstain based on a weighted combination of:
      - top1 similarity score
      - average of top-K scores
      - margin between top1 and top2

    Steps:
      1) If df is empty → reject.
      2) Compute scores descending.
      3) Compute:
           top1, avg_topk, margin = top1 - top2
      4) confidence = w1*top1 + w2*avg_topk + w3*margin
      5) If top1 < MIN_TOP1_SCORE or confidence < MIN_CONFIDENCE → reject.
    """
    if df.empty:
        return True

    scores = sorted(df["score"].astype(float), reverse=True)
    top1 = scores[0]

    # average top-K scores
    k = min(CONF_TOPK_AVG, len(scores))
    avg_topk = sum(scores[:k]) / k if k > 0 else 0.0

    # margin between top1 and top2
    if len(scores) > 1:
        margin = max(top1 - scores[1], 0.0)
    else:
        margin = top1  # if only one score, margin = top1 itself

    # weighted confidence
    confidence = (
        CONF_W_TOP1 * top1
        + CONF_W_AVG_TOPK * avg_topk
        + CONF_W_MARGIN * margin
    )

    print(
        f"\n[CONFIDENCE] top1={top1:.3f}, "
        f"avg_topk={avg_topk:.3f}, "
        f"margin={margin:.3f}, "
        f"confidence={confidence:.3f}"
    )

    # Hard guardrail: if even best match is weak, don't answer
    if top1 < MIN_TOP1_SCORE:
        return True

    # Global confidence threshold
    if confidence < MIN_CONFIDENCE:
        return True

    return False


def rag_answer(
    query_text: str,
    model: SentenceTransformer,
    table: lancedb.table.LanceTable,
    top_k: int = 8,
    where: Optional[str] = None,
    latest_only: bool = True,
) -> Dict[str, Any]:
    """
    RAG pipeline with:
      - vector retrieval + scores
      - latest-only version routing
      - confidence-based abstention
    """
    df = retrieve_relevant_chunks(table, query_text, model, top_k=top_k, where=where)

    if latest_only:
        df = filter_latest_versions(df)

    # --- NEW: confidence-based abstention ---
    if _should_reject(df):
        return {
            "query": query_text,
            "answer": (
                "I do not have enough high-confidence information in the indexed documents "
                "to answer this question. Please refine your question or add more relevant documents."
            ),
            "context": "",
            "sources": [],
            "raw_chunks_df": df,
            "latest_only": latest_only,
            "rejected": True,
        }
    # ----------------------------------------

    context = build_context_from_chunks(df)
    sources = extract_sources(df)
    answer = generate_answer(query_text, context)

    return {
        "query": query_text,
        "answer": answer,
        "context": context,
        "sources": sources,
        "raw_chunks_df": df,
        "latest_only": latest_only,
        "rejected": False,
    }
