from functools import lru_cache
from typing import Optional

import numpy as np
import pandas as pd
from sentence_transformers import CrossEncoder

from .config import RERANKER_MODEL_NAME, RERANK_TOP_N, USE_RERANKER


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    """
    Lazily load the CrossEncoder reranker model once.
    """
    print(f"[RERANK] Loading CrossEncoder model: {RERANKER_MODEL_NAME}")
    model = CrossEncoder(RERANKER_MODEL_NAME)
    return model


def rerank_chunks_for_query(query: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    Rerank retrieved chunks for a given query using a CrossEncoder.

    - Takes the existing df from LanceDB (with 'text' and 'score' columns).
    - Scores (query, chunk_text) pairs with CrossEncoder.
    - Adds a 'rerank_score' column.
    - Returns df sorted by rerank_score (descending).

    NOTE:
      - We only rerank the top-N rows (RERANK_TOP_N) to save time.
    """
    if not USE_RERANKER:
        # Safety: if disabled, just return original ordering
        return df

    if df.empty:
        return df

    model = get_reranker()

    # Work on a copy to avoid mutating caller df unexpectedly
    df = df.copy()

    # Limit to top N by original score to keep cost bounded
    # (We assume df already sorted by 'score' descending from retrieval.)
    top_n = min(RERANK_TOP_N, len(df))
    df_head = df.head(top_n).copy()
    df_tail = df.iloc[top_n:].copy()  # untouched remainder (if any)

    # Prepare (query, chunk_text) pairs
    pairs = [(query, text) for text in df_head["text"].tolist()]

    # CrossEncoder gives a relevance score for each pair
    # Higher score = more relevant to the query
    print(f"[RERANK] Scoring {len(pairs)} chunks for query via CrossEncoder...")
    scores = model.predict(pairs)
    df_head["rerank_score"] = scores

    # Sort by rerank_score (desc)
    df_head = df_head.sort_values("rerank_score", ascending=False).reset_index(drop=True)

    # Concatenate reranked head with original tail (unchanged)
    if not df_tail.empty:
        # For rows not reranked, keep rerank_score = NaN
        df_tail["rerank_score"] = np.nan
        df_out = pd.concat([df_head, df_tail], axis=0).reset_index(drop=True)
    else:
        df_out = df_head

    return df_out
