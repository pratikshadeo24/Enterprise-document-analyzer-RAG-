from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import lancedb


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    denom = (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
    if denom == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / denom)


def retrieve_relevant_chunks(
    table: lancedb.table.LanceTable,
    query_text: str,
    model: SentenceTransformer,
    top_k: int = 8,
    where: Optional[str] = None,
) -> pd.DataFrame:
    print(f"\n[RETRIEVE] Query: {query_text!r}")
    if where:
        print(f"[RETRIEVE] Filter: {where}")

    query_emb = model.encode([query_text], convert_to_numpy=True)[0]

    search_builder = table.search(query_emb)
    if where:
        search_builder = search_builder.where(where)

    df: pd.DataFrame = (
        search_builder
        .select([
            "doc_key",
            "doc_version",
            "page_number",        # <-- NEW
            "chunk_id",
            "doc_type",
            "source",
            "created_at",
            "text",
            "embedding",
        ])
        .limit(top_k)
        .to_pandas()
    )

    if df.empty:
        return df

    scores = []
    for emb_list in df["embedding"]:
        emb_vec = np.array(emb_list, dtype=np.float32)
        scores.append(cosine_similarity(query_emb, emb_vec))
    df["score"] = scores

    print("\n[RETRIEVE] Raw results:")
    for idx, row in df.iterrows():
        print(
            f"  rank={idx + 1}, doc_key={row['doc_key']}, "
            f"v={row['doc_version']}, chunk={row['chunk_id']}, score={row['score']:.3f}"
        )

    return df


def filter_latest_versions(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    max_versions = df.groupby("doc_key")["doc_version"].transform("max")
    latest_df = df[df["doc_version"] == max_versions].copy()

    print("\n[VERSIONING] Keeping only latest versions per doc_key:")
    for idx, row in latest_df.iterrows():
        print(
            f"  rank={idx + 1}, doc_key={row['doc_key']}, "
            f"v={row['doc_version']}, chunk={row['chunk_id']}, score={row['score']:.3f}"
        )

    return latest_df


def build_context_from_chunks(df: pd.DataFrame) -> str:
    if df.empty:
        return ""

    lines = []
    for idx, row in df.iterrows():
        rank = idx + 1
        page_display = (
            f"p{int(row['page_number'])}" if not pd.isna(row.get("page_number", None)) else "p?"
        )
        tag = (
            f"[{rank}] ({row['doc_key']}@v{row['doc_version']}:{page_display}"
            f"#chunk{row['chunk_id']} score={row['score']:.3f})"
        )
        lines.append(f"{page_display} {tag} {row['text']}")
    return "\n\n".join(lines)


def extract_sources(df: pd.DataFrame) -> List[Dict[str, Any]]:
    sources: List[Dict[str, Any]] = []
    for idx, row in df.iterrows():
        sources.append(
            {
                "rank": idx + 1,
                "doc_key": row["doc_key"],
                "doc_version": int(row["doc_version"]),
                "chunk_id": int(row["chunk_id"]),
                "doc_type": row["doc_type"],
                "source": row["source"],
                "score": float(row["score"]),
                "page_number": (
                    int(row["page_number"]) if "page_number" in row and pd.notna(row["page_number"]) else None
                ),
                "snippet": row["text"],
            }
        )
    return sources
