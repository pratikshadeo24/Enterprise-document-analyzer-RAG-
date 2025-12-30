import json
from pathlib import Path
from typing import List, Dict, Any

from rag_core.config import (
    BASE_DIR,
    DB_DIR,
    TABLE_NAME,
    EMBEDDING_MODEL_NAME,
    CONF_TOPK_AVG,
    CONF_W_TOP1,
    CONF_W_AVG_TOPK,
    CONF_W_MARGIN,
    MIN_TOP1_SCORE,
    MIN_CONFIDENCE,
)
from rag_core.db import connect_lancedb
from rag_core.embeddings import get_embedding_model
from rag_core.pipeline import rag_answer


PRINT_ANSWERS = True          # set False if logs get too noisy
PRINT_ONLY_WHEN_NOT_REJECTED = True
TRUNCATE_ANSWER_CHARS = 400   # just to keep things readable

def compute_confidence_from_scores(scores: List[float]) -> Dict[str, float]:
    """
    Given a list of scores (descending), compute:
      - top1
      - avg_topk
      - margin
      - confidence
    using the same logic as the abstention function.
    """
    if not scores:
        return {
            "top1": 0.0,
            "avg_topk": 0.0,
            "margin": 0.0,
            "confidence": 0.0,
        }

    scores_sorted = sorted(scores, reverse=True)
    top1 = scores_sorted[0]

    k = min(CONF_TOPK_AVG, len(scores_sorted))
    avg_topk = sum(scores_sorted[:k]) / k if k > 0 else 0.0

    if len(scores_sorted) > 1:
        margin = max(top1 - scores_sorted[1], 0.0)
    else:
        margin = top1

    confidence = (
        CONF_W_TOP1 * top1
        + CONF_W_AVG_TOPK * avg_topk
        + CONF_W_MARGIN * margin
    )

    return {
        "top1": top1,
        "avg_topk": avg_topk,
        "margin": margin,
        "confidence": confidence,
    }


def load_eval_queries(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main():
    eval_path = BASE_DIR / "eval_queries.json"
    if not eval_path.exists():
        raise FileNotFoundError(f"eval_queries.json not found at: {eval_path}")

    eval_queries = load_eval_queries(eval_path)
    print(f"Loaded {len(eval_queries)} eval queries from {eval_path}")

    # Prepare model + DB + table
    model = get_embedding_model(EMBEDDING_MODEL_NAME)
    db = connect_lancedb(DB_DIR)
    table = db.open_table(TABLE_NAME)

    # Metrics
    answerable_total = 0
    answerable_answered = 0
    answerable_hit_expected = 0

    unanswerable_total = 0
    unanswerable_rejected = 0

    rows_summary = []

    for q in eval_queries:
        qid = q["id"]
        query = q["query"]
        answerable = bool(q["answerable"])
        expected_keys = set(q.get("expected_doc_keys", []))

        print(f"\n=== Evaluating {qid}: {query!r} (answerable={answerable}) ===")

        result = rag_answer(query, model, table, top_k=8, latest_only=True)
        df = result["raw_chunks_df"]
        rejected = result.get("rejected", False)
        if PRINT_ANSWERS and (not PRINT_ONLY_WHEN_NOT_REJECTED or not rejected):
            answer_text = result.get("answer", "")
            if TRUNCATE_ANSWER_CHARS and len(answer_text) > TRUNCATE_ANSWER_CHARS:
                display_answer = answer_text[:TRUNCATE_ANSWER_CHARS] + "... [truncated]"
            else:
                display_answer = answer_text

            print("\n  --- MODEL ANSWER ---")
            print(display_answer)
            print("  --------------------")
        if df is None or df.empty:
            scores = []
            retrieved_keys = set()
        else:
            scores = [float(s) for s in df["score"].tolist()]
            retrieved_keys = set(df["doc_key"].tolist())

        conf = compute_confidence_from_scores(scores)
        top1 = conf["top1"]
        avg_topk = conf["avg_topk"]
        margin = conf["margin"]
        confidence = conf["confidence"]

        hit_expected = bool(expected_keys and (retrieved_keys & expected_keys))

        print(
            f"  rejected={rejected}, "
            f"hit_expected={hit_expected}, "
            f"top1={top1:.3f}, avg_topk={avg_topk:.3f}, "
            f"margin={margin:.3f}, confidence={confidence:.3f}"
        )

        # Aggregate metrics
        if answerable:
            answerable_total += 1
            if not rejected:
                answerable_answered += 1
            if hit_expected:
                answerable_hit_expected += 1
        else:
            unanswerable_total += 1
            if rejected:
                unanswerable_rejected += 1

        rows_summary.append(
            {
                "id": qid,
                "query": query,
                "answerable": answerable,
                "rejected": rejected,
                "hit_expected": hit_expected,
                "top1": top1,
                "avg_topk": avg_topk,
                "margin": margin,
                "confidence": confidence,
            }
        )

    print("\n================= SUMMARY =================")

    if answerable_total > 0:
        coverage = answerable_answered / answerable_total
        hit_rate = answerable_hit_expected / answerable_total
        print(
            f"Answerable queries: {answerable_total}\n"
            f"  Coverage (not rejected)     : {coverage:.3f}\n"
            f"  Retrieval hit rate (expected doc in top-k): {hit_rate:.3f}"
        )
    else:
        print("No answerable queries in eval set.")

    if unanswerable_total > 0:
        abstention_rate = unanswerable_rejected / unanswerable_total
        print(
            f"\nUnanswerable queries: {unanswerable_total}\n"
            f"  Abstention rate (rejected)  : {abstention_rate:.3f}"
        )
    else:
        print("No unanswerable queries in eval set.")

    print("\n(You can extend this script to write rows_summary to CSV for deeper analysis.)")


        # ---- Detailed misclassification view ----

    wrong_answerable = [
        r for r in rows_summary
        if r["answerable"] and r["rejected"]
    ]
    wrong_unanswerable = [
        r for r in rows_summary
        if (not r["answerable"]) and (not r["rejected"])
    ]

    print("\n=============== WRONG ANSWERABLE (should NOT be rejected) ===============")
    for r in wrong_answerable:
        print(
            f"{r['id']} | top1={r['top1']:.3f}, conf={r['confidence']:.3f}, margin={r['margin']:.3f}, "
            f"rejected={r['rejected']} | query={r['query']}"
        )

    print("\n=============== WRONG UNANSWERABLE (should be rejected) =================")
    for r in wrong_unanswerable:
        print(
            f"{r['id']} | top1={r['top1']:.3f}, conf={r['confidence']:.3f}, margin={r['margin']:.3f},"
            f"rejected={r['rejected']} | query={r['query']}"
        )


if __name__ == "__main__":
    main()
