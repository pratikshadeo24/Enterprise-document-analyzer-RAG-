import requests

from .config import USE_OLLAMA, OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_TIMEOUT_SECONDS


def dummy_llm_generate_answer(query: str, context: str) -> str:
    """
    Fallback LLM: deterministic, uses only retrieved context.
    """
    if not context:
        return (
            "I could not find any relevant information in the knowledge base "
            "to answer this question confidently."
        )

    lines = context.splitlines()
    bullet_points = []

    for line in lines:
        if ") " in line:
            _, text_part = line.split(") ", maxsplit=1)
        else:
            text_part = line
        bullet_points.append(text_part)

    bullet_points = bullet_points[:4]
    bullets_str = "\n- ".join(bullet_points)

    answer = (
        f"Question: {query}\n\n"
        "Based on the latest document versions in the knowledge base, key points are:\n"
        f"- {bullets_str}\n\n"
        "This answer is synthesized from the top-matching chunks."
    )
    return answer


def _call_ollama_chat(system_prompt: str, user_prompt: str) -> str:
    """
    Low-level Ollama HTTP /api/chat call.
    """
    url = f"{OLLAMA_BASE_URL.rstrip('/')}/api/chat"
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
    }

    print(f"[LLM] Calling Ollama at {url} with model={OLLAMA_MODEL!r}")
    resp = requests.post(url, json=payload, timeout=OLLAMA_TIMEOUT_SECONDS)
    resp.raise_for_status()
    data = resp.json()

    # Ollama non-streaming: data["message"]["content"]
    msg = data.get("message", {})
    content = msg.get("content", "")

    if isinstance(content, list):
        # newer Ollama can return list-of-parts
        text = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    else:
        text = str(content)

    return text.strip()


def ollama_generate_answer(query: str, context: str) -> str:
    """
    High-level wrapper for Ollama-based answer generation.
    Falls back to dummy if something goes wrong.
    """
    if not context:
        # Even with Ollama, don't answer without context; keep it grounded.
        return (
            "I do not have enough high-confidence information in the indexed documents "
            "to answer this question."
        )

    system_prompt = (
        "You are a RAG assistant. You must answer ONLY using the provided context. "
        "If the context is insufficient, say you don't know and do not hallucinate. "
        "Cite concepts from the context naturally in your answer."
    )
    user_prompt = (
        f"Question:\n{query}\n\n"
        f"Context:\n{context}\n\n"
        "Using only this context, provide a concise, well-structured answer. "
        "If the context does not actually contain the answer, explicitly say you "
        "cannot answer from the given documents."
    )

    try:
        answer = _call_ollama_chat(system_prompt, user_prompt)
        if not answer.strip():
            print("[LLM] Ollama returned empty content; falling back to dummy LLM.")
            return dummy_llm_generate_answer(query, context)
        return answer
    except Exception as e:
        print(f"[LLM] Ollama call failed: {e}. Falling back to dummy LLM.")
        return dummy_llm_generate_answer(query, context)


def generate_answer(query: str, context: str) -> str:
    """
    Public entry point used by the RAG pipeline.
    Chooses between Ollama and dummy based on USE_OLLAMA.
    """
    if USE_OLLAMA:
        return ollama_generate_answer(query, context)
    return dummy_llm_generate_answer(query, context)
