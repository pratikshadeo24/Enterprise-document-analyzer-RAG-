from typing import List


def word_chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Split text into overlapping word-based chunks.
    """
    words = text.split()
    if not words:
        return []

    chunks: List[str] = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        if not chunk_words:
            break
        chunks.append(" ".join(chunk_words))

        step = chunk_size - overlap
        if step <= 0:
            step = chunk_size
        start += step

    return chunks
