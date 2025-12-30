import re
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer
import spacy

from functools import lru_cache

@lru_cache(maxsize=1)
def get_spacy_nlp():
    """
    Load spaCy English model once and cache it.

    Using lru_cache avoids reloading the model on every call and
    keeps code simple.
    """
    return spacy.load("en_core_web_sm")

def split_into_sentences(text: str) -> List[str]:
    """
    Sentence splitter using spaCy's sentencizer.

    This gives more accurate sentence boundaries than a regex-based
    approach, especially for technical text with abbreviations, etc.
    """
    text = text.strip()
    if not text:
        return []

    nlp = get_spacy_nlp()
    doc = nlp(text.replace("\n", " "))

    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    return sentences


def embedding_semantic_chunk_text(
    text: str,
    model: SentenceTransformer,
    target_chunk_size_words: int = 200,
    min_chunk_size_words: int = 80,
    similarity_percentile: float = 30.0,
) -> List[str]:
    """
    Embedding-based semantic chunker.

    Steps:
      1. Split text into sentences.
      2. Embed each sentence.
      3. Compute cosine similarity between consecutive sentence embeddings.
      4. Detect semantic breakpoints where similarity is low (topic shifts).
      5. Group sentences into chunks, respecting semantic breakpoints and
         approximate target chunk size in words.

    Args:
        text: Raw document text.
        model: SentenceTransformer embedding model (already used in your project).
        target_chunk_size_words: Desired chunk size in words (soft constraint).
        min_chunk_size_words: Minimum chunk size to avoid very tiny chunks.
        similarity_percentile: Percentile (0-100) to determine the similarity
            threshold. Sentence pairs with similarity below this threshold
            are considered breakpoints (semantic cuts).

    Returns:
        List of chunk strings.
    """
    sentences = split_into_sentences(text)
    if not sentences:
        return []

    if len(sentences) == 1:
        # Single sentence, return as one chunk
        return [sentences[0]]

    # 2) Embed each sentence (normalized so dot product = cosine similarity)
    embeddings = model.encode(
        sentences,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    # 3) Compute similarities between consecutive sentences
    sims = []
    for i in range(len(sentences) - 1):
        sim = float(np.dot(embeddings[i], embeddings[i + 1]))
        sims.append(sim)

    if not sims:
        # Shouldn't happen if len(sentences) > 1, but be safe
        return [" ".join(sentences)]

    sims_arr = np.array(sims, dtype=float)

    # 4) Determine breakpoint threshold based on percentile
    # Lower similarity = more likely a semantic shift.
    # Example: percentile=30 -> splits on the "lowest 30% similarity" pairs.
    threshold = float(np.percentile(sims_arr, similarity_percentile))

    # Indices i where we cut AFTER sentence i (i.e., between i and i+1)
    break_indices = {i for i, s in enumerate(sims) if s < threshold}

    chunks: List[str] = []
    current_sentences: List[str] = []
    current_word_count = 0

    def flush_chunk():
        nonlocal current_sentences, current_word_count
        if not current_sentences:
            return
        chunk_text = " ".join(current_sentences).strip()
        if chunk_text:
            chunks.append(chunk_text)
        current_sentences = []
        current_word_count = 0

    for i, sent in enumerate(sentences):
        sent_word_count = len(sent.split())
        current_sentences.append(sent)
        current_word_count += sent_word_count

        is_break_here = i in break_indices

        # 5) Decide when to cut:
        #    - either a semantic breakpoint AND we already have a decent chunk
        #    - or the chunk is getting too large
        if is_break_here and current_word_count >= min_chunk_size_words:
            flush_chunk()
            continue

        # Hard guard against overly large chunks
        if current_word_count >= int(target_chunk_size_words * 1.5):
            flush_chunk()
            continue

    # Flush remaining sentences
    flush_chunk()

    # If we somehow ended up with a tiny last chunk, merge it into previous
    if len(chunks) >= 2:
        last = chunks[-1]
        if len(last.split()) < min_chunk_size_words:
            chunks[-2] = chunks[-2] + " " + last
            chunks.pop()

    return chunks
