from typing import List
from sentence_transformers import SentenceTransformer


def get_embedding_model(model_name: str) -> SentenceTransformer:
    print(f"Loading embedding model: {model_name}")
    return SentenceTransformer(model_name)


def compute_embeddings(model: SentenceTransformer, texts: List[str]):
    print(f"Computing embeddings for {len(texts)} texts/chunks...")
    return model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
