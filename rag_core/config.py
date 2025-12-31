from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parents[1]

# Paths
DATA_DIR = BASE_DIR / "data"
DB_DIR = BASE_DIR / "lancedb_data"

# Embeddings
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# LanceDB
TABLE_NAME = "doc_chunks_versions"

# Chunking
CHUNK_STRATEGY = "semantic"  # currently only "word" is supported
CHUNK_SIZE_WORDS = 80
CHUNK_OVERLAP_WORDS = 20

# LLM (we keep Ollama optional, disabled by default)
USE_OLLAMA = True
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
OLLAMA_TIMEOUT_SECONDS = int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "300"))

# Confidence / abstention settings
CONF_TOPK_AVG = 3                # how many scores to average
CONF_W_TOP1 = 0.45
CONF_W_AVG_TOPK = 0.25
CONF_W_MARGIN = 0.30
MIN_TOP1_SCORE = 0.30      # was ~0.36–0.40 earlier
MIN_CONFIDENCE = 0.27      # was ~0.32–0.38 earlier


USE_RERANKER = os.getenv("USE_RERANKER", "false").lower() == "true"

RERANKER_MODEL_NAME = os.getenv(
    "RERANKER_MODEL_NAME",
    "cross-encoder/ms-marco-MiniLM-L-6-v2",  # good, small reranker
)
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "5"))  # how many to rerank within top-k
