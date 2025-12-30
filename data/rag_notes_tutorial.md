# RAG System Notes

Retrieval-Augmented Generation (RAG) combines a vector search step with a large language model
to answer questions using external documents instead of relying only on parametric memory.

In this project, we are using LanceDB as the vector store and SBERT-based embeddings to power
semantic search over enterprise documents.

The system also supports metadata like doc_type and doc_version to enable routing and
version-aware retrieval.
