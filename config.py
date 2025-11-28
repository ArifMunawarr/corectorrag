"""Configuration settings for the STT Corrector RAG system."""

import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # Embedding model
    EMBEDDING_MODEL: str = os.getenv(
        "EMBEDDING_MODEL",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    
    # ChromaDB
    CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")
    COLLECTION_NAME: str = "stt_corrections"
    
    # Server
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "llama3.2:latest")

    # RAG settings
    TOP_K_RESULTS: int = 5
    SIMILARITY_THRESHOLD: float = 0.3
    # Ambang similarity untuk direct match (0-1). Lebih rendah = lebih agresif koreksi.
    DIRECT_MATCH_THRESHOLD: float = 0.7


config = Config()
