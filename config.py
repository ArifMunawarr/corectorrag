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
    
    # ============================================================
    # LLM Backend: Pilih SALAH SATU dengan uncomment yang sesuai
    # ============================================================
    
    # === OPSI 1: Ollama (default) ===
    LLM_BACKEND: str = os.getenv("LLM_BACKEND", "ollama")
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "llama3.2:latest")
    
    # === OPSI 2: llama.cpp ===
    # Uncomment 2 baris di bawah, comment 3 baris di atas (OPSI 1)
    # LLM_BACKEND: str = os.getenv("LLM_BACKEND", "llama_cpp")
    # LLAMA_CPP_URL: str = os.getenv("LLAMA_CPP_URL", "http://localhost:8080")

    # RAG settings
    TOP_K_RESULTS: int = 5
    SIMILARITY_THRESHOLD: float = 0.3
    # Ambang similarity untuk direct match (0-1). Lebih rendah = lebih agresif koreksi.
    DIRECT_MATCH_THRESHOLD: float = 0.7


config = Config()
