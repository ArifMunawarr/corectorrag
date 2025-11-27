"""Main entry point for STT Corrector API."""

import uvicorn
from config import config

if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║           STT Corrector - RAG System                     ║
    ║   Koreksi Kesalahan Speech-to-Text dengan Knowledge Base ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    uvicorn.run(
        "src.api:app",
        host=config.HOST,
        port=config.PORT,
        reload=True
    )
