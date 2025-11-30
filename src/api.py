"""FastAPI REST API for STT Corrector (Backend Only)."""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import os

from src.corrector import get_corrector
from config import config


# Pydantic models
class CorrectionRequest(BaseModel):
    text: str = Field(..., description="Teks STT yang akan dikoreksi")


class BatchCorrectionRequest(BaseModel):
    texts: List[str] = Field(..., description="Daftar teks STT")
    use_llm: bool = Field(True, description="Gunakan LLM untuk koreksi")


class AddCorrectionRequest(BaseModel):
    correct_phrase: str = Field(..., description="Frasa yang benar")
    common_mistakes: List[str] = Field(..., description="Daftar kesalahan umum")
    context: str = Field("", description="Konteks penggunaan")
    category: str = Field("", description="Kategori")


class CorrectionResponse(BaseModel):
    input_text: str
    corrected_text: str
    correction_made: bool
    method: str
    confidence: Optional[float] = None
    candidates: Optional[List[dict]] = None


class PlainCorrectionResponse(BaseModel):
    corrected_text: str


# Create FastAPI app
app = FastAPI(
    title="STT Corrector API",
    description="Backend API untuk mengoreksi kesalahan Speech-to-Text menggunakan RAG",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Initialize corrector and load knowledge base on startup."""
    corrector = get_corrector()
    kb_path = os.path.join(os.path.dirname(__file__), "..", "data", "knowledge_base.json")
    if os.path.exists(kb_path):
        corrector.load_knowledge_base(kb_path)
    print("✓ STT Corrector API ready")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "STT Corrector API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "stt-corrector"}


@app.get("/stats")
async def get_stats():
    """Get system statistics."""
    corrector = get_corrector()
    return corrector.get_stats()


@app.post("/correct", response_model=CorrectionResponse)
async def correct_text(request: CorrectionRequest):
    """Correct a single STT text."""
    try:
        corrector = get_corrector()
        result = corrector.correct(
            input_text=request.text,
            # use_llm di-hardcode True agar klien cukup mengirim {"text": "..."}
            use_llm=True,
        )
        return CorrectionResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/correct/plain", response_model=PlainCorrectionResponse)
async def correct_text_plain(request: CorrectionRequest):
    try:
        corrector = get_corrector()
        # Gunakan mode n-gram supaya hanya frasa salah dengar yang diganti
        result = corrector.correct_in_text(
            input_text=request.text,
            # Selalu gunakan LLM untuk normalisasi akhir
            use_llm=True,
        )
        return PlainCorrectionResponse(corrected_text=result["corrected_text"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/correct/batch")
async def correct_batch(request: BatchCorrectionRequest):
    """Correct multiple STT texts."""
    try:
        corrector = get_corrector()
        results = corrector.correct_batch(
            texts=request.texts,
            use_llm=request.use_llm
        )
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/knowledge/add")
async def add_correction(request: AddCorrectionRequest):
    """Add a new correction to the knowledge base."""
    try:
        corrector = get_corrector()
        doc_id = corrector.add_correction(
            correct_phrase=request.correct_phrase,
            common_mistakes=request.common_mistakes,
            context=request.context,
            category=request.category
        )
        return {"success": True, "doc_id": doc_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/knowledge/reload")
async def reload_knowledge_base():
    """Reload the knowledge base from file."""
    try:
        corrector = get_corrector()
        kb_path = os.path.join(os.path.dirname(__file__), "..", "data", "knowledge_base.json")
        count = corrector.load_knowledge_base(kb_path)
        return {"success": True, "loaded_entries": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
