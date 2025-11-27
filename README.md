# STT Corrector - RAG System

Sistem koreksi kesalahan Speech-to-Text (STT) menggunakan Retrieval-Augmented Generation (RAG).

## 🎯 Fitur

- **Koreksi STT Otomatis**: Mengoreksi kesalahan pengenalan suara berdasarkan knowledge base
- **RAG Pipeline**: Menggunakan vector similarity search + LLM untuk koreksi yang akurat
- **Knowledge Base Dinamis**: Tambahkan koreksi baru melalui API atau UI
- **Web Interface**: UI modern untuk testing dan demo
- **REST API**: Integrasi mudah dengan sistem lain

## 📁 Struktur Proyek

```
corector/
├── config.py              # Konfigurasi sistem
├── main.py                # Entry point aplikasi
├── requirements.txt       # Dependencies
├── .env                   # Environment variables
├── README.md
│
├── data/
│   ├── knowledge_base.json    # Knowledge base koreksi
│   └── chroma_db/             # Vector database (auto-generated)
│
├── src/
│   ├── __init__.py
│   ├── api.py             # FastAPI endpoints
│   ├── corrector.py       # Main corrector logic
│   ├── embeddings.py      # Embedding model
│   ├── llm.py             # Ollama LLM wrapper
│   └── vector_store.py    # ChromaDB vector store
│
├── static/
│   └── index.html         # Web interface
│
└── scripts/
    ├── init_db.py         # Initialize database
    └── test_corrector.py  # Test script
```

## 🚀 Instalasi

### 1. Clone dan Setup Environment

```bash
cd ~/corector
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Pastikan Ollama Running

```bash
# Cek Ollama service
ollama list

# Pastikan model tersedia
# hf.co/ojisetyawan/gemma2-9b-cpt-sahabatai-v1-instruct-Q4_K_M-GGUF:latest
```

### 3. Jalankan Aplikasi

```bash
# Jalankan server
python main.py

# Atau dengan uvicorn langsung
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Akses Aplikasi

- **Web Interface**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

## 📖 Penggunaan

### Web Interface

1. Buka http://localhost:8000
2. Masukkan teks STT yang salah (misal: "start eating")
3. Klik "Koreksi Teks"
4. Sistem akan mengoreksi menjadi "start meeting"

### REST API

#### Koreksi Teks

```bash
curl -X POST http://localhost:8000/correct \
  -H "Content-Type: application/json" \
  -d '{"text": "start eating", "use_llm": true}'
```

Response:
```json
{
  "input_text": "start eating",
  "corrected_text": "start meeting",
  "correction_made": true,
  "method": "direct_match",
  "confidence": 0.92
}
```

#### Tambah Koreksi Baru

```bash
curl -X POST http://localhost:8000/knowledge/add \
  -H "Content-Type: application/json" \
  -d '{
    "correct_phrase": "book appointment",
    "common_mistakes": ["book a point meant", "book a pointment"],
    "context": "Membuat janji",
    "category": "scheduling"
  }'
```

#### Cek Status

```bash
curl http://localhost:8000/stats
```

## 🔧 Konfigurasi

Edit file `.env` untuk mengubah konfigurasi:

```env
# Model Ollama
OLLAMA_MODEL=hf.co/ojisetyawan/gemma2-9b-cpt-sahabatai-v1-instruct-Q4_K_M-GGUF:latest

# Embedding model
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

# Server
HOST=0.0.0.0
PORT=8000
```

## 📝 Knowledge Base

Edit `data/knowledge_base.json` untuk menambah/mengubah koreksi:

```json
{
  "corrections": [
    {
      "correct_phrase": "start meeting",
      "common_mistakes": ["start eating", "start meting"],
      "context": "Memulai rapat",
      "category": "meeting"
    }
  ]
}
```

## 🧪 Testing

```bash
# Test corrector
python scripts/test_corrector.py

# Initialize database
python scripts/init_db.py
```

## 🛠️ Tech Stack

- **LLM**: Ollama dengan gemma2-9b-cpt-sahabatai
- **Embeddings**: SentenceTransformers (multilingual)
- **Vector Store**: ChromaDB
- **Backend**: FastAPI + Uvicorn
- **Frontend**: HTML + TailwindCSS

## 📄 License

MIT License
