# STT Corrector - RAG System

Sistem koreksi kesalahan Speech-to-Text (STT) berbasis **knowledge base + embedding + vector search (ChromaDB)**,
dengan opsi tambahan **LLM (Ollama / llama.cpp)** untuk normalisasi teks berbasis konteks.

## 🎯 Fitur

- **Koreksi STT Otomatis**: Mengoreksi kesalahan pengenalan suara berdasarkan knowledge base
- **RAG Pipeline**: Menggunakan embedding `sentence-transformers` + vector similarity search di ChromaDB
- **Knowledge Base Dinamis**: Tambahkan koreksi baru melalui API
- **Backend-only REST API**: Mudah diintegrasikan ke pipeline STT / aplikasi lain
- **LLM Opsional**: Support **Ollama** atau **llama.cpp** untuk normalisasi teks

## 📁 Struktur Proyek

```
corector/
├── config.py              # Konfigurasi sistem (embedding, Chroma, server)
├── main.py                # Entry point aplikasi (menjalankan FastAPI)
├── requirements.txt       # Dependencies Python
├── .env                   # Environment variables (HOST, PORT, EMBEDDING_MODEL, dst.)
├── README.md
│
├── data/
│   ├── knowledge_base.json    # Knowledge base koreksi (correct_phrase + common_mistakes)
│   └── chroma_db/             # Vector database (auto-generated oleh Chroma)
│
├── src/
│   ├── __init__.py
│   ├── api.py             # FastAPI endpoints (REST API backend-only)
│   ├── corrector.py       # Main corrector logic (n-gram + vector store)
│   ├── embeddings.py      # Embedding model (SentenceTransformers)
│   └── vector_store.py    # ChromaDB vector store wrapper
│
└── scripts/
    ├── init_db.py         # Initialize database dari knowledge_base.json (opsional)
    └── test_corrector.py  # Test script (opsional)
```

## 🚀 Instalasi & Menjalankan

### 1. Clone Repo

```bash
git clone https://github.com/ArifMunawarr/corectorrag.git
cd corectorrag
```

### 2. Buat Virtualenv & Install Dependencies

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Konfigurasi Dasar

Buat file `.env` di root proyek (jika belum ada), misalnya:

```env
# Embedding model
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

# ChromaDB
CHROMA_PERSIST_DIR=./data/chroma_db

# Server
HOST=0.0.0.0
PORT=8888
```

Sesuaikan `PORT` jika perlu.

### 4. Jalankan Server Secara Manual

```bash
source venv/bin/activate
python main.py
```

Secara default server akan berjalan di `http://0.0.0.0:PORT` (misal `http://localhost:8888`).

## 📖 Penggunaan

### REST API

Untuk mengaktifkan LLM (RAG + LLM normalizer), set `use_llm` ke `true`:

```bash
curl -X POST http://localhost:8888/correct \
  -H "Content-Type: application/json" \
  -d '{"text": "beso kit mulai pelatihan nek ji"}'
```

#### Koreksi Teks (output sederhana)

Endpoint khusus yang hanya mengembalikan teks koreksi:

```bash
curl -X POST http://localhost:8888/correct/plain \
  -H "Content-Type: application/json" \
  -d '{"text": "start eating"}'
```

Response:

```json
{ "corrected_text": "start meeting" }
```

#### Tambah Koreksi Baru

```bash
curl -X POST http://localhost:8888/knowledge/add \
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
curl http://localhost:8888/stats
```

## 🔧 Konfigurasi

### Environment Variables (`.env`)

```env
# Embedding model
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

# ChromaDB
CHROMA_PERSIST_DIR=./data/chroma_db

# Server
HOST=0.0.0.0
PORT=8888
```

### LLM Backend (`config.py`)

Edit `config.py` untuk memilih backend LLM:

#### OPSI 1: Ollama (default)

```python
LLM_BACKEND: str = os.getenv("LLM_BACKEND", "ollama")
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL: str = os.getenv("LLM_MODEL", "llama3.2:latest")
```

Jalankan Ollama:
```bash
ollama run llama3.2:latest
```

#### OPSI 2: llama.cpp

Comment OPSI 1, uncomment OPSI 2 di `config.py`:

```python
LLM_BACKEND: str = os.getenv("LLM_BACKEND", "llama_cpp")
LLAMA_CPP_URL: str = os.getenv("LLAMA_CPP_URL", "http://localhost:8080")
```

Jalankan llama.cpp server:
```bash
./llama-server -m /path/to/model.gguf --port 8080
```

#### Tanpa LLM

Jika tidak ingin pakai LLM, cukup set `use_llm: false` di request API.

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

## 🛠️ Tech Stack

- **Embeddings**: SentenceTransformers (`paraphrase-multilingual-MiniLM-L12-v2`)
- **Vector Store**: ChromaDB
- **Backend**: FastAPI + Uvicorn
- **LLM**: Ollama atau llama.cpp (opsional)