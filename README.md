# STT Corrector - RAG System

Sistem koreksi kesalahan Speech-to-Text (STT) berbasis **knowledge base + embedding + vector search (ChromaDB)**.
Tidak lagi menggunakan Ollama/LLM, seluruh koreksi ditentukan oleh daftar frasa di `knowledge_base.json`.

## 🎯 Fitur

- **Koreksi STT Otomatis**: Mengoreksi kesalahan pengenalan suara berdasarkan knowledge base
- **RAG Pipeline Sederhana**: Menggunakan embedding `sentence-transformers` + vector similarity search di ChromaDB
- **Knowledge Base Dinamis**: Tambahkan koreksi baru melalui API
- **Backend-only REST API**: Mudah diintegrasikan ke pipeline STT / aplikasi lain

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

Atau jika Anda sudah punya folder `/home/olama/corector`, cukup pastikan remote sudah mengarah ke repo tersebut.

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

### 5. (Opsional) Jalankan sebagai systemd Service

Contoh file service `/etc/systemd/system/corector.service`:

```ini
[Unit]
Description=STT Corrector RAG Service
After=network.target

[Service]
User=olama
Group=olama
WorkingDirectory=/home/olama/corector
ExecStart=/home/olama/corector/venv/bin/python /home/olama/corector/main.py
Environment=PYTHONUNBUFFERED=1
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Aktifkan dan jalankan:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now corector.service
sudo systemctl status corector.service
```

Setelah itu service akan otomatis berjalan di background.

## 📖 Penggunaan

### REST API

#### Koreksi Teks

```bash
# Output JSON lengkap
curl -X POST http://localhost:8888/correct \
  -H "Content-Type: application/json" \
  -d '{"text": "start eating"}'
```

Response:

```json
{
  "input_text": "start eating",
  "corrected_text": "start meeting",
  "correction_made": true,
  "method": "direct_match",
  "confidence": 1.0,
  "candidates": [
    {
      "correct_phrase": "start meeting",
      "matched_text": "start eating",
      "common_mistakes": ["start eating", "start meting", ...],
      "context": "Memulai rapat atau pertemuan",
      "category": "meeting",
      "similarity": 1.0
    }
  ]
}
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

Edit file `.env` untuk mengubah konfigurasi:

```env
# Embedding model
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

# ChromaDB
CHROMA_PERSIST_DIR=./data/chroma_db

# Server
HOST=0.0.0.0
PORT=8888
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

- **Embeddings**: SentenceTransformers (`paraphrase-multilingual-MiniLM-L12-v2`)
- **Vector Store**: ChromaDB (persistent, `data/chroma_db/`)
- **Backend**: FastAPI + Uvicorn
- **Config**: `.env` + `config.py`

## 📄 License

MIT License
