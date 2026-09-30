# 📚 Multimodal Research Paper Question Answering System

A high-performance, document-isolated Retrieval-Augmented Generation (RAG) system engineered to analyze complex scientific papers. The system extracts text, structured tables, and figures/diagrams (with EasyOCR enrichment), generates multimodal CLIP embeddings, stores them in ChromaDB with document isolation, and delivers grounded answers with page citations using Groq LLMs.

---

## 🚀 Key Features

- **📄 Document Isolation**: Vector search is strictly filtered by `paper_id` so that questions about one paper never retrieve chunks from unrelated papers.
- **🖼️ Multimodal Extraction & OCR**: Extracts text, tables, and figures using **PyMuPDF**, **Unstructured**, and **EasyOCR**.
- **⚡ Multimodal Vector Embeddings**: Uses **CLIP (ViT-B-32)** to embed text and images into a shared semantic space.
- **🤖 Configurable Groq LLM Generation**: Configurable via `LLM_MODEL` in `.env` (e.g., `openai/gpt-oss-120b` or `llama-3.3-70b-versatile`).
- **📌 Page Citations & Groundedness**: Every answer cites specific source page numbers (e.g., *Page 3*, *Page 4*) and LaTeX mathematical formulas.
- **💻 Modern Dark-Mode Web UI**: Built-in glassmorphism frontend featuring drag-and-drop PDF upload, real-time progress bar, paper list sidebar, and interactive Q&A chat.
- **🧩 Decoupled Architecture**: Service layer (`PaperService`, `PaperRepository`, `MultimodalRetriever`, `MultimodalGenerator`) decoupled from FastAPI routes for future MCP (Model Context Protocol) tool integration.

---

## 🏗️ System Architecture & Workflow

```mermaid
flowchart TD
    User((User / Web UI)) -->|Upload PDF| API_Upload[POST /api/v1/papers/upload]
    API_Upload -->|Save PDF & Init Repo| Service[PaperService]
    
    subgraph Background Processing Pipeline
        Service -->|1. Layout & Table Extraction| Unstructured[Unstructured PDF Parser]
        Service -->|2. Image & Figure Extraction| PyMuPDF[PyMuPDF fitz Engine]
        PyMuPDF -->|3. Run Text Recognition| EasyOCR[EasyOCR Engine]
        
        Unstructured --> Chunks[Chunking with paper_id & page_number Metadata]
        EasyOCR --> Chunks
    end
    
    Chunks -->|4. Generate 512d Vector Embeddings| CLIP[CLIP ViT-B-32 Embedder]
    CLIP -->|5. Store Chunks & Vectors| Chroma[(ChromaDB)]
    
    User -->|Ask Question| API_Query[POST /api/v1/papers/{paper_id}/query]
    API_Query -->|Document Isolated Retrieval where paper_id| Retriever[MultimodalRetriever]
    Retriever -->|Metadata Filter Search| Chroma
    Chroma -->|Relevant Text/Tables/Images| Retriever
    Retriever -->|Format Context| Generator[MultimodalGenerator]
    Generator -->|API Request| Groq[Groq LLM API]
    Groq -->|Grounded Answer + Page Citations| User
```

---

## 🛠️ Installation & Setup

### Prerequisites
- **Python 3.10+** (Python 3.10 - 3.14 supported)
- **Groq API Key**: Get a free API key from [Groq Console](https://console.groq.com/).

### 1. Environment Configuration
Create a `.env` file in the project root (or copy `.env.example`):
```ini
GROQ_API_KEY=your_groq_api_key_here

# LLM Model Configuration
LLM_MODEL=openai/gpt-oss-120b

# Paths for data storage
VECTOR_DB_PATH=./data/chroma
UPLOAD_DIR=./data/uploads
PROCESSED_DATA_PATH=./data/processed
IMAGE_EXTRACTION_PATH=./data/processed/images
RAW_DATA_PATH=./sample_documents

# Upload limits
MAX_UPLOAD_SIZE_MB=50

# Logging
LOG_LEVEL=INFO
```

### 2. Local Setup (Virtual Environment)
```bash
# Clone the repository
git clone <repo-url>
cd Multimodal-RAG-Pipeline-Text-Img

# Create virtual environment
python -m venv .venv

# Activate virtual environment (Windows PowerShell)
.\.venv\Scripts\activate
# Activate virtual environment (Linux / macOS)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Docker Deployment
```bash
# Build and run with docker-compose
docker-compose up --build
```

---

## 🖥️ Running in VS Code

### Option A: Press F5 (One-Click Run)
1. Open the repository in VS Code.
2. Press **`F5`** (or go to **Run and Debug** `Ctrl+Shift+D`).
3. Select **`FastAPI: Run Server`** and press Play.

### Option B: VS Code Terminal
```powershell
.venv\Scripts\python.exe -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

- **Web Assistant UI**: [http://localhost:8000](http://localhost:8000)
- **Swagger API Documentation**: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 📡 API Reference Specification

### Paper Management & Q&A (`/api/v1/papers`)

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/api/v1/papers/upload` | Upload a research paper PDF (returns `paper_id`). |
| `POST` | `/api/v1/papers/{paper_id}/process` | Start background processing (text/tables/images/embeddings). |
| `GET` | `/api/v1/papers/{paper_id}/status` | Check processing status, progress percentage, and page counts. |
| `GET` | `/api/v1/papers` | List all uploaded papers. |
| `GET` | `/api/v1/papers/{paper_id}` | Get detailed paper metadata. |
| `DELETE` | `/api/v1/papers/{paper_id}` | Delete paper PDF, extracted images, vector store records, and metadata. |
| `POST` | `/api/v1/papers/{paper_id}/query` | Ask paper-isolated questions with page citations. |
| `GET` | `/api/v1/health` | Health check endpoint and vector store status. |

### Legacy Compatibility Endpoints
- `GET /status` - Quick server and collection status.
- `POST /ingest` - Triggers ingestion for documents in `./sample_documents`.
- `POST /query` - Queries the most recently indexed paper.

---

## 🧪 Running Automated Tests

Run the full pytest suite covering PDF upload validation, processing, status tracking, document isolation, Q&A generation, citations, 404 handling, and cleanup:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_api.py -v
```

---

## 📂 Project Structure

```text
Multimodal-RAG-Pipeline-Text-Img/
├── src/
│   ├── api/
│   │   └── main.py              # FastAPI endpoints and static UI mounting
│   ├── services/
│   │   ├── paper_service.py     # High-level business logic orchestrator
│   │   └── paper_repository.py  # JSON file persistence for paper states
│   ├── ingestion/
│   │   ├── document_parser.py   # PDF text, table, and image extractor
│   │   └── image_processor.py   # EasyOCR text recognition
│   ├── embeddings/
│   │   └── model_loader.py      # CLIP ViT-B-32 multimodal embedder
│   ├── vector_store/
│   │   └── chroma_manager.py    # LangChain ChromaDB manager with filtering & deletion
│   ├── retrieval/
│   │   └── retriever.py         # Paper-isolated vector retriever
│   ├── generation/
│   │   └── generator.py         # Groq LLM grounded answer generator
│   └── static/
│       ├── index.html           # Dark-mode Web UI HTML
│       ├── styles.css           # Glassmorphism styling & badges
│       └── app.js               # Frontend interactive application logic
├── data/
│   ├── uploads/                 # Uploaded research paper PDFs
│   ├── processed/               # Extracted images and figures
│   ├── chroma/                  # ChromaDB vector store directory
│   └── papers.json              # Paper metadata and state database
├── sample_documents/            # Sample research paper PDFs
├── tests/
│   ├── test_api.py              # Comprehensive API & RAG test suite
│   ├── test_integrated.py       # Integration tests
│   └── manual_stress_test.py    # Stress testing script
├── .env.example                 # Environment variable template
├── .vscode/                     # VS Code launch & debug settings
├── Dockerfile                   # System Docker containerization
├── docker-compose.yml           # Multi-container orchestration
└── requirements.txt             # Python dependencies
```
