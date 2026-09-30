import os
import glob
from pathlib import Path
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks, status
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from src.services.paper_service import PaperService

# Load environment variables
load_dotenv()

app = FastAPI(
    title="Multimodal Research Paper Question Answering API",
    description="API for uploading, processing, isolating, and asking questions about research papers with multimodal RAG.",
    version="2.0.0"
)

# Initialize Paper Service
paper_service = PaperService()

# --- Request / Response Pydantic Models ---

class PaperUploadResponse(BaseModel):
    paper_id: str
    filename: str
    status: str
    message: str

class PaperStatusResponse(BaseModel):
    paper_id: str
    status: str
    progress: int
    message: Optional[str] = None
    pages: Optional[int] = 0
    chunks: Optional[int] = 0
    images: Optional[int] = 0
    error: Optional[str] = None

class PaperSummary(BaseModel):
    paper_id: str
    filename: str
    status: str
    pages: int
    chunks: int
    images: int

class QueryRequest(BaseModel):
    query: str = Field(..., description="The question about the research paper.")
    n_results: Optional[int] = Field(5, description="Number of context snippets to retrieve.")

class SourceSnippet(BaseModel):
    page: int
    content_type: str
    snippet: Optional[str] = None
    image_path: Optional[str] = None

class QueryPaperResponse(BaseModel):
    paper_id: str
    question: str
    answer: str
    sources: List[SourceSnippet]

class LegacySource(BaseModel):
    document_id: str
    page_number: int
    content_type: str
    snippet: Optional[str] = None
    image_path: Optional[str] = None

class LegacyQueryResponse(BaseModel):
    answer: str
    sources: List[LegacySource]

# --- Static Files & Web UI ---
static_dir = Path(__file__).parent.parent / "static"
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    index_file = static_dir / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return HTMLResponse("<h1>Multimodal Research Paper Q&A System API</h1><p>Visit <a href='/docs'>/docs</a> for Swagger UI.</p>")

# --- API v1 Endpoints ---

@app.get("/api/v1/health")
def get_health():
    return {
        "status": "healthy",
        "service": "Multimodal Research Paper QA System",
        "vector_db_count": paper_service.vector_store.get_count(),
        "model": paper_service.generator.model_name
    }

@app.post("/api/v1/papers/upload", response_model=PaperUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_paper(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    Upload a research paper (PDF).
    Validates file format, size, and generates a unique paper ID.
    """
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded.")

    try:
        content = await file.read()
        res = paper_service.upload_paper(
            file_bytes=content,
            original_filename=file.filename,
            content_type=file.content_type
        )
        
        # Trigger background processing automatically after upload
        background_tasks.add_task(paper_service.process_paper, res["paper_id"])
        
        return PaperUploadResponse(**res)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/api/v1/papers/{paper_id}/process", response_model=PaperStatusResponse)
async def process_paper(paper_id: str, background_tasks: BackgroundTasks):
    """
    Start background processing (text, table, image extraction, OCR, embedding) for an uploaded paper.
    """
    paper = paper_service.get_paper(paper_id)
    if not paper:
        raise HTTPException(status_code=404, detail=f"Paper '{paper_id}' not found.")

    if paper["status"] == "processing":
        return PaperStatusResponse(**paper)

    background_tasks.add_task(paper_service.process_paper, paper_id)
    
    paper["status"] = "processing"
    paper["message"] = "Processing started"
    return PaperStatusResponse(**paper)

@app.get("/api/v1/papers/{paper_id}/status", response_model=PaperStatusResponse)
def get_paper_status(paper_id: str):
    """
    Check the processing status and progress of a paper.
    """
    paper = paper_service.get_paper(paper_id)
    if not paper:
        raise HTTPException(status_code=404, detail=f"Paper '{paper_id}' not found.")
    return PaperStatusResponse(**paper)

@app.get("/api/v1/papers", response_model=List[PaperSummary])
def list_papers():
    """
    List all uploaded papers and their statuses.
    """
    papers = paper_service.list_papers()
    summaries = []
    for p in papers:
        summaries.append(PaperSummary(
            paper_id=p["paper_id"],
            filename=p["filename"],
            status=p["status"],
            pages=p.get("pages", 0),
            chunks=p.get("chunks", 0),
            images=p.get("images", 0)
        ))
    return summaries

@app.get("/api/v1/papers/{paper_id}")
def get_paper(paper_id: str):
    """
    Get detailed metadata for a specific paper.
    """
    paper = paper_service.get_paper(paper_id)
    if not paper:
        raise HTTPException(status_code=404, detail=f"Paper '{paper_id}' not found.")
    return paper

@app.delete("/api/v1/papers/{paper_id}")
def delete_paper(paper_id: str):
    """
    Delete a paper, including its stored PDF, extracted images, vector store embeddings, and metadata.
    """
    success = paper_service.delete_paper(paper_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Paper '{paper_id}' not found.")
    return {"message": f"Paper {paper_id} deleted successfully"}

@app.post("/api/v1/papers/{paper_id}/query", response_model=QueryPaperResponse)
def query_paper(paper_id: str, request: QueryRequest):
    """
    Ask a question about a specific paper with document isolation and page citations.
    """
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query string cannot be empty.")

    try:
        result = paper_service.query_paper(
            paper_id=paper_id,
            query=request.query,
            n_results=request.n_results or 5
        )
        return QueryPaperResponse(**result)
    except KeyError as ke:
        raise HTTPException(status_code=404, detail=str(ke).strip("'"))
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

# --- Legacy Compatibility Endpoints ---

@app.get("/status")
def get_legacy_status():
    return {
        "status": "Ready",
        "document_count": paper_service.vector_store.get_count(),
        "collection_name": "multimodal_rag"
    }

@app.post("/ingest")
async def ingest_documents(background_tasks: BackgroundTasks):
    raw_path = os.getenv("RAW_DATA_PATH", "./sample_documents")
    files = glob.glob(os.path.join(raw_path, "*.pdf"))
    
    if not files:
        return {"status": "error", "message": f"No valid PDF documents found in {raw_path}"}

    processed_ids = []
    for f in files:
        with open(f, "rb") as fp:
            res = paper_service.upload_paper(fp.read(), os.path.basename(f), "application/pdf")
            background_tasks.add_task(paper_service.process_paper, res["paper_id"])
            processed_ids.append(res["paper_id"])

    return {
        "status": "success", 
        "message": f"Ingestion started for {len(files)} files.",
        "paper_ids": processed_ids
    }

@app.post("/query", response_model=LegacyQueryResponse)
async def legacy_query(request: QueryRequest):
    papers = paper_service.list_papers()
    ready_papers = [p for p in papers if p["status"] == "ready"]
    
    if not ready_papers:
        return LegacyQueryResponse(
            answer="No ready documents found in the database. Please upload and process a paper first.",
            sources=[]
        )

    # Pick the most recent ready paper
    paper_id = ready_papers[-1]["paper_id"]
    res = paper_service.query_paper(paper_id=paper_id, query=request.query, n_results=request.n_results)
    
    legacy_sources = [
        LegacySource(
            document_id=res["paper_id"],
            page_number=s["page"],
            content_type=s["content_type"],
            snippet=s.get("snippet"),
            image_path=s.get("image_path")
        )
        for s in res["sources"]
    ]
    return LegacyQueryResponse(answer=res["answer"], sources=legacy_sources)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
