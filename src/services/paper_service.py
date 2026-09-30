import os
import uuid
import re
import fitz # PyMuPDF
from pathlib import Path
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

from src.services.paper_repository import PaperRepository
from src.ingestion.document_parser import PDFParser
from src.embeddings.model_loader import MultimodalEmbedder, LangChainCLIPEmbeddings
from src.vector_store.chroma_manager import ChromaManager
from src.retrieval.retriever import MultimodalRetriever
from src.generation.generator import MultimodalGenerator

load_dotenv()

class PaperService:
    def __init__(
        self,
        repository: Optional[PaperRepository] = None,
        parser: Optional[PDFParser] = None,
        embedder: Optional[MultimodalEmbedder] = None,
        vector_store: Optional[ChromaManager] = None,
        retriever: Optional[MultimodalRetriever] = None,
        generator: Optional[MultimodalGenerator] = None
    ):
        self.repository = repository or PaperRepository()
        self.upload_dir = Path(os.getenv("UPLOAD_DIR", "./data/uploads"))
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.max_size_mb = int(os.getenv("MAX_UPLOAD_SIZE_MB", "50"))

        # Shared RAG components
        self.clip_lc = LangChainCLIPEmbeddings()
        self.embedder = embedder or self.clip_lc.embedder
        self.vector_store = vector_store or ChromaManager(embedding_function=self.clip_lc)
        self.retriever = retriever or MultimodalRetriever(self.embedder, self.vector_store)
        self.generator = generator or MultimodalGenerator()
        self.parser = parser or PDFParser()

    def sanitize_filename(self, filename: str) -> str:
        """
        Sanitizes input filename to prevent directory traversal and invalid characters.
        """
        clean_name = os.path.basename(filename)
        clean_name = re.sub(r'[^a-zA-Z0-9_.-]', '_', clean_name)
        if not clean_name.lower().endswith(".pdf"):
            clean_name += ".pdf"
        return clean_name

    def upload_paper(self, file_bytes: bytes, original_filename: str, content_type: str) -> Dict[str, Any]:
        """
        Validates and saves uploaded PDF, creating a new paper record.
        """
        # Validate extension
        if not original_filename or not original_filename.lower().endswith(".pdf"):
            raise ValueError("Only PDF files are supported.")

        # Validate MIME type if provided
        if content_type and "pdf" not in content_type.lower() and content_type != "application/octet-stream":
            raise ValueError(f"Invalid content type: {content_type}. Must be application/pdf.")

        # Validate file size
        if len(file_bytes) == 0:
            raise ValueError("Empty file uploaded. Please select a valid PDF.")

        max_bytes = self.max_size_mb * 1024 * 1024
        if len(file_bytes) > max_bytes:
            raise ValueError(f"File size exceeds maximum limit of {self.max_size_mb}MB.")

        # Generate unique paper_id
        paper_id = f"paper_{uuid.uuid4().hex[:12]}"
        safe_filename = self.sanitize_filename(original_filename)
        saved_file_path = self.upload_dir / f"{paper_id}.pdf"

        # Save to disk
        with open(saved_file_path, "wb") as f:
            f.write(file_bytes)

        # Count pages quickly via PyMuPDF
        try:
            doc = fitz.open(saved_file_path)
            total_pages = len(doc)
            doc.close()
        except Exception:
            total_pages = 0

        paper_record = {
            "paper_id": paper_id,
            "filename": safe_filename,
            "original_filename": original_filename,
            "file_path": str(saved_file_path),
            "file_size_bytes": len(file_bytes),
            "status": "uploaded",
            "progress": 0,
            "message": "Paper uploaded successfully",
            "pages": total_pages,
            "chunks": 0,
            "images": 0,
            "error": None
        }

        self.repository.save_paper(paper_record)

        return {
            "paper_id": paper_id,
            "filename": safe_filename,
            "status": "uploaded",
            "message": "Paper uploaded successfully"
        }

    def process_paper(self, paper_id: str) -> Dict[str, Any]:
        """
        Executes PDF ingestion, extraction, embedding, and vector storage for paper_id.
        """
        paper = self.repository.get_paper(paper_id)
        if not paper:
            raise ValueError(f"Paper with ID '{paper_id}' not found.")

        file_path = paper["file_path"]
        filename = paper["filename"]

        try:
            # Step 1: Update status to processing
            self.repository.update_paper(paper_id, {
                "status": "processing",
                "progress": 15,
                "message": "Extracting text, tables, and images..."
            })

            # Step 2: Extract content chunks
            chunks = self.parser.extract_content(file_path, paper_id=paper_id, filename=filename)
            
            if not chunks:
                self.repository.update_paper(paper_id, {
                    "status": "failed",
                    "progress": 0,
                    "message": "No indexable content found in PDF",
                    "error": "Empty extraction"
                })
                return self.repository.get_paper(paper_id)

            self.repository.update_paper(paper_id, {
                "progress": 45,
                "message": f"Extracted {len(chunks)} chunks. Generating embeddings..."
            })

            # Count components
            image_count = sum(1 for c in chunks if c.get("type") == "image")
            
            # Step 3: Embed and add to ChromaDB
            all_ids = []
            all_embeddings = []
            all_metadatas = []
            all_documents = []

            for i, chunk in enumerate(chunks):
                chunk_id = f"{paper_id}_{i}_{chunk['type']}_p{chunk['page']}"
                
                if chunk["type"] == "image":
                    embedding = self.embedder.encode_image(chunk["content"]).tolist()
                    doc_text = chunk.get("ocr_text") or f"Figure/Image from {filename} page {chunk['page']}"
                else:
                    embedding = self.embedder.encode_text(chunk["content"]).tolist()
                    doc_text = chunk["content"]

                all_ids.append(chunk_id)
                all_embeddings.append(embedding)
                all_metadatas.append(chunk["metadata"])
                all_documents.append(doc_text)

            self.repository.update_paper(paper_id, {
                "progress": 75,
                "message": "Storing embeddings in vector database..."
            })

            # Batch insert to ChromaDB
            batch_size = 50
            for j in range(0, len(all_ids), batch_size):
                end = min(j + batch_size, len(all_ids))
                self.vector_store.add_embeddings(
                    ids=all_ids[j:end],
                    embeddings=all_embeddings[j:end],
                    metadatas=all_metadatas[j:end],
                    documents=all_documents[j:end]
                )

            # Step 4: Complete
            final_record = self.repository.update_paper(paper_id, {
                "status": "ready",
                "progress": 100,
                "message": "Paper processed and ready for questions",
                "chunks": len(chunks),
                "images": image_count,
                "error": None
            })

            print(f"[+] Paper {paper_id} ready! Chunks: {len(chunks)}, Images: {image_count}")
            return final_record

        except Exception as e:
            print(f"[!] Processing failed for paper {paper_id}: {e}")
            self.repository.update_paper(paper_id, {
                "status": "failed",
                "progress": 0,
                "message": f"Processing failed: {str(e)}",
                "error": str(e)
            })
            raise RuntimeError(f"Processing failed: {str(e)}")

    def query_paper(self, paper_id: str, query: str, n_results: int = 5) -> Dict[str, Any]:
        """
        Queries a specific paper_id with document isolation.
        """
        paper = self.repository.get_paper(paper_id)
        if not paper:
            raise KeyError(f"Paper '{paper_id}' not found.")

        if paper["status"] == "processing":
            raise ValueError(f"Paper '{paper_id}' is currently processing ({paper.get('progress', 0)}%). Please try again when ready.")

        if paper["status"] != "ready":
            raise ValueError(f"Paper '{paper_id}' is not ready (status: {paper['status']}).")

        if not query or not query.strip():
            raise ValueError("Query string cannot be empty.")

        # 1. Retrieve paper-isolated items
        relevant_items = self.retriever.retrieve(query=query, paper_id=paper_id, n_results=n_results)

        # 2. Generate LLM answer
        gen_result = self.generator.generate_answer(query=query, context_items=relevant_items)

        # 3. Format sources
        formatted_sources = []
        for item in relevant_items:
            meta = item.get("metadata", {})
            snippet = item.get("content", "")
            if len(snippet) > 300:
                snippet = snippet[:297] + "..."
                
            formatted_sources.append({
                "page": meta.get("page_number", 1),
                "content_type": meta.get("content_type", "text"),
                "snippet": snippet,
                "image_path": meta.get("image_path")
            })

        return {
            "paper_id": paper_id,
            "question": query,
            "answer": gen_result["answer"],
            "sources": formatted_sources
        }

    def list_papers(self) -> List[Dict[str, Any]]:
        return self.repository.list_papers()

    def get_paper(self, paper_id: str) -> Optional[Dict[str, Any]]:
        return self.repository.get_paper(paper_id)

    def delete_paper(self, paper_id: str) -> bool:
        paper = self.repository.get_paper(paper_id)
        if not paper:
            return False

        # 1. Delete PDF file
        file_path = paper.get("file_path")
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"[!] Warning deleting PDF file {file_path}: {e}")

        # 2. Delete extracted images for paper_id
        try:
            image_dir = Path(os.getenv("PROCESSED_DATA_PATH", "./data/processed")) / "images"
            if image_dir.exists():
                safe_id = "".join(c if c.isalnum() else "_" for c in paper_id)
                for img_file in image_dir.glob(f"{safe_id}_*"):
                    try:
                        os.remove(img_file)
                    except Exception:
                        pass
        except Exception as e:
            print(f"[!] Warning cleaning extracted images for {paper_id}: {e}")

        # 3. Delete ChromaDB vector entries
        self.vector_store.delete_paper(paper_id)

        # 4. Delete metadata from repo
        return self.repository.delete_paper(paper_id)
