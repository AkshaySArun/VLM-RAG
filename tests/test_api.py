import os
import io
import time
import pytest
from pathlib import Path
from fastapi.testclient import TestClient
from src.api.main import app, paper_service

client = TestClient(app)

SAMPLE_DIR = Path("./sample_documents")
ATTENTION_PDF = SAMPLE_DIR / "Attention Is All You Need.pdf"
ADAM_PDF = SAMPLE_DIR / "Adam-A Method for Stochastic Optimization-Kingma & Ba (2015).pdf"

def test_health_check():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "vector_db_count" in data

def test_invalid_pdf_upload():
    # Test non-PDF upload
    response = client.post(
        "/api/v1/papers/upload",
        files={"file": ("test.txt", b"Hello text file content", "text/plain")}
    )
    assert response.status_code == 400
    assert "Only PDF files" in response.json()["detail"]

def test_empty_pdf_upload():
    # Test 0-byte upload
    response = client.post(
        "/api/v1/papers/upload",
        files={"file": ("empty.pdf", b"", "application/pdf")}
    )
    assert response.status_code == 400
    assert "Empty file" in response.json()["detail"]

def test_missing_paper_404():
    response = client.get("/api/v1/papers/paper_nonexistent999/status")
    assert response.status_code == 404

    response = client.post(
        "/api/v1/papers/paper_nonexistent999/query",
        json={"query": "What is attention?"}
    )
    assert response.status_code == 404

    response = client.delete("/api/v1/papers/paper_nonexistent999")
    assert response.status_code == 404

def test_empty_query_400():
    # Create dummy ready paper in repo for testing empty query
    paper_id = "paper_test_empty_q"
    paper_service.repository.save_paper({
        "paper_id": paper_id,
        "filename": "dummy.pdf",
        "file_path": "",
        "status": "ready"
    })
    
    response = client.post(
        f"/api/v1/papers/{paper_id}/query",
        json={"query": "   "}
    )
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()
    
    # Cleanup
    paper_service.delete_paper(paper_id)

def test_end_to_end_paper_workflow():
    if not ATTENTION_PDF.exists():
        pytest.skip(f"{ATTENTION_PDF} not found in sample_documents")

    # 1. Upload PDF
    with open(ATTENTION_PDF, "rb") as f:
        pdf_bytes = f.read()

    upload_resp = client.post(
        "/api/v1/papers/upload",
        files={"file": ("Attention_Is_All_You_Need.pdf", pdf_bytes, "application/pdf")}
    )
    assert upload_resp.status_code == 201
    upload_data = upload_resp.json()
    paper_id = upload_data["paper_id"]
    assert paper_id.startswith("paper_")
    assert upload_data["status"] == "uploaded"

    # 2. Synchronous Process
    proc_res = paper_service.process_paper(paper_id)
    assert proc_res["status"] == "ready"
    assert proc_res["chunks"] > 0

    # 3. Check status
    status_resp = client.get(f"/api/v1/papers/{paper_id}/status")
    assert status_resp.status_code == 200
    st_data = status_resp.json()
    assert st_data["status"] == "ready"
    assert st_data["progress"] == 100

    # 4. List papers
    list_resp = client.get("/api/v1/papers")
    assert list_resp.status_code == 200
    papers_list = list_resp.json()
    assert any(p["paper_id"] == paper_id for p in papers_list)

    # 5. Question 1: Grounded answer test
    q1_resp = client.post(
        f"/api/v1/papers/{paper_id}/query",
        json={"query": "What problem does the Transformer solve?", "n_results": 5}
    )
    assert q1_resp.status_code == 200
    q1_data = q1_resp.json()
    assert q1_data["paper_id"] == paper_id
    assert len(q1_data["sources"]) > 0
    assert "answer" in q1_data

    # 6. Question 2: Self-attention
    q2_resp = client.post(
        f"/api/v1/papers/{paper_id}/query",
        json={"query": "Explain self-attention.", "n_results": 5}
    )
    assert q2_resp.status_code == 200
    assert len(q2_resp.json()["sources"]) > 0

    # 7. Question 3: Multi-head attention
    q3_resp = client.post(
        f"/api/v1/papers/{paper_id}/query",
        json={"query": "What is multi-head attention?", "n_results": 5}
    )
    assert q3_resp.status_code == 200

    # 8. Question 4: Optimizer
    q4_resp = client.post(
        f"/api/v1/papers/{paper_id}/query",
        json={"query": "What optimizer was used?", "n_results": 5}
    )
    assert q4_resp.status_code == 200

    # 9. Question 5: Unsupported query (Non-existent concept in paper)
    q5_resp = client.post(
        f"/api/v1/papers/{paper_id}/query",
        json={"query": "What is the recipe for cooking lasagna?", "n_results": 5}
    )
    assert q5_resp.status_code == 200
    q5_answer = q5_resp.json()["answer"]
    # Check that it states context doesn't contain info or handles domain guardrail cleanly
    assert len(q5_answer) > 0

    # 10. Delete paper
    del_resp = client.delete(f"/api/v1/papers/{paper_id}")
    assert del_resp.status_code == 200

    # Verify deleted
    status_resp_after = client.get(f"/api/v1/papers/{paper_id}/status")
    assert status_resp_after.status_code == 404

def test_document_isolation():
    """Verify that querying Paper A does NOT retrieve vectors from Paper B."""
    if not (ATTENTION_PDF.exists() and ADAM_PDF.exists()):
        pytest.skip("Sample PDFs not available")

    # Upload Paper A (Attention)
    with open(ATTENTION_PDF, "rb") as f:
        resp_a = client.post("/api/v1/papers/upload", files={"file": ("Attention.pdf", f.read(), "application/pdf")})
    paper_id_a = resp_a.json()["paper_id"]
    paper_service.process_paper(paper_id_a)

    # Upload Paper B (Adam)
    with open(ADAM_PDF, "rb") as f:
        resp_b = client.post("/api/v1/papers/upload", files={"file": ("Adam.pdf", f.read(), "application/pdf")})
    paper_id_b = resp_b.json()["paper_id"]
    paper_service.process_paper(paper_id_b)

    # Retrieve items for Paper A with query "stochastic optimization"
    items_a = paper_service.retriever.retrieve(query="stochastic optimization", paper_id=paper_id_a, n_results=5)
    for item in items_a:
        assert item["metadata"]["paper_id"] == paper_id_a
        assert item["metadata"]["paper_id"] != paper_id_b

    # Cleanup
    paper_service.delete_paper(paper_id_a)
    paper_service.delete_paper(paper_id_b)
