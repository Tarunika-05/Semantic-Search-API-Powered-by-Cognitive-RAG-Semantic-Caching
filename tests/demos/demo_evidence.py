import pytest
import numpy as np
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock


# Mock the entire loading process before importing the app
@pytest.fixture(autouse=True)
def mock_startup():
    with patch("app.main.get_model"), \
         patch("app.main.load_embeddings") as mock_le, \
         patch("app.main.load_index") as mock_li, \
         patch("app.main.load_clustering") as mock_lc, \
         patch("app.main.load_bm25") as mock_lb, \
         patch("app.api.process_query") as mock_pq, \
         patch("app.api.get_result_from_corpus") as mock_grfc, \
         patch("app.api.get_hybrid_result") as mock_ghr, \
         patch("app.llm.generate_answer") as mock_ga:
         
        # Setup mocks to return dummy data for startup
        mock_le.return_value = (np.array([[0.1, 0.2]]), ["Dummy Doc"], [0])
        mock_li.return_value = MagicMock()
        mock_lc.return_value = (MagicMock(), MagicMock(), np.array([[1.0]]), [0])
        mock_lb.return_value = MagicMock()
        
        # Setup mocks for API endpoints
        mock_pq.return_value = (np.array([0.1, 0.2]), 0, np.array([1.0]))
        mock_grfc.return_value = ("Dummy result", ["Dummy result doc"])
        mock_ghr.return_value = ("Hybrid result", ["Hybrid doc"], [{"dense_score": 0.5}])
        mock_ga.return_value = "Generated answer"
        
        from app.main import app
        yield app

def test_query_endpoint(mock_startup):
    with TestClient(mock_startup) as client:
        response = client.post("/query", json={"query": "test query", "generate": True})
        assert response.status_code == 200
        data = response.json()
        assert data["query"] == "test query"
        assert data["result"] == "Dummy result"
        assert data["generated_answer"] == "Generated answer"
        assert data["cache_hit"] is False
        assert "dominant_cluster" in data

def test_hybrid_query_endpoint(mock_startup):
    with TestClient(mock_startup) as client:
        response = client.post("/hybrid-query", json={"query": "test hybrid", "alpha": 0.5, "generate": True})
        assert response.status_code == 200
        data = response.json()
        assert data["search_mode"] == "hybrid"
        assert data["result"] == "Hybrid result"
        assert data["generated_answer"] == "Generated answer"

def test_filtered_query_endpoint(mock_startup):
    with TestClient(mock_startup) as client:
        response = client.post("/filtered-query", json={"query": "test filter", "category": 1, "generate": False})
        assert response.status_code == 200
        data = response.json()
        assert data["search_mode"] == "filtered"
        assert data["category_filter"] == 1
        assert data["generated_answer"] is None

def test_cache_endpoints(mock_startup):
    with TestClient(mock_startup) as client:
        # Check stats
        response = client.get("/cache/stats")
        assert response.status_code == 200
        
        headers = {"X-Admin-Key": "super-secret-admin-key"}
        
        # Update threshold
        response = client.patch("/cache/threshold", params={"threshold": 0.95}, headers=headers)
        assert response.status_code == 200
        assert response.json()["threshold"] == 0.95
        
        # Clear cache
        response = client.delete("/cache", headers=headers)
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

def test_health_endpoint(mock_startup):
    with TestClient(mock_startup) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
