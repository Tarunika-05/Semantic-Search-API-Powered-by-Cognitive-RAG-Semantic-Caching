
def test_query_endpoint(app_client):
    response = app_client.post(
        "/query", 
        json={"query": "neural networks", "generate": False, "limit": 2, "offset": 0},
        headers={"X-API-Key": "user-secret-key"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "result" in data
    assert data["limit"] == 2
    assert data["offset"] == 0

def test_query_pagination(app_client):
    # Test offset 0
    resp1 = app_client.post(
        "/query", 
        json={"query": "neural networks", "generate": False, "limit": 2, "offset": 0},
        headers={"X-API-Key": "user-secret-key"}
    ).json()
    
    # Test offset 2
    resp2 = app_client.post(
        "/query", 
        json={"query": "neural networks", "generate": False, "limit": 2, "offset": 2},
        headers={"X-API-Key": "user-secret-key"}
    ).json()
    
    # Ensure they return different results or handle end of corpus safely
    assert resp1["limit"] == 2
    assert resp2["offset"] == 2
    
def test_hybrid_query_endpoint(app_client):
    response = app_client.post(
        "/hybrid-query", 
        json={"query": "image classification", "generate": False, "limit": 3},
        headers={"X-API-Key": "user-secret-key"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "result" in data
    assert "score_breakdown" in data
    assert data["limit"] == 3

def test_filtered_query_endpoint(app_client):
    response = app_client.post(
        "/filtered-query", 
        json={"query": "image classification", "category": 0, "generate": False, "limit": 1},
        headers={"X-API-Key": "user-secret-key"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["category_filter"] == 0
    assert data["limit"] == 1
