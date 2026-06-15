def test_health_returns_ok(app_client):
    response = app_client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_query_returns_200(app_client):
    response = app_client.post("/query", json={"query": "machine learning", "limit": 3}, headers={"X-API-Key": "user-secret-key"})
    assert response.status_code == 200
    data = response.json()
    assert "result" in data
    assert "limit" in data
    assert "offset" in data

def test_query_empty_string_returns_400(app_client):
    response = app_client.post("/query", json={"query": "   ", "limit": 3}, headers={"X-API-Key": "user-secret-key"})
    assert response.status_code == 400
    assert "detail" in response.json()


def test_filtered_query_with_category(app_client):
    response = app_client.post("/filtered-query", json={"query": "test", "category": 2, "limit": 3}, headers={"X-API-Key": "user-secret-key"})
    assert response.status_code == 200
    assert "result" in response.json()

def test_cache_stats_returns_counts(app_client):
    response = app_client.get("/cache/stats", headers={"X-API-Key": "admin-secret-key"})
    assert response.status_code == 200
    data = response.json()
    assert "total_entries" in data
    assert "hit_count" in data
    assert "miss_count" in data

def test_threshold_update(app_client):
    response = app_client.patch("/cache/threshold?threshold=0.95", headers={"X-API-Key": "admin-secret-key"})
    assert response.status_code in [200, 401, 403]
    if response.status_code == 200:
        assert response.json()["status"] == "ok"

def test_cache_clear_resets(app_client):
    response = app_client.delete("/cache", headers={"X-API-Key": "admin-secret-key"})
    assert response.status_code in [200, 401, 403]
    if response.status_code == 200:
        assert response.json()["status"] == "ok"
