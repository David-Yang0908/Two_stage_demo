from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_read_main():
    response = client.get("/")
    assert response.status_code == 200

def test_get_test_list():
    response = client.get("/api/get_test_list")
    assert response.status_code == 200
    assert "files" in response.json()