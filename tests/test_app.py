def test_home(client):

    response = client.get("/")

    assert response.status_code == 200
    assert "Receipt OCR" in response.text


def test_liveness(client):

    response = client.get(
        "/health/live"
    )

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "alive"


def test_health(client):

    response = client.get(
        "/health"
    )

    assert response.status_code == 200

    data = response.json()

    assert data["model_loaded"] is True
    assert data["pipeline_ready"] is True


def test_readiness(client):

    response = client.get(
        "/health/ready"
    )

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ready"


def test_api_info(client):

    response = client.get(
        "/api/info"
    )

    assert response.status_code == 200

    data = response.json()

    assert data["success"] is True
    assert "ReceiptInferencePipeline" in data["pipeline"]


def test_pipeline_info(client):

    response = client.get(
        "/api/pipeline/info"
    )

    assert response.status_code == 200

    data = response.json()

    assert data["success"] is True
    assert data["pipeline"] == "ReceiptInferencePipeline"