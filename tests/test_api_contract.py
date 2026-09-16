import io


def test_openapi_document(client):

    response = client.get(
        "/api/openapi.json"
    )

    assert response.status_code == 200

    data = response.json()

    assert "openapi" in data
    assert "paths" in data

    assert "/health" in data["paths"]
    assert "/predict/upload" in data["paths"]


def test_swagger_docs(client):

    response = client.get(
        "/api/docs"
    )

    assert response.status_code == 200


def test_upload_rejects_unsupported_extension(
    client
):

    response = client.post(
        "/predict/upload",
        files={
            "file": (
                "malware.exe",
                io.BytesIO(
                    b"test"
                ),
                "application/octet-stream",
            )
        },
    )

    assert response.status_code in {
        400,
        415,
    }


def test_upload_empty_file(
    client
):

    response = client.post(
        "/predict/upload",
        files={
            "file": (
                "receipt.jpg",
                io.BytesIO(
                    b""
                ),
                "image/jpeg",
            )
        },
    )

    assert response.status_code in {
        400,
        500,
    }