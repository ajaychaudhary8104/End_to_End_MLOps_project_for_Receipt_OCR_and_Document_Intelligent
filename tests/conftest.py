import pytest
from fastapi.testclient import TestClient


class MockInference:

    def __init__(self):
        self.config = type(
            "Config",
            (),
            {
                "model_path": "tests/mock-model",
                "input_dir": "data/inference",
                "output_dir": "artifacts/inference",
            },
        )()

    def predict_one(self, image_path):

        return {
            "receipt_id": image_path.stem,
            "store_name": "TEST STORE",
            "date": "2025-01-15",
            "items": [
                {
                    "name": "TEST ITEM",
                    "price": 10.00,
                }
            ],
            "item_count": 1,
            "expected_item_count": 1,
            "item_sum": 10.00,
            "total_amount": 10.00,
            "overall_confidence": 0.96,
            "reliability": "high",
            "review_required": False,
            "status": "success",
        }


@pytest.fixture()
def mock_inference():

    return MockInference()


@pytest.fixture()
def client(monkeypatch, mock_inference):

    import app

    monkeypatch.setattr(
        app,
        "initialize_inference",
        lambda: mock_inference,
    )

    with TestClient(app.app) as test_client:
        yield test_client