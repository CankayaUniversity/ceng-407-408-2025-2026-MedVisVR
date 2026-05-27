from fastapi.testclient import TestClient

from ai_assistant.api.main import app


client = TestClient(app)


def test_cases_endpoint_returns_items():
    resp = client.get("/cases")
    assert resp.status_code == 200
    payload = resp.json()
    assert isinstance(payload, list)
    assert any(item["case_id"] == "BraTS20_Validation_001" for item in payload)


def test_case_detail_endpoint_returns_quality_panel():
    resp = client.get("/cases/BraTS20_Validation_001")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["case_id"] == "BraTS20_Validation_001"
    assert payload["quality_panel"]["total_questions"] >= 1
    assert payload["context"]["segmentation_metrics"]["total_tumor_volume_mm3"] > 0


def test_qa_from_case_uses_run_segmentation_metrics():
    resp = client.post(
        "/qa/ask-from-case",
        json={
            "case_id": "BraTS20_Validation_001",
            "question": "Which component has the largest volume?",
        },
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert "largest burden component" in payload["answer"].lower()
    assert "mm3" in payload["answer"].lower()
    assert len(payload["evidence_ids"]) > 0
