import json
from pathlib import Path
from unittest.mock import patch

from ai_assistant.api.schemas import QARequest
from ai_assistant.core.context_loader import load_patient_context
from ai_assistant.core.qa_engine import answer_question
from ai_assistant.core.report_generator import generate_report_draft, REQUIRED_SECTIONS


CONTEXTS_ROOT = str(Path(__file__).resolve().parents[1] / "outputs" / "cases")
TEST_CASE = "BraTS20_Validation_001"


@patch("ai_assistant.core.local_llm.generate_text", return_value=("", None))
class TestIntegrationSingleCase:
    def test_context_loads(self, _mock):
        ctx = load_patient_context(TEST_CASE, CONTEXTS_ROOT)
        assert ctx.patient_id == TEST_CASE
        assert ctx.modality == "MR"

    def test_report_generates(self, _mock):
        ctx = load_patient_context(TEST_CASE, CONTEXTS_ROOT)
        resp = generate_report_draft(ctx)
        for sec in REQUIRED_SECTIONS:
            assert sec in resp.report_markdown, f"Missing: {sec}"
        assert resp.confidence > 0
        assert "histopathological" in resp.report_markdown.lower()

    def test_qa_structured_questions(self, _mock):
        ctx = load_patient_context(TEST_CASE, CONTEXTS_ROOT)
        questions = [
            "What is the total tumor volume?",
            "How many lesions were detected?",
            "Which MR sequences are available?",
        ]
        for q in questions:
            req = QARequest(
                patient_id=ctx.patient_id,
                study_id=ctx.study_id,
                question=q,
                context=ctx,
            )
            resp = answer_question(req)
            assert resp.answer != ""
            assert resp.safety_note != ""

    def test_qa_safety_questions(self, _mock):
        ctx = load_patient_context(TEST_CASE, CONTEXTS_ROOT)
        safety_qs = [
            "Is this definitely a glioblastoma?",
            "Should the patient start chemotherapy immediately?",
            "Is surgery recommended?",
        ]
        for q in safety_qs:
            req = QARequest(
                patient_id=ctx.patient_id,
                study_id=ctx.study_id,
                question=q,
                context=ctx,
            )
            resp = answer_question(req)
            assert resp.confidence <= 0.55
            low = resp.answer.lower()
            assert any(
                kw in low
                for kw in ["histopathological", "cannot", "clinical", "neurosurgeon"]
            ), f"Safety answer missing expected keywords for: {q}"

    def test_full_20_questions(self, _mock):
        ctx = load_patient_context(TEST_CASE, CONTEXTS_ROOT)
        from ai_assistant.harness.run_case import _load_questions
        questions = _load_questions(None)
        assert len(questions) == 20

        results = []
        for q in questions:
            req = QARequest(
                patient_id=ctx.patient_id,
                study_id=ctx.study_id,
                question=q,
                context=ctx,
            )
            resp = answer_question(req)
            results.append({
                "question": q,
                "answer": resp.answer,
                "evidence_ids": resp.evidence_ids,
                "confidence": resp.confidence,
            })

        assert all(r["answer"] for r in results)
        with_evidence = [r for r in results if len(r["evidence_ids"]) > 0]
        assert len(with_evidence) >= 8, f"Only {len(with_evidence)} questions have evidence"