import json
import os
import time
from pathlib import Path

import pytest

from ai_assistant.api.schemas import QARequest
from ai_assistant.core.context_loader import load_patient_context
from ai_assistant.core.local_llm import health_check
from ai_assistant.core.qa_engine import answer_question
from ai_assistant.core.report_generator import generate_report_draft

CONTEXTS_ROOT = str(Path(__file__).resolve().parents[1] / "outputs" / "cases")
TEST_CASE = "BraTS20_Validation_001"
STRUCTURED_QA_MAX_S = 0.25
REPORT_MAX_S = 0.50


def _env_info():
    return {
        "AI_USE_LLM": os.getenv("AI_USE_LLM"),
        "AI_LLM_PROVIDER": os.getenv("AI_LLM_PROVIDER"),
        "AI_LLAMACPP_URL": os.getenv("AI_LLAMACPP_URL"),
        "AI_MODEL_DEFAULT": os.getenv("AI_MODEL_DEFAULT"),
    }


def _load_context():
    return load_patient_context(TEST_CASE, CONTEXTS_ROOT)


def test_structured_qa_latency_smoke(monkeypatch):
    monkeypatch.setenv("AI_USE_LLM", "0")
    ctx = _load_context()
    req = QARequest(
        patient_id=ctx.patient_id,
        study_id=ctx.study_id,
        question="What is the total tumor volume?",
        context=ctx,
    )
    t0 = time.perf_counter()
    resp = answer_question(req)
    elapsed = time.perf_counter() - t0
    assert "mm3" in resp.answer
    assert elapsed <= STRUCTURED_QA_MAX_S


def test_report_generation_latency_smoke(monkeypatch):
    monkeypatch.setenv("AI_USE_LLM", "0")
    ctx = _load_context()
    t0 = time.perf_counter()
    resp = generate_report_draft(ctx)
    elapsed = time.perf_counter() - t0
    assert "# Findings" in resp.report_markdown
    assert elapsed <= REPORT_MAX_S


def test_optional_llm_latency_smoke():
    if os.getenv("AI_RUN_LLM_PERF", "0") != "1":
        pytest.skip("Set AI_RUN_LLM_PERF=1 to run live LLM latency smoke test.")
    if not health_check(timeout_s=5):
        pytest.skip("Embedded LLM endpoint is not healthy.")

    ctx = _load_context()
    req = QARequest(
        patient_id=ctx.patient_id,
        study_id=ctx.study_id,
        question="Summarize the key imaging findings for this patient.",
        context=ctx,
    )
    t0 = time.perf_counter()
    resp = answer_question(req)
    elapsed = time.perf_counter() - t0
    assert resp.answer
    assert elapsed <= float(os.getenv("AI_LLM_QA_MAX_S", "30"))


def run_performance_test():
    print("\n=== PERFORMANCE TEST ===")
    print(f"Environment: {json.dumps(_env_info())}")

    llm_ok = health_check(timeout_s=5)
    print(f"LLM health check: {'OK' if llm_ok else 'FAIL (running without LLM)'}")

    ctx = _load_context()

    q1 = "Summarize the key imaging findings for this patient."
    req1 = QARequest(patient_id=ctx.patient_id, study_id=ctx.study_id, question=q1, context=ctx)

    t0 = time.perf_counter()
    resp1 = answer_question(req1)
    t_first_qa = time.perf_counter() - t0
    print(f"\n[1] First QA (cold):  {t_first_qa:.2f}s")
    print(f"    Answer preview:   {resp1.answer[:100]}...")

    q2 = "What additional clinical information would help interpret these findings?"
    req2 = QARequest(patient_id=ctx.patient_id, study_id=ctx.study_id, question=q2, context=ctx)

    t0 = time.perf_counter()
    resp2 = answer_question(req2)
    t_warm_qa = time.perf_counter() - t0
    print(f"\n[2] Warm QA:          {t_warm_qa:.2f}s")
    print(f"    Answer preview:   {resp2.answer[:100]}...")

    q3 = "What is the total tumor volume?"
    req3 = QARequest(patient_id=ctx.patient_id, study_id=ctx.study_id, question=q3, context=ctx)

    t0 = time.perf_counter()
    resp3 = answer_question(req3)
    t_structured = time.perf_counter() - t0
    print(f"\n[3] Structured QA:    {t_structured:.4f}s")
    print(f"    Answer:           {resp3.answer}")

    t0 = time.perf_counter()
    report = generate_report_draft(ctx)
    t_report = time.perf_counter() - t0
    print(f"\n[4] Report generation: {t_report:.2f}s")
    print(f"    Report length:    {len(report.report_markdown)} chars")

    print("\n=== SUMMARY ===")
    print(f"First QA (cold):     {t_first_qa:.2f}s  {'PASS' if t_first_qa <= 30 else 'SLOW'}")
    print(f"Warm QA:             {t_warm_qa:.2f}s  {'PASS' if t_warm_qa <= 15 else 'SLOW'}")
    print(f"Structured QA:       {t_structured:.4f}s  {'PASS' if t_structured <= 1 else 'SLOW'}")
    print(f"Report generation:   {t_report:.2f}s")
    print(f"LLM active:          {llm_ok}")

    results = {
        "case_id": TEST_CASE,
        "llm_active": llm_ok,
        "env": _env_info(),
        "first_qa_cold_s": round(t_first_qa, 2),
        "warm_qa_s": round(t_warm_qa, 2),
        "structured_qa_s": round(t_structured, 4),
        "report_generation_s": round(t_report, 2),
    }
    out_path = Path(__file__).resolve().parents[1] / "outputs" / "performance_test.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {out_path}")


if __name__ == "__main__":
    run_performance_test()
