import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ai_assistant.core.case_repository import load_case_context
from ai_assistant.core.paths import default_contexts_root, default_runs_root
from ai_assistant.core.qa_engine import answer_question
from ai_assistant.api.schemas import QARequest


BENCHMARK_FILE = Path(__file__).parent / "brain_qa_benchmark.json"


def load_benchmark(ids: list[int] | None, category: str | None) -> list[dict]:
    data = json.loads(BENCHMARK_FILE.read_text(encoding="utf-8"))
    qs = data["questions"]
    if ids:
        qs = [q for q in qs if q["id"] in ids]
    if category:
        qs = [q for q in qs if q.get("category") == category]
    return qs, data.get("default_case_id", "BraTS20_Validation_001")


def run_question(case_id: str, question: str, session_id: str) -> dict:
    ctx_root = default_contexts_root()
    runs_root = default_runs_root()
    context = load_case_context(case_id, ctx_root, runs_root)
    req = QARequest(
        patient_id=context.patient_id,
        study_id=context.study_id,
        question=question,
        context=context,
        session_id=session_id,
    )
    return answer_question(req)


def check_result(resp, q: dict) -> tuple[bool, list[str]]:
    failures = []
    answer_lower = resp.answer.lower()

    for kw in q.get("expected_keywords", []):
        if kw.lower() not in answer_lower:
            failures.append(f"missing keyword: '{kw}'")

    for kw in q.get("must_not_contain", []):
        if kw.lower() in answer_lower:
            failures.append(f"contains forbidden: '{kw}'")

    expected_intent = q.get("expected_intent")
    if expected_intent:
        got_intent = (resp.resolved_intent or {}).get("intent_class", "")
        if got_intent != expected_intent:
            failures.append(f"intent: expected '{expected_intent}', got '{got_intent}'")

    expected_tiers = q.get("expected_implication_tier")
    if expected_tiers:
        got_tier = (resp.clinical_implication or {}).get("overall_tier", "") if resp.clinical_implication else ""
        if isinstance(resp.clinical_implication, dict):
            got_tier = resp.clinical_implication.get("overall_tier", "")
        elif resp.clinical_implication:
            got_tier = getattr(resp.clinical_implication, "overall_tier", "")
        if got_tier not in expected_tiers:
            failures.append(f"implication tier: expected one of {expected_tiers}, got '{got_tier}'")

    if resp.confidence < 0.15 and q.get("expected_status") != "guardrail_blocked":
        failures.append(f"confidence too low: {resp.confidence:.2f}")

    return len(failures) == 0, failures


def run_benchmark(questions: list[dict], default_case: str, verbose: bool = False):
    passed = 0
    failed = 0
    errors = 0
    results = []

    print(f"\n{'═'*70}")
    print(f"  Brain QA Benchmark — {len(questions)} questions")
    print(f"{'═'*70}\n")

    for q in questions:
        case_id = q.get("case_id", default_case)
        session_id = f"bench-{q['id']}"
        question = q["question"]

        t0 = time.time()
        try:
            resp = run_question(case_id, question, session_id)
            elapsed = time.time() - t0
            ok, failures = check_result(resp, q)

            status = "PASS" if ok else "FAIL"
            if ok:
                passed += 1
            else:
                failed += 1

            results.append({
                "id": q["id"],
                "category": q.get("category"),
                "case_id": case_id,
                "question": question,
                "status": status,
                "failures": failures,
                "confidence": round(resp.confidence, 3),
                "intent_got": (resp.resolved_intent or {}).get("intent_class", ""),
                "intent_expected": q.get("expected_intent", ""),
                "elapsed_s": round(elapsed, 2),
            })

            color = "\033[92m" if ok else "\033[91m"
            reset = "\033[0m"
            print(f"  [{color}{status}{reset}] #{q['id']:02d} [{q.get('category','?'):20s}] {question[:55]:<55s}  conf={resp.confidence:.2f}  {elapsed:.1f}s")
            if not ok and verbose:
                for f in failures:
                    print(f"         ↳ {f}")
                if resp.answer:
                    print(f"         answer: {resp.answer[:120]}…")

        except Exception as exc:
            errors += 1
            elapsed = time.time() - t0
            results.append({"id": q["id"], "status": "ERROR", "error": str(exc), "elapsed_s": round(elapsed, 2)})
            print(f"  [\033[93mERROR\033[0m] #{q['id']:02d} [{q.get('category','?'):20s}] {question[:55]:<55s}  {exc}")

    total = len(questions)
    print(f"\n{'─'*70}")
    print(f"  Results: {passed}/{total} passed  |  {failed} failed  |  {errors} errors")
    print(f"  Pass rate: {passed/total*100:.1f}%")
    print(f"{'═'*70}\n")

    out_path = Path(__file__).parent / "benchmark_results.json"
    out_path.write_text(json.dumps({"summary": {"total": total, "passed": passed, "failed": failed, "errors": errors}, "results": results}, indent=2, ensure_ascii=False))
    print(f"  Results saved to: {out_path}\n")

    return passed, failed, errors


def main():
    parser = argparse.ArgumentParser(description="Brain QA Benchmark Runner")
    parser.add_argument("--case", help="Override case ID for all questions")
    parser.add_argument("--category", help="Filter by category (e.g. mass_effect, volumetric)")
    parser.add_argument("--ids", help="Comma-separated question IDs to run (e.g. 1,5,16)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show failure details")
    args = parser.parse_args()

    ids = [int(x) for x in args.ids.split(",")] if args.ids else None
    questions, default_case = load_benchmark(ids, args.category)

    if args.case:
        for q in questions:
            q["case_id"] = args.case
        default_case = args.case

    passed, failed, errors = run_benchmark(questions, default_case, verbose=args.verbose)
    sys.exit(0 if failed == 0 and errors == 0 else 1)


if __name__ == "__main__":
    main()
