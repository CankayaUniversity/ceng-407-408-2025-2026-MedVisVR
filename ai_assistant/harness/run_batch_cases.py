import argparse
import json
from pathlib import Path

from ai_assistant.core.context_builder import build_patient_context_from_case, save_patient_context
from ai_assistant.core.versioning import build_run_fingerprint
from ai_assistant.eval.metrics import evaluate_case
from ai_assistant.harness.generate_contexts import discover_cases
from ai_assistant.harness.run_case import run_case


def build_summary(runs_root: Path) -> dict[str, object]:
    case_dirs = sorted([p for p in runs_root.iterdir() if p.is_dir()]) if runs_root.exists() else []
    case_results = [evaluate_case(case_dir) for case_dir in case_dirs]
    n = len(case_results)
    return {
        "num_cases": n,
        "required_section_missing_rate": round(
            sum(0 if c["has_required_sections"] else 1 for c in case_results) / n if n else 1.0,
            4,
        ),
        "qa_evidence_coverage": round(
            sum(c["qa_evidence_coverage"] for c in case_results) / n if n else 0.0,
            4,
        ),
        "unsupported_statement_rate": round(
            sum(c["unsupported_statement_rate"] for c in case_results) / n if n else 1.0,
            4,
        ),
        "contradiction_rate": round(
            sum(1 for c in case_results if c["contradiction_flag"]) / n if n else 0.0,
            4,
        ),
        "per_case": case_results,
    }


def _run_meta_path(runs_root: Path, case_id: str) -> Path:
    return runs_root / case_id / "run_meta.json"


def _case_already_matches_policy(runs_root: Path, case_id: str, current_fingerprint: dict[str, object]) -> bool:
    meta_path = _run_meta_path(runs_root, case_id)
    if not meta_path.exists():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return False

    llm_policy = meta.get("llm_policy", {})
    if not isinstance(llm_policy, dict):
        return False

    if not (llm_policy.get("report_use_llm") and llm_policy.get("qa_nonstructured_use_llm")):
        return False

    prior_fingerprint = meta.get("run_fingerprint", {})
    if not isinstance(prior_fingerprint, dict):
        return False

    return prior_fingerprint.get("composite_hash") == current_fingerprint.get("composite_hash")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full harness pipeline for all discovered brain cases")
    parser.add_argument("--brain-root", default="brain_data", help="Root folder containing per-case MRI sequences")
    parser.add_argument("--contexts-root", default="outputs/cases", help="Per-case patient context root")
    parser.add_argument("--runs-root", default="outputs/runs", help="Per-case run artifact root")
    parser.add_argument("--seg-root", default="output_brain", help="Segmentation mask root")
    parser.add_argument("--questions", default="", help="Optional JSON file with test questions")
    parser.add_argument("--metrics-out", default="outputs/metrics_summary.json", help="Summary output JSON")
    parser.add_argument("--max-cases", type=int, default=0, help="Optional limit over the sorted discovered cases")
    parser.add_argument("--resume", action="store_true", help="Skip cases whose run_meta already matches the current LLM policy and code fingerprint")
    args = parser.parse_args()

    brain_root = Path(args.brain_root)
    contexts_root = Path(args.contexts_root)
    runs_root = Path(args.runs_root)
    contexts_root.mkdir(parents=True, exist_ok=True)
    runs_root.mkdir(parents=True, exist_ok=True)

    cases = discover_cases(brain_root)
    if args.max_cases and args.max_cases > 0:
        cases = cases[:args.max_cases]
    current_fingerprint = build_run_fingerprint()
    for case_id in cases:
        if args.resume and _case_already_matches_policy(runs_root, case_id, current_fingerprint):
            print(f"SKIP case={case_id} reason=already_matches_policy")
            continue
        context = build_patient_context_from_case(
            patient_id=case_id,
            dicom_case_dir=str(brain_root / case_id),
            nifti_case_dir=str(brain_root / case_id),
            modality="MR",
        )
        save_patient_context(context, str(contexts_root / case_id / "patient_context.json"))
        run_case(
            case_id=case_id,
            contexts_root=str(contexts_root),
            out_root=str(runs_root),
            seg_root=args.seg_root,
            questions_file=args.questions,
        )

    summary = build_summary(runs_root)
    metrics_out = Path(args.metrics_out)
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    with metrics_out.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=True, indent=2)

    print(f"DONE cases={len(cases)} metrics={metrics_out}")


if __name__ == "__main__":
    main()
