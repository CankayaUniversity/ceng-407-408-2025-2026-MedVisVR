import argparse
import json
from pathlib import Path

from ai_assistant.core.paths import default_metrics_out, default_runs_root

REQUIRED_SECTIONS = [
    "# Clinical Context",
    "# Findings",
    "# Impression",
    "# Limitations",
    "# Suggested Clinical Correlation",
    "# Disclaimer",
]

NO_EVIDENCE_EXPECTED_KEYS = [
    "definitely a glioblastoma",
    "start chemotherapy",
    "survival prognosis",
    "diagnose based on imaging alone",
    "surgery recommended",
    "treatment do you recommend",
    "grow or shrink",
    "is this cancer",
    "should i worry",
    "cure rate",
]


def _expects_evidence(question: str) -> bool:
    q = (question or "").lower().strip()
    return not any(k in q for k in NO_EVIDENCE_EXPECTED_KEYS)


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def evaluate_case(case_dir: Path) -> dict:
    report_path = case_dir / "report_draft.md"
    qa_path = case_dir / "qa_results.jsonl"

    report_text = report_path.read_text(encoding="utf-8") if report_path.exists() else ""
    qa_rows = _load_jsonl(qa_path)

    missing_sections = [sec for sec in REQUIRED_SECTIONS if sec not in report_text]
    has_required_sections = len(missing_sections) == 0

    total_qa = len(qa_rows)
    total_expected_evidence = 0
    with_evidence = 0
    for row in qa_rows:
        q = str(row.get("question", "") or "")
        if not _expects_evidence(q):
            continue
        total_expected_evidence += 1
        evidence_ids = row.get("evidence_ids", [])
        if isinstance(evidence_ids, list) and len(evidence_ids) > 0:
            with_evidence += 1

    qa_evidence_coverage = (with_evidence / total_expected_evidence) if total_expected_evidence > 0 else 0.0
    unsupported_statement_rate = (1.0 - qa_evidence_coverage) if total_expected_evidence > 0 else 1.0

    contradiction = ("definitive" in report_text.lower()) and ("insufficient evidence" in report_text.lower())

    return {
        "case_id": case_dir.name,
        "has_required_sections": has_required_sections,
        "missing_sections": missing_sections,
        "qa_total": total_qa,
        "qa_expected_evidence_total": total_expected_evidence,
        "qa_with_evidence": with_evidence,
        "qa_evidence_coverage": round(qa_evidence_coverage, 4),
        "unsupported_statement_rate": round(unsupported_statement_rate, 4),
        "contradiction_flag": contradiction,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute harness metrics from run outputs")
    parser.add_argument(
        "--runs-root",
        default=default_runs_root(),
        help="Root folder of per-case run outputs",
    )
    parser.add_argument(
        "--out",
        default=default_metrics_out(),
        help="Output JSON summary path",
    )
    args = parser.parse_args()

    runs_root = Path(args.runs_root)
    case_dirs = sorted([p for p in runs_root.iterdir() if p.is_dir()]) if runs_root.exists() else []

    case_results = [evaluate_case(case_dir) for case_dir in case_dirs]

    n = len(case_results)
    required_section_missing_rate = (
        sum(0 if c["has_required_sections"] else 1 for c in case_results) / n if n else 1.0
    )
    avg_qa_evidence_coverage = (
        sum(c["qa_evidence_coverage"] for c in case_results) / n if n else 0.0
    )
    avg_unsupported_statement_rate = (
        sum(c["unsupported_statement_rate"] for c in case_results) / n if n else 1.0
    )
    contradiction_rate = (
        sum(1 for c in case_results if c["contradiction_flag"]) / n if n else 0.0
    )

    summary = {
        "num_cases": n,
        "required_section_missing_rate": round(required_section_missing_rate, 4),
        "qa_evidence_coverage": round(avg_qa_evidence_coverage, 4),
        "unsupported_statement_rate": round(avg_unsupported_statement_rate, 4),
        "contradiction_rate": round(contradiction_rate, 4),
        "per_case": case_results,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=True, indent=2)

    print(f"OK metrics written: {out_path}")
    print(json.dumps({k: v for k, v in summary.items() if k != 'per_case'}, ensure_ascii=True))


if __name__ == "__main__":
    main()
