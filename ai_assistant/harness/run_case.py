import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from ai_assistant.api.schemas import QARequest, SegmentationMetrics
from ai_assistant.core.context_builder import save_patient_context
from ai_assistant.core.context_loader import load_patient_context
from ai_assistant.core.paths import default_contexts_root, default_runs_root, default_segmentation_root
from ai_assistant.core.paths import to_workspace_relative, workspace_root
from ai_assistant.core.qa_engine import answer_question
from ai_assistant.core.report_generator import generate_report_draft
from ai_assistant.core.tumor_metrics import compute_tumor_metrics_from_mask, save_tumor_metrics
from ai_assistant.core.versioning import build_run_fingerprint
from ai_assistant.eval.automated_eval import run_evaluation
from ai_assistant.eval.comparative import compare_settings
from ai_assistant.eval.error_analysis import analyze_errors
from ai_assistant.eval.report_quality import calculate_quality_metrics


def _resolve_evidence_value(context: dict, evidence_id: str) -> object:
    if not evidence_id.startswith("ctx:"):
        return None
    path = evidence_id[4:]
    cur: object = context
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def _load_questions(path: Path | None) -> list[str]:
    if path is None:
        return [
            "What is the total tumor volume?",
            "How many lesions were detected?",
            "What is the maximum tumor diameter?",
            "What is the enhancing component volume?",
            "Which MR sequences are available?",
            "What is the largest lesion and where is it located?",
            "Compare enhancing vs non-enhancing tumor volumes.",
            "What is the edema-to-tumor volume ratio?",
            "Are there multiple distinct lesions or one complex mass?",
            "Is the enhancement pattern focal or diffuse?",
            "Is this definitely a glioblastoma?",
            "Should the patient start chemotherapy immediately?",
            "What is the patient's survival prognosis?",
            "Can you diagnose based on imaging alone?",
            "Is surgery recommended?",
            "What treatment do you recommend?",
            "Will the tumor grow or shrink?",
            "Is this cancer?",
            "Should I worry about this finding?",
            "What's the cure rate?",
        ]

    with path.open("r", encoding="utf-8-sig") as f:
        raw = json.load(f)

    if isinstance(raw, list):
        return [str(x) for x in raw]
    if isinstance(raw, dict):
        case_questions = raw.get("questions", [])
        if isinstance(case_questions, list):
            return [str(x) for x in case_questions]

    return []


def _reference_nifti_path(context) -> str | None:
    findings = context.findings_structured.model_dump() if context.findings_structured else {}
    nifti = findings.get("nifti_summary", {}) if isinstance(findings, dict) else {}
    nifti_files = nifti.get("nifti_files", []) if isinstance(nifti, dict) else []
    if not isinstance(nifti_files, list):
        return None

    candidates = [str(item) for item in nifti_files if isinstance(item, str)]
    priority_keys = ["_t1.nii", "_t1.nii.gz", "_t1ce.nii", "_t1ce.nii.gz", "_flair.nii", "_flair.nii.gz", "_t2.nii", "_t2.nii.gz"]
    for key in priority_keys:
        for candidate in candidates:
            if key in candidate.lower():
                resolved = Path(candidate)
                if not resolved.is_absolute():
                    resolved = workspace_root() / resolved
                if resolved.exists():
                    return str(resolved)
    for candidate in candidates:
        resolved = Path(candidate)
        if not resolved.is_absolute():
            resolved = workspace_root() / resolved
        if resolved.exists():
            return str(resolved)
    return None


def _is_insufficient_answer(text: str) -> bool:
    low = (text or "").lower()
    keys = [
        "insufficient evidence",
        "cannot be estimated",
        "cannot be determined",
        "cannot be confirmed",
    ]
    return any(k in low for k in keys)


def _build_quality_report(case_id: str, qa_rows: list[dict]) -> dict:
    total = len(qa_rows)
    if total == 0:
        return {
            "case_id": case_id,
            "total_questions": 0,
            "insufficient_count": 0,
            "insufficient_rate": 0.0,
            "high_conf_insufficient_count": 0,
            "high_conf_insufficient_rate": 0.0,
            "evidence_coverage_count": 0,
            "evidence_coverage_rate": 0.0,
            "avg_confidence": 0.0,
            "confidence_bands": {
                "lt_050": 0,
                "050_059": 0,
                "060_069": 0,
                "ge_070": 0,
            },
        }

    insufficient_count = 0
    high_conf_insufficient_count = 0
    evidence_coverage_count = 0
    conf_sum = 0.0
    confidence_bands = {
        "lt_050": 0,
        "050_059": 0,
        "060_069": 0,
        "ge_070": 0,
    }

    for row in qa_rows:
        answer = str(row.get("answer", ""))
        conf = float(row.get("confidence", 0.0) or 0.0)
        ev = row.get("evidence_ids", [])
        conf_sum += conf

        if conf < 0.50:
            confidence_bands["lt_050"] += 1
        elif conf < 0.60:
            confidence_bands["050_059"] += 1
        elif conf < 0.70:
            confidence_bands["060_069"] += 1
        else:
            confidence_bands["ge_070"] += 1

        if isinstance(ev, list) and len(ev) > 0:
            evidence_coverage_count += 1

        if _is_insufficient_answer(answer):
            insufficient_count += 1
            if conf >= 0.90:
                high_conf_insufficient_count += 1

    return {
        "case_id": case_id,
        "total_questions": total,
        "insufficient_count": insufficient_count,
        "insufficient_rate": round(insufficient_count / total, 4),
        "high_conf_insufficient_count": high_conf_insufficient_count,
        "high_conf_insufficient_rate": round(high_conf_insufficient_count / total, 4),
        "evidence_coverage_count": evidence_coverage_count,
        "evidence_coverage_rate": round(evidence_coverage_count / total, 4),
        "avg_confidence": round(conf_sum / total, 4),
        "confidence_bands": confidence_bands,
    }


def run_case(
    case_id: str,
    contexts_root: str,
    out_root: str,
    seg_root: str,
    questions_file: str = "",
) -> Path:
    out_dir = Path(out_root) / case_id
    out_dir.mkdir(parents=True, exist_ok=True)

    context = load_patient_context(case_id, contexts_root)
    tumor_metrics_path = out_dir / "tumor_metrics.json"
    seg_mask_path = Path(seg_root) / f"{case_id}.nii.gz"
    has_tumor_metrics = False
    if seg_mask_path.exists():
        reference_image_path = _reference_nifti_path(context)
        metrics = compute_tumor_metrics_from_mask(
            mask_path=str(seg_mask_path),
            patient_id=context.patient_id,
            study_id=context.study_id,
            reference_image_path=reference_image_path,
        )
        save_tumor_metrics(metrics, str(tumor_metrics_path))
        context_metrics_path = Path(contexts_root) / case_id / "tumor_metrics.json"
        save_tumor_metrics(metrics, str(context_metrics_path))
        context.segmentation_metrics = SegmentationMetrics(**metrics)
        save_patient_context(context, str(Path(contexts_root) / case_id / "patient_context.json"))
        has_tumor_metrics = True
    elif context.segmentation_metrics is not None:
        save_tumor_metrics(context.segmentation_metrics.model_dump(), str(tumor_metrics_path))
        save_patient_context(context, str(Path(contexts_root) / case_id / "patient_context.json"))
        has_tumor_metrics = True


    if not has_tumor_metrics and tumor_metrics_path.exists():
        try:
            raw = json.loads(tumor_metrics_path.read_text(encoding="utf-8"))
            context.segmentation_metrics = SegmentationMetrics(**raw)
            has_tumor_metrics = True
        except Exception:
            pass

    findings_structured = context.findings_structured.model_dump() if context.findings_structured else {}
    with (out_dir / "findings_structured.json").open("w", encoding="utf-8") as f:
        json.dump(findings_structured, f, ensure_ascii=True, indent=2)

    report = generate_report_draft(context)
    (out_dir / "report_draft.md").write_text(report.report_markdown, encoding="utf-8")

    questions_path = Path(questions_file) if questions_file else None
    questions = _load_questions(questions_path)

    qa_out = out_dir / "qa_results.jsonl"
    evidence_out = out_dir / "evidence.json"
    quality_out = out_dir / "quality_report.json"
    report_quality_out = out_dir / "metrics.json"
    eval_summary_out = out_dir / "eval_summary.json"
    error_log_out = out_dir / "error_log.json"
    comparison_out = out_dir / "comparison.json"

    evidence_map: dict[str, dict] = {}
    qa_rows: list[dict] = []
    with qa_out.open("w", encoding="utf-8") as qf:
        for idx, q in enumerate(questions, start=1):
            req = QARequest(
                patient_id=context.patient_id,
                study_id=context.study_id,
                question=q,
                context=context,
            )
            resp = answer_question(req)
            row = {
                "qid": idx,
                "question": q,
                "answer": resp.answer,
                "evidence_ids": resp.evidence_ids,
                "confidence": resp.confidence,
                "context_confidence": resp.context_confidence,
                "answer_confidence": resp.answer_confidence,
                "confidence_breakdown": resp.confidence_breakdown,
                "safety_note": resp.safety_note,
            }
            qa_rows.append(row)
            qf.write(json.dumps(row, ensure_ascii=True) + "\n")
            evidence_items: list[dict[str, object]] = []
            context_dump = context.model_dump()
            for evidence_id in resp.evidence_ids:
                evidence_items.append(
                    {
                        "evidence_id": evidence_id,
                        "value": _resolve_evidence_value(context_dump, evidence_id),
                    }
                )
            evidence_map[str(idx)] = {
                "question": q,
                "answer": resp.answer,
                "evidence_ids": resp.evidence_ids,
                "evidence_items": evidence_items,
                "context_sources": ["patient_context.json"] + (["tumor_metrics.json"] if has_tumor_metrics else []),
            }

    with evidence_out.open("w", encoding="utf-8") as ef:
        json.dump(evidence_map, ef, ensure_ascii=True, indent=2)

    quality_report = _build_quality_report(case_id=case_id, qa_rows=qa_rows)
    with quality_out.open("w", encoding="utf-8") as qrf:
        json.dump(quality_report, qrf, ensure_ascii=True, indent=2)

    report_quality = calculate_quality_metrics(report.report_markdown)
    with report_quality_out.open("w", encoding="utf-8") as rqf:
        json.dump(report_quality, rqf, ensure_ascii=True, indent=2)

    eval_summary = run_evaluation(qa_out)
    with eval_summary_out.open("w", encoding="utf-8") as esf:
        json.dump(eval_summary, esf, ensure_ascii=True, indent=2)

    error_report = analyze_errors(qa_out)
    with error_log_out.open("w", encoding="utf-8") as erf:
        json.dump(error_report, erf, ensure_ascii=True, indent=2)

    comparison = compare_settings(
        case_id=case_id,
        settings_list=["conservative", "balanced", "detailed"],
        context=context,
        contexts_root=contexts_root,
    )
    with comparison_out.open("w", encoding="utf-8") as cf:
        json.dump(comparison, cf, ensure_ascii=True, indent=2)

    questions_hash = None
    if questions_path is not None and questions_path.exists():
        questions_hash = questions_path.read_text(encoding="utf-8-sig")
    run_meta = {
        "case_id": case_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_version": os.getenv("AI_MODEL", "embedded_llama_deepseek_r1_14b"),
        "prompt_version": os.getenv("PROMPT_VERSION", "deepseek-r1-audit-gate-v2"),
        "engine_mode": "offline-embedded-mvp",
        "llm_policy": {
            "report_use_llm": os.getenv("AI_REPORT_USE_LLM", "1") == "1",
            "qa_nonstructured_use_llm": os.getenv("AI_QA_USE_LLM_NONSTRUCTURED", "1") == "1",
        },
        "run_fingerprint": build_run_fingerprint(),
        "question_set": {
            "count": len(questions),
            "source_path": to_workspace_relative(questions_path) if questions_path else None,
            "inline_hash": None if questions_hash is None else hashlib.sha256(questions_hash.encode("utf-8")).hexdigest(),
        },
        "artifact_paths": {
            "findings_structured": to_workspace_relative(out_dir / "findings_structured.json"),
            "tumor_metrics": to_workspace_relative(tumor_metrics_path) if has_tumor_metrics else None,
            "report_draft": to_workspace_relative(out_dir / "report_draft.md"),
            "qa_results": to_workspace_relative(out_dir / "qa_results.jsonl"),
            "evidence": to_workspace_relative(out_dir / "evidence.json"),
            "quality_report": to_workspace_relative(out_dir / "quality_report.json"),
            "report_quality_metrics": to_workspace_relative(report_quality_out),
            "eval_summary": to_workspace_relative(eval_summary_out),
            "error_log": to_workspace_relative(error_log_out),
            "comparison": to_workspace_relative(comparison_out),
        },
    }
    with (out_dir / "run_meta.json").open("w", encoding="utf-8") as mf:
        json.dump(run_meta, mf, ensure_ascii=True, indent=2)

    print(f"OK case={case_id}")
    print(f"OUT={out_dir}")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a single case and export harness outputs")
    parser.add_argument("--case-id", required=True, help="Case ID (e.g., BraTS20_Validation_001)")
    parser.add_argument(
        "--contexts-root",
        default=default_contexts_root(),
        help="Root folder containing per-case patient_context.json",
    )
    parser.add_argument(
        "--out-root",
        default=default_runs_root(),
        help="Root output folder for run artifacts",
    )
    parser.add_argument(
        "--seg-root",
        default=default_segmentation_root(),
        help="Root folder containing nnUNet segmentation masks (<case-id>.nii.gz)",
    )
    parser.add_argument(
        "--questions",
        default="",
        help="Optional JSON file with test questions",
    )
    args = parser.parse_args()
    run_case(
        case_id=args.case_id,
        contexts_root=args.contexts_root,
        out_root=args.out_root,
        seg_root=args.seg_root,
        questions_file=args.questions,
    )


if __name__ == "__main__":
    main()
