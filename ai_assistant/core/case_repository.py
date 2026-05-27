import json
from pathlib import Path
from typing import Any

from ai_assistant.api.schemas import (
    CaseAnswerSummary,
    CaseDetailResponse,
    CaseListItem,
    CaseQualityPanel,
    FindingsStructured,
    PatientContext,
    SegmentationMetrics,
)
from ai_assistant.core.context_loader import load_patient_context
from ai_assistant.core.paths import workspace_root


def _existing_dir(path_like: str | Path) -> Path | None:
    path = Path(path_like)
    return path if path.exists() else None


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def _read_text(path: Path) -> str | None:
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8-sig")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            rows.append(item)
    return rows


def _is_insufficient(answer: str) -> bool:
    low = (answer or "").strip().lower()
    return low.startswith("insufficient evidence")


def _artifact_dir(case_id: str, contexts_root: str, runs_root: str | None = None) -> Path:
    runs_path = _existing_dir(runs_root) if runs_root else None
    if runs_path:
        preferred = runs_path / case_id
        if preferred.exists():
            return preferred
    return Path(contexts_root) / case_id


def _load_case_index(case_id: str) -> dict[str, Any]:
    path = workspace_root() / "output_brain" / "case_index" / f"{case_id}.json"
    return _read_json(path)


def _merge_case_index_into_metrics(metrics: dict[str, Any], case_index: dict[str, Any]) -> dict[str, Any]:
    if not case_index:
        return metrics
    result = dict(metrics)
    if "mass_effect" not in result and "mass_effect" in case_index:
        result["mass_effect"] = case_index["mass_effect"]
    if "critical_structure_proximity" not in result and "critical_proximity" in case_index:
        prox = case_index["critical_proximity"]
        result["critical_structure_proximity"] = [
            {"structure": k.replace("_distance_mm", ""), "distance_mm": float(v)}
            for k, v in prox.items()
            if k.endswith("_distance_mm") and isinstance(v, (int, float))
        ]
    if "location" not in result and "location" in case_index:
        result["location"] = case_index["location"]
    return result


def _merge_model(existing: Any, incoming: dict[str, Any], model_cls: type) -> Any:
    if not incoming:
        return existing

    base: dict[str, Any] = {}
    if existing is not None:
        if hasattr(existing, "model_dump"):
            base = existing.model_dump()
        elif isinstance(existing, dict):
            base = existing

    return model_cls(**{**base, **incoming})


def _normalize_brats_metrics(raw: dict[str, Any]) -> dict[str, Any]:
    if not raw or "tumor_burden" not in raw:
        return raw  

    burden = raw.get("tumor_burden", {})
    location = raw.get("location", {})
    mass_effect = raw.get("mass_effect", {})

    total_vol = float(burden.get("total_volume_mm3", 0) or 0)
    enh_vol   = float(burden.get("enhancing_volume_mm3", 0) or 0)
    edema_vol = float(burden.get("edema_volume_mm3", 0) or 0)
    nec_vol   = float(burden.get("necrotic_volume_mm3", 0) or 0)
    max_diam  = float(burden.get("max_diameter_mm", 0) or 0)

    hemi       = location.get("hemisphere", "")
    lobe       = location.get("lobe", "")
    depth_cls  = location.get("depth_class", "")
    depth_mm   = float(location.get("depth_mm", 0) or 0)
    lobe_conf  = location.get("lobe_confidence", "low")
    depth_conf = location.get("depth_confidence", "low")

    hemi_t = hemi.title() if hemi else ""
    lobe_t = lobe.title() if lobe else ""
    loc_parts = [p for p in [hemi_t, f"{lobe_t} lobe" if lobe_t else "", depth_cls] if p]
    dominant_loc = ", ".join(loc_parts) if loc_parts else None

    location_labels = [x for x in [hemi, lobe] if x]


    atlas_top_regions: list[dict[str, Any]] = []
    if hemi_t and lobe_t:
        atlas_top_regions.append({
            "region": f"{hemi_t} {lobe_t}",
            "fraction": 1.0,
            "source": "heuristic_centroid",
        })
    if depth_cls and depth_cls not in ("cortical",):
        structure = f"{hemi_t} Cerebral {'White Matter' if depth_cls == 'subcortical' else 'Cortex'}"
        atlas_top_regions.append({
            "region": structure,
            "fraction": 0.5,
            "source": "heuristic_depth",
        })


    lesion_list: list[dict[str, Any]] = []
    if total_vol > 0:
        lesion_anatomy = {
            "hemisphere": hemi_t or "Unknown",
            "description": dominant_loc or "anatomic localization unavailable",
            "atlas_confidence": lobe_conf,
            "atlas_source": "heuristic_centroid",
            "atlas_top_regions": atlas_top_regions,
            "depth_class": depth_cls,
            "depth_mm": depth_mm,
            "depth_confidence": depth_conf,
            "registration_quality_flag": depth_conf,
            "registration_quality_score": float(location.get("lobe_confidence_score", 0.5) or 0.5),
        }
        for comp_vol, comp_label in [(enh_vol, "enhancing"), (edema_vol, "edema"), (nec_vol, "necrotic")]:
            if comp_vol > 0:
                lesion_list.append({
                    "lesion_id": len(lesion_list) + 1,
                    "label": comp_label,
                    "volume_mm3": comp_vol,
                    "max_diameter_mm": max_diam if comp_label == "enhancing" else 0.0,
                    "anatomy": lesion_anatomy,
                })

    midline_ms = _as_midline_dict(mass_effect)

    normalized: dict[str, Any] = {
        "case_id": raw.get("case_id"),
        "total_tumor_volume_mm3": total_vol,
        "volume_mm3": total_vol,
        "max_diameter_mm": max_diam,
        "tumor_count": 1 if total_vol > 0 else 0,
        "segmented_component_count": len(lesion_list),
        "label_volumes_mm3": {
            "enhancing": enh_vol,
            "edema": edema_vol,
            "necrotic": nec_vol,
        },
        "lesion_list": lesion_list,
        "dominant_anatomic_location": dominant_loc,
        "dominant_atlas_confidence": lobe_conf,
        "dominant_registration_quality_flag": depth_conf,
        "dominant_registration_quality_score": float(location.get("lobe_confidence_score", 0.5) or 0.5),
        "location_labels": location_labels,
        "midline_shift": midline_ms,
        "tumor_burden": burden,
        "location": location,
        "mass_effect": mass_effect,
    }

    for k, v in raw.items():
        if k not in normalized:
            normalized[k] = v

    return normalized


def _as_midline_dict(mass_effect: dict[str, Any]) -> dict[str, Any]:
    if not mass_effect:
        return {}
    ms_mm = float(mass_effect.get("midline_shift_mm", 0) or 0)
    nested = mass_effect.get("midline_shift", {})
    if isinstance(nested, dict) and nested:
        return nested
    return {
        "midline_shift_mm": ms_mm,
        "method": "centroid_to_midline",
        "confidence": float(mass_effect.get("midline_shift", {}).get("confidence", 0.5) if isinstance(mass_effect.get("midline_shift"), dict) else 0.5),
    }


def load_case_context(case_id: str, contexts_root: str, runs_root: str | None = None) -> PatientContext:
    root = Path(contexts_root)
    context_dir = root / case_id
    if not context_dir.exists():
        raise FileNotFoundError(f"case not found: {context_dir}")

    context = load_patient_context(case_id, str(root))
    artifact_dir = _artifact_dir(case_id, contexts_root, runs_root)

    findings_structured = _read_json(artifact_dir / "findings_structured.json")
    if findings_structured:
        context.findings_structured = _merge_model(
            context.findings_structured,
            findings_structured,
            FindingsStructured,
        )

    tumor_metrics = _read_json(artifact_dir / "tumor_metrics.json")
    tumor_metrics = _normalize_brats_metrics(tumor_metrics)
    case_index = _load_case_index(case_id)
    tumor_metrics = _merge_case_index_into_metrics(tumor_metrics, case_index)
    if tumor_metrics:
        context.segmentation_metrics = _merge_model(
            context.segmentation_metrics,
            tumor_metrics,
            SegmentationMetrics,
        )

    return context


def list_cases(contexts_root: str, runs_root: str | None = None) -> list[CaseListItem]:
    root = Path(contexts_root)
    if not root.exists():
        return []

    cases: list[CaseListItem] = []
    for case_dir in sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: p.name):
        try:
            context = load_case_context(case_dir.name, str(root), runs_root)
        except Exception:
            continue

        seg = context.segmentation_metrics
        artifact_dir = _artifact_dir(case_dir.name, contexts_root, runs_root)
        cases.append(
            CaseListItem(
                case_id=case_dir.name,
                patient_id=context.patient_id,
                study_id=context.study_id,
                study_date=context.study_date,
                modality=context.modality,
                has_report=(artifact_dir / "report_draft.md").exists(),
                has_quality_report=(artifact_dir / "quality_report.json").exists(),
                has_evidence_index=(artifact_dir / "evidence.json").exists(),
                tumor_count=int(seg.tumor_count if seg else 0),
                segmented_component_count=int(seg.segmented_component_count if seg else 0),
                total_tumor_volume_mm3=float(seg.total_tumor_volume_mm3 if seg else 0.0),
                max_diameter_mm=float(seg.max_diameter_mm if seg else 0.0),
            )
        )
    return cases


def load_case_detail(case_id: str, contexts_root: str, runs_root: str | None = None) -> CaseDetailResponse:
    context = load_case_context(case_id, contexts_root, runs_root)
    case_dir = _artifact_dir(case_id, contexts_root, runs_root)
    qa_rows = _read_jsonl(case_dir / "qa_results.jsonl")

    question_results: list[CaseAnswerSummary] = []
    unsupported_answer_qids: list[int] = []

    for row in qa_rows:
        qid_raw = row.get("qid")
        qid = int(qid_raw) if isinstance(qid_raw, int) else None
        evidence_ids = row.get("evidence_ids", [])
        if not isinstance(evidence_ids, list):
            evidence_ids = []
        answer = str(row.get("answer", "") or "")
        insufficient = _is_insufficient(answer)

        if insufficient:
            evidence_status = "insufficient"
        elif evidence_ids:
            evidence_status = "supported"
        else:
            evidence_status = "missing"
            if qid is not None:
                unsupported_answer_qids.append(qid)

        question_results.append(
            CaseAnswerSummary(
                qid=qid,
                question=str(row.get("question", "") or ""),
                answer=answer,
                evidence_ids=evidence_ids,
                confidence=float(row.get("confidence", 0.0) or 0.0),
                safety_note=str(row.get("safety_note", "") or ""),
                evidence_status=evidence_status,
                insufficient=insufficient,
            )
        )

    total_questions = len(question_results)
    insufficient_count = sum(1 for row in question_results if row.insufficient)
    supported_count = sum(1 for row in question_results if row.evidence_status == "supported")
    evidence_coverage_rate = (supported_count / total_questions) if total_questions else 0.0

    return CaseDetailResponse(
        case_id=case_id,
        context=context,
        report_markdown=_read_text(case_dir / "report_draft.md"),
        quality_report=_read_json(case_dir / "quality_report.json"),
        eval_summary=_read_json(case_dir / "eval_summary.json"),
        findings_structured=_read_json(case_dir / "findings_structured.json"),
        tumor_metrics=_read_json(case_dir / "tumor_metrics.json"),
        question_results=question_results,
        evidence_index=_read_json(case_dir / "evidence.json"),
        quality_panel=CaseQualityPanel(
            total_questions=total_questions,
            insufficient_count=insufficient_count,
            evidence_coverage_rate=round(evidence_coverage_rate, 4),
            unsupported_answer_count=len(unsupported_answer_qids),
            unsupported_answer_qids=unsupported_answer_qids,
        ),
    )
