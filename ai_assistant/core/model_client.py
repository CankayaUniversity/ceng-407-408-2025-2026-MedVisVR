import json
import os
import re
from pathlib import Path
from typing import Any

from ai_assistant.core.local_llm import generate_text
from ai_assistant.core.report_descriptor import build_report_descriptor

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


def _load_prompt(filename: str) -> str:
    path = _PROMPTS_DIR / filename
    try:
        return path.read_text(encoding="utf-8").strip()
    except Exception:
        return ""

REQUIRED_REPORT_SECTIONS = [
    "# Clinical Context",
    "# Findings",
    "# Impression",
    "# Limitations",
    "# Suggested Clinical Correlation",
    "# Disclaimer",
]


def _llm_enabled() -> bool:
    if os.getenv("AI_USE_LLM", "1") != "1":
        return False
    provider = os.getenv("AI_LLM_PROVIDER", "embedded_llama").strip().lower()
    return provider in {"embedded_llama", "llama_cpp", "llama.cpp"}


def _model_name(use_high: bool) -> str:
    default_model = os.getenv("AI_MODEL_DEFAULT", "deepseek-r1:14b")
    high_model = os.getenv("AI_MODEL_HIGH", "deepseek-r1:14b")
    return high_model if use_high else default_model


def _report_has_required_sections(text: str) -> bool:
    t = text or ""
    if not all(sec in t for sec in REQUIRED_REPORT_SECTIONS):
        return False
    required_phrases = [
        "histopathological",
        "does not constitute a medical diagnosis",
    ]
    low = t.lower()
    return all(p in low for p in required_phrases)


def _as_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return {}


def _facts_snapshot(context: dict[str, Any]) -> dict[str, Any]:
    ctx = _as_dict(context)
    findings = _as_dict(ctx.get("findings_structured"))
    dicom = _as_dict(findings.get("dicom_summary"))
    nifti = _as_dict(findings.get("nifti_summary"))
    seg = _as_dict(ctx.get("segmentation_metrics"))
    label_volumes = _as_dict(seg.get("label_volumes_mm3"))

    return {
        "patient_id": ctx.get("patient_id"),
        "study_id": ctx.get("study_id"),
        "study_date": ctx.get("study_date"),
        "modality": ctx.get("modality"),
        "dicom_count": int(dicom.get("dicom_count", 0) or 0),
        "dicom_series_count": len(dicom.get("series_dirs", [])),
        "nifti_count": len(nifti.get("nifti_files", [])),
        "tumor_count": int(seg.get("tumor_count", 0) or 0),
        "segmented_component_count": int(seg.get("segmented_component_count", 0) or 0),
        "total_tumor_volume_mm3": float(seg.get("total_tumor_volume_mm3", seg.get("volume_mm3", 0)) or 0),
        "max_diameter_mm": float(seg.get("max_diameter_mm", 0) or 0),
        "enhancing_volume_mm3": float(label_volumes.get("enhancing", 0) or 0),
        "nonenhancing_volume_mm3": float(label_volumes.get("nonenhancing", 0) or 0),
        "edema_volume_mm3": float(label_volumes.get("edema", 0) or 0),
    }


def _audit_gate_snapshot(context: dict[str, Any]) -> dict[str, Any]:
    ctx = _as_dict(context)
    seg = _as_dict(ctx.get("segmentation_metrics"))
    label_volumes = _as_dict(seg.get("label_volumes_mm3"))
    total_tumor_volume_mm3 = float(seg.get("total_tumor_volume_mm3", seg.get("volume_mm3", 0)) or 0)
    label_volume_sum_mm3 = sum(float(v or 0.0) for v in label_volumes.values() if isinstance(v, (int, float)))
    snapshot = {
        "organ": ctx.get("organ") or ctx.get("modality"),
        "source_mask_path": seg.get("source_mask_path"),
        "model_version": seg.get("model_version"),
        "voxel_spacing_xyz_mm": seg.get("voxel_spacing_xyz_mm"),
        "image_shape_zyx": seg.get("image_shape_zyx"),
        "total_tumor_volume_mm3": total_tumor_volume_mm3,
        "label_volume_sum_mm3": round(label_volume_sum_mm3, 4),
        "measurement_uncertainty": _as_dict(seg.get("uncertainty")),
        "audit_gate": _as_dict(seg.get("audit_gate")),
    }
    if not snapshot["audit_gate"] and total_tumor_volume_mm3 > 0:
        snapshot["audit_gate"] = {
            "volume_consistency_delta_mm3": round(total_tumor_volume_mm3 - label_volume_sum_mm3, 4),
            "numbers_are_deterministic": True,
            "narrative_must_not_modify_numeric_truth": True,
        }
    return snapshot


def _truncate_text(value: Any, max_chars: int = 400) -> str:
    text = str(value or "").strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _qa_context_snapshot(context: dict[str, Any]) -> dict[str, Any]:
    ctx = _as_dict(context)
    findings = _as_dict(ctx.get("findings_structured"))
    nifti = _as_dict(findings.get("nifti_summary"))
    seg = _as_dict(ctx.get("segmentation_metrics"))
    lesion_list = seg.get("lesion_list", [])
    if not isinstance(lesion_list, list):
        lesion_list = []

    compact_lesions: list[dict[str, Any]] = []
    for lesion in lesion_list[:3]:
        lesion_dict = _as_dict(lesion)
        anatomy = _as_dict(lesion_dict.get("anatomy"))
        compact_lesions.append(
            {
                "lesion_id": lesion_dict.get("lesion_id"),
                "label_name": lesion_dict.get("label_name"),
                "volume_mm3": lesion_dict.get("volume_mm3"),
                "max_diameter_mm": lesion_dict.get("max_diameter_mm"),
                "anatomic_description": anatomy.get("description"),
                "mapping_basis": anatomy.get("mapping_basis"),
                "atlas_confidence": anatomy.get("atlas_confidence"),
            }
        )

    return {
        "patient_id": ctx.get("patient_id"),
        "study_id": ctx.get("study_id"),
        "study_date": ctx.get("study_date"),
        "modality": ctx.get("modality"),
        "clinical_note": _truncate_text(ctx.get("clinical_note")),
        "prior_reports": [_truncate_text(item, 300) for item in (ctx.get("prior_reports") or [])[:2]],
        "available_nifti_files": nifti.get("nifti_files", []),
        "segmentation_metrics": {
            "tumor_count": seg.get("tumor_count"),
            "segmented_component_count": seg.get("segmented_component_count"),
            "total_tumor_volume_mm3": seg.get("total_tumor_volume_mm3", seg.get("volume_mm3")),
            "max_diameter_mm": seg.get("max_diameter_mm"),
            "dominant_anatomic_location": seg.get("dominant_anatomic_location"),
            "label_volumes_mm3": seg.get("label_volumes_mm3", {}),
            "audit_gate": _as_dict(seg.get("audit_gate")),
            "measurement_uncertainty": _as_dict(seg.get("uncertainty")),
            "lesion_list_top3": compact_lesions,
        },
    }


def _report_context_snapshot(context: dict[str, Any]) -> dict[str, Any]:
    ctx = _as_dict(context)
    descriptor = build_report_descriptor(ctx)
    overview = _as_dict(descriptor.get("case_overview"))
    seg = _as_dict(descriptor.get("segmentation_overview"))
    component = _as_dict(descriptor.get("component_profile"))
    dominant = _as_dict(descriptor.get("dominant_lesion"))
    spatial = _as_dict(descriptor.get("spatial_distribution"))
    semantic = _as_dict(descriptor.get("semantic_segmentation"))
    reporting_cards = _as_dict(descriptor.get("clinical_reporting_cards"))
    hints = _as_dict(descriptor.get("narrative_hints"))
    return {
        "case_overview": overview,
        "segmentation_overview": seg,
        "component_profile": component,
        "dominant_lesion": dominant,
        "spatial_distribution": spatial,
        "semantic_segmentation": semantic,
        "clinical_reporting_cards": reporting_cards,
        "narrative_hints": hints,
        "audit_gate": _audit_gate_snapshot(ctx),
        "clinical_note": _truncate_text(ctx.get("clinical_note")),
        "prior_reports": [_truncate_text(item, 300) for item in (ctx.get("prior_reports") or [])[:2]],
    }


def generate_report_text(
    context: dict[str, Any],
    fallback_report: str,
    report_descriptor: dict[str, Any] | None = None,
    use_high_model: bool = False,
) -> tuple[str, str | None]:
    if not _llm_enabled():
        return fallback_report, None

    model = _model_name(use_high_model)
    facts = _facts_snapshot(context)
    descriptor = report_descriptor or build_report_descriptor(context)
    compact_context = _report_context_snapshot(context)
    audit_gate = _audit_gate_snapshot(context)
    system_prompt = _load_prompt("report_prompt.txt") or (
        "You are a clinical reporting assistant. Respond only in English. "
        "Use only provided facts and context. Do not invent values. "
        "Do not provide definitive diagnosis or treatment recommendation."
    )
    user_prompt = (
        "Generate the report now using EXACT markdown headers in this order:\n"
        "# Clinical Context\n# Findings\n# Impression\n# Limitations\n# Suggested Clinical Correlation\n# Disclaimer\n\n"
        "FACTS_JSON:\n"
        + json.dumps(facts, ensure_ascii=True)
        + "\n\n"
        "REPORT_DESCRIPTOR_JSON:\n"
        + json.dumps(descriptor, ensure_ascii=True)
        + "\n\n"
        "AUDIT_GATE_JSON:\n"
        + json.dumps(audit_gate, ensure_ascii=True)
        + "\n\n"
        "FULL_CONTEXT_JSON:\n"
        + json.dumps(compact_context, ensure_ascii=True)
    )

    try:
        text, model_used = generate_text(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            use_high_model=use_high_model,
        )
        text = _strip_reasoning_blocks(text)
        if text and _report_has_required_sections(text):
            return text, (model_used or model)
        return fallback_report, None
    except Exception:
        return fallback_report, None


def _extract_json_object(text: str) -> dict[str, Any]:
    if not text:
        return {}
    candidate = text.strip()
    try:
        parsed = json.loads(candidate)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        pass
    match = re.search(r"\{.*\}", candidate, flags=re.DOTALL)
    if not match:
        return {}
    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _strip_reasoning_blocks(text: str) -> str:
    cleaned = str(text or "").strip()
    if not cleaned:
        return ""
    cleaned = re.sub(r"<think>.*?</think>\s*", "", cleaned, flags=re.IGNORECASE | re.DOTALL)
    return cleaned.strip()


def _is_safe_narrative_text(text: str) -> bool:
    if not text:
        return False
    if re.search(r'\d', text):
        return False
    low = text.lower()
    forbidden = [
        "glioblastoma",
        "chemotherapy",
        "surgery is recommended immediately",
        "definitely",
        "definitive diagnosis is",
    ]
    return not any(term in low for term in forbidden)


def _is_safe_qa_narrative_text(text: str) -> bool:
    if not text:
        return False
    low = text.lower()
    forbidden = [
        "glioblastoma",
        "chemotherapy",
        "surgery is recommended immediately",
        "definitely",
        "certainly",
    ]
    return not any(term in low for term in forbidden)


def generate_report_section_narratives(
    context: dict[str, Any],
    report_descriptor: dict[str, Any],
    fallback_clinical_context_narrative: str,
    fallback_findings_narrative: str,
    fallback_impression_narrative: str,
    use_high_model: bool = False,
) -> tuple[dict[str, str], str | None]:
    if not _llm_enabled():
        return {
            "clinical_context_narrative": fallback_clinical_context_narrative,
            "findings_narrative": fallback_findings_narrative,
            "impression_narrative": fallback_impression_narrative,
        }, None

    model = _model_name(use_high_model)
    facts = _facts_snapshot(context)
    compact_context = _report_context_snapshot(context)
    audit_gate = _audit_gate_snapshot(context)
    system_prompt = _load_prompt("report_narrative_prompt.txt") or (
        "You generate short English narrative paragraphs for a brain tumor report. "
        "Do not generate any numbers. Return JSON only."
    )
    user_prompt = (
        "Return ONLY valid JSON with keys clinical_context_narrative, findings_narrative, and impression_narrative.\n\n"
        "FACTS_JSON:\n"
        + json.dumps(facts, ensure_ascii=True)
        + "\n\n"
        "REPORT_DESCRIPTOR_JSON:\n"
        + json.dumps(report_descriptor, ensure_ascii=True)
        + "\n\n"
        "AUDIT_GATE_JSON:\n"
        + json.dumps(audit_gate, ensure_ascii=True)
        + "\n\n"
        "FULL_CONTEXT_JSON:\n"
        + json.dumps(compact_context, ensure_ascii=True)
    )

    try:
        text, model_used = generate_text(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            use_high_model=use_high_model,
        )
        text = _strip_reasoning_blocks(text)
    except Exception:
        return {
            "clinical_context_narrative": fallback_clinical_context_narrative,
            "findings_narrative": fallback_findings_narrative,
            "impression_narrative": fallback_impression_narrative,
        }, None

    payload = _extract_json_object(text)
    clinical_context_narrative = str(payload.get("clinical_context_narrative", "") or "").strip()
    findings_narrative = str(payload.get("findings_narrative", "") or "").strip()
    impression_narrative = str(payload.get("impression_narrative", "") or "").strip()

    if not _is_safe_narrative_text(clinical_context_narrative):
        clinical_context_narrative = fallback_clinical_context_narrative
    if not _is_safe_narrative_text(findings_narrative):
        findings_narrative = fallback_findings_narrative
    if not _is_safe_narrative_text(impression_narrative):
        impression_narrative = fallback_impression_narrative

    return {
        "clinical_context_narrative": clinical_context_narrative,
        "findings_narrative": findings_narrative,
        "impression_narrative": impression_narrative,
    }, (model_used or model)


def generate_qa_narrative_text(
    question: str,
    context: dict[str, Any],
    fallback_narrative: str,
    use_high_model: bool = False,
    session_context: str = "",
) -> tuple[str, str | None]:
    if not _llm_enabled():
        return fallback_narrative, None

    model = _model_name(use_high_model)
    facts = _facts_snapshot(context)
    descriptor = build_report_descriptor(context)
    compact_context = _qa_context_snapshot(context)
    audit_gate = _audit_gate_snapshot(context)
    system_prompt = _load_prompt("qa_narrative_prompt.txt") or (
        "You write one short English clinical paragraph from provided case descriptors only. "
        "Do not include numbers, units, dates, counts, or percentages. "
        "Do not provide definitive diagnosis or treatment recommendation."
    )
    user_prompt = (
        f"Question: {question}\n\n"
        + (f"SESSION_CONTEXT:\n{session_context}\n\n" if session_context else "")
        + "FACTS_JSON:\n"
        + json.dumps(facts, ensure_ascii=True)
        + "\n\n"
        + "CASE_DESCRIPTOR_JSON:\n"
        + json.dumps(descriptor, ensure_ascii=True)
        + "\n\n"
        + "AUDIT_GATE_JSON:\n"
        + json.dumps(audit_gate, ensure_ascii=True)
        + "\n\n"
        + "FULL_CONTEXT_JSON:\n"
        + json.dumps(compact_context, ensure_ascii=True)
    )

    try:
        text, model_used = generate_text(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            use_high_model=use_high_model,
        )
        text = _strip_reasoning_blocks(text)
        if text and _is_safe_qa_narrative_text(text):
            return text.strip(), (model_used or model)
        return fallback_narrative, None
    except Exception:
        return fallback_narrative, None


def generate_qa_text(
    question: str,
    context: dict[str, Any],
    fallback_answer: str,
    use_high_model: bool = False,
    session_context: str = "",
) -> tuple[str, str | None]:
    if not _llm_enabled():
        return fallback_answer, None

    model = _model_name(use_high_model)
    facts = _facts_snapshot(context)
    compact_context = _qa_context_snapshot(context)
    descriptor = build_report_descriptor(context)
    audit_gate = _audit_gate_snapshot(context)
    system_prompt = _load_prompt("qa_prompt.txt") or (
        "You are a patient-specific QA assistant. Respond only in English. "
        "Use only provided facts/context and never invent. "
        "If insufficient, answer with 'insufficient evidence'."
    )
    user_prompt = (
        f"Question: {question}\n\n"
        + (f"SESSION_CONTEXT:\n{session_context}\n\n" if session_context else "")
        + "FACTS_JSON:\n"
        + json.dumps(facts, ensure_ascii=True)
        + "\n\n"
        + "CASE_DESCRIPTOR_JSON:\n"
        + json.dumps(descriptor, ensure_ascii=True)
        + "\n\n"
        + "AUDIT_GATE_JSON:\n"
        + json.dumps(audit_gate, ensure_ascii=True)
        + "\n\n"
        + "FULL_CONTEXT_JSON:\n"
        + json.dumps(compact_context, ensure_ascii=True)
    )

    try:
        text, model_used = generate_text(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            use_high_model=use_high_model,
        )
        text = _strip_reasoning_blocks(text)
        if text:
            return text, (model_used or model)
        return fallback_answer, None
    except Exception:
        return fallback_answer, None


def _ct_facts_snapshot(context: dict[str, Any]) -> dict[str, Any]:
    ctx = _as_dict(context)
    findings = _as_dict(ctx.get("findings_structured"))
    nifti = _as_dict(findings.get("nifti_summary"))
    seg = _as_dict(ctx.get("segmentation_metrics"))
    label_volumes = _as_dict(seg.get("label_volumes_mm3"))
    lesion_list = seg.get("lesion_list", [])
    if not isinstance(lesion_list, list):
        lesion_list = []

    compact_lesions = []
    for lesion in lesion_list[:5]:
        ld = _as_dict(lesion)
        anatomy = _as_dict(ld.get("anatomy"))
        compact_lesions.append({
            "lesion_id": ld.get("lesion_id"),
            "label_name": ld.get("label_name"),
            "volume_mm3": ld.get("volume_mm3"),
            "max_diameter_mm": ld.get("max_diameter_mm"),
            "size_category": ld.get("size_category"),
            "anatomic_description": anatomy.get("description"),
            "couinaud_group": anatomy.get("couinaud_group") or anatomy.get("couinaud_estimate"),
            "pulmonary_lobe": anatomy.get("pulmonary_lobe") or anatomy.get("lobe"),
            "side": anatomy.get("side") or anatomy.get("hemisphere"),
            "mapping_basis": anatomy.get("mapping_basis"),
            "atlas_confidence": anatomy.get("atlas_confidence"),
        })

    uncertainty = _as_dict(seg.get("uncertainty"))
    audit_gate = _as_dict(seg.get("audit_gate"))
    vr_readiness = _as_dict(seg.get("vr_readiness"))
    critical_structure_assessment = _as_dict(seg.get("critical_structure_assessment"))
    dominant_lesion = compact_lesions[0] if compact_lesions else {}

    return {
        "patient_id": ctx.get("patient_id"),
        "study_id": ctx.get("study_id"),
        "study_date": ctx.get("study_date"),
        "modality": ctx.get("modality"),
        "organ": ctx.get("organ"),
        "nifti_count": len(nifti.get("nifti_files", [])),
        "tumor_count": int(seg.get("tumor_count", 0) or 0),
        "total_volume_mm3": float(seg.get("total_tumor_volume_mm3", seg.get("volume_mm3", 0)) or 0),
        "total_volume_cm3": round(float(seg.get("total_tumor_volume_mm3", seg.get("volume_mm3", 0)) or 0) / 1000.0, 3),
        "max_diameter_mm": float(seg.get("max_diameter_mm", 0) or 0),
        "label_volumes_mm3": dict(label_volumes),
        "measurement_uncertainty": dict(uncertainty),
        "dominant_anatomic_location": seg.get("dominant_anatomic_location"),
        "dominant_lesion": dominant_lesion,
        "audit_gate": audit_gate,
        "critical_structure_assessment": critical_structure_assessment,
        "vr_readiness": vr_readiness,
        "lesion_list_top5": compact_lesions,
        "clinical_note": _truncate_text(ctx.get("clinical_note")),
    }


CT_REQUIRED_REPORT_SECTIONS = [
    "# Clinical Context",
    "# Findings",
    "# Impression",
    "# Limitations",
    "# Suggested Clinical Correlation",
    "# Disclaimer",
]

CT_REQUIRED_SUBSECTIONS = [
    "## Patient Demographics & Scan Metadata",
    "## Detailed Volumetric Analysis",
    "## Spatial Localization & Critical Structures",
    "## Confidence Metrics & Automated Segmentation Validation",
    "## VR-Readiness Assessment",
]

CT_FORBIDDEN_SPECIFIC_DIAGNOSIS_TERMS = {
    "adenocarcinoma",
    "squamous cell carcinoma",
    "small cell carcinoma",
    "neuroendocrine tumor",
    "hepatocellular carcinoma",
    "cholangiocarcinoma",
    "focal nodular hyperplasia",
    "hepatic adenoma",
    "hemangioma",
}


def _ct_report_has_required_sections(text: str) -> bool:
    t = text or ""
    if not all(sec in t for sec in CT_REQUIRED_REPORT_SECTIONS):
        return False
    if not all(sec in t for sec in CT_REQUIRED_SUBSECTIONS):
        return False
    low = t.lower()
    required_phrases = ["does not constitute a medical diagnosis", "histopathological"]
    return all(p in low for p in required_phrases)


def _ct_report_is_academically_safe(text: str) -> bool:
    low = (text or "").lower()
    if not low:
        return False
    if "suggestive, not diagnostic" not in low:
        return False
    if any(term in low for term in CT_FORBIDDEN_SPECIFIC_DIAGNOSIS_TERMS):
        return False
    return True


def generate_ct_report_text(
    context: dict[str, Any],
    fallback_report: str,
    organ: str = "lung",
    use_high_model: bool = False,
) -> tuple[str, str | None]:
    if not _llm_enabled():
        return fallback_report, None

    model = _model_name(use_high_model)
    facts = _ct_facts_snapshot(context)
    audit_gate = _audit_gate_snapshot(context)
    prompt_file = f"report_ct_{organ}_prompt.txt"
    system_prompt = _load_prompt(prompt_file) or (
        f"You are a clinical reporting assistant for {organ} CT cases. "
        "Respond only in English. Use only provided facts and context. "
        "Do not invent values. Do not provide definitive diagnosis or treatment recommendation."
    )
    user_prompt = (
        "Generate the report now using EXACT markdown headers in this order:\n"
        "# Clinical Context\n# Findings\n# Impression\n# Limitations\n# Suggested Clinical Correlation\n# Disclaimer\n\n"
        "FACTS_JSON:\n"
        + json.dumps(facts, ensure_ascii=True)
        + "\n\n"
        "AUDIT_GATE_JSON:\n"
        + json.dumps(audit_gate, ensure_ascii=True)
        + "\n\n"
        "FULL_CONTEXT_JSON:\n"
        + json.dumps(facts, ensure_ascii=True)
    )

    try:
        text, model_used = generate_text(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            use_high_model=use_high_model,
        )
        text = _strip_reasoning_blocks(text)
        if text and _ct_report_has_required_sections(text) and _ct_report_is_academically_safe(text):
            return text, (model_used or model)
        return fallback_report, None
    except Exception:
        return fallback_report, None


def generate_intent_text(
    question: str,
    system_prompt: str,
) -> tuple[str, str | None]:

    if not _llm_enabled():
        return question, None

    model = _model_name(use_high=False)
    try:
        text, model_used = generate_text(
            system_prompt=system_prompt,
            user_prompt=question,
            use_high_model=False,
        )
        text = _strip_reasoning_blocks(text)
        if text and len(text.strip()) > 0:
            first_line = text.strip().splitlines()[0].strip()
            if len(first_line) >= 5:
                return first_line, (model_used or model)
    except Exception:
        pass
    return question, None


def generate_intent_json(
    question: str,
    session_context: str = "",
    previous_intent: dict[str, Any] | None = None,
    use_high_model: bool = False,
) -> tuple[dict[str, Any], str | None]:
    if not _llm_enabled():
        return {}, None

    model = _model_name(use_high_model)
    system_prompt = (
        "You are an intent arbitration module for clinical brain CT/MRI QA. "
        "Return JSON only. Resolve intent classes and slots without inventing case facts. "
        "Allowed intent_class values: measurement_query, localization_query, mass_effect_query, "
        "critical_proximity_query, differential_request, urgency_assessment, followup_recommendation, "
        "report_summarization, structured_reporting_query, comparison_query, limitation_query, unknown."
    )
    schema_hint = {
        "intent_class": "unknown",
        "target_compartment": None,
        "target_measurement": None,
        "anatomic_target": None,
        "reporting_system": "none",
        "reporting_subtask": None,
        "urgency_signal": "none",
        "temporal_scope": "current_scan",
        "router_confidence": 0.0,
        "explanation": "",
    }
    user_prompt = (
        f"QUESTION:\n{question}\n\n"
        + (f"SESSION_CONTEXT:\n{session_context}\n\n" if session_context else "")
        + f"PREVIOUS_INTENT_JSON:\n{json.dumps(previous_intent or {}, ensure_ascii=True)}\n\n"
        + f"RETURN_SCHEMA_JSON:\n{json.dumps(schema_hint, ensure_ascii=True)}"
    )
    try:
        text, model_used = generate_text(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            use_high_model=use_high_model,
        )
        parsed = _extract_json_object(_strip_reasoning_blocks(text))
        return parsed, (model_used or model) if parsed else (None and None)
    except Exception:
        return {}, None
