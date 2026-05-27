import os
import re

from ai_assistant.api.schemas import PatientContext, ReportResponse
from ai_assistant.core.anatomic_mapper import anatomy_from_lesion
from ai_assistant.core.guardrails import (
    confidence_from_context_quality,
    safety_note_for_low_evidence,
    validate_generated_clinical_text,
    validate_patient_context,
)
from ai_assistant.core.model_client import generate_report_section_narratives
from ai_assistant.core.report_descriptor import build_report_descriptor
from ai_assistant.core.uncertainty import add_confidence_intervals


REQUIRED_SECTIONS = [
    "# Clinical Context",
    "# Findings",
    "# Impression",
    "# Limitations",
    "# Suggested Clinical Correlation",
    "# Disclaimer",
]

REQUIRED_FINDINGS_SUBSECTIONS = [
    "## Detailed Volumetric Analysis",
    "## Spatial Localization & Critical Structures",
    "## Confidence Metrics & Automated Segmentation Validation",
    "## VR-Readiness Assessment",
    "## Localization",
    "## Mass Effect",
    "## Midline Shift",
    "## Enhancement Pattern",
    "## Multifocality",
]

FORBIDDEN_REPORT_PATTERNS = [
    "this is definitely glioblastoma",
    "definitive diagnosis is glioblastoma",
    "possible glioblastoma",
    "start chemotherapy immediately",
    "surgery is recommended immediately",
]


def _as_dict(value: object) -> dict:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return {}


def _extract_sequences(nifti_files: list[str]) -> list[str]:
    seq_map = [
        ("t1ce", "T1ce"),
        ("flair", "FLAIR"),
        ("t2", "T2"),
        ("t1", "T1"),
    ]
    found: list[str] = []
    for f in nifti_files:
        low = f.lower()
        for key, label in seq_map:
            if key in low and label not in found:
                found.append(label)
    return found


def _pct(n: float, d: float) -> float:
    if d <= 0:
        return 0.0
    return (n / d) * 100.0


def _component_name(enh_mm3: float, nonenh_mm3: float, edema_mm3: float) -> tuple[str, float]:
    parts = [
        ("enhancing", enh_mm3),
        ("non-enhancing", nonenh_mm3),
        ("edema", edema_mm3),
    ]
    return max(parts, key=lambda item: item[1])


def _enhancement_style(enh_pct: float) -> str:
    if enh_pct >= 60.0:
        return "predominantly enhancing"
    if enh_pct >= 30.0:
        return "mixed enhancing and infiltrative"
    return "predominantly non-enhancing/infiltrative"


def _edema_severity(edema_pct: float) -> str:
    if edema_pct >= 40.0:
        return "substantial edema burden"
    if edema_pct >= 15.0:
        return "moderate edema burden"
    if edema_pct > 0:
        return "limited edema burden"
    return "no meaningful edema burden"


def _segment_metrics(context: PatientContext) -> dict:
    seg = _as_dict(context.segmentation_metrics)
    if not _as_dict(seg.get("uncertainty")):
        seg = add_confidence_intervals(seg)
    return seg


def _has_segmentation_metrics(context: PatientContext) -> bool:
    seg = _as_dict(context.segmentation_metrics)
    if not seg:
        return False
    lesion_list = [item for item in seg.get("lesion_list", []) if isinstance(item, dict)]
    label_volumes = _as_dict(seg.get("label_volumes_mm3"))
    if lesion_list or any(float(value or 0.0) > 0.0 for value in label_volumes.values()):
        return True
    if float(seg.get("total_tumor_volume_mm3", seg.get("volume_mm3", 0.0)) or 0.0) > 0.0:
        return True
    if float(seg.get("max_diameter_mm", 0.0) or 0.0) > 0.0:
        return True
    if int(seg.get("tumor_count", 0) or 0) > 0 or int(seg.get("segmented_component_count", 0) or 0) > 0:
        return True
    return bool(seg.get("source_mask_path") or seg.get("model_version") or seg.get("uncertainty"))


def _lesion_count(seg: dict) -> int:
    return int(seg.get("tumor_count", 0) or 0)


def _component_count(seg: dict) -> int:
    lesion_list = seg.get("lesion_list", [])
    lesion_dicts = [item for item in lesion_list if isinstance(item, dict)]
    return int(seg.get("segmented_component_count", len(lesion_dicts)) or 0)


def _dominant_lesion(seg: dict) -> dict:
    lesion_list = seg.get("lesion_list", [])
    lesion_dicts = [item for item in lesion_list if isinstance(item, dict)]
    if not lesion_dicts:
        return {}
    dominant = max(lesion_dicts, key=lambda lesion: float(lesion.get("volume_mm3", 0.0) or 0.0))
    if not dominant.get("anatomy"):
        image_shape = seg.get("image_shape_zyx", [])
        shape_tuple = tuple(int(v) for v in image_shape) if isinstance(image_shape, list) and len(image_shape) == 3 else None
        dominant["anatomy"] = anatomy_from_lesion(dominant, image_shape_zyx=shape_tuple)
    return dominant


def _largest_lesion_line(seg: dict, total_mm3: float) -> str:
    largest = _dominant_lesion(seg)
    if not largest:
        return "- Largest lesion descriptor: Not available from current segmentation payload."
    burden_pct = _pct(float(largest.get("volume_mm3", 0.0) or 0.0), total_mm3)
    label_name = str(largest.get("label_name", "unknown") or "unknown")
    diameter_mm = float(largest.get("max_diameter_mm", 0) or 0)
    lesion_id = largest.get("lesion_id")
    lesion_prefix = f"Lesion #{lesion_id}" if lesion_id is not None else "Largest lesion"
    anatomy_desc = str((largest.get("anatomy") or {}).get("description") or "anatomic localization unavailable")
    return (
        f"- {lesion_prefix} (dominant component): {label_name} component measuring "
        f"{float(largest.get('volume_mm3', 0.0) or 0.0):.2f} mm3 ({burden_pct:.1f}% of total burden) "
        f"with max diameter {diameter_mm:.2f} mm, centered in {anatomy_desc}."
    )


def _atlas_backed_localization(seg: dict) -> bool:
    basis = str(seg.get("anatomic_mapping_basis", "") or "").lower()
    return basis.startswith("harvard_oxford") or basis.startswith("mni152")


def _registration_summary(seg: dict, anatomy: dict) -> dict:
    summary = _as_dict(seg.get("registration_summary"))
    if summary:
        return summary
    fallback = {
        "quality_flag": anatomy.get("registration_quality_flag"),
        "quality_score": anatomy.get("registration_quality_score"),
        "normalized_cross_correlation": anatomy.get("registration_ncc"),
        "foreground_dice": anatomy.get("registration_foreground_dice"),
        "center_distance_mm": anatomy.get("registration_center_distance_mm"),
    }
    return {k: v for k, v in fallback.items() if v is not None}


def _differential_lines(enh_pct: float, nonenh_pct: float, edema_pct: float) -> list[str]:
    ranked = [
        "1. Infiltrative high-grade glial process is most compatible with the mixed volumetric pattern.",
        "2. Lower-grade glial or treatment-altered infiltrative process is less typical due to the component burden profile.",
        "3. Metastatic or other non-glial pathology cannot be excluded, but the segmented composition is less specific for that pattern alone.",
    ]
    if enh_pct >= 60.0:
        ranked[0] = "1. More solid high-grade enhancing neoplasm is most compatible with the enhancing-dominant pattern."
    elif nonenh_pct >= 70.0:
        ranked[0] = "1. Infiltrative glial process is most compatible with the non-enhancing-dominant pattern."
    if edema_pct >= 40.0:
        ranked[2] = "3. Alternative aggressive pathology with marked vasogenic response cannot be excluded because edema burden is substantial."
    return ranked


def _has_required_sections(text: str) -> bool:
    return all(sec in text for sec in REQUIRED_SECTIONS)


def _has_required_findings_subsections(text: str) -> bool:
    return all(sec in text for sec in REQUIRED_FINDINGS_SUBSECTIONS)


def _contains_forbidden_pattern(text: str) -> bool:
    low = text.lower()
    return any(p in low for p in FORBIDDEN_REPORT_PATTERNS)


def _has_mandatory_disclaimer(text: str) -> bool:
    low = text.lower()
    required_phrases = [
        "does not provide a medical diagnosis",
        "definitive diagnosis requires",
        "histopathological",
        "clinical correlation",
        "qualified radiologist/clinician",
    ]
    return all(p in low for p in required_phrases)


def _replace_impression(text: str, impression_body: str) -> str:
    header = "# Impression"
    next_headers = ["# Limitations", "# Suggested Clinical Correlation", "# Disclaimer"]
    start = text.find(header)
    if start < 0:
        return text
    body_start = start + len(header)
    end = len(text)
    for h in next_headers:
        idx = text.find(h, body_start)
        if idx >= 0:
            end = min(end, idx)
    prefix = text[:body_start]
    suffix = text[end:]
    return prefix + "\n" + impression_body.strip() + "\n\n" + suffix.lstrip("\n")


def _detail_mode_profile(detail_mode: str) -> dict[str, bool]:
    normalized = (detail_mode or "balanced").strip().lower()
    return {
        "name": normalized,
        "include_mass_effect_estimate": normalized != "conservative",
        "include_midline_shift_estimate": normalized != "conservative",
        "include_multifocal_description": normalized != "conservative",
        "include_pattern_detail": normalized != "conservative",
        "include_ci": normalized != "conservative",
        "include_detailed_narrative": normalized == "detailed",
    }


def _mass_effect_severity(total_mm3: float, max_diameter_mm: float) -> str:
    total_cm3 = total_mm3 / 1000.0
    if total_cm3 >= 90.0 or max_diameter_mm >= 70.0:
        return "Severe"
    if total_cm3 >= 40.0 or max_diameter_mm >= 40.0:
        return "Moderate"
    if total_cm3 > 0:
        return "Mild"
    return "Minimal"


def _mass_effect_estimate(severity: str, total_mm3: float) -> str:
    total_cm3 = total_mm3 / 1000.0
    if severity == "Severe":
        return f"Heuristic estimate from lesion burden ({total_cm3:.1f} cm3) suggests clinically relevant regional mass effect risk."
    if severity == "Moderate":
        return f"Heuristic estimate from lesion burden ({total_cm3:.1f} cm3) suggests possible local sulcal and ventricular effacement."
    if severity == "Mild":
        return f"Heuristic estimate from lesion burden ({total_cm3:.1f} cm3) suggests only mild mass effect if present."
    return "Heuristic mass effect estimate is limited because measurable tumor burden is minimal."


def _midline_shift_probability(hemisphere: str, total_mm3: float) -> str:
    if hemisphere == "Unknown":
        return "Indeterminate"
    total_cm3 = total_mm3 / 1000.0
    if total_cm3 >= 90.0:
        return "High"
    if total_cm3 >= 40.0:
        return "Intermediate"
    return "Low"


def _midline_shift_estimate(probability: str, hemisphere: str) -> str:
    if probability == "High":
        return f"Heuristic burden-based estimate suggests elevated risk of midline displacement from the {hemisphere.lower()} hemisphere."
    if probability == "Intermediate":
        return f"Heuristic burden-based estimate suggests possible but unconfirmed midline shift from the {hemisphere.lower()} hemisphere."
    if probability == "Low":
        return "Heuristic burden-based estimate is less suggestive of substantial midline shift."
    return "Midline shift cannot be heuristically estimated from the available payload."


def _enhancement_description(enh_pct: float, component_count: int) -> str:
    if enh_pct >= 60.0:
        return "Predominantly solid enhancing morphology."
    if enh_pct >= 30.0:
        return "Heterogeneous mixed enhancing and infiltrative morphology."
    if component_count >= 5:
        return "Predominantly infiltrative non-enhancing pattern with fragmented satellite components."
    return "Predominantly infiltrative non-enhancing pattern."


def _multifocality_type(lesion_count: int, dominant_pct: float) -> str:
    if lesion_count <= 1:
        return "Unifocal"
    if dominant_pct >= 65.0:
        return "Dominant+satellites"
    return "Multifocal"


def _uncertainty_line(seg: dict) -> str:
    uncertainty = _as_dict(seg.get("uncertainty"))
    total_ci = _as_dict(uncertainty.get("total_tumor_volume_cm3"))
    if not total_ci:
        return "- Confidence: Uncertainty estimate unavailable from current metrics payload."
    confidence_label = uncertainty.get("confidence_label", "Moderate")
    error_pct = float(
        total_ci.get("uncertainty_pct")
        or uncertainty.get("measurement_uncertainty_pct", 0.0)
        or 0.0
    )
    return (
        f"- Confidence: {confidence_label} (measurement uncertainty: +/-{error_pct:.1f}%). "
        f"95% CI for total volume: {float(total_ci.get('lower', 0.0)):.2f}-{float(total_ci.get('upper', 0.0)):.2f} cm3."
    )


def _build_template_impression(
    total_mm3: float,
    max_diameter_mm: float,
    lesion_count: int,
    enh_mm3: float,
    nonenh_mm3: float,
    edema_mm3: float,
    quality_issues: list[str],
    localization_desc: str,
    mass_effect_severity: str,
    midline_shift_probability: str,
    impression_narrative: str,
) -> str:
    enh_pct = _pct(enh_mm3, total_mm3)
    nonenh_pct = _pct(nonenh_mm3, total_mm3)
    edema_pct = _pct(edema_mm3, total_mm3)
    multifocality = _multifocality_type(lesion_count, _pct(max(enh_mm3, nonenh_mm3, edema_mm3), total_mm3))

    summary_block = (
        f"Quantitative support: total segmented burden is {total_mm3 / 1000.0:.1f} cm3 with maximum diameter {max_diameter_mm:.0f} mm. "
        f"Composition is {_enhancement_style(enh_pct)}, with non-enhancing burden {nonenh_pct:.1f}% and edema burden {edema_pct:.1f}%. "
        f"Lesion architecture is best described as {multifocality.lower()}."
    )
    secondary_flags_block = (
        f"Secondary automated flags suggest {mass_effect_severity.lower()} mass effect risk and "
        f"{midline_shift_probability.lower()} midline shift risk, but these were not directly measured."
    )
    differential_block = "Differential considerations:\n" + "\n".join(_differential_lines(enh_pct, nonenh_pct, edema_pct))
    closing_block = (
        "These findings are suggestive, not diagnostic. Definitive diagnosis requires histopathological "
        "confirmation and clinical correlation."
    )
    if quality_issues:
        closing_block += "\nContext quality warnings: " + "; ".join(quality_issues) + "."
    return f"{impression_narrative}\n\n{summary_block}\n{secondary_flags_block}\n\n{differential_block}\n\n{closing_block}"


def _is_report_safe_and_valid(report: str, modality: str = "MR") -> bool:
    if not _has_required_sections(report):
        return False
    if modality.upper() not in ("CT",) and not _has_required_findings_subsections(report):
        return False
    if _contains_forbidden_pattern(report):
        return False
    if not _has_mandatory_disclaimer(report):
        return False
    return True


def _report_llm_enabled() -> bool:
    return os.getenv("AI_USE_LLM", "1") == "1" and os.getenv("AI_REPORT_USE_LLM", "1") == "1"


def _narrative_is_safe(text: str) -> bool:
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


def _soft_repair_report_narrative(text: str, fallback: str) -> str:
    candidate = str(text or "").strip()
    if not candidate:
        return fallback

    repaired = candidate
    replacements = {
        "glioblastoma": "aggressive glial neoplasm pattern",
        "definitely": "",
        "certainly": "",
        "chemotherapy": "definitive management",
        "surgery is recommended immediately": "management requires multidisciplinary review",
        "start treatment immediately": "management should be guided by multidisciplinary review",
        "treatment is recommended immediately": "management should be guided by multidisciplinary review",
    }
    for old, new in replacements.items():
        repaired = re.sub(old, new, repaired, flags=re.IGNORECASE)

    repaired = repaired.replace("  ", " ").strip()
    sentences = re.split(r"(?<=[.!?])\s+", repaired)
    safe_sentences: list[str] = []
    for sentence in sentences:
        s = sentence.strip()
        if not s:
            continue
        low = s.lower()
        if "treatment recommendation" in low:
            continue
        if "medical diagnosis" in low:
            continue
        safe_sentences.append(s)

    repaired = " ".join(safe_sentences).strip()
    repaired = re.sub(r"\s+", " ", repaired).strip()
    if not repaired:
        return fallback
    if re.search(r'\d', repaired):
        return fallback
    if repaired[-1] not in ".!?":
        repaired += "."
    return repaired


def _fallback_clinical_context_narrative(
    modality: str,
    sequences: list[str],
    study_date: str,
    has_clinical_note: bool,
    has_prior_reports: bool,
) -> str:
    sequence_text = ", ".join(sequences) if sequences else "available imaging sequences"
    note_text = (
        "A clinical note is available for contextual correlation."
        if has_clinical_note
        else "No clinical note accompanies the current package."
    )
    prior_text = (
        "Prior comparison material is available for interval context."
        if has_prior_reports
        else "No prior comparison material accompanies the current package."
    )
    date_text = (
        "The study date is unavailable in the current metadata."
        if study_date == "Unknown"
        else f"The study date is documented for this {modality.lower()} case."
    )
    return (
        f"This {modality.lower()} brain tumor review is based on {sequence_text}. "
        f"{date_text} {note_text} {prior_text}"
    )


def _fallback_findings_narrative(
    location_desc: str,
    lesion_count: int,
    component_count: int,
    enh_pct: float,
    edema_pct: float,
    top_regions: list[dict] | None = None,
) -> str:
    lesion_text = "a single lesion group" if lesion_count == 1 else "multiple lesion groups"
    component_text = (
        "with fragmented internal segmentation components"
        if component_count > max(lesion_count, 1)
        else "with internally coherent segmentation structure"
    )
    location_text = (
        f"The dominant pattern localizes to {location_desc}. "
        if location_desc and location_desc != "anatomic localization unavailable"
        else "The dominant lesion location remains approximate. "
    )
    region_text = ""
    valid_regions = [item for item in (top_regions or []) if isinstance(item, dict) and item.get("region")]
    if valid_regions:
        named_regions = ", ".join(str(item.get("region")) for item in valid_regions[:3])
        region_text = f"Approximate regional overlap is concentrated in {named_regions}. "
    return (
        f"{location_text}The segmented appearance is best described as {lesion_text} {component_text}. "
        f"The qualitative burden pattern is {_enhancement_style(enh_pct)} with {_edema_severity(edema_pct)}. "
        f"{region_text}"
        "Any mass effect or midline shift language in this draft should be read as heuristic rather than directly measured."
    )


def _fallback_impression_narrative(
    location_desc: str,
    lesion_count: int,
    enh_pct: float,
    edema_pct: float,
    atlas_backed: bool,
    top_regions: list[dict] | None = None,
) -> str:
    lesion_text = "one dominant lesion pattern" if lesion_count <= 1 else "a dominant lesion with additional satellite groups"
    location_text = (
        f"The overall imaging pattern centers in {location_desc}."
        if location_desc and location_desc != "anatomic localization unavailable"
        else "The overall imaging pattern has only approximate anatomic localization."
    )
    atlas_text = (
        " Localization is atlas-backed but remains approximate."
        if atlas_backed
        else " Localization remains heuristic."
    )
    region_text = ""
    valid_regions = [item for item in (top_regions or []) if isinstance(item, dict) and item.get("region")]
    if valid_regions:
        region_text = " Approximate regional distribution is concentrated in " + ", ".join(
            str(item.get("region")) for item in valid_regions[:2]
        ) + "."
    return (
        f"{location_text} Overall, the segmented appearance is {_enhancement_style(enh_pct)} with {_edema_severity(edema_pct)} "
        f"and is most compatible with {lesion_text}.{atlas_text}{region_text} "
        "These findings remain suggestive rather than diagnostic, and definitive characterization still requires histopathological confirmation with clinical correlation."
    )


def _regional_distribution_lines(top_regions: list[dict]) -> list[str]:
    lines = ["## Regional Distribution"]
    valid_regions = [item for item in top_regions if isinstance(item, dict) and str(item.get("region") or "").strip()]
    if not valid_regions:
        lines.append("- Atlas-supported regional overlap summary: unavailable or not reliable enough for named-region reporting.")
        return lines
    for item in valid_regions[:5]:
        region = str(item.get("region"))
        pct = float(item.get("estimated_overlap_pct_of_total_tumor", 0.0) or 0.0)
        source = str(item.get("source") or "unknown")
        lines.append(
            f"- Approximate atlas-supported overlap in {region}: {pct:.1f}% of the total segmented burden ({source} atlas support)."
        )
    return lines


def _critical_structure_lines(critical_structure_assessment: dict, top_regions: list[dict]) -> list[str]:
    lines = ["## Spatial Localization & Critical Structures"]
    valid_regions = [item for item in top_regions if isinstance(item, dict) and str(item.get("region") or "").strip()]
    if valid_regions:
        lines.append(
            "- Approximate atlas-supported regional burden is concentrated in "
            + ", ".join(str(item.get("region")) for item in valid_regions[:3])
            + "."
        )
    notes = critical_structure_assessment.get("notes", [])
    if isinstance(notes, list):
        for note in notes[:3]:
            lines.append(f"- {str(note)}")
    if not bool(critical_structure_assessment.get("direct_distance_measurements_available")):
        lines.append(
            "- Direct proximity measurements to eloquent cortex, white-matter tracts, ventricles, or vascular structures are not available in the current structured package."
        )
    return lines


def _confidence_validation_lines(seg: dict, anatomy: dict, registration: dict, audit_gate: dict) -> list[str]:
    lines = ["## Confidence Metrics & Automated Segmentation Validation"]
    uncertainty_line = _uncertainty_line(seg)
    lines.append(uncertainty_line)
    atlas_confidence = str(anatomy.get("atlas_confidence") or seg.get("dominant_atlas_confidence") or "unknown")
    registration_flag = str(registration.get("quality_flag") or seg.get("dominant_registration_quality_flag") or "unknown")
    registration_score = float(registration.get("quality_score", seg.get("dominant_registration_quality_score", 0.0)) or 0.0)
    lines.append(f"- Atlas localization confidence: {atlas_confidence.title()}.")
    lines.append(f"- Registration quality: {registration_flag.title()} (score {registration_score:.2f}).")
    volume_delta = float(_as_dict(audit_gate).get("volume_consistency_delta_mm3", 0.0) or 0.0)
    lines.append(f"- Audit gate volume consistency delta: {volume_delta:.2f} mm3.")
    lines.append("- Quantitative metrics remain deterministic and the narrative layer is constrained not to alter audited numeric truth.")
    return lines


def _vr_readiness_lines(vr_readiness: dict) -> list[str]:
    lines = ["## VR-Readiness Assessment"]
    status = str(vr_readiness.get("status") or "Not available")
    component_count = int(vr_readiness.get("mesh_candidate_component_count", 0) or 0)
    largest_component_diameter_mm = float(vr_readiness.get("largest_component_diameter_mm", 0.0) or 0.0)
    lines.append(f"- Mesh readiness status: {status}.")
    lines.append(
        f"- Candidate segmented mesh components: {component_count}; largest component diameter for mesh planning context: {largest_component_diameter_mm:.2f} mm."
    )
    lines.append(f"- {str(vr_readiness.get('note') or 'VR planning suitability is not available from the current payload.')}")
    return lines


def _build_missing_segmentation_report(
    context: PatientContext,
    profile: dict[str, bool],
    study_date: str,
    dicom_series_count: int,
    nifti_count: int,
    sequences: list[str],
) -> str:
    modality_line = f"{context.modality} ({', '.join(sequences)} sequences)" if sequences else f"{context.modality} (sequence detail unavailable)"
    clinical_context_lines = [
        f"- Patient ID: {context.patient_id}",
        f"- Study Date: {study_date}",
        f"- Modality: {modality_line}",
        f"- Available imaging inputs: {dicom_series_count} DICOM series directories and {nifti_count} NIfTI files.",
        f"- Clinical note present: {'yes' if (context.clinical_note or '').strip() else 'no'}; prior reports present: {'yes' if context.prior_reports else 'no'}.",
        f"- Report detail mode: {profile['name']}.",
        "- Purpose: case-level brain tumor imaging assessment with automated segmentation review.",
    ]
    findings_block = "\n".join(
        [
            "- Segmentation-derived tumor metrics are unavailable for this case package.",
            "- Quantitative tumor burden, lesion count, enhancement composition, and diameter cannot be verified.",
            "",
            "## Localization",
            "- Anatomic localization: not available because no segmentation-derived lesion map was provided.",
            "- Laterality: indeterminate from the current structured payload.",
            "",
            "## Mass Effect",
            "- Heuristic estimate: unavailable because lesion burden was not quantified.",
            "",
            "## Midline Shift",
            "- Heuristic estimate: unavailable because lesion burden and laterality were not quantified.",
            "",
            "## Enhancement Pattern",
            "- Description: unavailable because segmentation component volumes were not provided.",
            "",
            "## Multifocality",
            "- Description: unavailable because lesion grouping could not be computed.",
        ]
    )
    return (
        "# Clinical Context\n"
        + "\n".join(clinical_context_lines)
        + "\n\n# Findings\n"
        + findings_block
        + "\n\n# Impression\n"
        + "Segmentation-derived quantitative analysis is unavailable in the current case package.\n\n"
        + "Imaging should be reviewed directly by a qualified radiologist/clinician, and definitive diagnosis still requires histopathological confirmation and clinical correlation."
        + "\n\n# Limitations\n"
        + "- Definitive diagnosis cannot be made from imaging-only automated analysis.\n"
        + "- Segmentation output is missing, so quantitative burden, multiplicity, and pattern-based assessment could not be generated.\n"
        + "- Clinical history and prior comparison remain necessary for interpretation.\n"
        + "\n\n# Suggested Clinical Correlation\n"
        + "1. Regenerate or supply the segmentation output for this case if quantitative review is required.\n"
        + "2. Review the source imaging directly with the treating radiology/clinical team.\n"
        + "3. Obtain tissue diagnosis and clinical correlation when clinically indicated.\n"
        + "\n\n# Disclaimer\n"
        + "This is an automated decision-support draft. All findings require verification by a qualified radiologist/clinician.\n"
        + "It does NOT provide a medical diagnosis. Definitive diagnosis requires histopathological examination and clinical correlation.\n"
        + "This report does not constitute a medical diagnosis or treatment recommendation.\n"
    )


def generate_report_draft(
    context: PatientContext,
    detail_mode: str = "balanced",
    allow_llm: bool = True,
) -> ReportResponse:
    findings = _as_dict(context.findings_structured)
    nifti = _as_dict(findings.get("nifti_summary"))
    dicom = _as_dict(findings.get("dicom_summary"))
    profile = _detail_mode_profile(detail_mode)

    nifti_files = nifti.get("nifti_files", [])
    nifti_files = nifti_files if isinstance(nifti_files, list) else []
    sequences = _extract_sequences([str(x) for x in nifti_files])

    quality_issues = validate_patient_context(context)
    confidence = confidence_from_context_quality(context)
    study_date = context.study_date if context.study_date and context.study_date.lower() != "unknown" else "Unknown"
    dicom_series_count = len(dicom.get("series_dirs", [])) if isinstance(dicom.get("series_dirs", []), list) else 0
    nifti_count = len(nifti_files)

    if not _has_segmentation_metrics(context):
        report = _build_missing_segmentation_report(
            context=context,
            profile=profile,
            study_date=study_date,
            dicom_series_count=dicom_series_count,
            nifti_count=nifti_count,
            sequences=sequences,
        )
        evidence_ids = [
            "ctx:patient_id",
            "ctx:study_date",
            "ctx:modality",
            "ctx:findings_structured.nifti_summary.nifti_files",
        ]
        effective_confidence = min(confidence, 0.4)
        return ReportResponse(
            report_markdown=report,
            evidence_ids=evidence_ids,
            confidence=effective_confidence,
            safety_note=safety_note_for_low_evidence(effective_confidence),
        )

    seg = _segment_metrics(context)
    lesion_count = _lesion_count(seg)
    component_count = _component_count(seg)
    total_mm3 = float(seg.get("total_tumor_volume_mm3", seg.get("volume_mm3", 0)) or 0)
    max_diameter_mm = float(seg.get("max_diameter_mm", 0) or 0)
    label_vol = seg.get("label_volumes_mm3", {})
    label_vol = label_vol if isinstance(label_vol, dict) else {}
    enh_mm3 = float(label_vol.get("enhancing", 0) or 0)
    nonenh_mm3 = float(label_vol.get("nonenhancing", 0) or 0)
    edema_mm3 = float(label_vol.get("edema", 0) or 0)

    enh_pct = _pct(enh_mm3, total_mm3)
    nonenh_pct = _pct(nonenh_mm3, total_mm3)
    edema_pct = _pct(edema_mm3, total_mm3)
    dominant_name, dominant_mm3 = _component_name(enh_mm3, nonenh_mm3, edema_mm3)
    dominant_pct = _pct(dominant_mm3, total_mm3)
    dominant_lesion = _dominant_lesion(seg)
    anatomy = _as_dict(dominant_lesion.get("anatomy"))
    descriptor = build_report_descriptor(context.model_dump())
    spatial_distribution = _as_dict(descriptor.get("spatial_distribution"))
    critical_structure_assessment = _as_dict(descriptor.get("critical_structure_assessment"))
    confidence_validation = _as_dict(descriptor.get("confidence_validation"))
    vr_readiness = _as_dict(descriptor.get("vr_readiness"))
    top_regions = spatial_distribution.get("regional_distribution_top5", [])
    top_regions = top_regions if isinstance(top_regions, list) else []
    registration = _registration_summary(seg, anatomy)
    location_desc = str(anatomy.get("description") or seg.get("dominant_anatomic_location") or "anatomic localization unavailable")
    hemisphere = str(anatomy.get("hemisphere") or "Unknown")
    atlas_confidence = str(anatomy.get("atlas_confidence") or seg.get("dominant_atlas_confidence") or "unknown")
    registration_flag = str(registration.get("quality_flag") or seg.get("dominant_registration_quality_flag") or "unknown")
    registration_score = float(registration.get("quality_score", seg.get("dominant_registration_quality_score", 0.0)) or 0.0)
    mass_effect_severity = _mass_effect_severity(total_mm3, max_diameter_mm)
    midline_shift_probability = _midline_shift_probability(hemisphere, total_mm3)
    enhancement_desc = _enhancement_description(enh_pct, component_count)
    multifocality_type = _multifocality_type(lesion_count, dominant_pct)

    if sequences:
        modality_line = f"{context.modality} ({', '.join(sequences)} sequences)"
    else:
        modality_line = f"{context.modality} (sequence detail unavailable)"

    scan_metadata_lines = [
        f"- Patient ID: {context.patient_id}",
        f"- Study ID: {context.study_id}",
        f"- Study Date: {study_date}",
        f"- Modality: {modality_line}",
        f"- Available imaging inputs: {dicom_series_count} DICOM series directories and {nifti_count} NIfTI files.",
        f"- Clinical note present: {'yes' if (context.clinical_note or '').strip() else 'no'}; prior reports present: {'yes' if context.prior_reports else 'no'}.",
        f"- Report detail mode: {profile['name']}.",
        "- Purpose: case-level brain tumor imaging assessment with automated segmentation review.",
    ]

    total_volume_line = f"- Total segmented tumor burden: {total_mm3 / 1000.0:.2f} cm3 ({total_mm3:.2f} mm3)."
    uncertainty_line = _uncertainty_line(seg)
    if profile["include_ci"]:
        total_volume_line = (
            f"- Total segmented tumor burden: {total_mm3 / 1000.0:.2f} cm3 ({total_mm3:.2f} mm3). "
            + uncertainty_line[2:]
        )

    volumetric_analysis_lines = [
        "## Detailed Volumetric Analysis",
        total_volume_line + f" Maximum segmented diameter: {max_diameter_mm:.2f} mm.",
        f"- Lesion architecture after class merge: {lesion_count} lesion groups across {component_count} segmented components.",
        f"- Component composition: enhancing {enh_mm3 / 1000.0:.2f} cm3 ({enh_pct:.1f}%), non-enhancing {nonenh_mm3 / 1000.0:.2f} cm3 ({nonenh_pct:.1f}%), edema {edema_mm3 / 1000.0:.2f} cm3 ({edema_pct:.1f}%).",
        f"- Dominant segmented component: {dominant_name} burden at {dominant_mm3 / 1000.0:.2f} cm3 ({dominant_pct:.1f}% of total).",
        f"- Imaging phenotype: {_enhancement_style(enh_pct)} with {_edema_severity(edema_pct)}.",
        _largest_lesion_line(seg, total_mm3),
    ]
    if quality_issues:
        volumetric_analysis_lines.append("- Data quality warnings: " + "; ".join(quality_issues))

    localization_lines = [
        "## Localization",
        f"- Anatomic location: {location_desc}.",
        f"- Laterality: {hemisphere}.",
    ]
    if _atlas_backed_localization(seg):
        localization_lines.append(f"- Atlas localization confidence: {atlas_confidence.title()}.")
        localization_lines.append(
            f"- Registration quality: {registration_flag.title()} (score {registration_score:.2f})."
        )
    elif registration_flag != "unknown":
        localization_lines.append(f"- Registration quality: {registration_flag.title()} (score {registration_score:.2f}).")
    if profile["include_detailed_narrative"]:
        localization_lines.append(
            (
                "- Localization note: Dominant component mapping is atlas-backed via MNI152 registration and Harvard-Oxford lookup; registration error remains possible."
                if _atlas_backed_localization(seg)
                else "- Localization note: Dominant component mapping is derived from centroid-space heuristics and should be interpreted alongside source images."
            )
        )
        if registration:
            localization_lines.append(
                "- Registration metrics: "
                f"NCC={float(registration.get('normalized_cross_correlation', 0.0) or 0.0):.2f}, "
                f"foreground overlap={float(registration.get('foreground_dice', 0.0) or 0.0):.2f}, "
                f"center distance={float(registration.get('center_distance_mm', 0.0) or 0.0):.1f} mm."
            )

    mass_effect_lines = [
        "## Mass Effect",
        f"- Secondary automated flag: {mass_effect_severity}.",
        (
            f"- Comment: {_mass_effect_estimate(mass_effect_severity, total_mm3)}"
            if profile["include_mass_effect_estimate"]
            else "- Comment: Not emphasized in conservative mode."
        ),
    ]
    if profile["include_detailed_narrative"]:
        mass_effect_lines.append(
            f"- Quantitative trigger: heuristic classification driven by total burden {total_mm3/1000.0:.1f} cm3 and maximum diameter {max_diameter_mm:.0f} mm."
        )

    midline_shift_lines = [
        "## Midline Shift",
        f"- Secondary automated flag: {midline_shift_probability}.",
        (
            f"- Comment: {_midline_shift_estimate(midline_shift_probability, hemisphere)}"
            if profile["include_midline_shift_estimate"]
            else "- Comment: Not emphasized in conservative mode."
        ),
    ]
    if profile["include_detailed_narrative"]:
        midline_shift_lines.append(
            "- Limitation note: this is a probability estimate from lesion burden and laterality, not a direct midline measurement."
        )

    enhancement_pattern_lines = [
        "## Enhancement Pattern",
        f"- Dominant: {dominant_name.title()}.",
        f"- Percentage: {max(enh_pct, nonenh_pct, edema_pct):.1f}%.",
        (
            f"- Description: {enhancement_desc}"
            if profile["include_pattern_detail"]
            else "- Description: Not emphasized in conservative mode."
        ),
    ]
    if profile["include_detailed_narrative"]:
        enhancement_pattern_lines.append(
            f"- Quantitative context: enhancing={enh_mm3/1000.0:.2f} cm3, non-enhancing={nonenh_mm3/1000.0:.2f} cm3, edema={edema_mm3/1000.0:.2f} cm3."
        )

    multifocality_lines = [
        "## Multifocality",
        f"- Type: {multifocality_type}.",
        f"- Lesion group count: {lesion_count}.",
        f"- Segmented component count: {component_count}.",
        (
            f"- Description: Dominant component accounts for {dominant_pct:.1f}% of total segmented burden."
            if profile["include_multifocal_description"]
            else "- Description: Not emphasized in conservative mode."
        ),
    ]
    if profile["include_detailed_narrative"]:
        multifocality_lines.append(
            "- Interpretation note: component multiplicity may reflect biological satellite spread, segmentation fragmentation, or both."
        )

    findings_block = "\n".join(
        volumetric_analysis_lines
        + [""]
        + _critical_structure_lines(critical_structure_assessment, top_regions)
        + [""]
        + _confidence_validation_lines(seg, anatomy, registration, confidence_validation.get("audit_gate", {}))
        + [""]
        + _vr_readiness_lines(vr_readiness)
        + [""]
        + localization_lines
        + [""]
        + _regional_distribution_lines(top_regions)
        + [""]
        + mass_effect_lines
        + [""]
        + midline_shift_lines
        + [""]
        + enhancement_pattern_lines
        + [""]
        + multifocality_lines
    )

    limitations_lines = [
        "- Definitive diagnosis cannot be made from imaging-only automated analysis.",
        f"- Clinical history availability: {'present' if (context.clinical_note or '').strip() else 'not available in current package'}.",
        f"- Prior imaging/report comparison: {'available' if context.prior_reports else 'not available in current package'}.",
        (
            "- Neuroanatomic localization is derived from MNI152 registration with Harvard-Oxford atlas lookup; "
            f"current registration quality is {registration_flag.lower()} and atlas assignment remains approximate."
            if _atlas_backed_localization(seg)
            else "- Neuroanatomic localization is heuristic and derived from segmentation-space centroids rather than registered atlas coordinates."
        ),
        "- Segmentation is automated and remains subject to thresholding, labeling, and fragmentation artifacts.",
    ]
    if study_date == "Unknown":
        limitations_lines.append("- Study date is unknown, so longitudinal progression timing cannot be assessed reliably.")
    if quality_issues:
        limitations_lines.append("- Context quality warnings: " + "; ".join(quality_issues) + ".")

    recommendation_lines = [
        "1. Obtain tissue diagnosis when clinically feasible for definitive characterization.",
        "2. Integrate histopathology and molecular marker testing (for example IDH and MGMT) with imaging findings.",
        "3. Review the case in multidisciplinary tumor board / neuro-oncology conference.",
        "4. Correlate with neurologic examination, symptoms, and full clinical history.",
    ]
    if not context.prior_reports:
        recommendation_lines.append("5. Add longitudinal comparison imaging when prior studies become available.")
    if not (context.clinical_note or "").strip():
        recommendation_lines.append("6. Incorporate the treating team's clinical note and neurologic examination findings.")

    clinical_context_narrative = _fallback_clinical_context_narrative(
        modality=context.modality,
        sequences=sequences,
        study_date=study_date,
        has_clinical_note=bool((context.clinical_note or "").strip()),
        has_prior_reports=bool(context.prior_reports),
    )
    findings_narrative = _fallback_findings_narrative(
        location_desc=location_desc,
        lesion_count=lesion_count,
        component_count=component_count,
        enh_pct=enh_pct,
        edema_pct=edema_pct,
        top_regions=top_regions,
    )
    impression_narrative = _fallback_impression_narrative(
        location_desc=location_desc,
        lesion_count=lesion_count,
        enh_pct=enh_pct,
        edema_pct=edema_pct,
        atlas_backed=_atlas_backed_localization(seg),
        top_regions=top_regions,
    )

    model_used: str | None = None
    if allow_llm and _report_llm_enabled():
        use_high_model = len(quality_issues) > 0 or component_count > 3
        section_narratives, model_used = generate_report_section_narratives(
            context=context.model_dump(),
            report_descriptor=descriptor,
            fallback_clinical_context_narrative=clinical_context_narrative,
            fallback_findings_narrative=findings_narrative,
            fallback_impression_narrative=impression_narrative,
            use_high_model=use_high_model,
        )
        proposed_context_narrative = section_narratives["clinical_context_narrative"]
        proposed_findings_narrative = section_narratives["findings_narrative"]
        proposed_impression_narrative = section_narratives.get("impression_narrative", "")
        clinical_context_narrative = (
            proposed_context_narrative
            if _narrative_is_safe(proposed_context_narrative)
            else _soft_repair_report_narrative(proposed_context_narrative, clinical_context_narrative)
        )
        findings_narrative = (
            proposed_findings_narrative
            if _narrative_is_safe(proposed_findings_narrative)
            else _soft_repair_report_narrative(proposed_findings_narrative, findings_narrative)
        )
        impression_narrative = (
            proposed_impression_narrative
            if _narrative_is_safe(proposed_impression_narrative)
            else _soft_repair_report_narrative(proposed_impression_narrative, impression_narrative)
        )

    template_impression = _build_template_impression(
        total_mm3=total_mm3,
        max_diameter_mm=max_diameter_mm,
        lesion_count=lesion_count,
        enh_mm3=enh_mm3,
        nonenh_mm3=nonenh_mm3,
        edema_mm3=edema_mm3,
        quality_issues=quality_issues,
        localization_desc=location_desc,
        mass_effect_severity=mass_effect_severity,
        midline_shift_probability=midline_shift_probability,
        impression_narrative=impression_narrative,
    )
    if profile["include_detailed_narrative"]:
        template_impression += (
            "\n\nDetailed narrative synthesis:\n"
            f"- Dominant component localizes to {location_desc}.\n"
            f"- Burden pattern favors {_enhancement_style(enh_pct)} behavior with {_edema_severity(edema_pct)}.\n"
            "- These observations should be integrated with radiologist review rather than treated as standalone diagnosis."
        )

    fallback_report = (
        "# Clinical Context\n"
        + clinical_context_narrative
        + "\n\n## Patient Demographics & Scan Metadata\n"
        + "\n".join(scan_metadata_lines)
        + "\n\n# Findings\n"
        + findings_narrative
        + "\n\n"
        + findings_block
        + "\n\n# Impression\n"
        + template_impression
        + "\n\n# Limitations\n"
        + "\n".join(limitations_lines)
        + "\n\n# Suggested Clinical Correlation\n"
        + "\n".join(recommendation_lines)
        + "\n\n# Disclaimer\n"
        + "This is an automated decision-support draft. All findings require verification by a qualified radiologist/clinician.\n"
        + "It does NOT provide a medical diagnosis. Definitive diagnosis requires histopathological examination and clinical correlation.\n"
        + "This report does not constitute a medical diagnosis or treatment recommendation.\n"
    )

    report = fallback_report
    generated_issues = validate_generated_clinical_text(report, context)
    if not _is_report_safe_and_valid(report, modality=context.modality) or generated_issues:
        report = fallback_report
        model_used = None

    evidence_ids = [
        "ctx:patient_id",
        "ctx:study_date",
        "ctx:modality",
        "ctx:findings_structured.nifti_summary.nifti_files",
        "ctx:segmentation_metrics.tumor_count",
        "ctx:segmentation_metrics.segmented_component_count",
        "ctx:segmentation_metrics.total_tumor_volume_mm3",
        "ctx:segmentation_metrics.max_diameter_mm",
        "ctx:segmentation_metrics.label_volumes_mm3.enhancing",
        "ctx:segmentation_metrics.label_volumes_mm3.nonenhancing",
        "ctx:segmentation_metrics.label_volumes_mm3.edema",
        "ctx:segmentation_metrics.lesion_list",
    ]

    effective_confidence = min(0.99, confidence + (0.02 if model_used else 0.0))
    return ReportResponse(
        report_markdown=report,
        evidence_ids=evidence_ids,
        confidence=effective_confidence,
        safety_note=safety_note_for_low_evidence(effective_confidence),
    )
