from __future__ import annotations

from typing import Any


def _as_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return {}


def _safe_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except Exception:
        return 0.0


def _safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except Exception:
        return 0


def _pct(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return (numerator / denominator) * 100.0


def _extract_sequences(nifti_files: list[str]) -> list[str]:
    seq_map = [
        ("t1ce", "T1ce"),
        ("flair", "FLAIR"),
        ("t2", "T2"),
        ("t1", "T1"),
    ]
    found: list[str] = []
    for path in nifti_files:
        low = path.lower()
        for key, label in seq_map:
            if key in low and label not in found:
                found.append(label)
    return found


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


def _multifocality_type(lesion_count: int, dominant_pct: float) -> str:
    if lesion_count <= 1:
        return "Unifocal"
    if dominant_pct >= 65.0:
        return "Dominant+satellites"
    return "Multifocal"


def _dominant_component(enhancing_mm3: float, nonenhancing_mm3: float, edema_mm3: float) -> tuple[str, float]:
    parts = [
        ("enhancing", enhancing_mm3),
        ("non-enhancing", nonenhancing_mm3),
        ("edema", edema_mm3),
    ]
    return max(parts, key=lambda item: item[1])


def _clean_region_name(region: str) -> str:
    text = str(region or "").strip()
    if not text:
        return ""
    low = text.lower()
    if "unlabeled standard-space region" in low:
        return ""
    return text


def _critical_structure_notes(regional_distribution: list[dict[str, Any]], anatomy: dict[str, Any]) -> list[str]:
    notes: list[str] = []
    seen: set[str] = set()
    regions = [str(item.get("region", "") or "") for item in regional_distribution[:3]]
    regions.append(str(anatomy.get("description") or ""))
    joined = " ".join(regions).lower()

    mapping = [
        ("frontal", "Approximate frontal-region involvement may place the lesion burden near executive or premotor cortical territory; direct eloquent-cortex distance is not measured."),
        ("precentral", "Approximate precentral-region involvement may place the lesion burden near motor cortex; direct corticospinal tract proximity is not measured."),
        ("temporal", "Approximate temporal-region involvement may place the lesion burden near language or auditory networks depending on dominance; direct language mapping is not available."),
        ("pariet", "Approximate parietal-region involvement may place the lesion burden near sensory-association cortex; direct functional distance is not available."),
        ("occip", "Approximate occipital-region involvement may place the lesion burden near visual cortex; direct optic pathway distance is not measured."),
        ("ventric", "Periventricular or ventricular adjacency may be relevant, but no direct ventricular contact measurement is provided."),
        ("white matter", "Subcortical white-matter overlap is present, but tract-level proximity is not directly quantified."),
    ]
    for key, note in mapping:
        if key in joined and note not in seen:
            notes.append(note)
            seen.add(note)

    if not notes:
        notes.append(
            "Current structured findings do not provide direct distances to eloquent cortex, white-matter tracts, or ventricular structures; only approximate atlas-backed localization is available."
        )
    return notes[:3]


def _vr_readiness_profile(seg: dict[str, Any], lesion_list: list[dict[str, Any]]) -> dict[str, Any]:
    component_count = _safe_int(seg.get("segmented_component_count", len(lesion_list)))
    has_mask = bool(seg.get("source_mask_path"))
    total_mm3 = _safe_float(seg.get("total_tumor_volume_mm3", seg.get("volume_mm3")))
    max_diameter_mm = _safe_float(seg.get("max_diameter_mm"))
    ready = has_mask and component_count > 0 and total_mm3 > 0.0
    status = "Ready with clinician QC" if ready else "Not ready"
    if ready and component_count >= 6:
        status = "Conditionally ready with fragmentation review"
    return {
        "status": status,
        "has_mask": has_mask,
        "mesh_candidate_component_count": component_count,
        "largest_component_diameter_mm": round(max_diameter_mm, 4),
        "note": (
            "Segmentation-derived closed surfaces can be generated for VR review, but mesh quality must be checked for fragmentation and atlas misregistration before planning use."
            if ready
            else "VR mesh export is limited because a valid tumour mask or volumetric burden summary is incomplete."
        ),
    }


def _regional_distribution(lesion_list: list[dict[str, Any]], total_mm3: float) -> list[dict[str, Any]]:
    if total_mm3 <= 0.0:
        return []

    ranked: dict[tuple[str, str], float] = {}
    for lesion in lesion_list:
        lesion_volume = _safe_float(lesion.get("volume_mm3"))
        if lesion_volume <= 0.0:
            continue
        anatomy = _as_dict(lesion.get("anatomy"))
        atlas_regions = anatomy.get("atlas_top_regions", [])
        if not isinstance(atlas_regions, list):
            continue
        for item in atlas_regions:
            item_dict = _as_dict(item)
            region = _clean_region_name(item_dict.get("region"))
            if not region:
                continue
            fraction = _safe_float(item_dict.get("fraction"))
            if fraction <= 0.0:
                continue
            source = str(item_dict.get("source") or anatomy.get("atlas_source") or "unknown")
            key = (region, source)
            ranked[key] = ranked.get(key, 0.0) + (lesion_volume * fraction)

    ordered = sorted(ranked.items(), key=lambda item: item[1], reverse=True)
    top_regions: list[dict[str, Any]] = []
    for (region, source), weighted_mm3 in ordered[:5]:
        top_regions.append(
            {
                "region": region,
                "source": source,
                "estimated_overlap_mm3": round(weighted_mm3, 4),
                "estimated_overlap_pct_of_total_tumor": round(_pct(weighted_mm3, total_mm3), 4),
            }
        )
    return top_regions


def build_report_descriptor(context: dict[str, Any]) -> dict[str, Any]:
    ctx = _as_dict(context)
    findings = _as_dict(ctx.get("findings_structured"))
    dicom = _as_dict(findings.get("dicom_summary"))
    nifti = _as_dict(findings.get("nifti_summary"))
    seg = _as_dict(ctx.get("segmentation_metrics"))
    label_volumes = _as_dict(seg.get("label_volumes_mm3"))

    nifti_files = [str(item) for item in nifti.get("nifti_files", []) if isinstance(item, str)]
    sequences = _extract_sequences(nifti_files)
    lesion_list = [item for item in seg.get("lesion_list", []) if isinstance(item, dict)]
    total_mm3 = _safe_float(seg.get("total_tumor_volume_mm3", seg.get("volume_mm3")))
    total_cm3 = total_mm3 / 1000.0
    lesion_count = _safe_int(seg.get("tumor_count"))
    component_count = _safe_int(seg.get("segmented_component_count", len(lesion_list)))
    max_diameter_mm = _safe_float(seg.get("max_diameter_mm"))
    enhancing_mm3 = _safe_float(label_volumes.get("enhancing"))
    nonenhancing_mm3 = _safe_float(label_volumes.get("nonenhancing"))
    edema_mm3 = _safe_float(label_volumes.get("edema"))
    enh_pct = _pct(enhancing_mm3, total_mm3)
    nonenh_pct = _pct(nonenhancing_mm3, total_mm3)
    edema_pct = _pct(edema_mm3, total_mm3)
    dominant_name, dominant_mm3 = _dominant_component(enhancing_mm3, nonenhancing_mm3, edema_mm3)
    dominant_pct = _pct(dominant_mm3, total_mm3)
    dominant_lesion = max(
        lesion_list,
        key=lambda lesion: _safe_float(lesion.get("volume_mm3")),
        default={},
    )
    anatomy = _as_dict(dominant_lesion.get("anatomy"))

    has_segmentation = bool(
        lesion_list
        or total_mm3 > 0.0
        or max_diameter_mm > 0.0
        or lesion_count > 0
        or component_count > 0
        or seg.get("source_mask_path")
    )

    missing_data_flags: list[str] = []
    if not has_segmentation:
        missing_data_flags.append("missing_segmentation_metrics")
    if not (ctx.get("clinical_note") or "").strip():
        missing_data_flags.append("missing_clinical_note")
    if not ctx.get("prior_reports"):
        missing_data_flags.append("missing_prior_reports")
    if not sequences:
        missing_data_flags.append("missing_sequence_detail")

    heuristic_flags: list[str] = []
    if has_segmentation:
        heuristic_flags.extend(
            [
                "atlas_location_is_approximate",
                "mass_effect_is_not_directly_measured",
                "midline_shift_is_not_directly_measured",
                "segmentation_is_automated",
            ]
        )

    regional_distribution = _regional_distribution(lesion_list, total_mm3)
    has_atlas_regions = bool(regional_distribution)
    critical_structure_notes = _critical_structure_notes(regional_distribution, anatomy)
    vr_readiness = _vr_readiness_profile(seg, lesion_list)
    audit_gate = _as_dict(seg.get("audit_gate"))
    uncertainty = _as_dict(seg.get("uncertainty"))
    semantic_segmentation = [
        {
            "component": "enhancing",
            "description": "Contrast-enhancing segmented component",
            "volume_mm3": round(enhancing_mm3, 4),
            "pct_of_total_tumor": round(enh_pct, 4),
        },
        {
            "component": "non-enhancing",
            "description": "Non-enhancing infiltrative or core-like segmented component",
            "volume_mm3": round(nonenhancing_mm3, 4),
            "pct_of_total_tumor": round(nonenh_pct, 4),
        },
        {
            "component": "edema",
            "description": "Peritumoral edema segmented component",
            "volume_mm3": round(edema_mm3, 4),
            "pct_of_total_tumor": round(edema_pct, 4),
        },
    ]

    return {
        "case_overview": {
            "patient_id": ctx.get("patient_id"),
            "study_id": ctx.get("study_id"),
            "study_date": ctx.get("study_date"),
            "modality": ctx.get("modality"),
            "available_sequences": sequences,
            "dicom_series_count": len(dicom.get("series_dirs", [])) if isinstance(dicom.get("series_dirs", []), list) else 0,
            "nifti_count": len(nifti_files),
            "has_clinical_note": bool((ctx.get("clinical_note") or "").strip()),
            "has_prior_reports": bool(ctx.get("prior_reports")),
        },
        "segmentation_overview": {
            "available": has_segmentation,
            "lesion_count": lesion_count,
            "segmented_component_count": component_count,
            "total_tumor_volume_mm3": round(total_mm3, 4),
            "total_tumor_volume_cm3": round(total_cm3, 4),
            "max_diameter_mm": round(max_diameter_mm, 4),
        },
        "component_profile": {
            "dominant_component": dominant_name,
            "dominant_component_volume_mm3": round(dominant_mm3, 4),
            "dominant_component_pct": round(dominant_pct, 4),
            "enhancing_volume_mm3": round(enhancing_mm3, 4),
            "nonenhancing_volume_mm3": round(nonenhancing_mm3, 4),
            "edema_volume_mm3": round(edema_mm3, 4),
            "enhancing_pct": round(enh_pct, 4),
            "nonenhancing_pct": round(nonenh_pct, 4),
            "edema_pct": round(edema_pct, 4),
            "enhancement_pattern": _enhancement_style(enh_pct),
            "edema_severity": _edema_severity(edema_pct),
            "multifocality_type": _multifocality_type(lesion_count, dominant_pct),
        },
        "dominant_lesion": {
            "lesion_id": dominant_lesion.get("lesion_id"),
            "label_name": dominant_lesion.get("label_name"),
            "volume_mm3": round(_safe_float(dominant_lesion.get("volume_mm3")), 4),
            "max_diameter_mm": round(_safe_float(dominant_lesion.get("max_diameter_mm")), 4),
            "anatomic_location": anatomy.get("description") or seg.get("dominant_anatomic_location"),
            "hemisphere": anatomy.get("hemisphere"),
            "atlas_confidence": anatomy.get("atlas_confidence") or seg.get("dominant_atlas_confidence"),
            "registration_quality_flag": anatomy.get("registration_quality_flag") or seg.get("dominant_registration_quality_flag"),
        },
        "spatial_distribution": {
            "has_atlas_supported_regions": has_atlas_regions,
            "regional_distribution_top5": regional_distribution,
            "distribution_summary": (
                "Atlas-supported regional overlap is available for the dominant segmented burden."
                if has_atlas_regions
                else "Atlas-supported regional overlap is limited or unavailable; localization remains approximate."
            ),
        },
        "critical_structure_assessment": {
            "notes": critical_structure_notes,
            "direct_distance_measurements_available": False,
        },
        "confidence_validation": {
            "atlas_confidence": anatomy.get("atlas_confidence") or seg.get("dominant_atlas_confidence"),
            "registration_quality_flag": anatomy.get("registration_quality_flag") or seg.get("dominant_registration_quality_flag"),
            "registration_quality_score": anatomy.get("registration_quality_score") or seg.get("dominant_registration_quality_score"),
            "measurement_uncertainty": uncertainty,
            "audit_gate": audit_gate,
        },
        "semantic_segmentation": {
            "components": semantic_segmentation,
            "dominant_component_interpretation": dominant_name,
        },
        "vr_readiness": vr_readiness,
        "clinical_reporting_cards": {
            "data_availability": (
                "Imaging sequences and segmentation-derived burden metrics are available."
                if has_segmentation
                else "Segmentation-derived burden metrics are unavailable."
            ),
            "localization_reliability": (
                "Atlas-supported but approximate"
                if has_atlas_regions
                else "Approximate localization only"
            ),
            "quantitative_reliability": "Segmentation-derived measurements should be treated as automated estimates.",
            "vr_readiness": vr_readiness.get("status"),
        },
        "narrative_hints": {
            "missing_data_flags": missing_data_flags,
            "heuristic_flags": heuristic_flags,
            "reporting_style": "natural_clinical_narrative_but_strictly_grounded",
        },
        "safety_constraints": {
            "no_definitive_diagnosis": True,
            "no_treatment_recommendation": True,
            "must_state_histopathology_required": True,
            "must_state_clinical_correlation_required": True,
            "must_not_invent_missing_values": True,
        },
    }
