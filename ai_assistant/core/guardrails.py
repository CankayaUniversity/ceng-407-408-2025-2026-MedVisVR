from typing import Any


_NON_ENGLISH_FALLBACK = (
    "The response was suppressed because it was not reliably generated in English. "
    "Clinical confirmation is required."
)
_TURKISH_CHARS = set("çğıöşüÇĞİÖŞÜ")
_TURKISH_HINTS = {
    "ve",
    "bir",
    "bu",
    "icin",
    "için",
    "ile",
    "gibi",
    "hasta",
    "bulgu",
    "degil",
    "değil",
    "gerekir",
}
_DISALLOWED_GENERATED_DIAGNOSIS_TERMS = {
    "glioblastoma",
    " gbm ",
}


def _tokenize_alpha(text: str) -> list[str]:
    token: list[str] = []
    out: list[str] = []
    for ch in text.lower():
        if ch.isalpha() or ch in _TURKISH_CHARS:
            token.append(ch)
        elif token:
            out.append("".join(token))
            token = []
    if token:
        out.append("".join(token))
    return out


def enforce_english_only(text: str) -> str:

    return text


def safety_note_for_low_evidence(confidence: float) -> str:
    if confidence < 0.5:
        return "Clinical confirmation required due to insufficient evidence."
    return "For decision support only; clinician confirmation required."


def _as_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return {}


def _canonical_measurement_variants(value: float, unit: str) -> list[str]:
    variants = {
        f"{value:.3f} {unit}".lower(),
        f"{value:.2f} {unit}".lower(),
        f"{value:.1f} {unit}".lower(),
    }
    if abs(value - round(value)) < 1e-9:
        variants.add(f"{round(value):.0f} {unit}".lower())
    return sorted(variants)


def _line_contains_expected_measurement(line: str, value: float, unit: str) -> bool:
    normalized = line.lower().replace(",", "").replace("³", "3")
    return any(token in normalized for token in _canonical_measurement_variants(value, unit))


def _line_mentions_check_keyword(line: str, check: dict[str, Any]) -> bool:
    low_line = line.lower().replace(",", "").replace("Â³", "3")
    if not any(keyword in low_line for keyword in check["keywords"]):
        return False

    if check["id"] == "enhancing_component" and any(
        keyword in low_line for keyword in ["non-enhancing component", "nonenhancing component"]
    ):
        return False

    return True


def _largest_lesion(seg: dict[str, Any]) -> dict[str, Any]:
    lesion_list = seg.get("lesion_list", [])
    if not isinstance(lesion_list, list):
        return {}
    lesion_dicts = [item for item in lesion_list if isinstance(item, dict)]
    if not lesion_dicts:
        return {}
    return max(lesion_dicts, key=lambda item: float(item.get("volume_mm3", 0) or 0))


def validate_generated_clinical_text(text: str, context: Any) -> list[str]:
    issues: list[str] = []
    if not text:
        return issues

    low = f" {text.lower().replace('³', '3')} "
    if any(term in low for term in _DISALLOWED_GENERATED_DIAGNOSIS_TERMS):
        issues.append("disallowed specific diagnosis language")

    ctx = _as_dict(context)
    seg = _as_dict(ctx.get("segmentation_metrics"))
    label_volumes = _as_dict(seg.get("label_volumes_mm3"))
    total_mm3 = float(seg.get("total_tumor_volume_mm3", seg.get("volume_mm3", 0)) or 0)
    enhancing_mm3 = float(label_volumes.get("enhancing", 0) or 0)
    nonenhancing_mm3 = float(label_volumes.get("nonenhancing", 0) or 0)
    edema_mm3 = float(label_volumes.get("edema", 0) or 0)
    max_diameter_mm = float(seg.get("max_diameter_mm", 0) or 0)
    largest = _largest_lesion(seg)
    largest_mm3 = float(largest.get("volume_mm3", 0) or 0)
    largest_diameter_mm = float(largest.get("max_diameter_mm", 0) or 0)

    checks = [
        {
            "id": "total_volume",
            "keywords": ["total segmented tumor volume", "total tumor volume", "total volume", "total segmented tumor burden", "tumor burden"],
            "pairs": [("mm3", total_mm3), ("cm3", total_mm3 / 1000.0)],
        },
        {
            "id": "enhancing_component",
            "keywords": ["enhancing component"],
            "pairs": [("mm3", enhancing_mm3), ("cm3", enhancing_mm3 / 1000.0)],
        },
        {
            "id": "nonenhancing_component",
            "keywords": ["non-enhancing component", "nonenhancing component", "non-enhancing burden", "nonenhancing burden"],
            "pairs": [("mm3", nonenhancing_mm3), ("cm3", nonenhancing_mm3 / 1000.0)],
        },
        {
            "id": "edema_component",
            "keywords": ["edema"],
            "pairs": [("mm3", edema_mm3), ("cm3", edema_mm3 / 1000.0)],
        },
        {
            "id": "maximum_diameter",
            "keywords": ["maximum diameter", "maximum lesion diameter", "max diameter"],
            "pairs": [("mm", max_diameter_mm)],
        },
        {
            "id": "largest_lesion",
            "keywords": ["largest segmented component", "largest lesion"],
            "pairs": [("mm3", largest_mm3), ("cm3", largest_mm3 / 1000.0), ("mm", largest_diameter_mm)],
        },
    ]

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        low_line = line.lower().replace(",", "").replace("³", "3")
        for check in checks:
            if not _line_mentions_check_keyword(line, check):
                continue
            for unit, value in check["pairs"]:
                if f" {unit}" not in low_line:
                    continue
                if value <= 0 and unit in {"mm3", "cm3"}:
                    continue
                if not _line_contains_expected_measurement(line, value, unit):
                    issues.append(f"inconsistent {check['id']} {unit} value")
                    break

    return issues


def validate_patient_context(context: Any) -> list[str]:
    issues: list[str] = []

    if not getattr(context, "patient_id", None):
        issues.append("missing patient_id")
    if not getattr(context, "study_id", None):
        issues.append("missing study_id")
    if not getattr(context, "study_date", None):
        issues.append("missing study_date")

    modality = (getattr(context, "modality", "") or "").upper()
    if modality not in {"CT", "MR"}:
        issues.append("invalid modality (expected CT or MR)")

    findings = _as_dict(getattr(context, "findings_structured", None))
    dicom = _as_dict(findings.get("dicom_summary"))
    nifti = _as_dict(findings.get("nifti_summary"))

    dicom_count = int(dicom.get("dicom_count", 0) or 0)
    _nifti_check = nifti.get("nifti_files", []) if isinstance(nifti, dict) else []
    if modality == "CT" and dicom_count <= 0 and len(_nifti_check) == 0:
        issues.append("no dicom files detected")

    nifti_files = nifti.get("nifti_files", []) if isinstance(nifti, dict) else []
    if not isinstance(nifti_files, list):
        nifti_files = []
    if len(nifti_files) == 0:
        issues.append("no nifti files detected")

    return issues


def confidence_from_context_quality(context: Any) -> float:
    findings = _as_dict(getattr(context, "findings_structured", None))
    dicom = _as_dict(findings.get("dicom_summary"))
    nifti = _as_dict(findings.get("nifti_summary"))
    seg = _as_dict(getattr(context, "segmentation_metrics", None))
    modality = (getattr(context, "modality", "") or "").upper()

    score = 0.30
    if modality == "CT" and int(dicom.get("dicom_count", 0) or 0) > 0:
        score += 0.15
    if len(nifti.get("nifti_files", []) if isinstance(nifti, dict) else []) > 0:
        score += 0.20
    if int(seg.get("tumor_count", 0) or 0) > 0:
        score += 0.10
    if modality in {"CT", "MR"}:
        score += 0.05
    if getattr(context, "study_date", None):
        sd = str(getattr(context, "study_date", "")).strip().lower()
        if sd and sd != "unknown":
            score += 0.05

    return min(0.75, round(score, 2))
