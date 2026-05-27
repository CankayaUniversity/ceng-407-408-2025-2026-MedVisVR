import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from ai_assistant.api.schemas import DicomSummary, FindingsStructured, NiftiSummary, PatientContext
from ai_assistant.core.paths import to_workspace_relative


def _parse_study_date(raw: str | None) -> str:
    if not raw:
        return "unknown"
    s = raw.strip()
    if re.fullmatch(r"\d{8}", s):
        return f"{s[0:4]}-{s[4:6]}-{s[6:8]}"
    m = re.match(r"(\d{2})-(\d{2})-(\d{4})", s)
    if m:
        mm, dd, yyyy = m.groups()
        return f"{yyyy}-{mm}-{dd}"
    try:
        return datetime.fromisoformat(s).date().isoformat()
    except ValueError:
        return "unknown"


def _first_dicom_file(case_dir: Path) -> Path | None:
    for f in case_dir.rglob("*.dcm"):
        return f
    return None


def _extract_dicom_metadata(first_dcm: Path | None) -> dict[str, Any]:
    if first_dcm is None:
        return {}
    try:
        import pydicom  

        ds = pydicom.dcmread(str(first_dcm), stop_before_pixels=True, force=True)
        return {
            "patient_id": str(getattr(ds, "PatientID", "")) or None,
            "study_id": str(getattr(ds, "StudyID", "")) or None,
            "study_instance_uid": str(getattr(ds, "StudyInstanceUID", "")) or None,
            "study_date": str(getattr(ds, "StudyDate", "")) or None,
            "modality": str(getattr(ds, "Modality", "")) or None,
            "series_description": str(getattr(ds, "SeriesDescription", "")) or None,
        }
    except Exception:
        return {}


def _collect_dicom_summary(case_dir: Path) -> dict[str, Any]:
    dcm_files = list(case_dir.rglob("*.dcm"))
    study_dirs = sorted({p.parent.parent.name for p in dcm_files if len(p.parents) >= 2})
    series_dirs = sorted({p.parent.name for p in dcm_files})
    return {
        "dicom_root": to_workspace_relative(case_dir),
        "dicom_count": len(dcm_files),
        "study_dirs": study_dirs,
        "series_dirs": series_dirs,
    }


def _collect_nifti_summary(case_dir: Path) -> dict[str, Any]:
    if not case_dir.exists():
        return {
            "nifti_root": to_workspace_relative(case_dir),
            "nifti_files": [],
            "nifti_json_sidecars": [],
        }
    nii_files = sorted(
        [to_workspace_relative(p) for p in case_dir.rglob("*.nii")]
        + [to_workspace_relative(p) for p in case_dir.rglob("*.nii.gz")]
    )
    json_files = sorted([to_workspace_relative(p) for p in case_dir.rglob("*.json")])
    return {
        "nifti_root": to_workspace_relative(case_dir),
        "nifti_files": nii_files,
        "nifti_json_sidecars": json_files,
    }


def build_patient_context_from_case(
    patient_id: str,
    dicom_case_dir: str,
    nifti_case_dir: str,
    modality: str = "CT",
) -> PatientContext:
    dicom_dir = Path(dicom_case_dir)
    nifti_dir = Path(nifti_case_dir)

    dicom_summary = _collect_dicom_summary(dicom_dir)
    nifti_summary = _collect_nifti_summary(nifti_dir)
    first_dcm = _first_dicom_file(dicom_dir)
    meta = _extract_dicom_metadata(first_dcm)

    fallback_study_dir = dicom_summary["study_dirs"][0] if dicom_summary["study_dirs"] else patient_id
    study_date = _parse_study_date(meta.get("study_date") or fallback_study_dir)
    study_id = meta.get("study_id") or meta.get("study_instance_uid") or fallback_study_dir
    source_evidence: list[str] = []
    if int(dicom_summary.get("dicom_count", 0) or 0) > 0:
        source_evidence.append("dicom")
    if len(nifti_summary.get("nifti_files", [])) > 0:
        source_evidence.append("nifti")
    if len(nifti_summary.get("nifti_json_sidecars", [])) > 0:
        source_evidence.append("nifti_sidecar")

    return PatientContext(
        patient_id=patient_id,
        study_id=str(study_id),
        study_date=study_date,
        modality=(meta.get("modality") or modality),
        findings_structured=FindingsStructured(
            dicom_summary=DicomSummary(**dicom_summary),
            nifti_summary=NiftiSummary(**nifti_summary),
            source_evidence=source_evidence,
        ),
    )


def save_patient_context(context: PatientContext, out_path: str) -> None:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(context.model_dump(), f, ensure_ascii=True, indent=2)
