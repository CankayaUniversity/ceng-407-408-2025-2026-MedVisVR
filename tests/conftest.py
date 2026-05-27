from unittest.mock import patch

import pytest

from ai_assistant.api.schemas import PatientContext, QARequest


def make_context(
    patient_id="TEST_001",
    study_id="STUDY_001",
    study_date="2025-01-15",
    modality="MR",
    nifti_files=None,
    seg=None,
):
    if nifti_files is None:
        nifti_files = [
            "BraTS20_TEST_001_t1.nii.gz",
            "BraTS20_TEST_001_t1ce.nii.gz",
            "BraTS20_TEST_001_t2.nii.gz",
            "BraTS20_TEST_001_flair.nii.gz",
        ]
    if seg is None:
        seg = {
            "tumor_count": 3,
            "segmented_component_count": 3,
            "total_tumor_volume_mm3": 45000.0,
            "max_diameter_mm": 52.0,
            "image_shape_zyx": [155, 240, 240],
            "label_volumes_mm3": {
                "enhancing": 12000.0,
                "nonenhancing": 18000.0,
                "edema": 15000.0,
            },
            "lesion_list": [
                {"label_name": "enhancing", "volume_mm3": 12000.0, "max_diameter_mm": 52.0, "centroid_xyz_voxel": [72.0, 54.0, 88.0]},
                {"label_name": "nonenhancing", "volume_mm3": 18000.0, "max_diameter_mm": 40.0, "centroid_xyz_voxel": [70.0, 60.0, 84.0]},
                {"label_name": "edema", "volume_mm3": 15000.0, "max_diameter_mm": 48.0, "centroid_xyz_voxel": [78.0, 68.0, 80.0]},
            ],
        }
    return PatientContext(
        patient_id=patient_id,
        study_id=study_id,
        study_date=study_date,
        modality=modality,
        findings_structured={"nifti_summary": {"nifti_files": nifti_files}},
        segmentation_metrics=seg,
    )


def make_qa_request(question, context=None):
    ctx = context or make_context()
    return QARequest(
        patient_id=ctx.patient_id,
        study_id=ctx.study_id,
        question=question,
        context=ctx,
    )


@pytest.fixture(autouse=True)
def disable_llm(monkeypatch):
    monkeypatch.setenv("AI_USE_LLM", "0")
    monkeypatch.setenv("AI_REPORT_USE_LLM", "0")
    monkeypatch.setenv("AI_QA_USE_LLM_NONSTRUCTURED", "0")
    monkeypatch.setenv("AI_USE_ATLAS_LOCALIZATION", "0")
    with (
        patch("ai_assistant.core.local_llm.generate_text", return_value=("", None)),
        patch("ai_assistant.core.model_client.generate_text", return_value=("", None)),
    ):
        yield
