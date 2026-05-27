import os
import uuid
import logging
from pathlib import Path
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter()

WORKSPACE_ROOT = Path(os.environ.get("AI_WORKSPACE_ROOT", os.getcwd()))

_safe_tmp = Path(os.environ.get("TEMP", os.environ.get("TMP", "C:/Temp")))
OUTPUTS_ROOT = _safe_tmp / "carvis_seg"
OUTPUTS_ROOT.mkdir(parents=True, exist_ok=True)


class SegmentationRequest(BaseModel):
    input_path: str
    case_id: str | None = None


class SegmentationResponse(BaseModel):
    job_id: str
    success: bool
    output_file: str | None
    labels: dict
    model: str
    error: str | None


def _make_output_dir(model_name: str, case_id: str | None) -> Path:
    suffix = case_id or str(uuid.uuid4())[:8]
    out = OUTPUTS_ROOT / model_name / suffix
    out.mkdir(parents=True, exist_ok=True)
    return out


def _model_exists(subdir: str) -> bool:
    return (WORKSPACE_ROOT / subdir).exists()


@router.get("/status")
def segmentation_status() -> dict:
    return {
        "lung": {
            "available": _model_exists("nnunet_lung"),
            "version": "nnUNet v2",
            "dataset": "Dataset601_LungTumor",
            "labels": {"0": "Background", "1": "Lung Tumor"},
        },
        "liver": {
            "available": _model_exists("nnunet_liver"),
            "version": "nnUNet v1 (1.7.1)",
            "dataset": "Task003_Liver",
            "labels": {"0": "Background", "1": "Liver", "2": "Liver Tumor"},
        },
        "brain": {
            "available": _model_exists("nnunet_brain"),
            "version": "nnUNet v2",
            "dataset": "nnunet_brain",
            "labels": {"0": "Background", "1": "Brain Tumor"},
        },
    }


@router.post("/lung", response_model=SegmentationResponse)
def segment_lung(req: SegmentationRequest) -> SegmentationResponse:
    if not _model_exists("nnunet_lung"):
        raise HTTPException(status_code=503, detail="Lung model not found")

    from ai_assistant.segmentation.segmentation_lung import run_lung_segmentation

    job_id = str(uuid.uuid4())[:8]
    output_dir = _make_output_dir("lung", req.case_id)

    logger.info("[%s] Lung segmentation starting: %s", job_id, req.input_path)
    result = run_lung_segmentation(req.input_path, str(output_dir))

    return SegmentationResponse(
        job_id=job_id,
        model="lung_nnunetv2",
        **result,
    )


@router.post("/liver", response_model=SegmentationResponse)
def segment_liver(req: SegmentationRequest) -> SegmentationResponse:
    if not _model_exists("nnunet_liver"):
        raise HTTPException(status_code=503, detail="Liver model not found")

    from ai_assistant.segmentation.segmentation_liver import run_liver_segmentation

    job_id = str(uuid.uuid4())[:8]
    output_dir = _make_output_dir("liver", req.case_id)

    logger.info("[%s] Liver segmentation starting: %s", job_id, req.input_path)
    result = run_liver_segmentation(req.input_path, str(output_dir))

    return SegmentationResponse(
        job_id=job_id,
        model="liver_nnunetv1",
        **result,
    )


@router.post("/brain", response_model=SegmentationResponse)
def segment_brain(req: SegmentationRequest) -> SegmentationResponse:
    if not _model_exists("nnunet_brain"):
        raise HTTPException(status_code=503, detail="Brain model not found")

    from ai_assistant.segmentation.segmentation_brain import run_brain_segmentation

    job_id = str(uuid.uuid4())[:8]
    output_dir = _make_output_dir("brain", req.case_id)

    logger.info("[%s] Brain segmentation starting: %s", job_id, req.input_path)
    result = run_brain_segmentation(req.input_path, str(output_dir))

    return SegmentationResponse(
        job_id=job_id,
        model="brain_nnunetv2",
        **result,
    )
