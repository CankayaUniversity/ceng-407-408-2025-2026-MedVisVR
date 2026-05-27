from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ai_assistant.api.routes_cases import router as cases_router
from ai_assistant.api.routes_ct_cases import router as ct_cases_router
from ai_assistant.api.routes_qa import router as qa_router
from ai_assistant.api.routes_report import router as report_router
from ai_assistant.api.routes_segmentation import router as segmentation_router
from ai_assistant.api.routes_upload import router as upload_router
from ai_assistant.api.routes_viewer import router as viewer_router

app = FastAPI(title="AI Assistant API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(cases_router,       prefix="/cases",        tags=["cases"])
app.include_router(ct_cases_router,    prefix="/ct-cases",     tags=["ct-cases"])
app.include_router(viewer_router,      prefix="/viewer",       tags=["viewer"])
app.include_router(report_router,      prefix="/report",       tags=["report"])
app.include_router(qa_router,          prefix="/qa",           tags=["qa"])
app.include_router(segmentation_router, prefix="/segmentation", tags=["segmentation"])
app.include_router(upload_router,      prefix="/upload",       tags=["upload"])


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


import json
import logging

from ai_assistant.core.model_checker import log_model_status

_startup_logger = logging.getLogger(__name__)


@app.on_event("startup")
async def startup_checks():
    from ai_assistant.core.paths import workspace_root
    ws = workspace_root()
    log_model_status(ws)

    try:
        brain_data_dir = ws / "brain_data"
        output_brain_dir = ws / "output_brain"
        cases_dir = ws / "outputs" / "cases"
        cases_dir.mkdir(parents=True, exist_ok=True)

        if brain_data_dir.exists():
            for case_dir in sorted(brain_data_dir.iterdir()):
                if not case_dir.is_dir():
                    continue
                case_id = case_dir.name
                out_dir = cases_dir / case_id
                if out_dir.exists() and (out_dir / "patient_context.json").exists():
                    continue

                mask_path = output_brain_dir / f"{case_id}.nii.gz"
                if not mask_path.exists():
                    continue

                nifti_files = [str(f.relative_to(ws)).replace("\\", "/") for f in sorted(case_dir.glob("*.nii*"))]
                patient_ctx = {
                    "patient_id": case_id, "study_id": case_id, "study_date": "unknown", "modality": "MR",
                    "clinical_note": None,
                    "findings_structured": {
                        "dicom_summary": {"dicom_root": f"brain_data/{case_id}", "dicom_count": 0, "study_dirs": [], "series_dirs": []},
                        "nifti_summary": {"nifti_root": f"brain_data/{case_id}", "nifti_files": nifti_files, "nifti_json_sidecars": []},
                        "source_evidence": ["nifti"]
                    },
                    "segmentation_metrics": None, "prior_reports": None
                }
                out_dir.mkdir(parents=True, exist_ok=True)
                (out_dir / "patient_context.json").write_text(
                    json.dumps(patient_ctx, indent=2, ensure_ascii=False), encoding="utf-8"
                )
                _startup_logger.info("[auto_discover] Created case entry: %s", case_id)
    except Exception as e:
        _startup_logger.warning("[auto_discover] Brain case discovery failed: %s", e)
