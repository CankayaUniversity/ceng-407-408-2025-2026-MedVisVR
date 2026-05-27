import json
import logging
import shutil
import threading
import uuid
from pathlib import Path

from typing import List

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)
router = APIRouter()

_JOBS: dict[str, dict] = {}
_JOBS_LOCK = threading.Lock()

REPO_ROOT  = None   
SEGS_ROOT  = None
CASES_ROOT = None


def _build_brain_qa(case_id: str, features: dict) -> list[dict]:
    burden  = features.get("tumor_burden", {})
    loc     = features.get("location", {})
    me      = features.get("mass_effect", {})
    labels  = features.get("label_volumes_mm3", {})
    safe    = "For decision support only; clinician confirmation required."

    total_mm3  = float(features.get("total_tumor_volume_mm3", burden.get("total_volume_mm3", 0)) or 0)
    enh_mm3    = float(labels.get("enhancing",  burden.get("enhancing_volume_mm3",  0)) or 0)
    edema_mm3  = float(labels.get("edema",      burden.get("edema_volume_mm3",      0)) or 0)
    nec_mm3    = float(labels.get("necrotic",   burden.get("necrotic_volume_mm3",   0)) or 0)
    max_diam   = float(features.get("max_diameter_mm", burden.get("max_diameter_mm", 0)) or 0)

    hemi      = loc.get("hemisphere", "unknown").title()
    lobe      = loc.get("lobe", "unknown").title()
    depth_cls = loc.get("depth_class", "unknown")
    depth_mm  = float(loc.get("depth_mm", 0) or 0)
    lobe_conf = loc.get("lobe_confidence", "low")

    ms_mm     = float(me.get("midline_shift_mm", 0) or 0)
    ms_nested = me.get("midline_shift", {})
    if isinstance(ms_nested, dict):
        ms_mm = float(ms_nested.get("midline_shift_mm", ms_mm) or ms_mm)
    ventr_comp = me.get("ventricular_compression", False)

    enh_pct   = round(enh_mm3  / total_mm3 * 100, 1) if total_mm3 > 0 else 0.0
    edema_pct = round(edema_mm3 / total_mm3 * 100, 1) if total_mm3 > 0 else 0.0
    nec_pct   = round(nec_mm3  / total_mm3 * 100, 1) if total_mm3 > 0 else 0.0

    dominant_comp = max(
        [("enhancing", enh_mm3), ("edema", edema_mm3), ("necrotic", nec_mm3)],
        key=lambda x: x[1],
    )[0]

    return [
        {
            "qid": 1, "question": "What is the total tumor volume?",
            "answer": (
                f"Total segmented brain tumour volume is {total_mm3:.0f} mm3 ({total_mm3/1000:.2f} cm3). "
                f"Composition: enhancing {enh_mm3/1000:.2f} cm3 ({enh_pct}%), "
                f"edema {edema_mm3/1000:.2f} cm3 ({edema_pct}%), "
                f"necrotic core {nec_mm3/1000:.2f} cm3 ({nec_pct}%)."
            ),
            "evidence_ids": ["ctx:segmentation_metrics.total_tumor_volume_mm3",
                             "ctx:segmentation_metrics.label_volumes_mm3.enhancing"],
            "confidence": 0.74, "safety_note": safe,
        },
        {
            "qid": 2, "question": "What is the maximum tumor diameter?",
            "answer": f"Maximum segmented tumour diameter is {max_diam:.1f} mm.",
            "evidence_ids": ["ctx:segmentation_metrics.max_diameter_mm"],
            "confidence": 0.72, "safety_note": safe,
        },
        {
            "qid": 3, "question": "Where is the tumor located?",
            "answer": (
                f"The dominant tumour localises to the {hemi} {lobe} lobe, {depth_cls} "
                f"({depth_mm:.1f} mm from cortical surface). "
                f"Lobe assignment confidence: {lobe_conf}. "
                f"Localisation is heuristic centroid-based and requires radiologist verification."
            ),
            "evidence_ids": ["ctx:segmentation_metrics.dominant_anatomic_location",
                             "ctx:segmentation_metrics.location_labels"],
            "confidence": 0.58, "safety_note": safe,
        },
        {
            "qid": 4, "question": "What is the dominant enhancement pattern?",
            "answer": (
                f"Dominant tumour component by volume is {dominant_comp}. "
                f"Enhancing fraction: {enh_pct}%, edema fraction: {edema_pct}%, "
                f"necrotic core: {nec_pct}%. "
                f"Pattern is {'predominantly enhancing' if enh_pct > 50 else 'predominantly infiltrative/non-enhancing' if edema_pct + nec_pct > enh_pct else 'mixed'}."
            ),
            "evidence_ids": ["ctx:segmentation_metrics.label_volumes_mm3.enhancing",
                             "ctx:segmentation_metrics.label_volumes_mm3.edema"],
            "confidence": 0.68, "safety_note": safe,
        },
        {
            "qid": 5, "question": "Is there evidence of mass effect or midline shift?",
            "answer": (
                f"Midline shift estimate: {ms_mm:.1f} mm (heuristic centroid-based, not directly measured). "
                f"Ventricular compression: {'suggested' if ventr_comp else 'not suggested'} by automated assessment. "
                f"These are automated heuristic flags and require radiologist confirmation."
            ),
            "evidence_ids": ["ctx:segmentation_metrics.mass_effect"],
            "confidence": 0.55, "safety_note": safe,
        },
        {
            "qid": 6, "question": "How many lesions were detected?",
            "answer": (
                f"Automated segmentation identified 1 dominant tumour region comprising "
                f"{len(features.get('lesion_list', []))} labelled subcomponents (ET, ED, NCR). "
                f"Multifocality assessment requires direct radiological review."
            ),
            "evidence_ids": ["ctx:segmentation_metrics.tumor_count",
                             "ctx:segmentation_metrics.segmented_component_count"],
            "confidence": 0.65, "safety_note": safe,
        },
        {
            "qid": 7, "question": "What imaging modality was used?",
            "answer": "MRI (BraTS multi-sequence: FLAIR, T1, T1ce, T2) was used for imaging.",
            "evidence_ids": ["ctx:modality"],
            "confidence": 0.95, "safety_note": safe,
        },
        {
            "qid": 8, "question": "What clinical correlation is recommended?",
            "answer": (
                "Correlation with neurological examination, full clinical history, and prior imaging is recommended. "
                "MDT review with neuro-oncology, neurosurgery, and radiation oncology is advised. "
                "Tissue sampling for histopathological diagnosis and molecular profiling (IDH, MGMT, 1p/19q) is required for definitive management."
            ),
            "evidence_ids": ["ctx:report_draft"],
            "confidence": 0.70, "safety_note": safe,
        },
        {
            "qid": 9, "question": "What is the edema extent?",
            "answer": (
                f"Peritumoral edema volume is {edema_mm3/1000:.2f} cm3 ({edema_pct}% of total tumour burden). "
                f"{'Moderate' if edema_pct > 20 else 'Limited'} edema burden based on automated segmentation."
            ),
            "evidence_ids": ["ctx:segmentation_metrics.label_volumes_mm3.edema"],
            "confidence": 0.67, "safety_note": safe,
        },
    ]


def _roots():
    global REPO_ROOT, SEGS_ROOT, CASES_ROOT
    if REPO_ROOT is None:
        import os
        REPO_ROOT  = Path(os.environ.get("AI_WORKSPACE_ROOT", Path(__file__).resolve().parents[2]))
        _safe_tmp  = Path(os.environ.get("TEMP", os.environ.get("TMP", "C:/Temp")))
        SEGS_ROOT  = _safe_tmp / "carvis_seg"
        CASES_ROOT = REPO_ROOT / "outputs" / "cases"
    return REPO_ROOT, SEGS_ROOT, CASES_ROOT


def _set_job(job_id: str, **kwargs):
    with _JOBS_LOCK:
        _JOBS[job_id].update(kwargs)


def _compute_seg_quality(mask_path: Path, organ: str) -> dict:
    try:
        import numpy as np
        import nibabel as nib
        img = nib.load(str(mask_path))
        data = np.asarray(img.dataobj, dtype=np.int32)
        voxel_vol = float(np.prod(img.header.get_zooms()[:3]))
        labels = {int(v): int(np.sum(data == v)) for v in np.unique(data) if v != 0}
        total_voxels = sum(labels.values())
        total_vol_ml = total_voxels * voxel_vol / 1000.0

        warnings = []
        if total_voxels == 0:
            warnings.append("No segmentation labels found — mask may be empty")
        if organ == "liver" and total_vol_ml > 5000:
            warnings.append(f"Liver volume {total_vol_ml:.0f}ml exceeds physiological range (>5L)")
        if organ == "liver" and total_vol_ml < 50:
            warnings.append(f"Liver volume {total_vol_ml:.0f}ml suspiciously small (<50ml)")
        if organ == "lung" and total_vol_ml > 3000:
            warnings.append(f"Lung tumor volume {total_vol_ml:.0f}ml exceeds physiological range")

        return {
            "label_voxel_counts": labels,
            "total_segmented_voxels": total_voxels,
            "total_volume_ml": round(total_vol_ml, 2),
            "voxel_volume_mm3": round(voxel_vol, 4),
            "quality_warnings": warnings,
            "quality_pass": len(warnings) == 0,
        }
    except Exception as e:
        return {"error": str(e), "quality_pass": False}


def _run_pipeline(job_id: str, nifti_path: Path, organ: str, case_id: str):
    repo, segs, cases = _roots()

    try:
        _set_job(job_id, progress=5, message="Starting segmentation...")

        seg_out = segs / organ / case_id
        seg_out.mkdir(parents=True, exist_ok=True)

        if organ == "liver":
            from ai_assistant.segmentation.segmentation_liver import run_liver_segmentation
            seg_result = run_liver_segmentation(str(nifti_path), str(seg_out))
        elif organ == "lung":
            from ai_assistant.segmentation.segmentation_lung import run_lung_segmentation
            seg_result = run_lung_segmentation(str(nifti_path), str(seg_out))
        elif organ == "brain":
            case_data_dir = nifti_path.parent
            mask_path_brain = repo / "output_brain" / f"{case_id}.nii.gz"

            if not mask_path_brain.exists():
                from ai_assistant.segmentation.segmentation_brain import run_brain_segmentation
                seg_result = run_brain_segmentation(str(nifti_path), str(seg_out))
                if not seg_result.get("success"):
                    err = seg_result.get("error", "Brain segmentation failed")
                    _set_job(job_id, status="error", progress=0, message=err, error=err)
                    return
                mask_files = list(seg_out.glob("*.nii.gz"))
                if not mask_files:
                    _set_job(job_id, status="error", progress=0, message="Brain mask not found after segmentation", error="mask missing")
                    return
                import shutil
                (repo / "output_brain").mkdir(parents=True, exist_ok=True)
                shutil.copy(str(mask_files[0]), str(mask_path_brain))

            _set_job(job_id, progress=70, message="Extracting brain features...")

            try:
                from ai_assistant.core.brats_feature_extractor import extract_case_features
                features = extract_case_features(case_dir=case_data_dir, mask_path=mask_path_brain)
            except Exception as e:
                features = {"case_id": case_id, "error": str(e)}

            _set_job(job_id, progress=80, message="Building brain report...")

            nifti_files = [str(f.relative_to(repo)).replace("\\", "/") for f in case_data_dir.glob("*.nii*")]
            patient_ctx = {
                "patient_id": case_id, "study_id": case_id, "study_date": "unknown", "modality": "MR",
                "clinical_note": None,
                "findings_structured": {
                    "dicom_summary": {"dicom_root": str(case_data_dir.relative_to(repo)), "dicom_count": 0, "study_dirs": [], "series_dirs": []},
                    "nifti_summary": {"nifti_root": str(case_data_dir.relative_to(repo)), "nifti_files": nifti_files, "nifti_json_sidecars": []},
                    "source_evidence": ["nifti"]
                },
                "segmentation_metrics": None, "prior_reports": None
            }

            if "tumor_burden" in features and "error" not in features:
                from ai_assistant.core.case_repository import _normalize_brats_metrics
                features = _normalize_brats_metrics(features)

            out_dir = cases / case_id
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "patient_context.json").write_text(json.dumps(patient_ctx, indent=2, ensure_ascii=False), encoding="utf-8")
            (out_dir / "tumor_metrics.json").write_text(json.dumps(features, indent=2, ensure_ascii=False), encoding="utf-8")
            (out_dir / "evidence.json").write_text(json.dumps({"case_id": case_id, "sources": ["segmentation_mask", "brats_feature_extractor"]}, indent=2), encoding="utf-8")

            try:
                quality_metrics = _compute_seg_quality(mask_path_brain, "brain")
            except Exception:
                quality_metrics = {"organ": "brain", "error": "quality check skipped"}
            (out_dir / "quality_report.json").write_text(json.dumps(quality_metrics, indent=2, ensure_ascii=False), encoding="utf-8")

            qa_rows = _build_brain_qa(case_id, features)
            (out_dir / "qa_results.jsonl").write_text(
                "\n".join(json.dumps(row, ensure_ascii=False) for row in qa_rows) + "\n",
                encoding="utf-8",
            )

            try:
                from ai_assistant.core.case_repository import load_case_context
                from ai_assistant.core.report_generator import generate_report_draft
                from ai_assistant.core.paths import default_contexts_root, default_runs_root
                ctx = load_case_context(case_id, default_contexts_root(), default_runs_root())
                report_result = generate_report_draft(ctx)
                report = report_result.report_markdown
            except Exception as _rpt_err:
                _burden = features.get("tumor_burden", {})
                _loc    = features.get("location", {})
                report = (
                    f"# {case_id} — Brain Tumor Report\n\n"
                    f"## Findings\n\n"
                    f"- **Total Volume:** {_burden.get('total_volume_mm3',0)/1000:.2f} cm³\n"
                    f"- **Max Diameter:** {_burden.get('max_diameter_mm',0):.2f} mm\n"
                    f"- **Location:** {_loc.get('hemisphere','?').title()} {_loc.get('lobe','?').title()}\n\n"
                    f"## Limitations\n\nAutomated segmentation output. Clinician confirmation required.\n"
                )
            (out_dir / "report_draft.md").write_text(report, encoding="utf-8")

            _set_job(job_id, status="done", progress=100, message="Brain analysis complete.", case_id=case_id, error=None)
            return
        else:
            _set_job(job_id, status="error", progress=0,
                     message=f"Unsupported organ: {organ}", error=f"Unsupported organ: {organ}")
            return

        if not seg_result.get("success"):
            err = seg_result.get("error", "Segmentation failed")
            _set_job(job_id, status="error", progress=0, message=err, error=err)
            return

        _set_job(job_id, progress=70, message="Segmentation complete. Computing metrics...")

        mask_file = seg_result.get("output_file")
        if not mask_file:
            mask_files = list(seg_out.glob("*.nii.gz"))
            mask_file = str(mask_files[0]) if mask_files else None
        if not mask_file or not Path(mask_file).exists():
            err = "Segmentation mask not found after inference"
            _set_job(job_id, status="error", progress=0, message=err, error=err)
            return

        mask_path = Path(mask_file)

        viewer_seg_dir = repo / "outputs" / "segmentations" / organ / case_id
        viewer_seg_dir.mkdir(parents=True, exist_ok=True)
        viewer_mask = viewer_seg_dir / mask_path.name
        try:
            import shutil as _shutil
            _shutil.copy2(str(mask_path), str(viewer_mask))
            logger.info(f"[{job_id}] Mask copied to viewer path: {viewer_mask}")
        except Exception as _cp_err:
            logger.warning(f"[{job_id}] Could not copy mask to viewer path: {_cp_err}")

        _set_job(job_id, progress=72, message="Running quality checks...")
        quality_metrics = _compute_seg_quality(mask_path, organ)

        from ai_assistant.harness.generate_ct_contexts import (
            _compute_ct_metrics,
            _build_patient_context,
            _build_lung_report,
            _build_liver_report,
            _build_lung_qa,
            _build_liver_qa,
            _try_llm_report,
            LUNG_LABELS,
            LIVER_LABELS,
        )

        label_map = LUNG_LABELS if organ == "lung" else LIVER_LABELS
        metrics   = _compute_ct_metrics(mask_path, case_id, label_map, organ)

        nifti_rel = str(nifti_path.relative_to(repo)).replace("\\", "/")
        context   = _build_patient_context(case_id, organ, nifti_rel, metrics)

        _set_job(job_id, progress=80, message="Building report and Q&A...")

        if organ == "lung":
            fallback = _build_lung_report(case_id, metrics)
            qa_rows  = _build_lung_qa(case_id, metrics)
        else:
            fallback = _build_liver_report(case_id, metrics)
            qa_rows  = _build_liver_qa(case_id, metrics)

        report = _try_llm_report(case_id, organ, context, fallback)

        out_dir = cases / case_id
        out_dir.mkdir(parents=True, exist_ok=True)

        (out_dir / "patient_context.json").write_text(
            json.dumps(context, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (out_dir / "tumor_metrics.json").write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (out_dir / "report_draft.md").write_text(report, encoding="utf-8")
        (out_dir / "qa_results.jsonl").write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in qa_rows) + "\n",
            encoding="utf-8",
        )
        (out_dir / "quality_report.json").write_text(
            json.dumps(quality_metrics, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        _set_job(job_id, status="done", progress=100,
                 message="Analysis complete.", case_id=case_id, error=None)
        logger.info(f"[{job_id}] Pipeline complete for {case_id}")

    except Exception as exc:
        logger.exception(f"[{job_id}] Pipeline error")
        _set_job(job_id, status="error", progress=0, message=str(exc), error=str(exc))



@router.post("/infer")
async def upload_and_infer(
    file: UploadFile = File(...),
    organ: str = Form(...),
    source_dir: str = Form(None),
):
    if organ not in ("lung", "liver", "brain"):
        raise HTTPException(status_code=400, detail='organ must be "lung", "liver", or "brain"')

    original_name = file.filename or "upload.nii.gz"
    if not (original_name.endswith(".nii.gz") or original_name.endswith(".nii")):
        raise HTTPException(status_code=400, detail="File must be .nii or .nii.gz")

    repo, _, cases_root = _roots()

    stem = original_name.replace(".nii.gz", "").replace(".nii", "")
    for brats_suf in ("_flair", "_t1ce", "_t1", "_t2"):
        if stem.endswith(brats_suf):
            stem = stem[: -len(brats_suf)]
            break
    case_id = stem

    data_dir = repo / f"{organ}_data" / case_id
    data_dir.mkdir(parents=True, exist_ok=True)
    dest = data_dir / original_name
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    if organ == "brain" and source_dir:
        src_path = Path(source_dir)
        if src_path.exists():
            for nii in src_path.glob("*.nii.gz"):
                target = data_dir / nii.name
                if not target.exists():
                    try:
                        shutil.copy(str(nii), str(target))
                        logger.info(f"Copied companion file: {nii.name}")
                    except Exception as e:
                        logger.warning(f"Could not copy {nii.name}: {e}")
            for nii in src_path.glob("*.nii"):
                if not nii.name.endswith(".nii.gz"):
                    target = data_dir / nii.name
                    if not target.exists():
                        try:
                            shutil.copy(str(nii), str(target))
                        except Exception as e:
                            logger.warning(f"Could not copy {nii.name}: {e}")

    job_id = str(uuid.uuid4())[:12]
    with _JOBS_LOCK:
        _JOBS[job_id] = {
            "status":   "running",
            "progress": 0,
            "message":  "Queued...",
            "case_id":  case_id,
            "error":    None,
        }

    t = threading.Thread(
        target=_run_pipeline,
        args=(job_id, dest, organ, case_id),
        daemon=True,
    )
    t.start()

    return {"job_id": job_id, "case_id": case_id}


@router.post("/infer-brain")
async def upload_brain_multi(
    files: List[UploadFile] = File(...),
):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    for f in files:
        fn = f.filename or ""
        if not (fn.endswith(".nii.gz") or fn.endswith(".nii")):
            raise HTTPException(status_code=400, detail=f"All files must be .nii or .nii.gz (got: {fn})")

    flair_file = next(
        (f for f in files if "_flair" in (f.filename or "").lower()),
        files[0]  
    )
    original_name = flair_file.filename or "upload.nii.gz"

    stem = original_name.replace(".nii.gz", "").replace(".nii", "")
    for brats_suf in ("_flair", "_t1ce", "_t1", "_t2"):
        if stem.lower().endswith(brats_suf):
            stem = stem[: -len(brats_suf)]
            break

    repo, _, cases_root = _roots()
    case_id = stem

    data_dir = repo / "brain_data" / case_id
    data_dir.mkdir(parents=True, exist_ok=True)

    flair_dest = None
    for f in files:
        fname = f.filename or f"file_{uuid.uuid4()}.nii.gz"
        dest = data_dir / fname
        with dest.open("wb") as out:
            shutil.copyfileobj(f.file, out)
        if f is flair_file:
            flair_dest = dest
        logger.info(f"Saved brain file: {fname}")

    job_id = str(uuid.uuid4())[:12]
    with _JOBS_LOCK:
        _JOBS[job_id] = {
            "status":   "running",
            "progress": 0,
            "message":  "Queued…",
            "case_id":  case_id,
            "error":    None,
        }

    threading.Thread(
        target=_run_pipeline,
        args=(job_id, flair_dest, "brain", case_id),
        daemon=True,
    ).start()

    return {"job_id": job_id, "case_id": case_id}


@router.get("/status/{job_id}")
def upload_status(job_id: str):
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job
