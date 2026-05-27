import os
import shutil
from pathlib import Path

from fastapi import APIRouter, HTTPException

from ai_assistant.api.schemas import CaseDetailResponse, CaseListItem
from ai_assistant.core.case_repository import list_cases, load_case_detail
from ai_assistant.core.paths import default_contexts_root, default_runs_root

router = APIRouter()


@router.get("", response_model=list[CaseListItem])
def get_cases() -> list[CaseListItem]:
    contexts_root = os.getenv("AI_CONTEXTS_ROOT", default_contexts_root())
    runs_root = os.getenv("AI_RUNS_ROOT", default_runs_root())
    all_cases = list_cases(contexts_root, runs_root)
    return [c for c in all_cases if (c.modality or "").upper() == "MR"]


@router.delete("/{case_id}")
def delete_case(case_id: str) -> dict:
    ws = Path(os.environ.get("AI_WORKSPACE_ROOT", Path(__file__).resolve().parents[2]))
    contexts_root = Path(os.getenv("AI_CONTEXTS_ROOT", default_contexts_root()))
    runs_root     = Path(os.getenv("AI_RUNS_ROOT",     default_runs_root()))

    deleted = []

    for root in [contexts_root, runs_root]:
        p = root / case_id
        if p.exists():
            shutil.rmtree(p, ignore_errors=True)
            deleted.append(str(p))

    for folder in ["brain_data", "lung_data", "liver_data"]:
        p = ws / folder / case_id
        if p.exists():
            shutil.rmtree(p, ignore_errors=True)
            deleted.append(str(p))

    for organ in ["brain", "lung", "liver"]:
        p = ws / "outputs" / "segmentations" / organ / case_id
        if p.exists():
            shutil.rmtree(p, ignore_errors=True)
            deleted.append(str(p))

    if not deleted:
        raise HTTPException(status_code=404, detail=f"Case '{case_id}' not found")

    return {"deleted": deleted}


@router.get("/{case_id}", response_model=CaseDetailResponse)
def get_case_detail(case_id: str) -> CaseDetailResponse:
    contexts_root = os.getenv("AI_CONTEXTS_ROOT", default_contexts_root())
    runs_root = os.getenv("AI_RUNS_ROOT", default_runs_root())
    try:
        return load_case_detail(case_id, contexts_root, runs_root)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
