import os
from pathlib import Path

from fastapi import APIRouter, HTTPException

from ai_assistant.api.schemas import ReportFromCaseRequest, ReportRequest, ReportResponse
from ai_assistant.core.case_repository import _artifact_dir, load_case_context
from ai_assistant.core.paths import default_contexts_root, default_runs_root
from ai_assistant.core.report_generator import generate_report_draft

router = APIRouter()


@router.post("/generate", response_model=ReportResponse)
def generate_report(payload: ReportRequest) -> ReportResponse:
    return generate_report_draft(payload.context)


@router.post("/generate-from-case", response_model=ReportResponse)
def generate_report_from_case(payload: ReportFromCaseRequest) -> ReportResponse:
    contexts_root = os.getenv(
        "AI_CONTEXTS_ROOT",
        default_contexts_root(),
    )
    runs_root = os.getenv(
        "AI_RUNS_ROOT",
        default_runs_root(),
    )
    try:
        context = load_case_context(payload.case_id, contexts_root, runs_root)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    result = generate_report_draft(context)

    if result.report_markdown:
        artifact_dir = Path(_artifact_dir(payload.case_id, contexts_root, runs_root))
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (artifact_dir / "report_draft.md").write_text(result.report_markdown, encoding="utf-8")

    return result
