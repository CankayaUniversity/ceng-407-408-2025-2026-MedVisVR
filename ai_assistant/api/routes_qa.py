import os

from fastapi import APIRouter, HTTPException

from ai_assistant.api.schemas import (
    QAFromCaseRequest,
    QARequest,
    QAResponse,
    QASessionClearRequest,
    QASessionClearResponse,
)
from ai_assistant.core.case_repository import load_case_context
from ai_assistant.core.paths import default_contexts_root, default_runs_root
from ai_assistant.core.qa_engine import answer_question
from ai_assistant.core.qa_session_memory import clear_session

router = APIRouter()


@router.post("/ask", response_model=QAResponse)
def ask_question(payload: QARequest) -> QAResponse:
    if payload.patient_id != payload.context.patient_id or payload.study_id != payload.context.study_id:
        raise HTTPException(status_code=400, detail="case lock mismatch")
    return answer_question(payload)


@router.post("/ask-from-case", response_model=QAResponse)
def ask_question_from_case(payload: QAFromCaseRequest) -> QAResponse:
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

    request = QARequest(
        patient_id=context.patient_id,
        study_id=context.study_id,
        question=payload.question,
        context=context,
        session_id=payload.session_id,
        reset_session=payload.reset_session,
    )
    return answer_question(request)


@router.post("/session/clear", response_model=QASessionClearResponse)
def clear_qa_session(payload: QASessionClearRequest) -> QASessionClearResponse:
    cleared = clear_session(payload.session_id)
    return QASessionClearResponse(session_id=payload.session_id, cleared=cleared)
