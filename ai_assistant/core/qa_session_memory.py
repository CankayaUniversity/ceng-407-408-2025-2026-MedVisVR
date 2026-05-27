from __future__ import annotations

import os
import threading
import time
from typing import Any


_LOCK = threading.Lock()
_SESSIONS: dict[str, dict[str, Any]] = {}


def _ttl_seconds() -> int:
    return int(os.getenv("AI_QA_SESSION_TTL_S", "1800") or 1800)


def _max_turns() -> int:
    return int(os.getenv("AI_QA_SESSION_MAX_TURNS", "6") or 6)


def purge_expired_sessions() -> None:
    now = time.time()
    ttl = max(60, _ttl_seconds())
    with _LOCK:
        expired = [
            session_id
            for session_id, payload in _SESSIONS.items()
            if (now - float(payload.get("updated_at", 0.0) or 0.0)) > ttl
        ]
        for session_id in expired:
            _SESSIONS.pop(session_id, None)


def clear_session(session_id: str | None) -> bool:
    if not session_id:
        return False
    with _LOCK:
        return _SESSIONS.pop(session_id, None) is not None


def append_turn(
    *,
    session_id: str | None,
    patient_id: str,
    study_id: str,
    original_question: str,
    resolved_question: str,
    answer: str,
    question_class: str,
    question_subtype: str,
    resolved_intent: dict[str, Any] | None = None,
) -> None:
    if not session_id:
        return
    purge_expired_sessions()
    turn = {
        "original_question": str(original_question or "").strip(),
        "resolved_question": str(resolved_question or "").strip(),
        "answer": str(answer or "").strip(),
        "question_class": str(question_class or "").strip(),
        "question_subtype": str(question_subtype or "").strip(),
        "resolved_intent": dict(resolved_intent or {}),
        "timestamp": round(time.time(), 3),
    }
    with _LOCK:
        session = _SESSIONS.setdefault(
            session_id,
            {
                "patient_id": patient_id,
                "study_id": study_id,
                "turns": [],
                "updated_at": time.time(),
            },
        )
        if session.get("patient_id") != patient_id or session.get("study_id") != study_id:
            session = {
                "patient_id": patient_id,
                "study_id": study_id,
                "turns": [],
                "updated_at": time.time(),
            }
            _SESSIONS[session_id] = session
        turns = session.setdefault("turns", [])
        if not isinstance(turns, list):
            turns = []
            session["turns"] = turns
        turns.append(turn)
        session["turns"] = turns[-_max_turns():]
        session["updated_at"] = time.time()


def session_turn_count(session_id: str | None) -> int:
    if not session_id:
        return 0
    purge_expired_sessions()
    with _LOCK:
        turns = (_SESSIONS.get(session_id) or {}).get("turns", [])
        return len(turns) if isinstance(turns, list) else 0


def latest_resolved_intent(session_id: str | None, *, patient_id: str, study_id: str) -> dict[str, Any]:
    if not session_id:
        return {}
    purge_expired_sessions()
    with _LOCK:
        session = _SESSIONS.get(session_id)
        if not isinstance(session, dict):
            return {}
        if session.get("patient_id") != patient_id or session.get("study_id") != study_id:
            return {}
        turns = session.get("turns", [])
        if not isinstance(turns, list) or not turns:
            return {}
        latest = turns[-1]
        if not isinstance(latest, dict):
            return {}
        resolved_intent = latest.get("resolved_intent", {})
        return dict(resolved_intent) if isinstance(resolved_intent, dict) else {}


def build_session_context(session_id: str | None, *, patient_id: str, study_id: str) -> str:
    if not session_id:
        return ""
    purge_expired_sessions()
    with _LOCK:
        session = _SESSIONS.get(session_id)
        if not isinstance(session, dict):
            return ""
        if session.get("patient_id") != patient_id or session.get("study_id") != study_id:
            return ""
        turns = session.get("turns", [])
        if not isinstance(turns, list) or not turns:
            return ""
        lines = ["Recent QA session context:"]
        for idx, turn in enumerate(turns[-_max_turns():], start=1):
            if not isinstance(turn, dict):
                continue
            original_question = str(turn.get("original_question", "") or "").strip()
            resolved_question = str(turn.get("resolved_question", "") or "").strip()
            answer = str(turn.get("answer", "") or "").strip()
            lines.append(f"{idx}. User question: {original_question}")
            if resolved_question and resolved_question != original_question:
                lines.append(f"   Resolved standalone question: {resolved_question}")
            if answer:
                lines.append(f"   Assistant answer: {answer}")
        return "\n".join(lines)
