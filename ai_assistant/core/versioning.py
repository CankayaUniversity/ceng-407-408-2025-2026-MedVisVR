import hashlib
import json
from pathlib import Path

from ai_assistant.core.paths import workspace_root


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def file_sha256(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _safe_hash(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    return file_sha256(path)


def build_run_fingerprint() -> dict[str, object]:
    root = workspace_root()
    tracked_files = {
        "model_client": root / "ai_assistant" / "core" / "model_client.py",
        "qa_engine": root / "ai_assistant" / "core" / "qa_engine.py",
        "qa_session_memory": root / "ai_assistant" / "core" / "qa_session_memory.py",
        "report_generator": root / "ai_assistant" / "core" / "report_generator.py",
        "report_descriptor": root / "ai_assistant" / "core" / "report_descriptor.py",
        "guardrails": root / "ai_assistant" / "core" / "guardrails.py",
        "schemas": root / "ai_assistant" / "api" / "schemas.py",
        "qa_prompt": root / "ai_assistant" / "prompts" / "qa_prompt.txt",
        "report_prompt": root / "ai_assistant" / "prompts" / "report_prompt.txt",
        "qa_narrative_prompt": root / "ai_assistant" / "prompts" / "qa_narrative_prompt.txt",
        "report_narrative_prompt": root / "ai_assistant" / "prompts" / "report_narrative_prompt.txt",
        "report_ct_lung_prompt": root / "ai_assistant" / "prompts" / "report_ct_lung_prompt.txt",
        "report_ct_liver_prompt": root / "ai_assistant" / "prompts" / "report_ct_liver_prompt.txt",
        "pyproject": root / "pyproject.toml",
    }
    file_hashes = {name: _safe_hash(path) for name, path in tracked_files.items()}
    payload = {
        "workspace": root.name,
        "repo_state": "no-git",
        "file_hashes": file_hashes,
    }
    payload["composite_hash"] = _sha256_bytes(
        json.dumps(payload["file_hashes"], sort_keys=True).encode("utf-8")
    )
    return payload
