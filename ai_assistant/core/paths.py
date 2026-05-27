import os
from pathlib import Path


def workspace_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_contexts_root() -> str:
    return os.getenv("AI_CONTEXTS_ROOT", str(workspace_root() / "outputs" / "cases"))


def default_runs_root() -> str:
    configured = os.getenv("AI_RUNS_ROOT")
    if configured:
        return configured

    root = workspace_root() / "outputs"
    preferred = root / "runs_first10_reportllm_relaxed_v1"
    if preferred.exists():
        return str(preferred)
    return str(root / "runs")


def default_metrics_out() -> str:
    return os.getenv("AI_METRICS_OUT", str(workspace_root() / "outputs" / "metrics_summary.json"))


def default_segmentation_root() -> str:
    return os.getenv("AI_SEG_ROOT", str(workspace_root() / "output_brain"))


def to_workspace_relative(path_like: str | Path) -> str:
    p = Path(path_like)
    try:
        rel = p.resolve(strict=False).relative_to(workspace_root().resolve(strict=False))
        return rel.as_posix()
    except Exception:
        return p.resolve(strict=False).as_posix()
