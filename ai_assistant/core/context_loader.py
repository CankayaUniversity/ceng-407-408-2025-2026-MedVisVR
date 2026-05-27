import json
from pathlib import Path

from ai_assistant.api.schemas import PatientContext, SegmentationMetrics


def load_patient_context(case_id: str, contexts_root: str) -> PatientContext:
    path = Path(contexts_root) / case_id / "patient_context.json"
    if not path.exists():
        raise FileNotFoundError(f"context file not found: {path}")

    with path.open("r", encoding="utf-8-sig") as f:
        raw = json.load(f)

    context = PatientContext(**raw)

    metrics_path = Path(contexts_root) / case_id / "tumor_metrics.json"
    if metrics_path.exists() and context.segmentation_metrics is None:
        with metrics_path.open("r", encoding="utf-8-sig") as f:
            context.segmentation_metrics = SegmentationMetrics(**json.load(f))

    return context
