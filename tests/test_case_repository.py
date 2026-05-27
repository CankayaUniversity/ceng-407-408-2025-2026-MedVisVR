from pathlib import Path

from ai_assistant.core.case_repository import list_cases, load_case_context, load_case_detail


CONTEXTS_ROOT = str(Path(__file__).resolve().parents[1] / "outputs" / "cases")
RUNS_ROOT = str(Path(__file__).resolve().parents[1] / "outputs" / "runs_first10_reportllm_relaxed_v1")


def test_list_cases_returns_existing_case():
    cases = list_cases(CONTEXTS_ROOT)
    case_ids = {case.case_id for case in cases}
    assert "BraTS20_Validation_001" in case_ids


def test_load_case_detail_includes_quality_panel():
    detail = load_case_detail("BraTS20_Validation_001", CONTEXTS_ROOT, RUNS_ROOT)
    assert detail.case_id == "BraTS20_Validation_001"
    assert detail.quality_panel.total_questions >= 1
    assert detail.quality_panel.evidence_coverage_rate >= 0.0
    assert detail.report_markdown


def test_load_case_context_merges_run_segmentation_metrics():
    context = load_case_context("BraTS20_Validation_001", CONTEXTS_ROOT, RUNS_ROOT)
    assert context.segmentation_metrics is not None
    assert context.segmentation_metrics.total_tumor_volume_mm3 > 0.0
    assert len(context.segmentation_metrics.lesion_list) > 0
