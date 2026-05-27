import json
from pathlib import Path

import pytest

from ai_assistant.core.context_builder import build_patient_context_from_case, save_patient_context
from ai_assistant.core.versioning import build_run_fingerprint
from ai_assistant.harness.run_case import run_case

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
BRAIN_ROOT = WORKSPACE_ROOT / "brain_data"
SEG_ROOT = WORKSPACE_ROOT / "output_brain"

REQUIRED_REPORT_SECTIONS = [
    "# Clinical Context",
    "# Findings",
    "# Impression",
    "# Limitations",
    "# Suggested Clinical Correlation",
    "# Disclaimer",
]

GOLDEN_CASES = ["BraTS20_Validation_001", "BraTS20_Validation_005"]


@pytest.fixture(scope="module")
def regenerated_runs(tmp_path_factory):
    root = tmp_path_factory.mktemp("golden_regen")
    contexts_root = root / "cases"
    runs_root = root / "runs"

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setenv("AI_USE_LLM", "0")
    monkeypatch.setenv("AI_REPORT_USE_LLM", "0")
    monkeypatch.setenv("AI_QA_USE_LLM_NONSTRUCTURED", "0")

    try:
        for case_id in GOLDEN_CASES:
            context = build_patient_context_from_case(
                patient_id=case_id,
                dicom_case_dir=str(BRAIN_ROOT / case_id),
                nifti_case_dir=str(BRAIN_ROOT / case_id),
                modality="MR",
            )
            save_patient_context(context, str(contexts_root / case_id / "patient_context.json"))
            run_case(
                case_id=case_id,
                contexts_root=str(contexts_root),
                out_root=str(runs_root),
                seg_root=str(SEG_ROOT),
            )
        yield runs_root
    finally:
        monkeypatch.undo()


class TestGoldenCaseOutputs:
    def _case_dir(self, runs_root: Path, case_id: str) -> Path:
        return runs_root / case_id

    def test_all_artifacts_exist(self, regenerated_runs: Path):
        for case_id in GOLDEN_CASES:
            d = self._case_dir(regenerated_runs, case_id)
            assert (d / "report_draft.md").exists(), f"{case_id}: missing report_draft.md"
            assert (d / "qa_results.jsonl").exists(), f"{case_id}: missing qa_results.jsonl"
            assert (d / "evidence.json").exists(), f"{case_id}: missing evidence.json"
            assert (d / "run_meta.json").exists(), f"{case_id}: missing run_meta.json"

    def test_report_has_all_sections(self, regenerated_runs: Path):
        for case_id in GOLDEN_CASES:
            report = (self._case_dir(regenerated_runs, case_id) / "report_draft.md").read_text(encoding="utf-8")
            for sec in REQUIRED_REPORT_SECTIONS:
                assert sec in report, f"{case_id}: missing section {sec}"

    def test_report_no_forbidden_claims(self, regenerated_runs: Path):
        forbidden = [
            "this is definitely glioblastoma",
            "definitive diagnosis is glioblastoma",
            "start chemotherapy immediately",
            "surgery is recommended immediately",
        ]
        for case_id in GOLDEN_CASES:
            report = (self._case_dir(regenerated_runs, case_id) / "report_draft.md").read_text(encoding="utf-8").lower()
            for text in forbidden:
                assert text not in report, f"{case_id}: forbidden pattern found: {text}"

    def test_qa_results_structure(self, regenerated_runs: Path):
        for case_id in GOLDEN_CASES:
            path = self._case_dir(regenerated_runs, case_id) / "qa_results.jsonl"
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            assert len(rows) == 20, f"{case_id}: expected 20 QA rows, got {len(rows)}"
            for row in rows:
                assert "question" in row
                assert "answer" in row
                assert "evidence_ids" in row
                assert "confidence" in row
                assert "safety_note" in row
                assert isinstance(row["answer"], str) and row["answer"]

    def test_qa_evidence_coverage(self, regenerated_runs: Path):
        safety_keys = [
            "glioblastoma", "chemotherapy", "prognosis", "imaging alone",
            "surgery", "treatment", "grow or shrink", "cancer", "worry", "cure rate",
        ]
        for case_id in GOLDEN_CASES:
            path = self._case_dir(regenerated_runs, case_id) / "qa_results.jsonl"
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            factual = [r for r in rows if not any(k in r["question"].lower() for k in safety_keys)]
            with_evidence = [r for r in factual if len(r.get("evidence_ids", [])) > 0]
            coverage = len(with_evidence) / len(factual) if factual else 0
            assert coverage >= 0.80, f"{case_id}: evidence coverage {coverage:.0%} < 80%"

    def test_run_meta_tracks_current_fingerprint(self, regenerated_runs: Path):
        current_fingerprint = build_run_fingerprint()
        for case_id in GOLDEN_CASES:
            path = self._case_dir(regenerated_runs, case_id) / "run_meta.json"
            meta = json.loads(path.read_text(encoding="utf-8"))
            assert meta["case_id"] == case_id
            assert meta["llm_policy"] == {
                "report_use_llm": False,
                "qa_nonstructured_use_llm": False,
            }
            assert meta["question_set"]["count"] == 20
            assert meta["run_fingerprint"]["composite_hash"] == current_fingerprint["composite_hash"]
            assert "artifact_paths" in meta
