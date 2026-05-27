import json
from pathlib import Path

from ai_assistant.core.paths import default_runs_root

CASES_ROOT = Path(__file__).resolve().parents[1] / "outputs" / "cases"
RUNS_ROOT = Path(default_runs_root())

TUMOR_METRICS_REQUIRED_KEYS = [
    "case_id", "modality", "tumor_burden",
]

TUMOR_BURDEN_REQUIRED_KEYS = [
    "total_volume_mm3", "max_diameter_mm",
]

QA_ROW_REQUIRED_KEYS = ["question", "answer", "evidence_ids", "confidence", "safety_note"]


class TestTumorMetricsSchema:
    def _find_metrics_files(self):
        files = []
        for root in [CASES_ROOT, RUNS_ROOT]:
            if root.exists():
                files.extend(root.rglob("tumor_metrics.json"))
        return files

    def test_at_least_one_metrics_file_exists(self):
        files = self._find_metrics_files()
        assert len(files) > 0, "No tumor_metrics.json found anywhere"

    def test_required_keys_present(self):
        for path in self._find_metrics_files():
            data = json.loads(path.read_text(encoding="utf-8"))
            for key in TUMOR_METRICS_REQUIRED_KEYS:
                assert key in data, f"{path}: missing key '{key}'"

    def test_types_correct(self):
        for path in self._find_metrics_files():
            data = json.loads(path.read_text(encoding="utf-8"))
            assert isinstance(data["case_id"], str)
            assert isinstance(data["modality"], str)
            assert isinstance(data["tumor_burden"], dict)

    def test_tumor_burden_schema(self):
        for path in self._find_metrics_files():
            data = json.loads(path.read_text(encoding="utf-8"))
            burden = data["tumor_burden"]
            for key in TUMOR_BURDEN_REQUIRED_KEYS:
                assert key in burden, f"{path}: tumor_burden missing '{key}'"
            assert isinstance(burden["total_volume_mm3"], (int, float))
            assert burden["total_volume_mm3"] >= 0
            assert isinstance(burden["max_diameter_mm"], (int, float))

    def test_volumes_non_negative(self):
        for path in self._find_metrics_files():
            data = json.loads(path.read_text(encoding="utf-8"))
            burden = data["tumor_burden"]
            for key in ["total_volume_mm3", "enhancing_volume_mm3",
                        "edema_volume_mm3", "necrotic_volume_mm3"]:
                if key in burden:
                    assert burden[key] >= 0, f"{path}: {key} < 0"


class TestQAOutputSchema:
    def _find_qa_files(self):
        files = []
        if RUNS_ROOT.exists():
            files.extend(RUNS_ROOT.rglob("qa_results.jsonl"))
        return files

    def test_at_least_one_qa_file_exists(self):
        files = self._find_qa_files()
        assert len(files) > 0, "No qa_results.jsonl found"

    def test_qa_row_schema(self):
        for path in self._find_qa_files():
            for line in path.read_text(encoding="utf-8").strip().split("\n"):
                if not line.strip():
                    continue
                row = json.loads(line)
                for key in QA_ROW_REQUIRED_KEYS:
                    assert key in row, f"{path}: QA row missing '{key}'"
                assert isinstance(row["answer"], str) and len(row["answer"]) > 0
                assert isinstance(row["evidence_ids"], list)
                assert isinstance(row["confidence"], (int, float))
                assert 0 <= row["confidence"] <= 1.0
