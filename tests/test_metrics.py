import json
import tempfile
from pathlib import Path

from ai_assistant.eval.metrics import evaluate_case, _expects_evidence, _load_jsonl


class TestExpectsEvidence:
    def test_factual_question_expects_evidence(self):
        assert _expects_evidence("What is the total tumor volume?") is True

    def test_safety_question_no_evidence(self):
        assert _expects_evidence("Is this definitely a glioblastoma?") is False

    def test_treatment_no_evidence(self):
        assert _expects_evidence("What treatment do you recommend?") is False

    def test_cancer_no_evidence(self):
        assert _expects_evidence("Is this cancer?") is False

    def test_empty_no_evidence(self):
        assert _expects_evidence("") is True


class TestLoadJsonl:
    def test_load_valid_jsonl(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
            f.write(json.dumps({"question": "Q1", "answer": "A1"}) + "\n")
            f.write(json.dumps({"question": "Q2", "answer": "A2"}) + "\n")
            path = Path(f.name)
        rows = _load_jsonl(path)
        assert len(rows) == 2
        path.unlink()

    def test_load_missing_file(self):
        rows = _load_jsonl(Path("nonexistent.jsonl"))
        assert rows == []

    def test_load_empty_lines(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
            f.write(json.dumps({"q": 1}) + "\n")
            f.write("\n")
            f.write(json.dumps({"q": 2}) + "\n")
            path = Path(f.name)
        rows = _load_jsonl(path)
        assert len(rows) == 2
        path.unlink()


class TestEvaluateCase:
    def _make_case_dir(self, tmp_path, report_text="", qa_rows=None):
        case_dir = tmp_path / "TEST_CASE"
        case_dir.mkdir()
        if report_text:
            (case_dir / "report_draft.md").write_text(report_text, encoding="utf-8")
        if qa_rows:
            with (case_dir / "qa_results.jsonl").open("w", encoding="utf-8") as f:
                for row in qa_rows:
                    f.write(json.dumps(row) + "\n")
        return case_dir

    def test_perfect_case(self, tmp_path):
        report = (
            "# Clinical Context\nctx\n# Findings\nfnd\n# Impression\nimp\n"
            "# Limitations\nlim\n# Suggested Clinical Correlation\nrec\n# Disclaimer\ndis\n"
        )
        qa = [
            {"question": "What is the total tumor volume?", "evidence_ids": ["ctx:seg"], "answer": "X"},
            {"question": "How many lesions?", "evidence_ids": ["ctx:count"], "answer": "Y"},
        ]
        result = evaluate_case(self._make_case_dir(tmp_path, report, qa))
        assert result["has_required_sections"] is True
        assert result["qa_evidence_coverage"] == 1.0
        assert result["unsupported_statement_rate"] == 0.0

    def test_missing_section(self, tmp_path):
        report = "# Clinical Context\nctx\n# Findings\nfnd\n"
        result = evaluate_case(self._make_case_dir(tmp_path, report))
        assert result["has_required_sections"] is False
        assert len(result["missing_sections"]) > 0

    def test_safety_questions_excluded_from_evidence(self, tmp_path):
        report = "\n".join([
            "# Clinical Context", "# Findings", "# Impression",
            "# Limitations", "# Suggested Clinical Correlation", "# Disclaimer"
        ])
        qa = [
            {"question": "Is this definitely a glioblastoma?", "evidence_ids": [], "answer": "safe"},
            {"question": "What is the total tumor volume?", "evidence_ids": ["ctx:vol"], "answer": "45000"},
        ]
        result = evaluate_case(self._make_case_dir(tmp_path, report, qa))
        assert result["qa_expected_evidence_total"] == 1
        assert result["qa_evidence_coverage"] == 1.0

    def test_no_files(self, tmp_path):
        case_dir = tmp_path / "EMPTY_CASE"
        case_dir.mkdir()
        result = evaluate_case(case_dir)
        assert result["has_required_sections"] is False
        assert result["qa_total"] == 0
