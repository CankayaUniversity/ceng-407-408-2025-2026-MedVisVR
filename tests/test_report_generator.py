import os
from unittest.mock import patch

from tests.conftest import make_context
from ai_assistant.core.report_generator import (
    generate_report_draft,
    _has_required_sections,
    _has_required_findings_subsections,
    _contains_forbidden_pattern,
    _has_mandatory_disclaimer,
    _replace_impression,
    REQUIRED_SECTIONS,
    REQUIRED_FINDINGS_SUBSECTIONS,
    FORBIDDEN_REPORT_PATTERNS,
)


class TestReportValidators:
    def test_has_required_sections_true(self):
        text = "\n".join(REQUIRED_SECTIONS)
        assert _has_required_sections(text) is True

    def test_has_required_sections_missing_one(self):
        text = "\n".join(REQUIRED_SECTIONS[:-1])
        assert _has_required_sections(text) is False

    def test_has_required_findings_subsections_true(self):
        text = "\n".join(REQUIRED_FINDINGS_SUBSECTIONS)
        assert _has_required_findings_subsections(text) is True

    def test_forbidden_pattern_detected(self):
        assert _contains_forbidden_pattern("This is definitely glioblastoma") is True

    def test_forbidden_pattern_clean(self):
        assert _contains_forbidden_pattern("Normal clinical report text.") is False

    def test_mandatory_disclaimer_present(self):
        text = (
            "does not provide a medical diagnosis. "
            "definitive diagnosis requires histopathological examination "
            "and clinical correlation by qualified radiologist/clinician."
        )
        assert _has_mandatory_disclaimer(text) is True

    def test_mandatory_disclaimer_missing(self):
        assert _has_mandatory_disclaimer("Some random text") is False


class TestReplaceImpression:
    def test_replaces_impression_section(self):
        original = "# Impression\nOld impression text\n\n# Limitations\nSome limits"
        result = _replace_impression(original, "New impression")
        assert "New impression" in result
        assert "Old impression text" not in result
        assert "# Limitations" in result

    def test_no_impression_header(self):
        original = "# Findings\nSome findings"
        result = _replace_impression(original, "New impression")
        assert result == original


class TestGenerateReport:
    def test_report_has_all_sections(self):
        ctx = make_context()
        resp = generate_report_draft(ctx)
        for sec in REQUIRED_SECTIONS:
            assert sec in resp.report_markdown, f"Missing section: {sec}"

    def test_report_no_forbidden_patterns(self):
        ctx = make_context()
        resp = generate_report_draft(ctx)
        for pattern in FORBIDDEN_REPORT_PATTERNS:
            assert pattern not in resp.report_markdown.lower()

    def test_report_has_disclaimer(self):
        ctx = make_context()
        resp = generate_report_draft(ctx)
        assert "does not provide a medical diagnosis" in resp.report_markdown.lower()
        assert "histopathological" in resp.report_markdown.lower()

    def test_report_contains_patient_id(self):
        ctx = make_context(patient_id="PATIENT_XYZ")
        resp = generate_report_draft(ctx)
        assert "PATIENT_XYZ" in resp.report_markdown

    def test_report_contains_volume_data(self):
        ctx = make_context()
        resp = generate_report_draft(ctx)
        assert "45.00 cm3" in resp.report_markdown or "45000.00 mm3" in resp.report_markdown

    def test_report_contains_sequences(self):
        ctx = make_context()
        resp = generate_report_draft(ctx)
        assert "T1ce" in resp.report_markdown

    def test_report_uses_more_fluid_fallback_narrative(self):
        ctx = make_context()
        resp = generate_report_draft(ctx)
        assert "this mr brain tumor review is based on" in resp.report_markdown.lower()
        assert "These findings remain suggestive rather than diagnostic" in resp.report_markdown

    def test_report_contains_richer_findings_detail(self):
        ctx = make_context()
        resp = generate_report_draft(ctx)
        assert "Dominant segmented component" in resp.report_markdown
        assert "dominant component" in resp.report_markdown
        assert "Clinical note present" in resp.report_markdown
        assert "## Localization" in resp.report_markdown
        assert "## Mass Effect" in resp.report_markdown
        assert "## Midline Shift" in resp.report_markdown
        assert "## Enhancement Pattern" in resp.report_markdown
        assert "## Multifocality" in resp.report_markdown

    def test_report_contains_uncertainty_interval(self):
        ctx = make_context()
        resp = generate_report_draft(ctx)
        assert "95% CI for total volume" in resp.report_markdown
        assert "measurement uncertainty" in resp.report_markdown

    def test_report_conservative_mode_uses_placeholders(self):
        ctx = make_context()
        resp = generate_report_draft(ctx, detail_mode="conservative")
        assert "Not emphasized in conservative mode" in resp.report_markdown

    def test_report_evidence_ids(self):
        ctx = make_context()
        resp = generate_report_draft(ctx)
        assert "ctx:patient_id" in resp.evidence_ids
        assert "ctx:segmentation_metrics.tumor_count" in resp.evidence_ids

    def test_report_confidence_range(self):
        ctx = make_context()
        resp = generate_report_draft(ctx)
        assert 0.0 < resp.confidence <= 1.0

    def test_report_safety_note_present(self):
        ctx = make_context()
        resp = generate_report_draft(ctx)
        assert resp.safety_note != ""

    def test_report_with_quality_issues(self):
        ctx = make_context(nifti_files=[], modality="XX")
        resp = generate_report_draft(ctx)
        for sec in REQUIRED_SECTIONS:
            assert sec in resp.report_markdown

    def test_report_llm_toggle_can_disable_model_use(self):
        ctx = make_context()
        with patch.dict(os.environ, {"AI_USE_LLM": "1", "AI_REPORT_USE_LLM": "0"}):
            with patch("ai_assistant.core.report_generator.generate_report_section_narratives") as mocked:
                resp = generate_report_draft(ctx)
        mocked.assert_not_called()
        assert "# Disclaimer" in resp.report_markdown

    def test_report_llm_receives_descriptor_json(self):
        ctx = make_context()
        captured = {}

        def _fake_generate_report_section_narratives(
            *,
            context,
            report_descriptor,
            fallback_clinical_context_narrative,
            fallback_findings_narrative,
            fallback_impression_narrative,
            use_high_model,
        ):
            captured["descriptor"] = report_descriptor
            return {
                "clinical_context_narrative": fallback_clinical_context_narrative,
                "findings_narrative": fallback_findings_narrative,
                "impression_narrative": fallback_impression_narrative,
            }, "qwen2.5:7b"

        with patch.dict(os.environ, {"AI_USE_LLM": "1", "AI_REPORT_USE_LLM": "1"}):
            with patch(
                "ai_assistant.core.report_generator.generate_report_section_narratives",
                side_effect=_fake_generate_report_section_narratives,
            ):
                resp = generate_report_draft(ctx)

        assert "# Findings" in resp.report_markdown
        assert captured["descriptor"]["segmentation_overview"]["lesion_count"] == 3
        assert captured["descriptor"]["segmentation_overview"]["segmented_component_count"] == 3
        assert captured["descriptor"]["component_profile"]["dominant_component"] == "non-enhancing"

    def test_report_llm_narrative_with_numbers_falls_back(self):
        ctx = make_context()
        with patch.dict(os.environ, {"AI_USE_LLM": "1", "AI_REPORT_USE_LLM": "1"}):
            with patch(
                "ai_assistant.core.report_generator.generate_report_section_narratives",
                return_value=(
                    {
                        "clinical_context_narrative": "This case includes 4 sequences and 3 lesions.",
                        "findings_narrative": "The tumor burden is 41 cm3.",
                    },
                    "qwen2.5:7b",
                ),
            ):
                resp = generate_report_draft(ctx)
        assert "This case includes 4 sequences and 3 lesions." not in resp.report_markdown
        assert "The tumor burden is 41 cm3." not in resp.report_markdown
        assert "based on T1, T1ce, T2, FLAIR" in resp.report_markdown or "based on FLAIR, T1, T1ce, T2" in resp.report_markdown

    def test_report_llm_narrative_with_glioblastoma_language_falls_back(self):
        ctx = make_context()
        with patch.dict(os.environ, {"AI_USE_LLM": "1", "AI_REPORT_USE_LLM": "1"}):
            with patch(
                "ai_assistant.core.report_generator.generate_report_section_narratives",
                return_value=(
                    {
                        "clinical_context_narrative": "This is a possible glioblastoma case.",
                        "findings_narrative": "This is definitely glioblastoma.",
                    },
                    "qwen2.5:7b",
                ),
            ):
                resp = generate_report_draft(ctx)
        assert "glioblastoma" not in resp.report_markdown.lower()

    def test_report_missing_segmentation_uses_unavailable_language(self):
        ctx = make_context(seg={})
        resp = generate_report_draft(ctx)
        assert "segmentation-derived tumor metrics are unavailable" in resp.report_markdown.lower()
        assert "0.00 mm3" not in resp.report_markdown
