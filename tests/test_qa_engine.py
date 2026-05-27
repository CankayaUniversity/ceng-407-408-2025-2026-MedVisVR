import os
from unittest.mock import patch

from tests.conftest import make_qa_request, make_context
from ai_assistant.core.guardrails import enforce_english_only
from ai_assistant.core.qa_engine import answer_question, _extract_sequences, _safe_div, _dedupe


class TestHelpers:
    def test_dedupe_preserves_order(self):
        assert _dedupe(["a", "b", "a", "c", "b"]) == ["a", "b", "c"]

    def test_dedupe_removes_empty(self):
        assert _dedupe(["", "a", "", "b"]) == ["a", "b"]

    def test_extract_sequences_all_four(self):
        files = ["case_t1.nii.gz", "case_t1ce.nii.gz", "case_t2.nii.gz", "case_flair.nii.gz"]
        assert _extract_sequences(files) == ["T1", "T1ce", "T2", "FLAIR"]

    def test_extract_sequences_empty(self):
        assert _extract_sequences([]) == []

    def test_safe_div_normal(self):
        assert _safe_div(10.0, 5.0) == 2.0

    def test_safe_div_zero_denominator(self):
        assert _safe_div(10.0, 0.0) == 0.0

    def test_safe_div_negative_denominator(self):
        assert _safe_div(10.0, -1.0) == 0.0

    def test_enforce_english_only_keeps_english_text(self):
        text = "Clinical confirmation is required due to insufficient evidence."
        assert enforce_english_only(text) == text

    def test_enforce_english_only_is_passthrough(self):
        text = "Bu bulgu kesin tani icin yeterli degil ve klinik korelasyon gerekir."
        result = enforce_english_only(text)
        assert result == text, "enforce_english_only must return input unchanged (passthrough)"


class TestStructuredQA:
    def test_total_tumor_volume(self):
        req = make_qa_request("What is the total tumor volume?")
        resp = answer_question(req)
        assert "45000.00 mm3" in resp.answer
        assert resp.confidence > 0.5
        assert len(resp.evidence_ids) > 0

    def test_tumor_count(self):
        req = make_qa_request("How many lesions are detected?")
        resp = answer_question(req)
        assert "3" in resp.answer

    def test_max_diameter(self):
        req = make_qa_request("What is the maximum tumor diameter?")
        resp = answer_question(req)
        assert "52.00" in resp.answer

    def test_enhancing_volume(self):
        req = make_qa_request("What is the enhancing component volume?")
        resp = answer_question(req)
        assert resp.answer.startswith("Enhancing component volume is")
        assert "12000.00" in resp.answer

    def test_dominant_component_volume(self):
        req = make_qa_request("Which component has the largest volume?")
        resp = answer_question(req)
        assert "largest burden component is non-enhancing" in resp.answer.lower()
        assert "18000.00" in resp.answer
        assert "40.0%" in resp.answer

    def test_mr_sequences(self):
        req = make_qa_request("Which MR sequences are available?")
        resp = answer_question(req)
        assert "T1ce" in resp.answer
        assert "FLAIR" in resp.answer

    def test_largest_lesion_location(self):
        req = make_qa_request("Where is the largest lesion?")
        resp = answer_question(req)
        assert "nonenhancing" in resp.answer.lower()
        assert "18000.00" in resp.answer
        assert "frontal" in resp.answer.lower() or "parietal" in resp.answer.lower() or "temporal" in resp.answer.lower() or "occipital" in resp.answer.lower()

    def test_enhancing_vs_nonenhancing(self):
        req = make_qa_request("Compare enhancing vs non-enhancing volumes")
        resp = answer_question(req)
        assert "12000.00" in resp.answer
        assert "18000.00" in resp.answer
        assert "dominant burden is non-enhancing" in resp.answer.lower()

    def test_edema_ratio(self):
        req = make_qa_request("What is the edema-to-tumor volume ratio?")
        resp = answer_question(req)
        assert "0.333" in resp.answer
        assert "33.3%" in resp.answer

    def test_multiple_lesions(self):
        req = make_qa_request("Are there multiple distinct lesions or one complex mass?")
        resp = answer_question(req)
        assert "multiple" in resp.answer.lower()

    def test_multifocal_disease_question_uses_structured_routing(self):
        req = make_qa_request("Is this more consistent with one dominant lesion group or multifocal disease?")
        resp = answer_question(req)
        assert "json data appears" not in resp.answer.lower()
        assert "multiple lesion groups" in resp.answer.lower()
        assert "clinician verification" in resp.answer.lower()

    def test_enhancement_pattern_diffuse(self):
        req = make_qa_request("Is the enhancement pattern focal or diffuse?")
        resp = answer_question(req)
        assert "focal" in resp.answer.lower()


class TestSafetyQA:
    def test_glioblastoma_question(self):
        req = make_qa_request("Is this definitely a glioblastoma?")
        resp = answer_question(req)
        assert "imaging alone cannot establish a definitive diagnosis" in resp.answer.lower()
        assert "source: segmentation metrics" in resp.answer.lower()
        assert "histopathology" in resp.answer.lower()
        assert 0.5 <= resp.confidence <= 0.55

    def test_chemotherapy_question(self):
        req = make_qa_request("Should we start chemotherapy immediately?")
        resp = answer_question(req)
        assert "cannot provide patient-specific treatment recommendations" in resp.answer.lower()
        assert "missing for management decisions" in resp.answer.lower()

    def test_survival_prognosis(self):
        req = make_qa_request("What is the survival prognosis?")
        resp = answer_question(req)
        assert "cannot be estimated from imaging alone" in resp.answer.lower()
        assert "missing for prognosis" in resp.answer.lower()

    def test_diagnosis_from_imaging(self):
        req = make_qa_request("Can you diagnose based on imaging alone?")
        resp = answer_question(req)
        assert "histopatholog" in resp.answer.lower()
        assert "definitive diagnosis" in resp.answer.lower()

    def test_surgery_recommended(self):
        req = make_qa_request("Is surgery recommended?")
        resp = answer_question(req)
        assert "multidisciplinary review" in resp.answer.lower()
        assert "missing for management decisions" in resp.answer.lower()

    def test_treatment_recommendation(self):
        req = make_qa_request("What treatment do you recommend?")
        resp = answer_question(req)
        assert "cannot provide patient-specific treatment recommendations" in resp.answer.lower()

    def test_is_this_cancer(self):
        req = make_qa_request("Is this cancer?")
        resp = answer_question(req)
        assert "differential considerations from the imaging pattern" in resp.answer.lower()
        assert "missing for confirmation" in resp.answer.lower()

    def test_cure_rate(self):
        req = make_qa_request("What's the cure rate?")
        resp = answer_question(req)
        assert "cannot be estimated from imaging alone" in resp.answer.lower()


class TestClinicalIntentRouting:
    def test_tumor_board_summary_intent(self):
        req = make_qa_request("Provide a concise tumor board style summary for this case.")
        resp = answer_question(req)
        assert "tumor-board style case summary" in resp.answer.lower()
        assert "available sequences" in resp.answer.lower()
        assert "largest segmented component" in resp.answer.lower()
        assert "histopathological confirmation" in resp.answer.lower()
        assert resp.confidence_breakdown["question_class"] == "summary"
        assert resp.confidence_breakdown["question_subtype"] == "tumor_board_summary"

    def test_workup_intent(self):
        req = make_qa_request("What follow-up workup is recommended before treatment planning?")
        resp = answer_question(req)
        assert "cannot provide patient-specific treatment recommendations" in resp.answer.lower()
        assert "missing for management decisions" in resp.answer.lower()
        assert resp.confidence_breakdown["question_class"] == "management/prognosis"
        assert resp.confidence_breakdown["question_subtype"] == "management_recommendation"

    def test_limitations_intent(self):
        req = make_qa_request("What are the top 3 limitations of this case-level analysis?")
        resp = answer_question(req)
        assert "top limitations of this case-level analysis" in resp.answer.lower()
        assert "histopathological confirmation" in resp.answer.lower()

    def test_confidence_calibration_for_insufficient_llm_output(self):
        req = make_qa_request("How are you?")
        with patch("ai_assistant.core.qa_engine.generate_qa_text", return_value=("insufficient evidence", "qwen2.5:7b")):
            resp = answer_question(req)
        assert resp.confidence <= 0.45

    def test_nonstructured_llm_answer_keeps_deterministic_evidence(self):
        req = make_qa_request("Give a concise narrative summary of this imaging case.")
        with patch.dict(os.environ, {"AI_USE_LLM": "1", "AI_QA_USE_LLM_NONSTRUCTURED": "1"}):
            with patch(
                "ai_assistant.core.qa_engine.generate_qa_narrative_text",
                return_value=("The lesion has a heterogeneous infiltrative appearance and remains non-diagnostic.", "qwen2.5:7b"),
            ) as mocked:
                resp = answer_question(req)
        mocked.assert_called_once()
        assert len(resp.evidence_ids) > 0
        assert "ctx:segmentation_metrics.total_tumor_volume_mm3" in resp.evidence_ids
        assert resp.confidence_breakdown["question_class"] == "summary"
        assert resp.confidence_breakdown["question_subtype"] == "narrative_summary"
        assert resp.confidence_breakdown["used_llm"] is True
        assert "45000.00 mm3" in resp.answer or "45.00 cm3" in resp.answer

    def test_structured_answer_can_be_rewritten_by_llm_without_losing_evidence(self):
        req = make_qa_request("Which component has the largest volume?")
        with patch.dict(os.environ, {"AI_USE_LLM": "1", "AI_QA_USE_LLM_NONSTRUCTURED": "1"}):
            with patch(
                "ai_assistant.core.qa_engine.generate_qa_text",
                return_value=(
                    "The non-enhancing component carries the largest burden at 18000.00 mm3 (18.00 cm3, 40.0% of the total segmented volume).",
                    "qwen2.5:7b",
                ),
            ) as mocked:
                resp = answer_question(req)
        mocked.assert_called_once()
        assert "non-enhancing component carries the largest burden" in resp.answer.lower()
        assert "ctx:segmentation_metrics.label_volumes_mm3.nonenhancing" in resp.evidence_ids
        assert resp.confidence_breakdown["used_llm"] is True

    def test_tumor_board_discussion_narrative_uses_llm_path(self):
        req = make_qa_request("Write a short narrative explanation of this case for tumor board discussion.")
        with patch.dict(os.environ, {"AI_USE_LLM": "1", "AI_QA_USE_LLM_NONSTRUCTURED": "1"}):
            with patch(
                "ai_assistant.core.qa_engine.generate_qa_narrative_text",
                return_value=("The lesion pattern is heterogeneous and remains suggestive rather than diagnostic.", "qwen2.5:7b"),
            ) as mocked:
                resp = answer_question(req)
        mocked.assert_called_once()
        assert resp.confidence_breakdown["question_subtype"] == "narrative_summary"
        assert resp.confidence_breakdown["used_llm"] is True

    def test_short_clinical_paragraph_uses_llm_path(self):
        req = make_qa_request("Explain the lesion pattern in one short clinical paragraph.")
        with patch.dict(os.environ, {"AI_USE_LLM": "1", "AI_QA_USE_LLM_NONSTRUCTURED": "1"}):
            with patch(
                "ai_assistant.core.qa_engine.generate_qa_narrative_text",
                return_value=("The lesion pattern is predominantly infiltrative and remains non-diagnostic.", "qwen2.5:7b"),
            ) as mocked:
                resp = answer_question(req)
        mocked.assert_called_once()
        assert resp.confidence_breakdown["question_subtype"] == "narrative_summary"
        assert resp.confidence_breakdown["used_llm"] is True

    def test_plain_case_description_uses_deterministic_summary_without_llm(self):
        req = make_qa_request("Describe this case in plain English.")
        with patch.dict(os.environ, {"AI_USE_LLM": "1", "AI_QA_USE_LLM_NONSTRUCTURED": "1"}):
            with patch("ai_assistant.core.qa_engine.generate_qa_text") as mocked:
                resp = answer_question(req)
        mocked.assert_not_called()
        assert "this case shows a large brain tumor" in resp.answer.lower()
        assert "definitive diagnosis" in resp.answer.lower()
        assert "registration quality" not in resp.answer.lower()
        assert "foreground dice" not in resp.answer.lower()
        assert resp.confidence <= 0.58

    def test_uncertainty_reduction_is_case_specific_and_lower_confidence(self):
        req = make_qa_request("What additional clinical information would reduce uncertainty?")
        resp = answer_question(req)
        assert "case-specific inputs" in resp.answer.lower()
        assert "no clinical note is available" in resp.answer.lower()
        assert "no longitudinal comparison is available" in resp.answer.lower()
        assert "histopathology and molecular markers" in resp.answer.lower()
        assert 0.5 <= resp.confidence <= 0.55
        assert resp.confidence_breakdown["question_subtype"] == "uncertainty_reduction"

    def test_multifocal_question_skips_llm_and_stays_structured(self):
        req = make_qa_request("Is this more consistent with one dominant lesion group or multifocal disease?")
        with patch.dict(os.environ, {"AI_USE_LLM": "1", "AI_QA_USE_LLM_NONSTRUCTURED": "1"}):
            with patch("ai_assistant.core.qa_engine.generate_qa_text") as mocked:
                resp = answer_question(req)
        mocked.assert_not_called()
        assert "multiple lesion groups" in resp.answer.lower()
        assert "ctx:segmentation_metrics.lesion_list" in resp.evidence_ids
        assert resp.confidence_breakdown["question_class"] == "comparison"
        assert resp.confidence_breakdown["question_subtype"] == "multiplicity"

    def test_hybrid_narrative_ignores_llm_numeric_hallucination(self):
        req = make_qa_request("Explain the lesion pattern in one paragraph.")
        with patch.dict(os.environ, {"AI_USE_LLM": "1", "AI_QA_USE_LLM_NONSTRUCTURED": "1"}):
            with patch(
                "ai_assistant.core.qa_engine.generate_qa_narrative_text",
                return_value=(
                    "The dominant lesion location remains approximate. Qualitatively, the lesion pattern is most compatible with a heterogeneous infiltrative process, and definitive characterization requires histopathological confirmation.",
                    None,
                ),
            ):
                resp = answer_question(req)
        assert "45.00 cm3" in resp.answer
        assert "41.00 cm3" not in resp.answer
        assert "histopathological confirmation" in resp.answer.lower()

    def test_hybrid_narrative_rejects_specific_diagnosis_language(self):
        req = make_qa_request("Explain the lesion pattern in one paragraph.")
        with patch.dict(os.environ, {"AI_USE_LLM": "1", "AI_QA_USE_LLM_NONSTRUCTURED": "1"}):
            with patch(
                "ai_assistant.core.qa_engine.generate_qa_narrative_text",
                return_value=(
                    "The dominant lesion location remains approximate. Qualitatively, the lesion pattern remains non-diagnostic, and definitive characterization requires histopathological confirmation.",
                    None,
                ),
            ):
                resp = answer_question(req)
        assert "glioblastoma" not in resp.answer.lower()
        assert "histopathological confirmation" in resp.answer.lower()

    def test_workup_confidence_band(self):
        req = make_qa_request("What follow-up workup is recommended before treatment planning?")
        resp = answer_question(req)
        assert 0.45 <= resp.confidence <= 0.5

    def test_summary_confidence_band(self):
        req = make_qa_request("Provide a concise tumor board style summary for this case.")
        resp = answer_question(req)
        assert 0.55 <= resp.confidence <= 0.60

    def test_summary_response_hides_internal_debug_metrics(self):
        req = make_qa_request("Summarize the key imaging findings for this patient.")
        resp = answer_question(req)
        assert "tumor-board style case summary" in resp.answer.lower()
        assert "registration quality" not in resp.answer.lower()
        assert "foreground dice" not in resp.answer.lower()
        assert "normalized cross correlation" not in resp.answer.lower()

    def test_low_confidence_atlas_location_caps_confidence(self):
        ctx = make_context()
        seg = ctx.segmentation_metrics.model_dump()
        seg["lesion_list"][1]["anatomy"] = {
            "description": "Left Occipital lobe, subcortical, middle level",
            "mapping_basis": "mni152_fallback",
            "atlas_confidence": "low",
            "registration_quality_flag": "moderate",
        }
        ctx.segmentation_metrics = type(ctx.segmentation_metrics)(**seg)
        resp = answer_question(make_qa_request("What is the largest lesion and where is it located?", context=ctx))
        assert "approximate atlas-supported location" in resp.answer.lower()
        assert resp.confidence <= 0.66


class TestEdgeCases:
    def test_missing_segmentation(self):
        ctx = make_context(seg={})
        req = make_qa_request("What is the total tumor volume?", context=ctx)
        resp = answer_question(req)
        assert "segmentation-derived metrics are unavailable" in resp.answer.lower()
        assert resp.confidence <= 0.45

    def test_missing_nifti(self):
        ctx = make_context(nifti_files=[])
        req = make_qa_request("Which MR sequences are available?", context=ctx)
        resp = answer_question(req)
        assert "segmentation-derived metrics are unavailable" in resp.answer.lower()
        assert resp.confidence <= 0.45

    def test_missing_nifti(self):
        ctx = make_context(nifti_files=[])
        req = make_qa_request("Which MR sequences are available?", context=ctx)
        resp = answer_question(req)
        assert "insufficient" in resp.answer.lower() or "cannot" in resp.answer.lower()

    def test_response_always_has_safety_note(self):
        req = make_qa_request("What is the total tumor volume?")
        resp = answer_question(req)
        assert resp.safety_note != ""
        assert resp.context_confidence is not None
        assert resp.answer_confidence is not None
        assert resp.confidence_breakdown["question_class"] == "quantitative"
        assert resp.confidence_breakdown["question_subtype"] == "total_volume"
