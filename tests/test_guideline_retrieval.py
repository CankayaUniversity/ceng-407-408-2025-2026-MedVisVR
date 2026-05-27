import pytest
from ai_assistant.core.guideline_retrieval import (
    _load_all_snippets,
    retrieve_guidelines,
    format_guideline_context,
)


@pytest.fixture(scope="module")
def all_snippets():
    return _load_all_snippets()


class TestCorpusLoading:
    def test_snippets_loaded(self, all_snippets):
        assert len(all_snippets) >= 30, "Expected at least 30 snippets across all corpora"

    def test_corpora_present(self, all_snippets):
        corpora = {s["_corpus"] for s in all_snippets}
        assert "btrads_snippets" in corpora
        assert "mass_effect_snippets" in corpora
        assert "brats_label_snippets" in corpora
        assert "neurosurgical_snippets" in corpora

    def test_all_snippets_have_text(self, all_snippets):
        for s in all_snippets:
            assert s.get("text"), f"Snippet {s.get('id')} has no text"

    def test_all_snippets_tokenized(self, all_snippets):
        for s in all_snippets:
            assert isinstance(s.get("_tokens"), list), f"Snippet {s.get('id')} not tokenized"
            assert len(s["_tokens"]) > 0, f"Snippet {s.get('id')} has empty tokens"


class TestRetrievalRelevance:
    def test_mls_query_returns_mass_effect_snippets(self):
        results = retrieve_guidelines("midline shift herniation risk", intent_class="mass_effect", top_k=3)
        assert len(results) > 0
        corpora = {r["corpus"] for r in results}
        assert "mass_effect_snippets" in corpora, "Mass effect query should hit mass_effect_snippets"

    def test_mls_top_result_is_mls_threshold(self):
        results = retrieve_guidelines("midline shift threshold 5mm 7mm", intent_class="mass_effect", top_k=1)
        assert results, "Should return at least one result"
        assert results[0]["id"] in {"mls_thresholds", "mls_brain_tumor"}, \
            f"Expected mls_thresholds as top result, got: {results[0]['id']}"

    def test_motor_cortex_query_returns_neurosurgical_snippets(self):
        results = retrieve_guidelines("motor cortex proximity surgical planning", intent_class="critical_proximity", top_k=3)
        assert len(results) > 0
        corpora = {r["corpus"] for r in results}
        assert "neurosurgical_snippets" in corpora

    def test_motor_cortex_top_result(self):
        results = retrieve_guidelines("motor cortex distance eloquent resection", intent_class="critical_proximity", top_k=1)
        assert results
        assert results[0]["id"] == "motor_cortex_proximity"

    def test_btrads_query_returns_btrads_snippets(self):
        results = retrieve_guidelines("BT-RADS score tumor progression response", intent_class="structured_reporting", top_k=3)
        assert len(results) > 0
        corpora = {r["corpus"] for r in results}
        assert "btrads_snippets" in corpora

    def test_brats_label_query(self):
        results = retrieve_guidelines("enhancing tumor label 4 necrotic core label 1", intent_class="quantitative", top_k=2)
        assert len(results) > 0
        assert any("brats" in r["corpus"] for r in results)

    def test_brainstem_query(self):
        results = retrieve_guidelines("brainstem proximity compression risk", intent_class="critical_proximity", top_k=2)
        assert len(results) > 0
        ids = {r["id"] for r in results}
        assert "brainstem_proximity" in ids

    def test_ventricular_compression_query(self):
        results = retrieve_guidelines("ventricular compression ICP hydrocephalus", intent_class="mass_effect", top_k=2)
        assert len(results) > 0
        ids = {r["id"] for r in results}
        assert "ventricular_compression" in ids

    def test_surgical_planning_query(self):
        results = retrieve_guidelines("extent of resection GTR surgical planning glioblastoma", intent_class="management/prognosis", top_k=2)
        assert len(results) > 0
        corpora = {r["corpus"] for r in results}
        assert "neurosurgical_snippets" in corpora

    def test_scores_are_ordered_descending(self):
        results = retrieve_guidelines("midline shift herniation basal cistern", intent_class="mass_effect", top_k=5)
        if len(results) > 1:
            for i in range(len(results) - 1):
                assert results[i]["score"] >= results[i+1]["score"], \
                    "Results should be ordered by score descending"


class TestScoreQuality:
    def test_highly_relevant_query_has_high_score(self):
        results = retrieve_guidelines("midline shift 7mm herniation risk critical", intent_class="mass_effect", top_k=1)
        assert results
        assert results[0]["score"] >= 3.0, f"Expected score ≥ 3.0, got {results[0]['score']}"

    def test_irrelevant_query_returns_no_results(self):
        results = retrieve_guidelines("python programming tutorial for loops", intent_class=None, top_k=3, min_score=2.0)
        assert len(results) == 0, "Unrelated query should not return high-scoring results"

    def test_intent_boost_applied(self):
        results_with_intent = retrieve_guidelines("tumor volume", intent_class="quantitative", top_k=5)
        results_no_intent = retrieve_guidelines("tumor volume", intent_class=None, top_k=5)
        if results_with_intent and results_no_intent:
            brats_with = next((r for r in results_with_intent if "brats" in r["corpus"]), None)
            brats_without = next((r for r in results_no_intent if "brats" in r["corpus"]), None)
            if brats_with and brats_without:
                assert brats_with["score"] >= brats_without["score"], \
                    "Intent boost should increase score for preferred corpus"


class TestHallucinationGuard:
    def test_retrieved_text_exists_in_corpus(self, all_snippets):
        corpus_texts = {s["text"] for s in all_snippets}
        results = retrieve_guidelines("midline shift herniation motor cortex brainstem", top_k=5)
        for r in results:
            assert r["text"] in corpus_texts, \
                f"Retrieved snippet '{r['id']}' text not found in corpus — possible fabrication"

    def test_retrieved_id_exists_in_corpus(self, all_snippets):
        corpus_ids = {s["id"] for s in all_snippets}
        results = retrieve_guidelines("tumor volume enhancing necrotic edema", top_k=5)
        for r in results:
            assert r["id"] in corpus_ids, \
                f"Retrieved snippet ID '{r['id']}' not found in corpus"

    def test_mls_threshold_values_are_correct(self):
        results = retrieve_guidelines("midline shift threshold clinical significance", intent_class="mass_effect", top_k=1)
        assert results
        text = results[0]["text"]
        assert "5" in text, "MLS threshold text should mention 5mm threshold"

    def test_motor_cortex_threshold_values_correct(self):
        results = retrieve_guidelines("motor cortex distance surgical high risk", intent_class="critical_proximity", top_k=1)
        assert results
        text = results[0]["text"]
        assert "5" in text, "Motor cortex proximity text should mention 5mm threshold"

    def test_no_hallucinated_drug_names(self):
        results = retrieve_guidelines("treatment chemotherapy surgery glioblastoma", top_k=5)
        for r in results:
            text_lower = r["text"].lower()
            assert "aspirin" not in text_lower, "Aspirin should not appear in brain tumor guidelines"
            assert "insulin" not in text_lower, "Insulin should not appear in brain tumor guidelines"


class TestFormatGuidelineContext:
    def test_empty_returns_empty_string(self):
        assert format_guideline_context([]) == ""

    def test_format_has_guideline_header(self):
        results = retrieve_guidelines("midline shift", intent_class="mass_effect", top_k=1)
        if results:
            ctx = format_guideline_context(results)
            assert "[GUIDELINE CONTEXT]" in ctx
            assert "[END GUIDELINE CONTEXT]" in ctx
            assert "[G1]" in ctx

    def test_format_includes_snippet_text(self):
        results = retrieve_guidelines("motor cortex surgical planning", intent_class="critical_proximity", top_k=1)
        if results:
            ctx = format_guideline_context(results)
            assert results[0]["text"][:30] in ctx


class TestEdgeCases:
    def test_empty_query_returns_empty(self):
        results = retrieve_guidelines("", top_k=3)
        assert results == []

    def test_stopwords_only_query_returns_empty(self):
        results = retrieve_guidelines("the a is are of in", top_k=3)
        assert results == []

    def test_unknown_intent_still_retrieves(self):
        results = retrieve_guidelines("midline shift herniation", intent_class="unknown_intent_xyz", top_k=3)
        assert isinstance(results, list)

    def test_top_k_respected(self):
        results = retrieve_guidelines("tumor brain glioma MRI", top_k=2)
        assert len(results) <= 2

    def test_result_has_required_fields(self):
        results = retrieve_guidelines("midline shift", intent_class="mass_effect", top_k=1)
        if results:
            r = results[0]
            assert "id" in r
            assert "text" in r
            assert "tags" in r
            assert "corpus" in r
            assert "score" in r
