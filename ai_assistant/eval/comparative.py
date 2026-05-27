from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Any

from ai_assistant.core.context_loader import load_patient_context
from ai_assistant.core.report_generator import generate_report_draft
from ai_assistant.eval.report_quality import calculate_quality_metrics


def _tokens(text: str) -> list[str]:
    return [token for token in text.lower().split() if token.strip()]


def _rouge1_f1(text_a: str, text_b: str) -> float:
    a = _tokens(text_a)
    b = _tokens(text_b)
    if not a or not b:
        return 0.0
    overlap = len(set(a) & set(b))
    precision = overlap / len(set(a))
    recall = overlap / len(set(b))
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _bleu_like(text_a: str, text_b: str) -> float:
    a = _tokens(text_a)
    b = _tokens(text_b)
    if not a or not b:
        return 0.0
    overlap = sum(1 for token in a if token in set(b))
    return overlap / len(a)


def _is_safe(report_text: str) -> float:
    low = report_text.lower()
    return 0.0 if "glioblastoma" in low or "chemotherapy immediately" in low else 1.0


def compare_settings(
    case_id: str,
    settings_list: list[str] | None = None,
    *,
    context: Any | None = None,
    contexts_root: str | Path = "outputs/cases",
) -> dict[str, Any]:
    settings = settings_list or ["conservative", "balanced", "detailed"]
    ctx = context or load_patient_context(case_id, str(contexts_root))

    variants: list[dict[str, Any]] = []
    reports: dict[str, str] = {}
    for setting in settings:
        response = generate_report_draft(ctx, detail_mode=setting, allow_llm=False)
        report_text = response.report_markdown
        quality = calculate_quality_metrics(report_text)
        reports[setting] = report_text
        variants.append(
            {
                "setting": setting,
                "coverage": quality["information_coverage"],
                "safety": _is_safe(report_text),
                "lexical_diversity": quality["lexical_diversity"],
                "avg_sentence_length": quality["avg_sentence_length"],
            }
        )

    pair_scores: list[float] = []
    for left, right in combinations(settings, 2):
        rouge = _rouge1_f1(reports[left], reports[right])
        bleu = _bleu_like(reports[left], reports[right])
        pair_scores.append((rouge + bleu) / 2.0)

    consistency_score = round(sum(pair_scores) / len(pair_scores), 4) if pair_scores else 1.0

    def _score(item: dict[str, Any]) -> float:
        safety = float(item["safety"])
        coverage = float(item["coverage"])
        lexical_diversity = float(item["lexical_diversity"])
        preference_bias = 0.01 if item["setting"] == "balanced" else 0.0
        return (coverage * 0.6) + (safety * 0.3) + (lexical_diversity * 0.1) + preference_bias

    recommended_variant = max(variants, key=_score) if variants else {"setting": "balanced"}
    return {
        "case_id": case_id,
        "variants": variants,
        "consistency_score": consistency_score,
        "recommended": recommended_variant["setting"],
    }
