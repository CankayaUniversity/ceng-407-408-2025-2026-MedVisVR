from __future__ import annotations

import json
import logging
import math
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)

_GUIDELINES_DIR = Path(__file__).resolve().parents[2] / "docs" / "guidelines"

_INTENT_CORPUS_MAP: dict[str, list[str]] = {
    "mass_effect":          ["mass_effect_snippets", "neurosurgical_snippets"],
    "critical_proximity":   ["neurosurgical_snippets", "mass_effect_snippets"],
    "structured_reporting": ["btrads_snippets", "brats_label_snippets"],
    "quantitative":         ["brats_label_snippets", "btrads_snippets"],
    "localization":         ["brats_label_snippets", "neurosurgical_snippets"],
    "comparison":           ["brats_label_snippets", "btrads_snippets"],
    "diagnosis/differential": ["brats_label_snippets", "btrads_snippets"],
    "management/prognosis": ["neurosurgical_snippets", "btrads_snippets"],
    "summary":              ["btrads_snippets", "brats_label_snippets"],
    "ct_organ":             ["ct_liver_snippets", "neurosurgical_snippets"],
    "measurement_query":    ["ct_liver_snippets", "brats_label_snippets"],
    "localization_query":   ["ct_liver_snippets", "brats_label_snippets"],
}

_STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "of", "in", "on", "at",
    "to", "for", "with", "by", "from", "this", "that", "which", "what",
    "how", "and", "or", "not", "it", "its", "there", "their", "they",
}


def _tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+(?:['-][a-z0-9]+)*", text.lower())
    return [t for t in tokens if t not in _STOP_WORDS and len(t) > 1]


@lru_cache(maxsize=1)
def _load_all_snippets() -> list[dict[str, Any]]:
    all_snippets: list[dict[str, Any]] = []
    if not _GUIDELINES_DIR.exists():
        LOGGER.warning("Guidelines directory not found: %s", _GUIDELINES_DIR)
        return all_snippets

    for path in sorted(_GUIDELINES_DIR.glob("*_snippets.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            corpus_name = path.stem
            for snippet in data.get("snippets", []):
                snippet = dict(snippet)
                snippet["_corpus"] = corpus_name
                snippet["_tokens"] = _tokenize(snippet.get("text", "") + " " + " ".join(snippet.get("tags", [])))
                all_snippets.append(snippet)
        except Exception as exc:
            LOGGER.warning("Failed to load guideline corpus %s: %s", path.name, exc)

    LOGGER.info("Loaded %d guideline snippets from %s", len(all_snippets), _GUIDELINES_DIR)
    return all_snippets


def _bm25_score(query_tokens: list[str], doc_tokens: list[str], avg_dl: float, k1: float = 1.5, b: float = 0.75) -> float:
    if not query_tokens or not doc_tokens:
        return 0.0
    dl = len(doc_tokens)
    freq: dict[str, int] = {}
    for t in doc_tokens:
        freq[t] = freq.get(t, 0) + 1
    score = 0.0
    for qt in query_tokens:
        tf = freq.get(qt, 0)
        if tf == 0:
            continue
        tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / max(avg_dl, 1)))
        score += tf_norm
    return score


def _intent_corpus_boost(snippet: dict[str, Any], preferred_corpora: list[str]) -> float:
    corpus = snippet.get("_corpus", "")
    if corpus in preferred_corpora:
        idx = preferred_corpora.index(corpus)
        return 1.5 - idx * 0.2  
    return 1.0


def retrieve_guidelines(
    query: str,
    intent_class: str | None = None,
    top_k: int = 3,
    min_score: float = 0.5,
) -> list[dict[str, Any]]:

    snippets = _load_all_snippets()
    if not snippets:
        return []

    query_tokens = _tokenize(query)
    if not query_tokens:
        return []

    avg_dl = sum(len(s["_tokens"]) for s in snippets) / len(snippets)
    preferred = _INTENT_CORPUS_MAP.get(intent_class or "", [])

    scored: list[tuple[float, dict[str, Any]]] = []
    for snippet in snippets:
        base = _bm25_score(query_tokens, snippet["_tokens"], avg_dl)
        boost = _intent_corpus_boost(snippet, preferred)
        final = base * boost
        if final >= min_score:
            scored.append((final, snippet))

    scored.sort(key=lambda x: x[0], reverse=True)

    results = []
    for score, snippet in scored[:top_k]:
        results.append({
            "id": snippet.get("id", ""),
            "text": snippet.get("text", ""),
            "tags": snippet.get("tags", []),
            "corpus": snippet.get("_corpus", ""),
            "score": round(score, 3),
        })
    return results


def format_guideline_context(snippets: list[dict[str, Any]]) -> str:
    if not snippets:
        return ""
    lines = ["[GUIDELINE CONTEXT]"]
    for i, s in enumerate(snippets, 1):
        lines.append(f"[G{i}] {s['text']}")
    lines.append("[END GUIDELINE CONTEXT]")
    return "\n".join(lines)
