from __future__ import annotations

import re


EXPECTED_REPORT_SECTIONS = [
    "# Clinical Context",
    "# Findings",
    "## Localization",
    "## Mass Effect",
    "## Midline Shift",
    "## Enhancement Pattern",
    "## Multifocality",
    "# Impression",
    "# Limitations",
    "# Suggested Clinical Correlation",
    "# Disclaimer",
]

CLINICAL_TERMS = {
    "tumor",
    "lesion",
    "enhancing",
    "non-enhancing",
    "nonenhancing",
    "edema",
    "mass",
    "midline",
    "shift",
    "multifocal",
    "glioma",
    "histopathological",
    "radiologist",
    "clinical",
    "parietal",
    "frontal",
    "temporal",
    "occipital",
    "periventricular",
    "subcortical",
}


def _words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9\-']+", text.lower())


def _sentences(text: str) -> list[str]:
    chunks = re.split(r"[.!?]+|\n+", text)
    return [chunk.strip() for chunk in chunks if chunk.strip()]


def _count_syllables(word: str) -> int:
    token = re.sub(r"[^a-z]", "", word.lower())
    if not token:
        return 1
    groups = re.findall(r"[aeiouy]+", token)
    count = len(groups)
    if token.endswith("e") and count > 1:
        count -= 1
    return max(1, count)


def _flesch_reading_ease(text: str) -> float:
    words = _words(text)
    sentences = _sentences(text)
    if not words or not sentences:
        return 0.0
    syllables = sum(_count_syllables(word) for word in words)
    words_per_sentence = len(words) / len(sentences)
    syllables_per_word = syllables / len(words)
    return round(206.835 - (1.015 * words_per_sentence) - (84.6 * syllables_per_word), 2)


def _section_body(text: str, header: str) -> str:
    start = text.find(header)
    if start < 0:
        return ""
    body_start = start + len(header)
    next_headers = [idx for idx in (text.find(marker, body_start) for marker in ["\n## ", "\n# "]) if idx >= 0]
    end = min(next_headers) if next_headers else len(text)
    return text[body_start:end].strip()


def _is_filled_section(text: str, header: str) -> bool:
    body = _section_body(text, header)
    if not body:
        return False
    low = body.lower()
    return "not emphasized in conservative mode" not in low and "unavailable" not in low


def calculate_quality_metrics(report_text: str) -> dict[str, float]:
    words = _words(report_text)
    sentences = _sentences(report_text)
    total_words = len(words)
    unique_words = len(set(words))
    clinical_terms = sum(1 for word in words if word in CLINICAL_TERMS)
    filled_sections = sum(1 for header in EXPECTED_REPORT_SECTIONS if _is_filled_section(report_text, header))

    lexical_diversity = (unique_words / total_words) if total_words else 0.0
    avg_sentence_length = (total_words / len(sentences)) if sentences else 0.0
    clinical_term_density = (clinical_terms / total_words) if total_words else 0.0
    information_coverage = (filled_sections / len(EXPECTED_REPORT_SECTIONS)) if EXPECTED_REPORT_SECTIONS else 0.0

    return {
        "lexical_diversity": round(lexical_diversity, 4),
        "flesch_reading_ease": _flesch_reading_ease(report_text),
        "avg_sentence_length": round(avg_sentence_length, 4),
        "clinical_term_density": round(clinical_term_density, 4),
        "information_coverage": round(information_coverage, 4),
    }
