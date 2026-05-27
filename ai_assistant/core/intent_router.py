from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


_INTENT_CLASSES = {
    "measurement_query",
    "localization_query",
    "mass_effect_query",
    "critical_proximity_query",
    "differential_request",
    "urgency_assessment",
    "followup_recommendation",
    "report_summarization",
    "structured_reporting_query",
    "comparison_query",
    "limitation_query",
    "unknown",
}


@dataclass
class QAIntent:
    intent_class: str = "unknown"
    clinical_domain: str = "brain_neuro_oncology"
    target_compartment: str | None = None
    target_measurement: str | None = None
    anatomic_target: str | None = None
    reporting_system: str = "none"
    reporting_subtask: str | None = None
    urgency_signal: str = "none"
    temporal_scope: str = "current_scan"
    requires_retrieval: bool = False
    requires_guideline_retrieval: bool = False
    requires_numeric_truth: bool = True
    router_confidence: float = 0.0
    explanation: str = ""
    intent_source: str = "local_classifier"
    retrieval_policy: str = "none"
    legacy_question_class: str = "unknown"
    legacy_question_subtype: str = "unknown"
    canonical_question: str = ""
    ontology_hits: dict[str, list[str]] = field(default_factory=dict)
    class_probabilities: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _normalize_question(text: str) -> str:
    return " ".join(str(text or "").strip().lower().replace("-", " ").replace("/", " ").split())


def _contains_any(text: str, phrases: list[str]) -> bool:
    return any(phrase in text for phrase in phrases)


_MEASUREMENT_ONTOLOGY: dict[str, list[str]] = {
    "total_volume_mm3": ["total tumor volume", "whole tumor volume", "tumor volume", "total volume"],
    "max_diameter_mm": ["maximum diameter", "max diameter", "largest diameter", "how big", "how large", "diameter"],
    "enhancing_volume_mm3": ["enhancing volume", "enhancing component volume"],
    "edema_volume_mm3": ["edema volume", "what about edema", "how much edema"],
    "midline_shift_mm": ["midline shift", "is there shift", "septum pellucidum", "falx shift"],
    "ventricular_compression": ["ventricular compression", "ventricle compression", "ventricle"],
    "sulcal_effacement": ["sulcal effacement", "sulci"],
    "cisternal_obliteration": ["cisternal obliteration", "cisternal effacement", "basal cistern", "cistern"],
    "distance_to_brainstem_mm": ["distance to brainstem", "brainstem", "close to brainstem"],
    "distance_to_motor_cortex_mm": ["distance to motor cortex", "motor cortex", "motor area", "precentral"],
    "distance_to_optic_tract_mm": ["distance to optic tract", "optic tract", "optic pathway"],
}

_COMPARTMENT_ONTOLOGY: dict[str, list[str]] = {
    "whole_tumor": ["tumor", "lesion", "mass"],
    "enhancing_tumor": ["enhancing", "enhancement"],
    "nonenhancing_core": ["nonenhancing", "non enhancing", "necrotic", "core"],
    "edema": ["edema", "oedema"],
    "cavity": ["cavity", "resection cavity", "post op cavity"],
    "ventricle": ["ventricle", "ventricular"],
    "cistern": ["cistern", "cisternal"],
    "midline": ["midline", "septum pellucidum", "falx"],
}

_ANATOMY_ONTOLOGY: dict[str, list[str]] = {
    "brainstem": ["brainstem"],
    "motor_cortex": ["motor cortex", "motor area", "precentral"],
    "optic_pathway": ["optic tract", "optic pathway", "optic"],
    "septum_pellucidum": ["septum pellucidum"],
    "falx": ["falx"],
    "lateral_ventricle": ["lateral ventricle", "ventricle", "frontal horn", "third ventricle"],
}

_REPORTING_SYSTEMS: dict[str, list[str]] = {
    "BT-RADS": ["bt rads", "btrads", "bt-rads"],
    "LI-RADS": ["li rads", "lirads", "li-rads"],
}

_REPORTING_SUBTASKS: dict[str, list[str]] = {
    "score_request": ["category", "score", "best fits", "classification"],
    "score_explanation": ["explain score", "how is score determined"],
    "category_justification": ["why would", "why not", "justify"],
    "field_mapping": ["mapped", "mapping"],
    "progression_criteria": ["progression", "progressive", "criteria"],
    "report_field_completion": ["currently supported", "available fields", "supported fields", "supported"],
}

_URGENCY_PHRASES = {
    "explicit": ["urgent", "emergency", "dangerous", "serious", "immediate attention"],
    "implicit": ["worrisome", "concerning", "should i worry", "high risk"],
}

_CLASS_PROTOTYPES: dict[str, list[str]] = {
    "measurement_query": ["what is the total tumor volume", "how big is it", "maximum diameter", "what about edema"],
    "localization_query": ["where is the lesion located", "which region is involved"],
    "mass_effect_query": ["is there shift", "any compression", "sulcal effacement", "cisternal obliteration"],
    "critical_proximity_query": ["how close is it to motor cortex", "distance to brainstem"],
    "comparison_query": ["compare enhancing versus edema", "current versus prior"],
    "structured_reporting_query": ["what bt rads category fits", "which bt rads fields are supported"],
    "differential_request": ["what are the differentials", "what could this be"],
    "urgency_assessment": ["is this urgent", "does this look worrisome right now"],
    "followup_recommendation": ["what follow up is recommended", "what should be done next"],
    "report_summarization": ["summarize this case", "tumor board summary", "explain the lesion pattern in one paragraph", "plain english summary"],
    "limitation_query": ["what are the limitations", "what information is missing"],
}


def _tokenize(text: str) -> set[str]:
    stopwords = {
        "a", "an", "the", "is", "it", "to", "for", "of", "and", "or", "what", "which",
        "how", "are", "there", "this", "that", "in", "on", "right", "now", "i", "need",
    }
    return {token for token in _normalize_question(text).replace("?", "").split() if token and token not in stopwords}


def _jaccard_score(a: str, b: str) -> float:
    ta = _tokenize(a)
    tb = _tokenize(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(len(ta | tb), 1)


def _best_match(text: str, options: dict[str, list[str]]) -> tuple[str | None, float, list[str]]:
    best_key: str | None = None
    best_score = 0.0
    hits: list[str] = []
    for key, phrases in options.items():
        direct_hits = [phrase for phrase in phrases if phrase in text]
        score = max((_jaccard_score(text, phrase) for phrase in phrases), default=0.0)
        if direct_hits:
            score = max(score, 0.86)
        if score > best_score:
            best_key = key
            best_score = score
            hits = direct_hits or phrases[:1]
    return best_key, best_score, hits


def _class_probabilities(text: str) -> dict[str, float]:
    scores: dict[str, float] = {}
    for key, phrases in _CLASS_PROTOTYPES.items():
        direct_bonus = 0.2 if _contains_any(text, phrases) else 0.0
        score = max((_jaccard_score(text, phrase) for phrase in phrases), default=0.0) + direct_bonus
        scores[key] = round(min(score, 0.99), 4)
    total = sum(scores.values())
    if total <= 0:
        return {"unknown": 1.0}
    return {key: round(val / total, 4) for key, val in scores.items()}


def _top_two(probabilities: dict[str, float]) -> tuple[tuple[str, float], tuple[str, float]]:
    ranked = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
    first = ranked[0] if ranked else ("unknown", 0.0)
    second = ranked[1] if len(ranked) > 1 else ("unknown", 0.0)
    return first, second


def _retrieval_policy(intent_class: str) -> tuple[bool, bool, str]:
    if intent_class in {"measurement_query", "mass_effect_query", "critical_proximity_query"}:
        return False, False, "current_case_only"
    if intent_class == "localization_query":
        return True, False, "current_case_primary_optional_case_retrieval"
    if intent_class in {"structured_reporting_query", "differential_request", "urgency_assessment", "followup_recommendation"}:
        return True, True, "current_case_plus_guidelines_plus_cases"
    if intent_class in {"report_summarization", "limitation_query", "comparison_query"}:
        return False, False, "current_case_only"
    return False, False, "none"


def _legacy_mapping(intent: QAIntent) -> tuple[str, str]:
    if intent.intent_class == "measurement_query":
        measurement_map = {
            "total_volume_mm3": "total_volume",
            "max_diameter_mm": "max_diameter",
            "enhancing_volume_mm3": "enhancing_volume",
            "edema_volume_mm3": "edema_volume",
        }
        return "quantitative", measurement_map.get(intent.target_measurement or "", "ambiguous_size")
    if intent.intent_class == "comparison_query":
        return "comparison", "enhancing_vs_nonenhancing"
    if intent.intent_class == "localization_query":
        return "localization", "largest_lesion_location"
    if intent.intent_class == "mass_effect_query":
        subtype_map = {
            "midline_shift_mm": "midline_shift",
            "ventricular_compression": "ventricular_compression",
            "sulcal_effacement": "sulcal_effacement",
            "cisternal_obliteration": "cisternal_obliteration",
        }
        return "mass_effect", subtype_map.get(intent.target_measurement or "", "mass_effect_unspecified")
    if intent.intent_class == "critical_proximity_query":
        subtype_map = {
            "distance_to_brainstem_mm": "brainstem",
            "distance_to_motor_cortex_mm": "motor_cortex",
            "distance_to_optic_tract_mm": "optic_tract",
        }
        return "critical_proximity", subtype_map.get(intent.target_measurement or "", "critical_structure")
    if intent.intent_class == "structured_reporting_query":
        system = (intent.reporting_system or "structured").lower().replace("-", "")
        subtask = intent.reporting_subtask or "score_request"
        return "structured_reporting", f"{system}_{subtask}"
    if intent.intent_class == "differential_request":
        return "diagnosis/differential", "diagnostic_interpretation"
    if intent.intent_class == "urgency_assessment":
        return "management/prognosis", "clinical_concern"
    if intent.intent_class == "followup_recommendation":
        return "management/prognosis", "management_recommendation"
    if intent.intent_class == "report_summarization":
        return "summary", "tumor_board_summary"
    if intent.intent_class == "limitation_query":
        return "summary", "limitations"
    return "unknown", "unknown"


def _inherit_from_session(intent: QAIntent, previous: dict[str, Any] | None) -> QAIntent:
    if not previous:
        return intent
    for key in ("target_compartment", "target_measurement", "anatomic_target", "reporting_system", "reporting_subtask"):
        if getattr(intent, key) in {None, "", "none"}:
            prior = previous.get(key)
            if prior not in {None, "", "none"}:
                setattr(intent, key, prior)
    if intent.intent_class == "unknown" and previous.get("intent_class") in _INTENT_CLASSES:
        intent.intent_class = str(previous.get("intent_class"))
        intent.intent_source = "session_followup"
        intent.router_confidence = max(intent.router_confidence, 0.58)
    return intent


def resolve_qa_intent_local(question: str, previous_intent: dict[str, Any] | None = None) -> QAIntent:
    normalized = _normalize_question(question)
    probabilities = _class_probabilities(normalized)
    (top_class, top_score), (_, second_score) = _top_two(probabilities)
    measurement, measurement_score, measurement_hits = _best_match(normalized, _MEASUREMENT_ONTOLOGY)
    compartment, compartment_score, compartment_hits = _best_match(normalized, _COMPARTMENT_ONTOLOGY)
    anatomy, anatomy_score, anatomy_hits = _best_match(normalized, _ANATOMY_ONTOLOGY)
    reporting_system, reporting_score, reporting_hits = _best_match(normalized, _REPORTING_SYSTEMS)
    reporting_subtask, reporting_subtask_score, reporting_subtask_hits = _best_match(normalized, _REPORTING_SUBTASKS)
    has_direct_signal = any(
        [
            measurement_score >= 0.86,
            compartment_score >= 0.86,
            anatomy_score >= 0.86,
            reporting_score >= 0.86,
            _contains_any(normalized, _URGENCY_PHRASES["explicit"] + _URGENCY_PHRASES["implicit"]),
        ]
    )

    if reporting_score >= 0.7:
        top_class = "structured_reporting_query"
        top_score = max(top_score, 0.82)
    elif measurement in {"midline_shift_mm", "ventricular_compression", "sulcal_effacement", "cisternal_obliteration"} and measurement_score >= 0.7:
        top_class = "mass_effect_query"
        top_score = max(top_score, 0.8)
    elif measurement in {"distance_to_brainstem_mm", "distance_to_motor_cortex_mm", "distance_to_optic_tract_mm"} and measurement_score >= 0.7:
        top_class = "critical_proximity_query"
        top_score = max(top_score, 0.8)
    elif _contains_any(normalized, ["where", "located", "location", "localization"]) and anatomy_score >= 0.5:
        top_class = "localization_query"
        top_score = max(top_score, 0.74)

    urgency_signal = "none"
    if _contains_any(normalized, _URGENCY_PHRASES["explicit"]):
        urgency_signal = "explicit"
        top_class = "urgency_assessment"
        top_score = max(top_score, 0.82)
    elif _contains_any(normalized, _URGENCY_PHRASES["implicit"]):
        urgency_signal = "implicit"
        top_class = "urgency_assessment"
        top_score = max(top_score, 0.72)

    if _contains_any(normalized, ["follow up", "follow-up", "next step", "recommended"]) and top_class != "structured_reporting_query":
        top_class = "followup_recommendation"
        top_score = max(top_score, 0.74)

    if _contains_any(normalized, ["summary", "summarize", "plain language", "tumor board", "one paragraph", "clinical paragraph"]):
        top_class = "report_summarization"
        top_score = max(top_score, 0.76)

    if _contains_any(normalized, ["limitation", "uncertainty", "missing information", "what is missing"]):
        top_class = "limitation_query"
        top_score = max(top_score, 0.76)

    if _contains_any(normalized, ["compare", "versus", "vs", "difference between"]):
        top_class = "comparison_query"
        top_score = max(top_score, 0.72)

    if _contains_any(normalized, ["differential", "what could this be", "most likely diagnosis"]):
        top_class = "differential_request"
        top_score = max(top_score, 0.78)

    if not has_direct_signal and top_score < 0.6:
        top_class = "unknown"
        top_score = min(top_score, 0.42) if top_score else 0.42

    if len(normalized.split()) <= 2 and top_score < 0.7 and normalized not in {"summary"}:
        top_class = "unknown"
        top_score = min(top_score, 0.44)

    intent = QAIntent(
        intent_class=top_class if top_class in _INTENT_CLASSES else "unknown",
        target_compartment=compartment if compartment_score >= 0.6 else None,
        target_measurement=measurement if measurement_score >= 0.58 else None,
        anatomic_target=anatomy if anatomy_score >= 0.58 else None,
        reporting_system=reporting_system if reporting_score >= 0.7 and reporting_system else "none",
        reporting_subtask=reporting_subtask if reporting_subtask_score >= 0.58 else None,
        urgency_signal=urgency_signal,
        router_confidence=round(max(top_score, probabilities.get(top_class, 0.0)), 4),
        canonical_question=normalized,
        ontology_hits={
            "measurement": measurement_hits,
            "compartment": compartment_hits,
            "anatomy": anatomy_hits,
            "reporting_system": reporting_hits,
            "reporting_subtask": reporting_subtask_hits,
        },
        class_probabilities=probabilities,
        explanation=f"Local ontology-aware classifier routed to {top_class}.",
    )
    intent = _inherit_from_session(intent, previous_intent)
    if "how big" in normalized or normalized == "size":
        if previous_intent and previous_intent.get("target_measurement") in {"max_diameter_mm", "total_volume_mm3"}:
            intent.intent_class = "measurement_query"
            intent.target_measurement = str(previous_intent.get("target_measurement"))
            intent.intent_source = "session_followup"
            intent.router_confidence = max(intent.router_confidence, 0.62)
        else:
            intent.intent_class = "measurement_query"
            intent.target_measurement = None
            intent.router_confidence = min(intent.router_confidence, 0.44) if intent.router_confidence else 0.44
    if intent.intent_class == "unknown" and normalized == "summary":
        intent.intent_class = "report_summarization"
        intent.router_confidence = 0.76
    if intent.intent_class == "unknown":
        intent.requires_numeric_truth = False
        intent.intent_source = "fallback_unknown"
    intent.requires_retrieval, intent.requires_guideline_retrieval, intent.retrieval_policy = _retrieval_policy(intent.intent_class)
    legacy_class, legacy_subtype = _legacy_mapping(intent)
    intent.legacy_question_class = legacy_class
    intent.legacy_question_subtype = legacy_subtype
    return intent


def intent_needs_arbiter(intent: QAIntent) -> bool:
    probabilities = sorted(intent.class_probabilities.values(), reverse=True)
    top = probabilities[0] if probabilities else 0.0
    second = probabilities[1] if len(probabilities) > 1 else 0.0
    close_margin = (top - second) < 0.08
    return any(
        [
            close_margin and top < 0.7,
            intent.intent_class == "structured_reporting_query" and intent.reporting_subtask is None,
            intent.intent_class == "urgency_assessment" and intent.urgency_signal == "implicit",
            intent.intent_class == "measurement_query" and intent.target_measurement is None,
        ]
    )


def apply_intent_arbiter_result(intent: QAIntent, arbiter_result: dict[str, Any] | None) -> QAIntent:
    if not arbiter_result:
        return intent
    for key in (
        "intent_class",
        "target_compartment",
        "target_measurement",
        "anatomic_target",
        "reporting_system",
        "reporting_subtask",
        "urgency_signal",
        "temporal_scope",
        "explanation",
    ):
        value = arbiter_result.get(key)
        if value not in {None, ""}:
            setattr(intent, key, value)
    confidence = arbiter_result.get("router_confidence")
    if isinstance(confidence, (int, float)):
        intent.router_confidence = float(confidence)
    intent.intent_source = "deepseek_arbiter"
    intent.requires_retrieval, intent.requires_guideline_retrieval, intent.retrieval_policy = _retrieval_policy(intent.intent_class)
    legacy_class, legacy_subtype = _legacy_mapping(intent)
    intent.legacy_question_class = legacy_class
    intent.legacy_question_subtype = legacy_subtype
    return intent
