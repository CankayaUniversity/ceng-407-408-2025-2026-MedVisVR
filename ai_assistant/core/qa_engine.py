import logging
import os
from typing import Any

from ai_assistant.api.schemas import QARequest, QAResponse
from ai_assistant.core.anatomic_mapper import anatomy_from_lesion
from ai_assistant.core.clinical_implications import (
    append_implication_to_answer,
    build_clinical_implication,
)
from ai_assistant.core.guardrails import (
    confidence_from_context_quality,
    enforce_english_only,
    safety_note_for_low_evidence,
    validate_generated_clinical_text,
    validate_patient_context,
)
from ai_assistant.core.intent_router import (
    QAIntent,
    _retrieval_policy,
    apply_intent_arbiter_result,
    intent_needs_arbiter,
    resolve_qa_intent_local,
)
from ai_assistant.core.model_client import (
    generate_intent_json,
    generate_intent_text,
    generate_qa_narrative_text,
    generate_qa_text,
)
from ai_assistant.core.qa_session_memory import (
    append_turn,
    build_session_context,
    clear_session,
    latest_resolved_intent,
    session_turn_count,
)

try:
    from rapidfuzz import fuzz as _fuzz
    _HAS_RAPIDFUZZ = True
except ImportError:
    _HAS_RAPIDFUZZ = False

try:
    from spellchecker import SpellChecker as _SpellChecker
    _spell = _SpellChecker()
    _MEDICAL_VOCAB = [
        "tumor", "tumour", "lesion", "lesions", "diameter", "enhancing",
        "nonenhancing", "non-enhancing", "edema", "glioblastoma", "chemotherapy",
        "prognosis", "segmentation", "sequences", "imaging", "surgery",
        "treatment", "diagnosis", "histopathological", "histopathology",
        "multidisciplinary", "neurosurgery", "oncology", "radiology",
        "neoplasm", "metastatic", "lymphoma", "infiltrative", "heterogeneous",
        "enhancement", "diffuse", "focal", "biopsy", "resection", "workup",
        "mri", "flair", "t1ce", "t1", "t2", "nifti", "voxel", "mm3", "cm3",
    ]
    _spell.word_frequency.load_words(_MEDICAL_VOCAB)
    _HAS_SPELLCHECKER = True
except ImportError:
    _HAS_SPELLCHECKER = False


LOGGER = logging.getLogger(__name__)

_SPELL_SKIP_PATTERNS = {"mr", "mri", "ct", "pet", "vs", "t1", "t2", "t1ce", "flair", "mm3", "cm3"}

_FIXED_QUESTION_INTENTS: dict[str, dict[str, list[str]]] = {
    "quantitative": {
        "edema_ratio": [
            "edema-to-tumor volume ratio", "edema to tumor ratio", "edema ratio", "edema volume ratio",
        ],
        "total_volume": [
            "total tumor volume", "whole tumor volume", "overall tumor volume",
            "tumor size", "tumor volume", "total volume",
            "how big is the tumor", "how large is the tumor", "size of the tumor", "size of tumor",
            "hacim", "boyut", "ölçü", "ne kadar", "kaç", "how much", "how large",
        ],
        "lesion_count": [
            "how many lesion", "how many lesions", "lesion count", "tumor count",
            "lesions are detected", "lesions were detected", "lesion are detected", "lesion were detected",
            "number of lesions", "number of tumors", "how many tumors",
            "lesion number", "lesion detected", "detected lesions",
        ],
        "component_count": [
            "how many components", "component count",
        ],
        "dominant_component": [
            "which component has the largest volume", "which component is the largest",
            "which component is largest", "what component has the largest volume",
            "dominant burden component",
            "largest burden component", "which component dominates",
        ],
        "max_diameter": [
            "maximum tumor diameter", "maximum lesion diameter", "max diameter",
            "largest diameter", "biggest diameter", "tumor diameter",
            "lesion diameter", "how wide is", "how large is the lesion",
            "size of the lesion", "diameter of the tumor", "diameter of tumor",
        ],
        "enhancing_volume": [
            "enhancing component volume",
        ],
        "edema_volume": [
            "edema volume", "edema component volume", "how much edema", "what about edema",
        ],
    },
    "comparison": {
        "enhancing_vs_nonenhancing": [
            "compare enhancing vs non-enhancing", "enhancing vs non-enhancing",
            "compare enhancing and non-enhancing", "enhancing vs non-enhancing distribution clinically",
            "enhancement versus non-enhancement", "enhancing vs infiltrative", "enhancement distribution",
            "active tumor vs infiltration",
        ],
        "multiplicity": [
            "multiple distinct lesions", "one complex mass",
            "multifocal disease versus a single dominant process", "multifocal disease",
            "is this multifocal", "multifocal or unifocal", "multiple lesions or one",
            "single lesion or multiple", "is there more than one lesion", "are there multiple tumors",
            "is this one tumor or multiple", "multifocality", "one dominant lesion group",
            "dominant lesion group", "dominant lesion or multifocal disease",
            "one dominant lesion or multifocal disease", "one dominant lesion group or multifocal disease",
        ],
        "enhancement_pattern": [
            "enhancement pattern",
        ],
    },
    "localization": {
        "largest_lesion_location": [
            "largest lesion and where", "where is the largest lesion", "where is the lesion",
            "largest lesion", "location of the lesion", "where is it located",
            "nerede", "bölge", "hangi", "yer", "lokasyon", "location", "where", "which lobe", "hangi lob",
        ],
    },
    "ct_organ": {
        "liver_volume": [
            "total liver volume", "liver volume", "how large is the liver", "liver size",
            "liver parenchyma volume",
        ],
        "liver_tumor_ratio": [
            "liver-to-tumor ratio", "liver to tumor ratio", "liver tumour ratio",
            "liver to tumour ratio", "tumor-to-liver ratio", "tumour-to-liver ratio",
            "tumor burden ratio", "tumour burden ratio", "tumor percentage of liver",
            "tumour percentage of liver", "liver lesion ratio",
        ],
        "lymph_node_involvement": [
            "lymph node", "lymph nodes", "nodal involvement", "lymphadenopathy",
            "mediastinal nodes", "hilar nodes", "nodal disease",
        ],
        "vascular_involvement": [
            "vascular involvement", "vessel involvement", "portal vein", "hepatic vein",
            "biliary involvement", "bile duct", "vascular invasion",
        ],
        "lobe_location": [
            "which lobe", "hepatic segment", "couinaud", "liver lobe",
            "lung lobe", "right lobe", "left lobe", "upper lobe", "lower lobe",
            "where is the tumor located", "where is the tumour located",
            "where is the lesion located", "tumor location", "tumour location",
            "lesion location",
        ],
        "lung_tumor_volume": [
            "lung tumor volume", "lung tumour volume", "lung lesion volume",
            "lung nodule volume", "how large is the lung tumor", "lung tumor size",
            "pulmonary tumor volume", "pulmonary lesion volume",
        ],
        "staging_ct": [
            "what stage", "ct staging", "tumor staging", "tnm stage",
            "is it operable", "resectable", "resectability",
        ],
        "ct_sequence": [
            "which ct", "ct protocol", "contrast ct", "imaging protocol",
            "what imaging modality", "what modality", "what scan",
        ],
    },
    "mass_effect": {
        "midline_shift": [
            "what is the midline shift", "midline shift", "mls",
            "how much midline shift", "degree of midline shift",
        ],
        "ventricular_compression": [
            "is there ventricular compression", "ventricular compression",
            "ventricle compressed", "ventricular involvement",
        ],
        "sulcal_effacement": [
            "is there sulcal effacement", "sulcal effacement", "sulcal obliteration",
        ],
        "cisternal_obliteration": [
            "cisternal obliteration", "is there cisternal obliteration",
        ],
        "herniation_risk": [
            "what is the herniation risk tier", "herniation risk tier",
            "herniation risk", "what is the herniation risk",
            "transtentorial herniation", "uncal herniation", "herniation",
        ],
    },
    "critical_proximity": {
        "motor_cortex": [
            "how close is the tumor to the motor cortex",
            "motor cortex proximity", "motor cortex distance",
            "distance to motor cortex", "near motor cortex",
        ],
        "brainstem": [
            "how close is the tumor to the brainstem",
            "brainstem proximity", "brainstem distance",
            "distance to brainstem", "near brainstem",
        ],
        "cortical_depth": [
            "how deep is the tumor from the cortical surface",
            "cortical surface depth", "depth from cortex", "cortical depth",
        ],
        "optic_tract": [
            "how close is the tumor to the optic tract",
            "optic tract distance", "optic tract proximity",
        ],
    },
    "structured_reporting": {
        "btrads_score_request": [
            "generate a bt-rads structured report", "bt-rads structured report",
            "bt-rads report", "btrads report", "btrads", "bt-rads",
            "bt-rads score", "btrads score", "structured report",
        ],
    },
    "summary": {
        "generic_summary": [
            "summary", "overall summary", "give me a summary", "provide a summary",
            "summarize this case", "summarize the case", "summary of findings",
            "overall findings", "overall impression", "brief summary",
            "overview", "overall", "genel", "özet", "rapor", "report summary", "özetle", "anlat",
        ],
        "sequence_inventory": [
            "which mr sequences", "mr sequences", "available sequences",
        ],
        "narrative_summary": [
            "give a concise narrative summary of this imaging case",
            "concise narrative summary of this imaging case",
            "concise narrative summary",
            "narrative summary of this imaging case",
            "narrative summary",
            "write a short narrative explanation of this case for tumor board discussion",
            "short narrative explanation of this case for tumor board discussion",
            "write a short narrative explanation of this case",
            "short narrative explanation of this case",
            "explain the lesion pattern in one short clinical paragraph",
            "lesion pattern in one short clinical paragraph",
            "short clinical paragraph",
            "natural language summary of this imaging case",
            "natural language summary",
            "summarize the imaging findings in natural language",
        ],
        "plain_language_summary": [
            "describe this case", "describe the case", "describe this case in plain",
            "describe this case in plain english", "plain english", "explain this case",
            "explain the case", "describe this imaging case",
        ],
        "tumor_board_summary": [
            "tumor board style summary", "concise tumor board style summary", "summarize this case",
            "case summary", "summarize the case", "summarize the key imaging findings",
            "key imaging findings", "key findings for this patient", "give me a summary",
            "provide a summary", "overall summary", "summary of findings", "brief summary",
            "tumor board", "mdt summary", "multidisciplinary team summary",
        ],
        "limitations": [
            "top 3 limitations of this case-level analysis", "limitations of this case-level analysis",
            "what are the limitations", "limitations of this analysis", "what can't this system do",
            "what cannot this system do", "limitations of imaging", "imaging limitations",
            "what are the constraints", "analysis limitations",
        ],
    },
    "diagnosis/differential": {
        "diagnostic_interpretation": [
            "definitely a glioblastoma", "is this glioblastoma", "is it glioblastoma",
            "could this be glioblastoma", "could it be glioblastoma", "might this be glioblastoma",
            "looks like glioblastoma", "glioblastoma possible", "glioblastoma pattern",
            "consistent with glioblastoma", "suggest glioblastoma",
            "is this cancer", "is it cancer", "could this be cancer",
            "is this malignant", "is it malignant", "could this be malignant",
            "is this a tumor", "is this tumor",
            "diagnose based on imaging alone", "can you diagnose based on imaging alone",
            "can imaging alone establish definitive diagnosis", "can imaging diagnose",
            "imaging alone diagnose", "diagnose from imaging", "diagnose from mri", "mri diagnose",
            "key differential diagnoses", "differential diagnoses ranked", "differential diagnosis",
            "what are the differentials", "possible diagnoses", "what could this be",
            "what is this likely", "most likely diagnosis", "probable diagnosis",
            "high-grade glioma pattern", "high grade glioma", "low-grade glioma",
            "low grade glioma", "glioma pattern", "glioma likely",
            "tanı", "ne olabilir", "differential", "ayırıcı", "olasılık",
        ],
    },
    "management/prognosis": {
        "management_recommendation": [
            "start chemotherapy immediately", "should chemotherapy be started", "chemotherapy recommended",
            "start chemo", "begin chemotherapy",
            "what treatment do you recommend", "what treatment is recommended", "recommend a treatment",
            "treatment options", "treatment plan", "treatment recommendation",
            "what should the treatment be", "how should this be treated", "management recommendation",
            "clinical management", "is surgery recommended", "should immediate surgery be recommended based only on this output",
            "surgery recommended", "should surgery be performed", "is surgical resection recommended",
            "surgical resection", "needs surgery", "require surgery",
            "follow-up workup is recommended", "before treatment planning", "next steps in management",
            "what are the next steps", "management plan", "clinical workup", "further workup",
            "radiation therapy", "radiotherapy recommended",
            "surgical planning considerations", "what are the surgical planning considerations",
            "surgical planning", "surgical approach", "surgical considerations",
            "neurosurgical planning", "presurgical planning",
            "ne yapmalı", "tedavi", "takip", "plan", "önerir", "öneri", "recommend",
        ],
        "prognosis_assessment": [
            "survival prognosis", "what's the cure rate", "what is the cure rate",
            "will the tumor grow or shrink", "what is the prognosis", "what's the prognosis",
            "prognosis for this patient", "how long will the patient live", "life expectancy",
            "survival rate", "survival outcome", "median survival", "will it grow",
            "will it spread", "tumor progression", "will this progress", "disease progression",
            "recurrence risk", "chance of recurrence", "will it come back",
        ],
        "clinical_concern": [
            "should i worry about this finding", "should i worry", "is this serious",
            "is this finding serious", "how serious is this", "is this dangerous", "is this urgent",
            "is this an emergency", "should the patient be concerned", "how concerning is this",
            "is this significant",
        ],
        "uncertainty_reduction": [
            "additional clinical information", "reduce diagnostic uncertainty", "what additional information",
            "what other information", "what clinical information", "what information would help",
            "what data is needed", "what else is needed", "missing information", "information needed",
            "reduce uncertainty", "improve certainty",
        ],
    },
}

_FIXED_QUESTION_CLASS_KEYS: dict[str, list[str]] = {
    question_class: [key for subtype_keys in subtype_map.values() for key in subtype_keys]
    for question_class, subtype_map in _FIXED_QUESTION_INTENTS.items()
}

_QUESTION_CLASS_PRIORITY: list[str] = [
    "ct_organ",
    "mass_effect",
    "critical_proximity",
    "structured_reporting",
    "localization",
    "comparison",
    "summary",
    "diagnosis/differential",
    "management/prognosis",
    "quantitative",
]


def _spell_correct_word(word: str) -> str:
    if not _HAS_SPELLCHECKER:
        return word
    if "-" in word or word in _SPELL_SKIP_PATTERNS or any(c.isdigit() for c in word):
        return word
    if len(word) <= 3:
        return word
    correction = _spell.correction(word)
    return correction if correction else word


def _normalize_question(q: str) -> str:
    corrections = {
        "tumor": ["tmor", "tumr", "tumer", "tumour", "tumor", "tumor"],
        "volume": ["volme", "volum", "volumee"],
        "lesion": ["leson", "leison", "lesoin", "lezyon"],
        "diameter": ["diamter", "dimeter", "diametr", "diametre"],
        "enhancing": ["enchancing", "enhanching", "enhacing"],
        "edema": ["edma", "edem"],
        "glioblastoma": ["glioblastma", "glioblstoma", "glioblastome"],
        "chemotherapy": ["chemoterapy", "kemotherapy"],
        "prognosis": ["prognsis", "prognos"],
        "segmentation": ["segmentaton", "segmantation"],
        "sequences": ["sequnces", "seqeunces"],
        "imaging": ["imging", "imagin", "imaing"],
        "surgery": ["surgey", "surger", "surjery"],
        "treatment": ["treatmnt", "tretment"],
        "diagnosis": ["diagnsis", "diagnos", "diagnossis"],
    }

    words = q.split()
    corrected = []
    for word in words:
        word_low = word.lower().strip("?.,!:;")
        matched = False

        for correct_term, typos in corrections.items():
            if word_low in typos:
                corrected.append(correct_term)
                matched = True
                break

        if matched:
            continue

        if _HAS_SPELLCHECKER:
            spell_result = _spell_correct_word(word_low)
            if spell_result != word_low:
                corrected.append(spell_result)
                matched = True

        if matched:
            continue

        if _HAS_RAPIDFUZZ:
            for correct_term in corrections:
                if "-" in word_low:
                    break
                score = _fuzz.ratio(word_low, correct_term)
                if len(word_low) >= 4 and score >= 85 and word_low != correct_term:
                    corrected.append(correct_term)
                    matched = True
                    break

        if not matched:
            corrected.append(word)

    return " ".join(corrected)


_INTENT_NORMALIZE_PROMPT = (
    "You are a medical NLP normalization assistant. "
    "Your ONLY job is to rewrite the user question in clean standard English "
    "while preserving the exact clinical intent. "
    "Rules:\n"
    "- Fix spelling errors (tmor -> tumor, whar -> what, etc.)\n"
    "- Fix grammar while keeping the same meaning\n"
    "- Do NOT add new clinical information\n"
    "- Do NOT answer the question\n"
    "- Return ONLY the rewritten question, nothing else\n"
    "- If the question is already clean, return it unchanged\n"
)

_SESSION_RESOLVE_PROMPT = (
    "You are a medical QA conversation resolver. "
    "Rewrite the CURRENT_QUESTION into a standalone English question using SESSION_CONTEXT only when needed. "
    "Rules:\n"
    "- Preserve the exact clinical intent.\n"
    "- Keep the same patient/case scope.\n"
    "- Resolve short follow-ups such as 'and what about edema?' or 'what about that one?'\n"
    "- Fix typos if needed.\n"
    "- Do NOT answer the question.\n"
    "- Do NOT add new clinical information.\n"
    "- Return ONLY the rewritten standalone question.\n"
)


def _llm_normalize_intent(question: str) -> str:
    try:
        normalized, model = generate_intent_text(
            question=question,
            system_prompt=_INTENT_NORMALIZE_PROMPT,
        )
        candidate = str(normalized or "").strip()
        if (
            candidate
            and len(candidate) > 0
            and "\n" not in candidate
            and "<think>" not in candidate.lower()
            and not candidate.startswith("#")
            and not candidate.startswith("- ")
            and len(candidate.split()) <= 18
        ):
            return candidate
    except Exception:
        pass
    return question


def _blocking_quality_issues(issues: list[str]) -> list[str]:
    blocking = {
        "missing patient_id",
        "missing study_id",
        "invalid modality (expected CT or MR)",
        "no nifti files detected",
    }
    return [issue for issue in issues if issue in blocking]


def _followup_like_question(question: str) -> bool:
    low = str(question or "").strip().lower()
    if not low:
        return False
    if len(low.split()) <= 4:
        return True
    followup_markers = [
        "what about",
        "and ",
        "that one",
        "this one",
        "those",
        "these",
        "it ",
        "its ",
        "same for",
        "how about",
    ]
    return any(marker in low for marker in followup_markers)


def _heuristic_followup_resolve(question: str) -> str:
    low = _normalize_question(str(question or "").lower())
    if not _followup_like_question(low):
        return str(question or "").strip()
    mapping = [
        (["edema"], "What is the edema component volume?"),
        (["enhancing"], "What is the enhancing component volume?"),
        (["non-enhancing", "nonenhancing"], "Compare enhancing vs non-enhancing tumor volumes."),
        (["diameter", "size", "wide"], "What is the maximum tumor diameter?"),
        (["location", "located", "where"], "What is the largest lesion and where is it located?"),
        (["sequence", "sequences", "scan", "modality"], "Which MR sequences are available?"),
        (["lesion", "lesions", "tumor count", "how many"], "How many lesions were detected?"),
        (["volume", "burden"], "What is the total tumor volume?"),
    ]
    for keys, rewritten in mapping:
        if any(key in low for key in keys):
            return rewritten
    return str(question or "").strip()


def _resolve_question_with_session(question: str, session_context: str, *, allow_llm: bool) -> str:
    cleaned = question.strip()
    if not cleaned or not session_context:
        return cleaned
    heuristic = _heuristic_followup_resolve(cleaned)
    if heuristic != cleaned:
        return heuristic
    if not allow_llm:
        return cleaned
    if not _followup_like_question(cleaned):
        return cleaned
    try:
        resolved, _ = generate_intent_text(
            question=f"SESSION_CONTEXT:\n{session_context}\n\nCURRENT_QUESTION:\n{cleaned}",
            system_prompt=_SESSION_RESOLVE_PROMPT,
        )
        if resolved:
            candidate = resolved.strip()
            if (
                len(candidate) >= 5
                and "SESSION_CONTEXT:" not in candidate
                and "CURRENT_QUESTION:" not in candidate
                and "\n" not in candidate
            ):
                return candidate
    except Exception:
        pass
    return cleaned


def _qa_llm_enabled() -> bool:
    return os.getenv("AI_USE_LLM", "1") == "1" and os.getenv("AI_QA_USE_LLM_NONSTRUCTURED", "1") == "1"


def _dedupe(items: list[str]) -> list[str]:
    out: list[str] = []
    for item in items:
        if item and item not in out:
            out.append(item)
    return out


def _extract_sequences(nifti_files: list[str]) -> list[str]:
    seq_map = [
        ("t1ce", "T1ce"),
        ("flair", "FLAIR"),
        ("t2", "T2"),
        ("t1", "T1"),
    ]
    found: list[str] = []
    for f in nifti_files:
        low = f.lower()
        for key, label in seq_map:
            if key in low and label not in found:
                found.append(label)
    return found


def _safe_div(num: float, den: float) -> float:
    if den <= 0:
        return 0.0
    return num / den


def _contains_any(q: str, keys: list[str]) -> bool:
    return any(k in q for k in keys)


def _as_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return {}


def _list_of_dicts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    out: list[dict[str, Any]] = []
    for item in value:
        item_dict = _as_dict(item)
        if item_dict:
            out.append(item_dict)
    return out


def _core_context_evidence_ids(payload: QARequest) -> list[str]:
    evidence = [
        "ctx:patient_id",
        "ctx:study_id",
        "ctx:study_date",
        "ctx:modality",
        "ctx:findings_structured.nifti_summary.nifti_files",
    ]
    if (payload.context.clinical_note or "").strip():
        evidence.append("ctx:clinical_note")
    if payload.context.prior_reports:
        evidence.append("ctx:prior_reports")
    return evidence


def _segmentation_evidence_ids(payload: QARequest) -> list[str]:
    seg = _as_dict(payload.context.segmentation_metrics)
    if not seg:
        return []
    evidence = [
        "ctx:segmentation_metrics.tumor_count",
        "ctx:segmentation_metrics.segmented_component_count",
        "ctx:segmentation_metrics.total_tumor_volume_mm3",
        "ctx:segmentation_metrics.max_diameter_mm",
        "ctx:segmentation_metrics.label_volumes_mm3.enhancing",
        "ctx:segmentation_metrics.label_volumes_mm3.nonenhancing",
        "ctx:segmentation_metrics.label_volumes_mm3.edema",
    ]
    if seg.get("lesion_list"):
        evidence.append("ctx:segmentation_metrics.lesion_list")
    return evidence


def _default_nonstructured_evidence_ids(payload: QARequest) -> list[str]:
    return _dedupe(_core_context_evidence_ids(payload) + _segmentation_evidence_ids(payload))


def _has_segmentation_metrics(context: Any) -> bool:
    seg = _as_dict(getattr(context, "segmentation_metrics", None))
    if not seg:
        return False
    lesion_list = _list_of_dicts(seg.get("lesion_list", []))
    label_volumes = _as_dict(seg.get("label_volumes_mm3"))
    if lesion_list or any(float(value or 0.0) > 0.0 for value in label_volumes.values()):
        return True
    if float(seg.get("total_tumor_volume_mm3", seg.get("volume_mm3", 0.0)) or 0.0) > 0.0:
        return True
    if float(seg.get("max_diameter_mm", 0.0) or 0.0) > 0.0:
        return True
    if int(seg.get("tumor_count", 0) or 0) > 0 or int(seg.get("segmented_component_count", 0) or 0) > 0:
        return True
    return bool(seg.get("source_mask_path") or seg.get("model_version") or seg.get("uncertainty"))


def _segmentation_unavailable_answer() -> str:
    return (
        "Insufficient evidence: segmentation-derived metrics are unavailable for this case package, "
        "so lesion burden, multiplicity, and component pattern cannot be determined."
    )



_FUZZY_THRESHOLD = 82


def _matches_intent(q: str, keys: list[str]) -> bool:
    if _contains_any(q, keys):
        return True

    if not _HAS_RAPIDFUZZ:
        return False

    for key in keys:
        if len(key) <= 6:
            continue
        score = _fuzz.token_set_ratio(q, key)
        if score >= _FUZZY_THRESHOLD:
            return True

    return False


def _is_insufficient_answer(text: str) -> bool:
    low = (text or "").lower().strip()
    keys = [
        "insufficient evidence",
        "cannot be estimated",
        "cannot be determined",
        "cannot be confirmed",
    ]
    return any(k in low for k in keys)


def _quality_issue_reason(issue: str) -> str:
    mapping = {
        "missing patient_id": "patient identifier missing",
        "missing study_id": "study identifier missing",
        "missing study_date": "study date missing",
        "invalid modality (expected CT or MR)": "modality metadata invalid",
        "no dicom files detected": "DICOM inputs unavailable",
        "no nifti files detected": "NIfTI case files unavailable",
    }
    return mapping.get(issue, issue.replace("_", " "))


def _resolved_intent(
    question: str,
    *,
    previous_intent: dict[str, Any] | None = None,
    can_use_arbiter: bool = False,
    session_context: str = "",
) -> QAIntent:
    q = " ".join(part.strip("?.,!:;") for part in _normalize_question(question.lower().strip()).split())
    legacy_class, legacy_subtype = "unknown", "unknown"
    for question_class in _QUESTION_CLASS_PRIORITY:
        subtype_map = _FIXED_QUESTION_INTENTS[question_class]
        if question_class == "ct_organ" and not any(term in q for term in ["ct", "lung", "liver", "hepatic", "couinaud", "portal vein", "bile duct"]):
            continue
        for question_subtype, keys in subtype_map.items():
            if _matches_intent(q, keys):
                legacy_class, legacy_subtype = question_class, question_subtype
                break
        if legacy_class != "unknown":
            break

    if legacy_class == "summary":
        if "narrative summary" in q or "narrative explanation" in q or "one paragraph" in q or "clinical paragraph" in q:
            legacy_subtype = "narrative_summary"
        elif "tumor board" in q or "mdt" in q:
            legacy_subtype = "tumor_board_summary"
        elif "plain english" in q or "plain language" in q:
            legacy_subtype = "plain_language_summary"

    intent = resolve_qa_intent_local(question, previous_intent=previous_intent)
    if can_use_arbiter and intent_needs_arbiter(intent):
        arbiter_result, _ = generate_intent_json(
            question=question,
            session_context=session_context,
            previous_intent=previous_intent,
            use_high_model=False,
        )
        intent = apply_intent_arbiter_result(intent, arbiter_result)

    if legacy_class != "unknown":
        new_specific_classes = {"mass_effect_query", "critical_proximity_query", "structured_reporting_query"}
        if (
            ("how big is it" in q and not previous_intent)
            or ("how big" in q and not previous_intent and legacy_class == "quantitative")
            or ("how big" in q and intent.intent_source == "session_followup")
        ):
            pass
        elif intent.intent_class not in new_specific_classes or intent.router_confidence < 0.9:
            intent.legacy_question_class = legacy_class
            intent.legacy_question_subtype = legacy_subtype
            if legacy_class == "quantitative":
                intent.intent_class = "measurement_query"
            elif legacy_class == "comparison":
                intent.intent_class = "comparison_query"
            elif legacy_class == "localization":
                intent.intent_class = "localization_query"
            elif legacy_class == "summary":
                intent.intent_class = "limitation_query" if legacy_subtype == "limitations" else "report_summarization"
            elif legacy_class == "diagnosis/differential":
                intent.intent_class = "differential_request"
            elif legacy_class == "management/prognosis":
                if legacy_subtype == "clinical_concern":
                    intent.intent_class = "urgency_assessment"
                else:
                    intent.intent_class = "followup_recommendation"
            elif legacy_class == "ct_organ":
                intent.intent_class = "measurement_query" if legacy_subtype in {"liver_volume", "liver_tumor_ratio", "lung_tumor_volume"} else "localization_query"
            intent.requires_retrieval, intent.requires_guideline_retrieval, intent.retrieval_policy = _retrieval_policy(intent.intent_class)
            return intent
    return intent


def _question_intent(question: str) -> tuple[str, str]:
    intent = _resolved_intent(question)
    if intent.legacy_question_class != "unknown":
        return intent.legacy_question_class, intent.legacy_question_subtype
    return "unknown", "unknown"


def _question_class(question: str) -> str:
    return _question_intent(question)[0]


def _question_subtype(question: str) -> str:
    return _question_intent(question)[1]


def _intent_route_question(intent: QAIntent, fallback_question: str) -> str:
    route_map = {
        ("quantitative", "total_volume"): "what is the total tumor volume",
        ("quantitative", "max_diameter"): "what is the maximum tumor diameter",
        ("quantitative", "enhancing_volume"): "what is the enhancing component volume",
        ("quantitative", "edema_volume"): "what is the edema component volume",
        ("quantitative", "ambiguous_size"): "how big is it",
        ("localization", "largest_lesion_location"): "where is the largest lesion",
        ("mass_effect", "midline_shift"): "is there midline shift",
        ("mass_effect", "ventricular_compression"): "is there ventricular compression",
        ("mass_effect", "sulcal_effacement"): "is there sulcal effacement",
        ("mass_effect", "cisternal_obliteration"): "is there cisternal obliteration",
        ("critical_proximity", "brainstem"): "how close is it to the brainstem",
        ("critical_proximity", "motor_cortex"): "how close is it to motor cortex",
        ("critical_proximity", "optic_tract"): "how close is it to the optic tract",
        ("structured_reporting", "btrads_score_request"): "what BT RADS category best fits this case",
        ("structured_reporting", "btrads_report_field_completion"): "which BT RADS fields are currently supported",
    }
    return route_map.get((intent.legacy_question_class, intent.legacy_question_subtype), fallback_question)


def _evidence_source_labels(evidence_ids: list[str]) -> list[str]:
    labels: list[str] = []
    for evidence_id in evidence_ids:
        if evidence_id.startswith("ctx:segmentation_metrics") and "segmentation metrics" not in labels:
            labels.append("segmentation metrics")
        elif evidence_id.startswith("ctx:findings_structured.nifti_summary") and "NIfTI sequence inventory" not in labels:
            labels.append("NIfTI sequence inventory")
        elif evidence_id == "ctx:clinical_note" and "clinical note" not in labels:
            labels.append("clinical note")
        elif evidence_id == "ctx:prior_reports" and "prior reports" not in labels:
            labels.append("prior reports")
    return labels


def _context_metrics(payload: QARequest) -> dict[str, float | int]:
    seg = _as_dict(payload.context.segmentation_metrics)
    label_volumes = _as_dict(seg.get("label_volumes_mm3"))
    total_mm3 = float(seg.get("total_tumor_volume_mm3", seg.get("volume_mm3", 0)) or 0)
    max_diameter_mm = float(seg.get("max_diameter_mm", 0) or 0)
    lesion_count = int(seg.get("tumor_count", 0) or 0)
    component_count = int(seg.get("segmented_component_count", len(_list_of_dicts(seg.get("lesion_list", [])))) or 0)
    enhancing_mm3 = float(label_volumes.get("enhancing", 0) or 0)
    nonenh_mm3 = float(label_volumes.get("nonenhancing", 0) or 0)
    edema_mm3 = float(label_volumes.get("edema", 0) or 0)
    return {
        "has_segmentation": 1 if _has_segmentation_metrics(payload.context) else 0,
        "total_mm3": total_mm3,
        "max_diameter_mm": max_diameter_mm,
        "tumor_count": lesion_count,
        "component_count": component_count,
        "enhancing_mm3": enhancing_mm3,
        "nonenh_mm3": nonenh_mm3,
        "edema_mm3": edema_mm3,
    }


def _component_counts(payload: QARequest) -> dict[str, int]:
    seg = _as_dict(payload.context.segmentation_metrics)
    lesion_list = _list_of_dicts(seg.get("lesion_list", []))
    counts = {"enhancing": 0, "nonenhancing": 0, "edema": 0}
    for lesion in lesion_list:
        if not isinstance(lesion, dict):
            continue
        label_name = str(lesion.get("label_name", "") or "")
        if label_name in counts:
            counts[label_name] += 1
    return counts


def _midline_shift_mm(payload: QARequest) -> float | None:
    seg = _as_dict(payload.context.segmentation_metrics)
    midline = _as_dict(seg.get("midline_shift"))
    if isinstance(midline.get("value_mm"), (int, float)):
        return float(midline["value_mm"])
    mass_effect = _as_dict(seg.get("mass_effect"))
    if isinstance(mass_effect.get("midline_shift_mm"), (int, float)):
        return float(mass_effect["midline_shift_mm"])
    return None


def _mass_effect_metric(payload: QARequest, key: str) -> Any:
    seg = _as_dict(payload.context.segmentation_metrics)
    mass_effect = _as_dict(seg.get("mass_effect"))
    return mass_effect.get(key)


def _critical_structure_distance(payload: QARequest, structure: str) -> float | None:
    seg = _as_dict(payload.context.segmentation_metrics)
    prox_list = seg.get("critical_structure_proximity", [])
    if not isinstance(prox_list, list):
        return None
    for item in prox_list:
        item_dict = _as_dict(item)
        if str(item_dict.get("structure", "")).lower() == structure.lower() and isinstance(item_dict.get("distance_mm"), (int, float)):
            return float(item_dict["distance_mm"])
    return None


def _largest_component(payload: QARequest) -> dict[str, Any]:
    seg = _as_dict(payload.context.segmentation_metrics)
    lesion_list = _list_of_dicts(seg.get("lesion_list", []))
    return max(
        lesion_list,
        key=lambda x: float(x.get("volume_mm3", 0) or 0),
        default={},
    )


def _format_centroid(lesion: dict[str, Any]) -> str:
    centroid = lesion.get("centroid_zyx_voxel", [])
    if not isinstance(centroid, list) or len(centroid) != 3:
        return "centroid unavailable"
    zyx = [float(x or 0) for x in centroid]
    return f"approximate centroid z/y/x {zyx[0]:.1f}/{zyx[1]:.1f}/{zyx[2]:.1f} vox"


def _location_confidence_label(lesion: dict[str, Any]) -> str:
    anatomy = _as_dict(lesion.get("anatomy"))
    mapping_basis = str(anatomy.get("mapping_basis") or "").lower()
    atlas_confidence = str(anatomy.get("atlas_confidence") or "").lower()
    registration_flag = str(anatomy.get("registration_quality_flag") or "").lower()

    if mapping_basis.startswith("harvard_oxford"):
        if atlas_confidence == "high" and registration_flag == "high":
            return "high-confidence atlas-supported location"
        return "approximate atlas-supported location"
    if mapping_basis.startswith("mni152"):
        return "approximate atlas-supported location"
    return "heuristic location only"


def _anatomic_description(payload: QARequest, lesion: dict[str, Any]) -> str:
    anatomy = _as_dict(lesion.get("anatomy"))
    if anatomy.get("description"):
        return str(anatomy["description"])

    seg = _as_dict(payload.context.segmentation_metrics)
    image_shape = seg.get("image_shape_zyx", [])
    shape_tuple = tuple(int(v) for v in image_shape) if isinstance(image_shape, list) and len(image_shape) == 3 else None
    derived = anatomy_from_lesion(lesion, image_shape_zyx=shape_tuple)
    if derived.get("description") and derived.get("mapping_basis") != "unavailable":
        return str(derived["description"])

    dominant_location = str(seg.get("dominant_anatomic_location") or "").strip()
    if dominant_location:
        return dominant_location
    return ""


def _anatomic_mapping_basis(lesion: dict[str, Any]) -> str:
    anatomy = _as_dict(lesion.get("anatomy"))
    return str(anatomy.get("mapping_basis") or "")


def _burden_bucket(total_mm3: float) -> str:
    if total_mm3 >= 100000:
        return "very high"
    if total_mm3 >= 50000:
        return "high"
    if total_mm3 >= 20000:
        return "moderate"
    if total_mm3 > 0:
        return "low"
    return "minimal"


def _enhancement_style(enh_pct: float) -> str:
    if enh_pct >= 60:
        return "predominantly enhancing"
    if enh_pct >= 30:
        return "mixed enhancing and infiltrative"
    return "predominantly non-enhancing/infiltrative"


def _edema_severity(edema_pct: float) -> str:
    if edema_pct >= 40:
        return "substantial edema burden"
    if edema_pct >= 15:
        return "moderate edema burden"
    if edema_pct > 0:
        return "limited edema burden"
    return "no meaningful edema burden"


def _fragmentation_note(component_count: int) -> str:
    if component_count >= 6:
        return "High component count may reflect fragmented segmentation rather than true multifocal anatomy."
    if component_count >= 3:
        return "Component count suggests a heterogeneous lesion pattern, but true multifocality still requires anatomic review."
    return "Component count is limited, which is more compatible with a dominant focal process."


def _case_profile_lines(payload: QARequest) -> list[str]:
    m = _context_metrics(payload)
    if not int(m["has_segmentation"]):
        return [
            "- Segmentation-derived burden metrics are unavailable.",
            "- Lesion count, component distribution, and enhancement profile cannot be verified from the current package.",
        ]
    counts = _component_counts(payload)
    total_mm3 = float(m["total_mm3"])
    lesion_count = int(m["tumor_count"])
    component_count = int(m["component_count"])
    enhancing_mm3 = float(m["enhancing_mm3"])
    nonenh_mm3 = float(m["nonenh_mm3"])
    edema_mm3 = float(m["edema_mm3"])
    enh_pct = _safe_div(enhancing_mm3, total_mm3) * 100.0
    non_pct = _safe_div(nonenh_mm3, total_mm3) * 100.0
    edema_pct = _safe_div(edema_mm3, total_mm3) * 100.0

    dominant_name = "mixed"
    dominant_value = max(enhancing_mm3, nonenh_mm3, edema_mm3)
    if dominant_value == nonenh_mm3:
        dominant_name = "non-enhancing burden"
    elif dominant_value == enhancing_mm3:
        dominant_name = "enhancing burden"
    elif dominant_value == edema_mm3:
        dominant_name = "edema burden"

    component_pattern = (
        f"enhancing={counts['enhancing']}, non-enhancing={counts['nonenhancing']}, edema={counts['edema']}"
    )
    return [
        f"- Overall segmented burden is {_burden_bucket(total_mm3)} at {total_mm3/1000.0:.2f} cm3.",
        f"- Lesion groups detected after class merge: {lesion_count}.",
        f"- Dominant pattern is {dominant_name} ({max(enh_pct, non_pct, edema_pct):.1f}% of total segmented volume).",
        f"- Enhancement behavior is {_enhancement_style(enh_pct)} with {_edema_severity(edema_pct)}.",
        f"- Component distribution by class: {component_pattern}.",
        f"- {_fragmentation_note(component_count)}",
    ]


def _supporting_context_lines(payload: QARequest) -> list[str]:
    findings = _as_dict(payload.context.findings_structured)
    nifti = _as_dict(findings.get("nifti_summary"))
    sequences = _extract_sequences([str(x) for x in nifti.get("nifti_files", []) if isinstance(x, str)])
    lines = [
        f"- Available sequences: {', '.join(sequences) if sequences else 'not available in context'}."
    ]
    largest = _largest_component(payload)
    if largest:
        label_name = str(largest.get("label_name", "unknown"))
        volume_mm3 = float(largest.get("volume_mm3", 0) or 0)
        max_diameter_mm = float(largest.get("max_diameter_mm", 0) or 0)
        location = _format_location_for_user(payload, largest)
        lines.append(
            "- Largest segmented component: "
            f"{label_name} burden measuring {volume_mm3:.2f} mm3 with max diameter {max_diameter_mm:.2f} mm."
        )
        if location:
            lines.append(f"- Dominant segmented component: {location}.")
    lines.append(
        f"- Clinical note present: {'yes' if (payload.context.clinical_note or '').strip() else 'no'}; "
        f"prior reports present: {'yes' if payload.context.prior_reports else 'no'}."
    )
    return lines


def _format_location_for_user(payload: QARequest, lesion: dict[str, Any]) -> str:
    location = _anatomic_description(payload, lesion).strip()
    if not location:
        return "anatomic location is not well characterized from the available data"
    label = _location_confidence_label(lesion)
    return f"{label}: {location}"


def _dominant_location_sentence(payload: QARequest) -> str:
    largest = _largest_component(payload)
    if not largest:
        return ""
    return _format_location_for_user(payload, largest)


def _component_pattern_sentence(payload: QARequest) -> str:
    m = _context_metrics(payload)
    if not int(m["has_segmentation"]):
        return "Segmentation-derived component pattern is unavailable."
    total_mm3 = float(m["total_mm3"])
    if total_mm3 <= 0:
        return "No measurable segmented tumor burden is available."
    enhancing_mm3 = float(m["enhancing_mm3"])
    nonenh_mm3 = float(m["nonenh_mm3"])
    edema_mm3 = float(m["edema_mm3"])
    enh_pct = _safe_div(enhancing_mm3, total_mm3) * 100.0
    non_pct = _safe_div(nonenh_mm3, total_mm3) * 100.0
    edema_pct = _safe_div(edema_mm3, total_mm3) * 100.0
    return (
        f"Most of the segmented burden is non-enhancing ({non_pct:.1f}%), "
        f"with a smaller enhancing component ({enh_pct:.1f}%) and "
        f"{'minimal' if edema_pct < 5.0 else 'additional'} edema ({edema_pct:.1f}%)."
    )


def _multifocality_sentence(payload: QARequest) -> str:
    m = _context_metrics(payload)
    if not int(m["has_segmentation"]):
        return "Multifocality cannot be assessed because segmentation-derived lesion grouping is unavailable."
    lesion_count = int(m["tumor_count"])
    component_count = int(m["component_count"])
    if lesion_count <= 1:
        if component_count > 1:
            return (
                f"Segmentation supports one dominant lesion with {component_count} internal components; "
                "this pattern is more compatible with heterogeneous composition than true multifocal disease."
            )
        return "The segmentation supports a single dominant lesion."
    if component_count >= 6:
        return (
            f"Multiple lesion groups are present ({lesion_count}), with {component_count} segmented components overall; "
            "some component fragmentation may still be present."
        )
    return f"Multiple lesion groups are present ({lesion_count})."


def _multifocality_direct_answer(payload: QARequest) -> str:
    m = _context_metrics(payload)
    if not int(m["has_segmentation"]):
        return _segmentation_unavailable_answer()

    lesion_count = int(m["tumor_count"])
    component_count = int(m["component_count"])
    total_mm3 = float(m["total_mm3"])
    largest = _largest_component(payload)
    largest_mm3 = float(largest.get("volume_mm3", 0.0) or 0.0)
    dominant_pct = _safe_div(largest_mm3, total_mm3) * 100.0

    if lesion_count <= 1:
        if component_count > 1:
            return (
                f"Segmentation is more consistent with one dominant lesion group than with multifocal disease. "
                f"After class merge, 1 lesion group is detected across {component_count} segmented components. "
                + _fragmentation_note(component_count)
                + " Clinician verification is still required because internal component fragmentation can mimic multiplicity."
            )
        return (
            "Segmentation is more consistent with one dominant lesion group than with multifocal disease. "
            "After class merge, 1 lesion group is detected."
        )

    if dominant_pct >= 65.0:
        return (
            "Segmentation is more consistent with multiple lesion groups with a dominant lesion plus satellite groups "
            "than with evenly distributed multifocal disease. "
            f"After class merge, {lesion_count} lesion groups are detected across {component_count} segmented components, "
            f"and the dominant component accounts for {dominant_pct:.1f}% of the total segmented burden. "
            + _fragmentation_note(component_count)
            + " Clinician verification is still required because anatomic continuity cannot be confirmed from segmentation alone."
        )

    return (
        f"Segmentation is more consistent with multiple lesion groups than with one dominant lesion group. "
        f"After class merge, {lesion_count} lesion groups are detected across {component_count} segmented components. "
        + _fragmentation_note(component_count)
        + " Clinician verification is still required because true multifocal anatomy depends on broader imaging review."
    )


def _plain_english_summary(payload: QARequest) -> str:
    m = _context_metrics(payload)
    if not int(m["has_segmentation"]):
        return (
            "This case package includes imaging context, but segmentation-derived tumor metrics are unavailable. "
            "A lesion burden summary cannot be generated from the current data, and definitive interpretation "
            "requires clinician review."
        )
    total_mm3 = float(m["total_mm3"])
    max_diameter_mm = float(m["max_diameter_mm"])
    location_sentence = _dominant_location_sentence(payload)
    lead = (
        f"This case shows a large brain tumor with a total segmented volume of {total_mm3/1000.0:.2f} cm3 "
        f"and a maximum diameter of {max_diameter_mm:.0f} mm."
    )
    location_detail = ""
    if location_sentence:
        if ": " in location_sentence:
            location_detail = location_sentence.split(": ", 1)[1]
        else:
            location_detail = location_sentence
    location = (
        f" The dominant segmented component is centered in the {location_detail}."
        if location_detail
        else ""
    )
    return (
        lead
        + location
        + " "
        + _component_pattern_sentence(payload)
        + " "
        + _multifocality_sentence(payload)
        + " Overall, this is a large heterogeneous lesion pattern, but imaging alone is not sufficient for a definitive diagnosis."
    )


def _tumor_board_summary(payload: QARequest) -> str:
    m = _context_metrics(payload)
    if not int(m["has_segmentation"]):
        lines = [
            "Tumor-board style case summary:",
            "- Segmentation-derived lesion metrics are unavailable for this case package.",
            f"- Available sequences: {', '.join(_extract_sequences([str(x) for x in _as_dict(_as_dict(payload.context.findings_structured).get('nifti_summary')).get('nifti_files', []) if isinstance(x, str)])) or 'not available in context'}",
            "- Quantitative burden, multiplicity, and enhancement composition cannot be verified.",
            "",
            "Interpretation support:",
            "- Imaging should be reviewed directly by the treating radiology/clinical team.",
            "- Definitive diagnosis requires histopathological confirmation and clinical correlation.",
        ]
        return "\n".join(lines)
    total_mm3 = float(m["total_mm3"])
    max_diameter_mm = float(m["max_diameter_mm"])
    lesion_count = int(m["tumor_count"])
    component_count = int(m["component_count"])
    largest = _largest_component(payload)
    location_sentence = _dominant_location_sentence(payload)
    sequences = _extract_sequences([
        str(x)
        for x in _as_dict(_as_dict(payload.context.findings_structured).get("nifti_summary")).get("nifti_files", [])
        if isinstance(x, str)
    ])
    lines = [
        "Tumor-board style case summary:",
        f"- Total segmented tumor volume: {total_mm3:.2f} mm3 ({total_mm3/1000.0:.2f} cm3)",
        f"- Maximum segmented lesion diameter: {max_diameter_mm:.2f} mm",
        f"- Lesion groups detected after class merge: {lesion_count}",
        f"- Segmented components detected: {component_count}",
        f"- Available sequences: {', '.join(sequences) if sequences else 'not available in context'}",
        f"- {_component_pattern_sentence(payload)}",
    ]
    if largest:
        lines.append(
            "- Largest segmented component: "
            f"{str(largest.get('label_name', 'unknown'))} burden measuring "
            f"{float(largest.get('volume_mm3', 0.0) or 0.0):.2f} mm3 with max diameter "
            f"{float(largest.get('max_diameter_mm', 0.0) or 0.0):.2f} mm"
        )
    if location_sentence:
        lines.append(f"- Dominant segmented component: {location_sentence}.")
    lines.append(f"- {_multifocality_sentence(payload)}")
    lines.extend(
        [
            "",
            "Interpretation support:",
            "- Imaging shows a large heterogeneous infiltrative lesion pattern.",
            "- Differential diagnosis remains open pending tissue diagnosis.",
            "- Definitive diagnosis requires histopathological confirmation and clinical correlation.",
        ]
    )
    return "\n".join(lines)


def _imaging_summary_lines(payload: QARequest) -> list[str]:
    m = _context_metrics(payload)
    if not int(m["has_segmentation"]):
        return [
            "- Segmentation-derived lesion metrics are unavailable.",
            "- Quantitative burden, multiplicity, and composition cannot be confirmed.",
        ]
    total_mm3 = float(m["total_mm3"])
    max_diameter_mm = float(m["max_diameter_mm"])
    lesion_count = int(m["tumor_count"])
    component_count = int(m["component_count"])
    enhancing_mm3 = float(m["enhancing_mm3"])
    nonenh_mm3 = float(m["nonenh_mm3"])
    edema_mm3 = float(m["edema_mm3"])
    enh_pct = _safe_div(enhancing_mm3, total_mm3) * 100.0
    non_pct = _safe_div(nonenh_mm3, total_mm3) * 100.0
    edema_pct = _safe_div(edema_mm3, total_mm3) * 100.0
    return [
        f"- Total segmented tumor volume: {total_mm3:.2f} mm3 ({total_mm3/1000.0:.2f} cm3)",
        f"- Maximum segmented lesion diameter: {max_diameter_mm:.2f} mm",
        f"- Lesion groups detected after class merge: {lesion_count}",
        f"- Segmented components detected: {component_count}",
        f"- Enhancing vs non-enhancing: {enhancing_mm3:.2f} mm3 ({enh_pct:.1f}%) vs {nonenh_mm3:.2f} mm3 ({non_pct:.1f}%)",
        f"- Edema volume: {edema_mm3:.2f} mm3 ({edema_pct:.1f}% of total)",
    ]


def _known_case_sentence(payload: QARequest, *, include_location: bool = False) -> str:
    m = _context_metrics(payload)
    if not int(m["has_segmentation"]):
        sequences = _extract_sequences([
            str(x)
            for x in _as_dict(_as_dict(payload.context.findings_structured).get("nifti_summary")).get("nifti_files", [])
            if isinstance(x, str)
        ])
        sequence_text = ", ".join(sequences) if sequences else "no confirmed MR sequence inventory"
        return f"Known from this case package: {sequence_text}."

    total_mm3 = float(m["total_mm3"])
    lesion_count = int(m["tumor_count"])
    enhancing_mm3 = float(m["enhancing_mm3"])
    nonenh_mm3 = float(m["nonenh_mm3"])
    edema_mm3 = float(m["edema_mm3"])
    dominant_name = "mixed burden"
    dominant_value = max(enhancing_mm3, nonenh_mm3, edema_mm3)
    if dominant_value == nonenh_mm3:
        dominant_name = "non-enhancing dominant pattern"
    elif dominant_value == enhancing_mm3:
        dominant_name = "enhancing dominant pattern"
    elif dominant_value == edema_mm3:
        dominant_name = "edema dominant pattern"
    parts = [
        f"total segmented volume {total_mm3/1000.0:.2f} cm3",
        f"{lesion_count} lesion group{'s' if lesion_count != 1 else ''}",
        dominant_name,
    ]
    if include_location:
        location = _dominant_location_sentence(payload)
        if location:
            parts.append(location)
    return "Known from this case package: " + "; ".join(parts) + "."


def _source_sentence(evidence_ids: list[str]) -> str:
    labels = _evidence_source_labels(evidence_ids)
    if not labels:
        return "Source: current case context."
    if len(labels) == 1:
        return f"Source: {labels[0]}."
    if len(labels) == 2:
        return f"Source: {labels[0]} and {labels[1]}."
    return "Source: " + ", ".join(labels[:-1]) + f", and {labels[-1]}."


def _fallback_narrative_summary(payload: QARequest) -> str:
    m = _context_metrics(payload)
    if not int(m["has_segmentation"]):
        return (
            "Qualitatively, the current package does not support a reliable lesion-pattern narrative. "
            "Imaging-only interpretation remains non-diagnostic and requires clinician review."
        )
    location = _dominant_location_sentence(payload)
    location_text = (
        f"The dominant component localizes to {location.split(': ', 1)[1]}."
        if location and ": " in location
        else "The dominant lesion location remains approximate."
    )
    return (
        f"{location_text} Qualitatively, the lesion pattern is {_component_pattern_sentence(payload).lower()} "
        f"{_multifocality_sentence(payload)} Imaging findings remain suggestive rather than diagnostic, "
        "and definitive characterization requires histopathological confirmation."
    )


def _hybrid_narrative_summary(
    payload: QARequest,
    *,
    allow_llm: bool,
    session_context: str = "",
) -> tuple[str, list[str], float, str | None] | None:
    _, q_subtype = _question_intent(payload.question)
    if q_subtype != "narrative_summary":
        return None

    m = _context_metrics(payload)
    evidence = _default_nonstructured_evidence_ids(payload)
    if not int(m["has_segmentation"]):
        answer = _segmentation_unavailable_answer() + " Clinical confirmation is required."
        return answer, _core_context_evidence_ids(payload), 0.42, None

    total_mm3 = float(m["total_mm3"])
    max_diameter_mm = float(m["max_diameter_mm"])
    enhancing_mm3 = float(m["enhancing_mm3"])
    nonenh_mm3 = float(m["nonenh_mm3"])
    edema_mm3 = float(m["edema_mm3"])
    location = _dominant_location_sentence(payload)
    location_clause = f" The dominant segmented component is {location}." if location else ""
    numeric_summary = (
        f"This case shows a total segmented tumor volume of {total_mm3:.2f} mm3 ({total_mm3/1000.0:.2f} cm3) "
        f"with a maximum segmented diameter of {max_diameter_mm:.2f} mm.{location_clause}"
    )
    composition_summary = (
        f"The segmented composition is {enhancing_mm3:.2f} mm3 enhancing, "
        f"{nonenh_mm3:.2f} mm3 non-enhancing, and {edema_mm3:.2f} mm3 edema."
    )

    fallback_narrative = _fallback_narrative_summary(payload)
    model_used: str | None = None
    qualitative_narrative = fallback_narrative
    if allow_llm:
        qualitative_narrative, model_used = generate_qa_narrative_text(
            question=payload.question,
            context=payload.context.model_dump(),
            fallback_narrative=fallback_narrative,
            use_high_model=False,
            session_context=session_context,
        )

    answer = f"{numeric_summary} {composition_summary} {qualitative_narrative}".strip()
    return answer, evidence, 0.60, model_used


def _uncertainty_reduction_answer(payload: QARequest) -> tuple[str, list[str], float]:
    evidence = _default_nonstructured_evidence_ids(payload)
    m = _context_metrics(payload)
    missing_items: list[str] = []

    if not (payload.context.clinical_note or "").strip():
        missing_items.append(
            "Clinical history and symptom timeline, because no clinical note is available in this package."
        )
    if not payload.context.prior_reports:
        missing_items.append(
            "Prior imaging or prior reports for interval change, because no longitudinal comparison is available."
        )

    study_date = str(payload.context.study_date or "").strip().lower()
    if study_date in {"", "unknown", "unavailable", "none", "n/a", "na"}:
        missing_items.append(
            "Confirmed study date or treatment timeline, because temporal context is currently incomplete."
        )

    missing_items.extend(
        [
            "Neurologic examination findings for clinicoradiologic correlation.",
            "Histopathology and molecular markers (for example IDH/MGMT) for definitive characterization.",
        ]
    )

    if int(m["tumor_count"]) > 1:
        missing_items.append(
            "Systemic oncologic history if multifocal or metastatic disease is a clinical consideration."
        )

    lines = [
        "Uncertainty would be reduced most by the following case-specific inputs:",
        f"- {_known_case_sentence(payload, include_location=True)}",
        f"- {_source_sentence(evidence)}",
    ]
    lines.extend(f"- {item}" for item in _dedupe(missing_items))
    lines.append("- Imaging-only outputs remain non-diagnostic without clinical correlation.")
    return "\n".join(lines), evidence, 0.54


def _balanced_clinical_support(payload: QARequest) -> tuple[str, list[str], float] | None:
    q_class, q_subtype = _question_intent(payload.question)
    evidence = [
        "ctx:segmentation_metrics.total_tumor_volume_mm3",
        "ctx:segmentation_metrics.max_diameter_mm",
        "ctx:segmentation_metrics.tumor_count",
        "ctx:segmentation_metrics.segmented_component_count",
        "ctx:segmentation_metrics.label_volumes_mm3.enhancing",
        "ctx:segmentation_metrics.label_volumes_mm3.nonenhancing",
        "ctx:segmentation_metrics.label_volumes_mm3.edema",
    ]
    summary = "\n".join(_imaging_summary_lines(payload))
    profile = "\n".join(_case_profile_lines(payload))
    m = _context_metrics(payload)
    lesion_count = int(m["tumor_count"])
    component_count = int(m["component_count"])
    total_mm3 = float(m["total_mm3"])
    enhancing_mm3 = float(m["enhancing_mm3"])
    nonenh_mm3 = float(m["nonenh_mm3"])
    edema_mm3 = float(m["edema_mm3"])
    enh_pct = _safe_div(enhancing_mm3, total_mm3) * 100.0

    if not int(m["has_segmentation"]):
        return (
            _segmentation_unavailable_answer() + " Clinical confirmation is required.",
            _core_context_evidence_ids(payload),
            0.42,
        )

    if q_class == "summary" and q_subtype == "plain_language_summary":
        return _plain_english_summary(payload), evidence, 0.58

    if q_class == "summary" and q_subtype == "generic_summary":
        return _tumor_board_summary(payload), evidence, 0.56

    if q_class == "summary" and q_subtype == "tumor_board_summary":
        return _tumor_board_summary(payload), evidence, 0.58

    if q_class == "diagnosis/differential" and q_subtype == "diagnostic_interpretation":
        answer = (
            "Imaging alone cannot establish a definitive diagnosis.\n"
            f"- {_known_case_sentence(payload, include_location=True)}\n"
            f"- {_source_sentence(evidence)}\n"
            "- Differential considerations from the imaging pattern: high-grade glial neoplasm is the leading pattern match; "
            "metastatic disease and primary CNS lymphoma remain less typical alternatives.\n"
            "- Missing for confirmation: histopathology, molecular markers, and full clinical correlation."
        )
        return answer, evidence, 0.52

    if q_class == "management/prognosis" and q_subtype == "management_recommendation":
        answer = (
            "I cannot provide patient-specific treatment recommendations from imaging alone.\n"
            f"- {_known_case_sentence(payload, include_location=True)}\n"
            f"- {_source_sentence(evidence)}\n"
            "- Missing for management decisions: histopathology, molecular markers, neurologic examination, and multidisciplinary review."
        )
        return answer, evidence, 0.48

    if q_class == "management/prognosis" and q_subtype == "prognosis_assessment":
        answer = (
            "A reliable prognosis cannot be estimated from imaging alone.\n"
            f"- {_known_case_sentence(payload)}\n"
            f"- {_source_sentence(evidence)}\n"
            "- Missing for prognosis: histopathology, molecular profile, baseline clinical status, and longitudinal follow-up."
        )
        return answer, evidence, 0.45

    if q_class == "management/prognosis" and q_subtype == "clinical_concern":
        answer = (
            "This is a clinically significant imaging finding, but urgency and risk still require clinician assessment.\n"
            f"- {_known_case_sentence(payload, include_location=True)}\n"
            f"- {_source_sentence(evidence)}\n"
            "- Missing before risk judgment: neurologic examination, full clinical history, and definitive tissue-based diagnosis."
        )
        return answer, evidence, 0.50

    if q_class == "management/prognosis" and q_subtype == "uncertainty_reduction":
        return _uncertainty_reduction_answer(payload)

    if q_class == "structured_reporting":
        system_name = "BT-RADS" if "btrads" in q_subtype else "structured reporting"
        answer = (
            f"{system_name} style questions are handled as structured-reporting support, not autonomous scoring.\n"
            f"- {_known_case_sentence(payload, include_location=True)}\n"
            f"- {_source_sentence(evidence)}\n"
            "- Numeric truth still comes from audited current-case measurements.\n"
            "- Guideline retrieval should be consulted before any final BT-RADS style categorization."
        )
        return answer, evidence, 0.53

    if q_class == "mass_effect":
        if q_subtype == "midline_shift":
            shift_mm = _midline_shift_mm(payload)
            if shift_mm is not None:
                return (
                    f"Measured midline shift is {shift_mm:.2f} mm based on current structured findings.",
                    ["ctx:segmentation_metrics.midline_shift", "ctx:segmentation_metrics.mass_effect.midline_shift_mm"],
                    0.71,
                )
            return (
                "Current structured findings do not show a direct measured midline shift value.",
                ["ctx:segmentation_metrics.midline_shift", "ctx:segmentation_metrics.mass_effect"],
                0.42,
            )
        if q_subtype == "ventricular_compression":
            value = _mass_effect_metric(payload, "ventricular_compression")
            if isinstance(value, bool):
                status = "present" if value else "not identified"
                return (
                    f"Ventricular compression is {status} in the current structured mass-effect fields.",
                    ["ctx:segmentation_metrics.mass_effect.ventricular_compression"],
                    0.69,
                )
        if q_subtype == "sulcal_effacement":
            value = _mass_effect_metric(payload, "sulcal_effacement")
            if isinstance(value, bool):
                status = "present" if value else "not identified"
                return (
                    f"Sulcal effacement is {status} in the current structured mass-effect fields.",
                    ["ctx:segmentation_metrics.mass_effect.sulcal_effacement"],
                    0.69,
                )
        if q_subtype == "cisternal_obliteration":
            value = _mass_effect_metric(payload, "cisternal_obliteration")
            if isinstance(value, bool):
                status = "present" if value else "not identified"
                return (
                    f"Cisternal obliteration is {status} in the current structured mass-effect fields.",
                    ["ctx:segmentation_metrics.mass_effect.cisternal_obliteration"],
                    0.69,
                )
        return (
            "Current structured findings do not contain a complete measured mass-effect profile for this specific question.",
            ["ctx:segmentation_metrics.mass_effect"],
            0.42,
        )

    if q_class == "critical_proximity":
        structure_map = {
            "brainstem": ("brainstem", "ctx:segmentation_metrics.critical_structure_proximity"),
            "motor_cortex": ("motor_cortex", "ctx:segmentation_metrics.critical_structure_proximity"),
            "optic_tract": ("optic_tract", "ctx:segmentation_metrics.critical_structure_proximity"),
        }
        structure_name, evidence_id = structure_map.get(q_subtype, ("", "ctx:segmentation_metrics.critical_structure_proximity"))
        if structure_name:
            distance_mm = _critical_structure_distance(payload, structure_name)
            if distance_mm is not None:
                return (
                    f"Distance to {structure_name.replace('_', ' ')} is {distance_mm:.2f} mm in the current structured proximity data.",
                    [evidence_id],
                    0.71,
                )
        return (
            "Current structured findings do not contain a measured critical-structure distance for that target.",
            ["ctx:segmentation_metrics.critical_structure_proximity"],
            0.42,
        )

    if q_class == "summary" and q_subtype == "limitations":
        has_note = bool((payload.context.clinical_note or "").strip())
        has_prior = bool(payload.context.prior_reports)
        answer = (
            "Top limitations of this case-level analysis:\n"
            f"1. Limited non-imaging context (clinical note present: {'yes' if has_note else 'no'}).\n"
            f"2. Longitudinal comparison constraints (prior reports present: {'yes' if has_prior else 'no'}).\n"
            "3. Imaging-based inference only; definitive diagnosis requires histopathological confirmation."
        )
        return answer, _default_nonstructured_evidence_ids(payload), 0.57

    if q_class == "unknown":
        answer = (
            "Insufficient evidence to interpret the request.\n"
            "- The question does not map clearly to a supported imaging or report query.\n"
            "- Please ask about a specific metric, location, lesion count, limitation, or structured reporting field."
        )
        return answer, [], 0.40

    return None


def _structured_answer(payload: QARequest) -> tuple[str, list[str]]:
    context = payload.context
    findings = _as_dict(context.findings_structured)
    nifti = _as_dict(findings.get("nifti_summary"))
    seg = _as_dict(context.segmentation_metrics)

    q = _normalize_question(payload.question.lower().strip())
    _, q_subtype = _question_intent(payload.question)
    nifti_files = nifti.get("nifti_files", [])
    nifti_files = nifti_files if isinstance(nifti_files, list) else []
    sequences = _extract_sequences([str(x) for x in nifti_files])
    has_segmentation = _has_segmentation_metrics(context)

    lesion_list = _list_of_dicts(seg.get("lesion_list", []))
    label_volumes = seg.get("label_volumes_mm3", {})
    label_volumes = label_volumes if isinstance(label_volumes, dict) else {}

    lesion_count = int(seg.get("tumor_count", 0) or 0)
    component_count = int(seg.get("segmented_component_count", len(lesion_list)) or 0)
    total_mm3 = float(seg.get("total_tumor_volume_mm3", seg.get("volume_mm3", 0)) or 0)
    max_diameter_mm = float(seg.get("max_diameter_mm", 0) or 0)
    enhancing_mm3 = float(label_volumes.get("enhancing", 0) or 0)
    nonenh_mm3 = float(label_volumes.get("nonenhancing", 0) or 0)
    edema_mm3 = float(label_volumes.get("edema", 0) or 0)

    if q_subtype == "sequence_inventory":
        if sequences:
            return (
                "Available MR sequences: " + ", ".join(sequences) + ".",
                ["ctx:findings_structured.nifti_summary.nifti_files"],
            )
        return (
            "MR sequences cannot be confirmed from current context (insufficient evidence).",
            ["ctx:findings_structured.nifti_summary.nifti_files"],
        )

    _organ = str(getattr(context, "organ", "") or "").lower()
    _is_ct = str(context.modality or "").upper() == "CT"

    if q_subtype == "ct_sequence" and _is_ct:
        _organ_label = {"lung": "lung", "liver": "abdominal/liver"}.get(_organ, "")
        _protocol = f"{_organ_label} protocol " if _organ_label else ""
        return (
            f"CT ({_protocol}imaging) was used for this case.",
            ["ctx:modality"],
        )

    if q_subtype == "liver_volume":
        _liver_vol = float((label_volumes or {}).get("liver", 0) or 0)
        if _liver_vol > 0:
            liver_vol_ml = _liver_vol / 1000.0  
            if liver_vol_ml > 2000:
                tier = "ENLARGED"; tier_cls = "alert"
            elif liver_vol_ml > 1500:
                tier = "BORDERLINE"; tier_cls = "caution"
            else:
                tier = "NORMAL"; tier_cls = "safe"
            return (
                f"Total segmented liver parenchyma volume is {liver_vol_ml:.0f} ml ({_liver_vol:.0f} mm³). "
                f"Clinical tier: {tier} (thresholds: NORMAL <1500 ml, BORDERLINE 1500–2000 ml, ENLARGED >2000 ml). "
                "Note: automated segmentation; direct radiological review required for clinical decisions.",
                ["ctx:segmentation_metrics.label_volumes_mm3.liver"],
            )
        return (
            "Insufficient evidence: total liver volume is not available as a separate label in the current segmentation.",
            ["ctx:segmentation_metrics.label_volumes_mm3"],
        )

    if q_subtype == "liver_tumor_ratio" and _is_ct:
        _liver_vol = float((label_volumes or {}).get("liver", 0) or 0)
        _tumor_vol = float(seg.get("total_tumor_volume_mm3", seg.get("volume_mm3", 0)) or 0)
        if _liver_vol > 0 and _tumor_vol > 0:
            liver_to_tumor = _liver_vol / max(_tumor_vol, 1e-9)
            tumor_pct = (_tumor_vol / _liver_vol) * 100.0
            ratio = _tumor_vol / _liver_vol
            if ratio > 0.30:
                tier = "HIGH RISK"
            elif ratio > 0.10:
                tier = "CAUTION"
            else:
                tier = "LOW"
            return (
                f"Liver-to-tumor volume ratio is {liver_to_tumor:.1f}:1, and tumor burden represents "
                f"{tumor_pct:.1f}% of total segmented liver volume. "
                f"Clinical tier: {tier} (thresholds: LOW <10%, CAUTION 10–30%, HIGH RISK >30%). "
                "Tumor-to-liver ratio >30% is associated with inadequate future liver remnant (FLR) for safe resection.",
                [
                    "ctx:segmentation_metrics.label_volumes_mm3.liver",
                    "ctx:segmentation_metrics.total_tumor_volume_mm3",
                ],
            )
        return (
            "Insufficient evidence: liver-to-tumor ratio cannot be derived because either liver volume or tumor volume is unavailable.",
            [
                "ctx:segmentation_metrics.label_volumes_mm3.liver",
                "ctx:segmentation_metrics.total_tumor_volume_mm3",
            ],
        )

    if q_subtype == "lung_tumor_volume" and _is_ct:
        _lung_tumor_vol = float(seg.get("total_tumor_volume_mm3", seg.get("volume_mm3", 0)) or 0)
        if _lung_tumor_vol > 0:
            lung_tumor_vol_ml = _lung_tumor_vol / 1000.0  
            if lung_tumor_vol_ml > 100:
                tier = "CRITICAL"
            elif lung_tumor_vol_ml > 30:
                tier = "WARNING"
            else:
                tier = "MONITOR"
            return (
                f"Total lung tumor volume is {lung_tumor_vol_ml:.1f} ml ({_lung_tumor_vol:.0f} mm³). "
                f"Clinical tier: {tier} (thresholds: MONITOR <30 ml, WARNING 30–100 ml, CRITICAL >100 ml). "
                "Refer to Fleischner Society and TNM 8th edition guidelines for staging and follow-up intervals.",
                ["ctx:segmentation_metrics.total_tumor_volume_mm3"],
            )
        return (
            "Insufficient evidence: lung tumor volume is not available in the current segmentation.",
            ["ctx:segmentation_metrics.total_tumor_volume_mm3"],
        )

    if q_subtype == "lymph_node_involvement" and _is_ct:
        return (
            "Insufficient evidence: mediastinal and hilar lymph node assessment is outside the scope "
            "of the current parenchymal segmentation. Direct radiological review is required.",
            [],
        )

    if q_subtype == "vascular_involvement" and _is_ct:
        return (
            "Insufficient evidence: hepatic vessel, portal vein, and biliary involvement cannot be "
            "assessed from the current parenchymal segmentation alone. Direct CT review is required.",
            [],
        )

    if q_subtype == "lobe_location" and _is_ct:
        dominant_loc = seg.get("dominant_anatomic_location", "")
        largest_anatomy: dict = {}
        if lesion_list:
            largest = max(
                (item for item in lesion_list if isinstance(item, dict)),
                key=lambda x: float(x.get("volume_mm3", 0) or 0),
                default={},
            )
            largest_anatomy = largest.get("anatomy", {}) if isinstance(largest, dict) else {}
        desc = (
            largest_anatomy.get("description")
            or dominant_loc
            or ""
        ).strip()
        note = largest_anatomy.get("localization_note", "")
        confidence = largest_anatomy.get("atlas_confidence", "Low")
        organ_val = (getattr(context, "organ", "") or "").lower()

        if desc:
            if organ_val == "liver":
                couinaud = largest_anatomy.get("couinaud_group", "")
                lobe = largest_anatomy.get("hepatic_lobe", "")
                location_text = f"Estimated hepatic location: {desc}."
                if couinaud:
                    location_text += f" Couinaud group: {couinaud}."
                if lobe:
                    location_text += f" Hepatic lobe: {lobe}."
            else:
                lobe = largest_anatomy.get("lobe", "")
                side = largest_anatomy.get("hemisphere", "")
                location_text = f"Estimated pulmonary location: {desc}."
                if lobe:
                    location_text += f" Lobe: {lobe}."
                if side:
                    location_text += f" Laterality: {side}."
            location_text += f" Localisation confidence: {confidence} (centroid heuristic)."
            if note:
                location_text += f" Note: {note}"
            return (location_text, ["ctx:segmentation_metrics.lesion_list[0].anatomy"])
        return (
            "Localisation data unavailable in the current package. Radiologist review required.",
            [],
        )

    if q_subtype == "staging_ct" and _is_ct:
        return (
            "Insufficient evidence: formal TNM staging cannot be determined from segmentation metrics alone. "
            "Staging requires full radiological review, pathology, and MDT assessment.",
            [],
        )

    if not has_segmentation:
        return (
            _segmentation_unavailable_answer(),
            _core_context_evidence_ids(payload),
        )

    if q_subtype == "midline_shift":
        shift_mm = _midline_shift_mm(payload)
        if shift_mm is not None:
            if shift_mm >= 7.5:
                tier = "CRITICAL — ≥7.5 mm; elevated herniation risk"
            elif shift_mm >= 5.0:
                tier = "WARNING — ≥5 mm; significant mass effect"
            else:
                tier = "NONE — <5 mm; below significance threshold"
            return (
                f"Midline shift (MLS): {shift_mm:.1f} mm (automated centroid-to-midline proxy). "
                f"Clinical tier: {tier}. "
                "Note: MLS is a secondary automated flag; direct neuroradiological review is required for clinical decisions.",
                ["ctx:segmentation_metrics.midline_shift", "ctx:segmentation_metrics.mass_effect.midline_shift_mm"],
            )
        return (
            "Insufficient evidence: midline shift data is not available in the current case package.",
            ["ctx:segmentation_metrics.midline_shift", "ctx:segmentation_metrics.mass_effect"],
        )
    if q_subtype == "ventricular_compression":
        value = _mass_effect_metric(payload, "ventricular_compression")
        prox = _mass_effect_metric(payload, "ventricle_proximity_mm")
        if isinstance(value, bool):
            status = "present" if value else "not detected"
            prox_text = f" Nearest ventricle proximity: {float(prox):.1f} mm." if isinstance(prox, (int, float)) else ""
            return (
                f"Ventricular compression: {status}.{prox_text} "
                "This is an automated surrogate; direct imaging review required.",
                ["ctx:segmentation_metrics.mass_effect.ventricular_compression"],
            )
        return (
            "Insufficient evidence: ventricular compression data is not available in the current case package.",
            ["ctx:segmentation_metrics.mass_effect"],
        )
    if q_subtype == "sulcal_effacement":
        value = _mass_effect_metric(payload, "sulcal_effacement")
        if isinstance(value, bool):
            status = "present" if value else "not detected"
            return (
                f"Sulcal effacement: {status} (automated hemispheric asymmetry proxy). "
                "Direct neuroradiological review is required for definitive assessment.",
                ["ctx:segmentation_metrics.mass_effect.sulcal_effacement"],
            )
        return (
            "Insufficient evidence: sulcal effacement data is not available in the current case package.",
            ["ctx:segmentation_metrics.mass_effect"],
        )
    if q_subtype == "cisternal_obliteration":
        value = _mass_effect_metric(payload, "cisternal_obliteration")
        if isinstance(value, bool):
            status = "present — critical herniation risk indicator" if value else "not detected"
            return (
                f"Cisternal obliteration: {status}. "
                "Direct neuroradiological review is required.",
                ["ctx:segmentation_metrics.mass_effect.cisternal_obliteration"],
            )
        return (
            "Insufficient evidence: cisternal obliteration data is not available in the current case package.",
            ["ctx:segmentation_metrics.mass_effect"],
        )
    if q_subtype == "herniation_risk":
        shift_mm = _midline_shift_mm(payload) or 0.0
        cist_obl = _mass_effect_metric(payload, "cisternal_obliteration") or False
        cist_eff = _mass_effect_metric(payload, "cisternal_effacement") or False
        vc = _mass_effect_metric(payload, "ventricular_compression") or False
        if shift_mm >= 7.5 or cist_obl:
            tier = "CRITICAL"
            rationale = f"MLS {shift_mm:.1f} mm" + (" + cisternal obliteration" if cist_obl else "")
        elif shift_mm >= 5.0 or cist_eff or vc:
            tier = "WARNING"
            parts = []
            if shift_mm >= 5.0:
                parts.append(f"MLS {shift_mm:.1f} mm")
            if cist_eff:
                parts.append("cisternal effacement")
            if vc:
                parts.append("ventricular compression")
            rationale = " + ".join(parts) if parts else "one or more mass effect indicators"
        else:
            tier = "NONE"
            rationale = f"MLS {shift_mm:.1f} mm, no cisternal obliteration, no ventricular compression"
        return (
            f"Herniation risk tier: {tier}. Basis: {rationale}. "
            "These are automated surrogate markers; clinical assessment and direct imaging review are required.",
            [
                "ctx:segmentation_metrics.mass_effect.midline_shift_mm",
                "ctx:segmentation_metrics.mass_effect.cisternal_obliteration",
                "ctx:segmentation_metrics.mass_effect.ventricular_compression",
            ],
        )
    if q_subtype == "brainstem":
        distance_mm = _critical_structure_distance(payload, "brainstem")
        if distance_mm is not None:
            if distance_mm <= 5.0:
                tier = "HIGH RISK — <5 mm; biopsy-only or SRS typically preferred"
            elif distance_mm <= 10.0:
                tier = "CAUTION — 5–10 mm; intermediate proximity corridor"
            else:
                tier = "SAFE — >10 mm; lower immediate concern"
            return (
                f"Brainstem distance: {distance_mm:.1f} mm (atlas-based proxy). "
                f"Surgical tier: {tier}. "
                "Note: atlas-based proximity only; direct imaging review required.",
                ["ctx:segmentation_metrics.critical_structure_proximity"],
            )
        return (
            "Insufficient evidence: brainstem proximity data is not available in the current case package.",
            ["ctx:segmentation_metrics.critical_structure_proximity"],
        )
    if q_subtype == "motor_cortex":
        distance_mm = _critical_structure_distance(payload, "motor_cortex")
        if distance_mm is not None:
            if distance_mm <= 5.0:
                tier = "HIGH RISK — <5 mm; awake craniotomy with cortical mapping typically indicated"
            elif distance_mm <= 8.0:
                tier = "CAUTION — 5–8 mm; intraoperative neurophysiological monitoring recommended"
            else:
                tier = "SAFE — >8 mm; lower risk corridor"
            return (
                f"Motor cortex (M1) distance: {distance_mm:.1f} mm (atlas-based proxy). "
                f"Surgical tier: {tier}. "
                "Note: atlas-based surface proximity only; confirm with fMRI/DTI before surgical planning.",
                ["ctx:segmentation_metrics.critical_structure_proximity"],
            )
        return (
            "Insufficient evidence: motor cortex proximity data is not available in the current case package.",
            ["ctx:segmentation_metrics.critical_structure_proximity"],
        )
    if q_subtype == "optic_tract":
        distance_mm = _critical_structure_distance(payload, "optic_tract")
        if distance_mm is not None:
            if distance_mm <= 5.0:
                tier = "HIGH RISK — <5 mm; visual field risk; neuro-ophthalmology review recommended"
            elif distance_mm <= 10.0:
                tier = "CAUTION — 5–10 mm; proximity warrants monitoring"
            else:
                tier = "SAFE — >10 mm"
            return (
                f"Optic tract distance: {distance_mm:.1f} mm (atlas-based proxy). "
                f"Tier: {tier}. Direct imaging review required.",
                ["ctx:segmentation_metrics.critical_structure_proximity"],
            )
        return (
            "Insufficient evidence: optic tract proximity data is not available in the current case package.",
            ["ctx:segmentation_metrics.critical_structure_proximity"],
        )
    if q_subtype == "cortical_depth":
        seg_d = _as_dict(payload.context.segmentation_metrics)
        depth = seg_d.get("cortical_depth_mm") or seg_d.get("depth_from_cortex_mm")
        if depth is None:
            loc = _as_dict(seg_d.get("location"))
            depth = loc.get("depth_mm") or loc.get("cortical_depth_mm")
        if depth is not None:
            depth_mm = float(depth)
            tier = "subcortical" if depth_mm > 10 else ("near-cortical" if depth_mm > 3 else "cortical/gyral")
            return (
                f"Estimated depth from cortical surface: {depth_mm:.1f} mm ({tier}). "
                "This is an automated atlas-based estimate; clinician verification required.",
                ["ctx:segmentation_metrics.cortical_depth_mm"],
            )
        return (
            "Insufficient evidence: cortical depth data is not available in the current case package.",
            [],
        )
    if q_subtype.startswith("btrads_"):
        total_cm3 = total_mm3 / 1000.0
        enh_cm3 = enhancing_mm3 / 1000.0
        nonenh_cm3 = nonenh_mm3 / 1000.0
        edema_cm3 = edema_mm3 / 1000.0
        shift_mm = _midline_shift_mm(payload) or 0.0
        mc_dist = _critical_structure_distance(payload, "motor_cortex")
        bs_dist = _critical_structure_distance(payload, "brainstem")
        vc = _mass_effect_metric(payload, "ventricular_compression")
        lines = [
            "BT-RADS STRUCTURED REPORT (automated decision-support — not a clinical diagnosis)",
            "─" * 60,
            f"Tumor Volume: {total_cm3:.2f} cm³  |  Enhancing: {enh_cm3:.2f} cm³  |  "
            f"Non-enhancing: {nonenh_cm3:.2f} cm³  |  Edema: {edema_cm3:.2f} cm³",
        ]
        if shift_mm > 0:
            lines.append(f"Midline Shift: {shift_mm:.1f} mm")
        if mc_dist is not None:
            lines.append(f"Motor Cortex Distance: {mc_dist:.1f} mm")
        if bs_dist is not None:
            lines.append(f"Brainstem Distance: {bs_dist:.1f} mm")
        if vc:
            lines.append("Ventricular Compression: present")
        lines += [
            "─" * 60,
            "Clinical confirmation and multidisciplinary review required before any treatment decision.",
        ]
        return (
            "\n".join(lines),
            [
                "ctx:segmentation_metrics.total_tumor_volume_mm3",
                "ctx:segmentation_metrics.label_volumes_mm3.enhancing",
                "ctx:segmentation_metrics.mass_effect.midline_shift_mm",
                "ctx:segmentation_metrics.critical_structure_proximity",
            ],
        )

    if q_subtype == "edema_ratio":
        ratio = _safe_div(edema_mm3, total_mm3)
        return (
            f"Edema-to-total-tumor volume ratio is {ratio:.3f} ({ratio*100.0:.1f}% of total segmented burden).",
            [
                "ctx:segmentation_metrics.label_volumes_mm3.edema",
                "ctx:segmentation_metrics.total_tumor_volume_mm3",
            ],
        )
    if q_subtype == "enhancing_vs_nonenhancing":
        enh_pct = _safe_div(enhancing_mm3, total_mm3) * 100.0
        non_pct = _safe_div(nonenh_mm3, total_mm3) * 100.0
        dominant = "non-enhancing" if nonenh_mm3 >= enhancing_mm3 else "enhancing"
        return (
            "Enhancing volume is "
            f"{enhancing_mm3:.2f} mm3 ({enh_pct:.1f}% of total), while non-enhancing volume is "
            f"{nonenh_mm3:.2f} mm3 ({non_pct:.1f}% of total). "
            f"The dominant burden is {dominant}.",
            [
                "ctx:segmentation_metrics.label_volumes_mm3.enhancing",
                "ctx:segmentation_metrics.label_volumes_mm3.nonenhancing",
                "ctx:segmentation_metrics.total_tumor_volume_mm3",
            ],
        )
    if q_subtype == "total_volume":
        return (
            f"Total segmented tumor volume is {total_mm3:.2f} mm3 ({total_mm3/1000.0:.2f} cm3).",
            ["ctx:segmentation_metrics.total_tumor_volume_mm3"],
        )
    if q_subtype == "lesion_count":
        if component_count > 0 and component_count != lesion_count:
            return (
                f"Detected lesion count is {lesion_count} after merging touching tumor labels across {component_count} segmented components.",
                ["ctx:segmentation_metrics.tumor_count", "ctx:segmentation_metrics.segmented_component_count"],
            )
        return (
            f"Detected lesion count is {lesion_count}.",
            ["ctx:segmentation_metrics.tumor_count"],
        )
    if q_subtype == "component_count":
        return (
            f"Segmented component count is {component_count}.",
            ["ctx:segmentation_metrics.segmented_component_count"],
        )
    if q_subtype == "dominant_component":
        dominant_name = "mixed burden"
        dominant_mm3 = max(enhancing_mm3, nonenh_mm3, edema_mm3)
        if dominant_mm3 == nonenh_mm3:
            dominant_name = "non-enhancing"
        elif dominant_mm3 == enhancing_mm3:
            dominant_name = "enhancing"
        elif dominant_mm3 == edema_mm3:
            dominant_name = "edema"
        dominant_pct = _safe_div(dominant_mm3, total_mm3) * 100.0
        return (
            f"The largest burden component is {dominant_name} at {dominant_mm3:.2f} mm3 "
            f"({dominant_mm3/1000.0:.2f} cm3, {dominant_pct:.1f}% of total segmented volume).",
            [
                "ctx:segmentation_metrics.label_volumes_mm3.enhancing",
                "ctx:segmentation_metrics.label_volumes_mm3.nonenhancing",
                "ctx:segmentation_metrics.label_volumes_mm3.edema",
                "ctx:segmentation_metrics.total_tumor_volume_mm3",
            ],
        )
    if q_subtype == "ambiguous_size":
        return (
            "Please clarify whether you mean total tumor volume or maximum lesion diameter.",
            [
                "ctx:segmentation_metrics.total_tumor_volume_mm3",
                "ctx:segmentation_metrics.max_diameter_mm",
            ],
        )
    if q_subtype == "max_diameter":
        return (
            f"Maximum segmented lesion diameter is {max_diameter_mm:.2f} mm.",
            ["ctx:segmentation_metrics.max_diameter_mm"],
        )
    if q_subtype == "sequence_inventory":
        if sequences:
            return (
                f"Available MR sequences are: {', '.join(sequences)}.",
                ["ctx:findings_structured.nifti_summary.nifti_files"],
            )
        return (
            "Sequence inventory is insufficiently specified in the current case package.",
            ["ctx:findings_structured.nifti_summary.nifti_files"],
        )
    if q_subtype == "enhancing_volume":
        return (
            f"Enhancing component volume is {enhancing_mm3:.2f} mm3 ({enhancing_mm3/1000.0:.2f} cm3).",
            ["ctx:segmentation_metrics.label_volumes_mm3.enhancing"],
        )
    if q_subtype == "edema_volume":
        edema_pct = _safe_div(edema_mm3, total_mm3) * 100.0
        return (
            f"Edema component volume is {edema_mm3:.2f} mm3 ({edema_mm3/1000.0:.2f} cm3, {edema_pct:.1f}% of total segmented burden).",
            [
                "ctx:segmentation_metrics.label_volumes_mm3.edema",
                "ctx:segmentation_metrics.total_tumor_volume_mm3",
            ],
        )
    if q_subtype == "largest_lesion_location":
        if lesion_list:
            largest = _largest_component(payload)
            label_name = str(largest.get("label_name", "unknown"))
            largest_mm3 = float(largest.get("volume_mm3", 0) or 0)
            diameter_mm = float(largest.get("max_diameter_mm", 0) or 0)
            lesion_id = int(largest.get("lesion_id", 0) or 0)
            burden_pct = _safe_div(largest_mm3, total_mm3) * 100.0
            location_desc = _anatomic_description(payload, largest)
            centroid_text = _format_centroid(largest)
            location_label = _location_confidence_label(largest)
            location_sentence = (
                f"{location_label}: {location_desc}. "
                if location_desc
                else f"Approximate location descriptor: {centroid_text}. "
            )
            return (
                "Largest segmented component is "
                f"lesion #{lesion_id} ({label_name}) with volume {largest_mm3:.2f} mm3, "
                f"representing {burden_pct:.1f}% of the total segmented burden, and max diameter {diameter_mm:.2f} mm. "
                + location_sentence
                + (
                    f"Centroid reference: {centroid_text}. "
                    if location_desc and centroid_text != "centroid unavailable"
                    else ""
                )
                + "Location should still be clinician-verified.",
                [
                    "ctx:segmentation_metrics.lesion_list",
                    "ctx:segmentation_metrics.location_labels",
                ],
            )
        return (
            "Largest lesion cannot be determined from current segmentation context.",
            ["ctx:segmentation_metrics.lesion_list"],
        )
    if q_subtype == "multiplicity" and ("multiple distinct lesions" in q or "one complex mass" in q):
        if lesion_count > 1:
            return (
                f"Segmentation indicates multiple distinct lesion groups ({lesion_count}) across {component_count} segmented components.",
                ["ctx:segmentation_metrics.tumor_count", "ctx:segmentation_metrics.segmented_component_count"],
            )
        if lesion_count == 1:
            return (
                (
                    f"Segmentation indicates one dominant lesion with {component_count} segmented component(s)."
                    if component_count > 1
                    else "Segmentation indicates one dominant lesion."
                ),
                ["ctx:segmentation_metrics.tumor_count", "ctx:segmentation_metrics.segmented_component_count"],
            )
        return (
            "Lesion multiplicity cannot be determined from current context.",
            ["ctx:segmentation_metrics.tumor_count"],
        )
    if q_subtype == "multiplicity":
        return (
            _multifocality_direct_answer(payload),
            [
                "ctx:segmentation_metrics.tumor_count",
                "ctx:segmentation_metrics.segmented_component_count",
                "ctx:segmentation_metrics.total_tumor_volume_mm3",
                "ctx:segmentation_metrics.lesion_list",
            ],
        )
    if q_subtype == "enhancement_pattern" and ("focal" in q or "diffuse" in q):
        enh_ratio = _safe_div(enhancing_mm3, max(total_mm3, 1e-9))
        if total_mm3 <= 0:
            pattern = "insufficient evidence"
        elif enh_ratio >= 0.4:
            pattern = "more diffuse/heterogeneous enhancement"
        else:
            pattern = "more focal-limited enhancement"
        return (
            "Based on volume fractions, the enhancement pattern appears "
            f"{pattern}. Case profile suggests {_enhancement_style(enh_ratio * 100.0)} behavior. "
            "This is imaging-only and requires radiologist confirmation.",
            [
                "ctx:segmentation_metrics.label_volumes_mm3.enhancing",
                "ctx:segmentation_metrics.total_tumor_volume_mm3",
            ],
        )

    return (
        "Insufficient evidence for a definitive clinical answer in this case-level QA.",
        [],
    )


def _is_structured_question(question: str) -> bool:
    question_class, question_subtype = _question_intent(question)
    return question_class in {"quantitative", "comparison", "localization", "ct_organ", "mass_effect", "critical_proximity", "structured_reporting"} or (
        question_class == "summary" and question_subtype == "sequence_inventory"
    )


def _requires_high_model(question: str) -> bool:
    q = _normalize_question(question.lower())
    high_risk_terms = [
        "diagnosis",
        "malignant",
        "benign",
        "treatment",
        "stage",
        "metastasis",
        "therapy",
        "prognosis",
        "cancer",
        "surgery",
    ]
    return any(term in q for term in high_risk_terms)


def _structured_llm_rewrite_allowed(question_subtype: str) -> bool:
    return question_subtype in {
        "total_volume",
        "lesion_count",
        "component_count",
        "dominant_component",
        "max_diameter",
        "enhancing_volume",
        "edema_volume",
        "enhancing_vs_nonenhancing",
        "edema_ratio",
        "largest_lesion_location",
    }


def _confidence_adjustment(payload: QARequest, answer: str, evidence_ids: list[str]) -> float:
    adj = 0.0
    if evidence_ids:
        adj += min(0.04, 0.01 * len(evidence_ids))
    if "exact neuroanatomic localization is not available" in answer.lower():
        adj -= 0.03
    if "high-confidence atlas-supported location" in answer.lower():
        adj += 0.01
    if "approximate atlas-supported location" in answer.lower():
        adj -= 0.02
    if "heuristic location only" in answer.lower():
        adj -= 0.04
    if "insufficient evidence" in answer.lower():
        adj -= 0.02

    tumor_count = int(_context_metrics(payload)["tumor_count"])
    if tumor_count >= 3 and len(evidence_ids) >= 2:
        adj += 0.01
    return adj


def _confidence_cap(payload: QARequest, answer: str, is_structured_question: bool) -> float | None:
    low_answer = answer.lower()
    if is_structured_question:
        if "centroid heuristic" in low_answer or "localisation confidence: low" in low_answer:
            return 0.68
        if "heuristic location only" in low_answer:
            return 0.64
        if "approximate atlas-supported location" in low_answer:
            return 0.66
        if "high-confidence atlas-supported location" in low_answer:
            return 0.72
        return None

    question_class, question_subtype = _question_intent(payload.question)
    if question_class == "summary" and question_subtype == "plain_language_summary":
        return 0.58
    if question_class == "summary" and question_subtype == "tumor_board_summary":
        return 0.60
    return None


def _structured_confidence_floor(question_subtype: str, answer: str, evidence_ids: list[str]) -> float | None:
    if _is_insufficient_answer(answer) or not evidence_ids:
        return None
    exact_metric_subtypes = {
        "total_volume",
        "lesion_count",
        "component_count",
        "max_diameter",
        "enhancing_volume",
        "edema_volume",
        "edema_ratio",
        "dominant_component",
        "liver_volume",
        "liver_tumor_ratio",
        "lung_tumor_volume",
    }
    if question_subtype in exact_metric_subtypes:
        return 0.72
    if question_subtype in {"sequence_inventory", "ct_sequence"}:
        return 0.74
    if question_subtype in {"enhancing_vs_nonenhancing", "largest_lesion_location", "multiplicity", "lobe_location"}:
        return 0.66
    return None


def _supporting_case_retrieval(question: str, resolved_intent: QAIntent) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    if resolved_intent.requires_guideline_retrieval:
        try:
            from ai_assistant.core.guideline_retrieval import retrieve_guidelines
            snippets = retrieve_guidelines(question, intent_class=resolved_intent.intent_class, top_k=2, min_score=0.8)
            if snippets:
                return [], {
                    "policy": "guideline_retrieval",
                    "retrieval_executed": True,
                    "requires_guideline_retrieval": True,
                    "guideline_snippets": [{"id": s.get("id"), "text": s.get("text", "")[:200]} for s in snippets],
                    "guideline_count": len(snippets),
                }
        except Exception as exc:
            LOGGER.warning("Guideline retrieval failed: %s", exc)
        return [], {"policy": "guideline_retrieval", "retrieval_executed": False, "requires_guideline_retrieval": True}
    if not resolved_intent.requires_retrieval:
        return [], {
            "policy": resolved_intent.retrieval_policy,
            "retrieval_executed": False,
            "requires_retrieval": False,
            "requires_guideline_retrieval": False,
            "reason": "intent does not require case retrieval",
        }

    try:
        from ai_assistant.core.retrieval_index import search

        supporting_cases = search(question, top_k=3)
        trace = {
            "policy": resolved_intent.retrieval_policy,
            "retrieval_executed": True,
            "requires_retrieval": True,
            "requires_guideline_retrieval": False,
            "result_count": len(supporting_cases),
            "reason": "supporting case retrieval completed",
        }
        return supporting_cases, trace
    except Exception as exc:
        LOGGER.warning("Supporting case retrieval failed for '%s': %s", question, exc)
        return [], {
            "policy": resolved_intent.retrieval_policy,
            "retrieval_executed": False,
            "requires_retrieval": True,
            "requires_guideline_retrieval": False,
            "error": str(exc),
            "reason": "supporting case retrieval failed",
        }


def answer_question(payload: QARequest) -> QAResponse:
    if payload.reset_session and payload.session_id:
        clear_session(payload.session_id)

    original_question = payload.question
    quality_issues = validate_patient_context(payload.context)
    blocking_issues = _blocking_quality_issues(quality_issues)
    can_use_qa_llm = _qa_llm_enabled()
    session_context = build_session_context(
        payload.session_id,
        patient_id=payload.patient_id,
        study_id=payload.study_id,
    )
    previous_intent = latest_resolved_intent(
        payload.session_id,
        patient_id=payload.patient_id,
        study_id=payload.study_id,
    )


    if can_use_qa_llm and not blocking_issues:
        _llm_q = _llm_normalize_intent(payload.question)
        if _llm_q != payload.question:
            payload = payload.model_copy(update={"question": _llm_q})
    resolved_question = _resolve_question_with_session(
        payload.question,
        session_context,
        allow_llm=can_use_qa_llm and not blocking_issues,
    )
    if resolved_question != payload.question:
        payload = payload.model_copy(update={"question": resolved_question})

    resolved_intent = _resolved_intent(
        payload.question,
        previous_intent=previous_intent,
        can_use_arbiter=can_use_qa_llm and not blocking_issues,
        session_context=session_context,
    )
    routed_payload = payload.model_copy(update={"question": _intent_route_question(resolved_intent, payload.question)})

    context_confidence = confidence_from_context_quality(payload.context)
    confidence = context_confidence
    is_structured_question = _is_structured_question(routed_payload.question)
    question_subtype = resolved_intent.legacy_question_subtype

    model_used: str | None = None
    answer, evidence_ids = _structured_answer(routed_payload)

    narrative_override = _hybrid_narrative_summary(
        routed_payload,
        allow_llm=can_use_qa_llm and not blocking_issues,
        session_context=session_context,
    )
    safe_override = _balanced_clinical_support(routed_payload) if narrative_override is None else None
    if narrative_override is not None:
        answer, evidence_ids, conf_cap, model_used = narrative_override
        confidence = min(confidence, conf_cap)
    elif safe_override is not None:
        answer, evidence_ids, conf_cap = safe_override
        confidence = min(confidence, conf_cap)
    elif (
        can_use_qa_llm
        and not blocking_issues
        and is_structured_question
        and _structured_llm_rewrite_allowed(question_subtype)
        and os.getenv("AI_QA_USE_LLM_NONSTRUCTURED", "0") == "1"
    ):
        fallback_answer = answer
        llm_answer, model_used = generate_qa_text(
            question=payload.question,
            context=payload.context.model_dump(),
            fallback_answer=fallback_answer,
            use_high_model=_requires_high_model(payload.question),
            session_context=session_context,
        )
        if model_used:
            generated_issues = validate_generated_clinical_text(llm_answer, payload.context)
            if not generated_issues:
                answer = llm_answer
                evidence_ids = _dedupe(evidence_ids + _default_nonstructured_evidence_ids(payload))
            else:
                model_used = None
    elif (
        can_use_qa_llm
        and not blocking_issues
        and question_subtype != "narrative_summary"
        and not is_structured_question
    ):
        fallback_answer = answer
        llm_answer, model_used = generate_qa_text(
            question=payload.question,
            context=payload.context.model_dump(),
            fallback_answer=fallback_answer,
            use_high_model=_requires_high_model(payload.question),
            session_context=session_context,
        )
        answer = llm_answer
        if model_used:
            generated_issues = validate_generated_clinical_text(llm_answer, payload.context)
            if generated_issues:
                answer = fallback_answer
                model_used = None
            else:
                evidence_ids = _dedupe(evidence_ids + _default_nonstructured_evidence_ids(payload))
        if model_used:
            evidence_ids = _dedupe(evidence_ids + _default_nonstructured_evidence_ids(payload))

    if blocking_issues:
        readable_issues = [_quality_issue_reason(issue) for issue in blocking_issues]
        answer = (
            "Insufficient evidence.\n"
            + "- Missing or low-quality context: "
            + "; ".join(readable_issues)
            + ".\n"
            + "- Source: current case context.\n"
            + "- Clinical confirmation is required."
        )
        confidence = min(confidence, 0.42)
        evidence_ids = []

    if _is_insufficient_answer(answer):
        confidence = min(confidence, 0.42)

    clinical_implication = None
    if os.getenv("AI_ENABLE_CLINICAL_IMPLICATIONS", "1") == "1":
        clinical_implication = build_clinical_implication(payload.context, resolved_intent)
        answer, clinical_implication = append_implication_to_answer(answer, clinical_implication, resolved_intent)

    llm_bonus = 0.02 if (model_used and not _is_insufficient_answer(answer)) else 0.0
    evidence_bonus = 0.0
    if is_structured_question and not _is_insufficient_answer(answer):
        evidence_bonus = _confidence_adjustment(routed_payload, answer, evidence_ids)
    effective_confidence = min(0.99, max(0.0, confidence + llm_bonus + evidence_bonus))
    confidence_cap = _confidence_cap(routed_payload, answer, is_structured_question)
    if confidence_cap is not None:
        effective_confidence = min(effective_confidence, confidence_cap)
    confidence_floor = _structured_confidence_floor(question_subtype, answer, evidence_ids)
    if confidence_floor is not None:
        effective_confidence = max(effective_confidence, confidence_floor)

    deduped_evidence = _dedupe(evidence_ids)
    answer_class = resolved_intent.legacy_question_class
    answer_subtype = resolved_intent.legacy_question_subtype
    turns_used = session_turn_count(payload.session_id)
    supporting_cases, retrieval_trace = _supporting_case_retrieval(payload.question, resolved_intent)

    if payload.session_id and not _is_insufficient_answer(answer):
        append_turn(
            session_id=payload.session_id,
            patient_id=payload.patient_id,
            study_id=payload.study_id,
            original_question=original_question,
            resolved_question=payload.question,
            answer=answer,
            question_class=answer_class,
            question_subtype=answer_subtype,
            resolved_intent=resolved_intent.to_dict(),
        )
        turns_used = session_turn_count(payload.session_id)

    return QAResponse(
        answer=answer,
        evidence_ids=deduped_evidence,
        confidence=effective_confidence,
        context_confidence=context_confidence,
        answer_confidence=effective_confidence,
        confidence_breakdown={
            "question_class": answer_class,
            "question_subtype": answer_subtype,
            "intent_class": resolved_intent.intent_class,
            "intent_confidence": round(float(resolved_intent.router_confidence or 0.0), 4),
            "intent_source": resolved_intent.intent_source,
            "retrieval_policy": resolved_intent.retrieval_policy,
            "context_completeness_confidence": round(context_confidence, 4),
            "answer_confidence": round(effective_confidence, 4),
            "evidence_count": len(deduped_evidence),
            "evidence_sources": _evidence_source_labels(deduped_evidence),
            "used_llm": bool(model_used),
            "session_turns_used": turns_used,
            "clinical_implication_count": len(clinical_implication.items) if clinical_implication else 0,
            "clinical_implication_overall_tier": clinical_implication.overall_tier if clinical_implication else "",
        },
        safety_note=safety_note_for_low_evidence(effective_confidence),
        session_id=payload.session_id,
        resolved_question=payload.question,
        resolved_intent=resolved_intent.to_dict(),
        retrieval_trace=retrieval_trace,
        supporting_cases=supporting_cases,
        clinical_implication=clinical_implication.to_dict() if clinical_implication else None,
        session_turns_used=turns_used,
    )
