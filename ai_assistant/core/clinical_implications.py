from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from ai_assistant.core.intent_router import QAIntent


_MLS_REFERENCES = [
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC6208863/",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC6651068/",
]
_CISTERN_REFERENCES = [
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC6651068/",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC6208863/",
]
_MOTOR_REFERENCES = [
    "https://pubmed.ncbi.nlm.nih.gov/20679917/",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC12954220/",
]
_VOLUME_REFERENCES = [
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC3264493/",
    "https://pubmed.ncbi.nlm.nih.gov/38337543/",
    "https://pubmed.ncbi.nlm.nih.gov/41265784/",
]
_BRAINSTEM_REFERENCES = [
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC6208863/",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC6651068/",
]

_OVERALL_TIER_RANK = {
    "none": 0,
    "safe": 0,
    "limited": 0,
    "low": 0,
    "warning": 1,
    "caution": 1,
    "moderate": 1,
    "notable": 1,
    "critical": 2,
    "high_risk": 2,
    "substantial": 2,
}


@dataclass
class ClinicalImplicationItem:
    implication_type: str
    source_metric: str
    metric_value: float | bool | str | None
    unit: str | None
    tier: str
    severity_rank: int
    rationale: str
    evidence_level: str
    literature_anchor: list[str] = field(default_factory=list)
    btrads_alignment: str = ""
    surfaced_in_answer: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ClinicalImplication:
    generated: bool
    summary: str
    overall_tier: str
    triggered_by_intent: bool
    items: list[ClinicalImplicationItem] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["items"] = [item.to_dict() for item in self.items]
        return data


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
    return [_as_dict(item) for item in value if _as_dict(item)]


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _bt_rads_alignment_for_tier(tier: str) -> str:
    if tier in {"critical", "high_risk", "substantial"}:
        return (
            "If this finding is new or increasing versus prior imaging, it is compatible with BT-RADS language "
            "that strengthens concern for clinically meaningful progression rather than stable-treatment effect."
        )
    if tier in {"warning", "caution", "moderate", "notable"}:
        return (
            "If interval worsening is present, this can strengthen BT-RADS progression-oriented language, "
            "but does not assign a BT-RADS category on its own."
        )
    return "This finding does not independently strengthen BT-RADS progression concern without interval change."


def _metric_from_seg(seg: dict[str, Any], key: str) -> float | None:
    return _safe_float(seg.get(key))


def _midline_shift_mm(seg: dict[str, Any]) -> float | None:
    midline = _as_dict(seg.get("midline_shift"))
    if isinstance(midline.get("value_mm"), (int, float)):
        return float(midline["value_mm"])
    mass_effect = _as_dict(seg.get("mass_effect"))
    if isinstance(mass_effect.get("midline_shift_mm"), (int, float)):
        return float(mass_effect["midline_shift_mm"])
    return None


def _mass_effect_metric(seg: dict[str, Any], key: str) -> Any:
    mass_effect = _as_dict(seg.get("mass_effect"))
    return mass_effect.get(key)


def _critical_distance_mm(seg: dict[str, Any], structure_name: str) -> float | None:
    candidates = seg.get("critical_structure_proximity")
    if candidates is None:
        candidates = seg.get("critical_proximity")
    for item in _list_of_dicts(candidates):
        structure = str(item.get("structure", "") or "").lower()
        if structure in {structure_name.lower(), structure_name.lower().replace("_", " ")}:
            return _safe_float(item.get("distance_mm"))
    return None


def _enhancing_volume_mm3(seg: dict[str, Any]) -> float | None:
    label_volumes = _as_dict(seg.get("label_volumes_mm3"))
    return _safe_float(label_volumes.get("enhancing"))


def _total_tumor_volume_mm3(seg: dict[str, Any]) -> float | None:
    value = _safe_float(seg.get("total_tumor_volume_mm3"))
    if value is not None:
        return value
    return _safe_float(seg.get("volume_mm3"))


def _midline_shift_implication(shift_mm: float) -> ClinicalImplicationItem:
    if shift_mm < 5.0:
        tier, severity_rank = "none", 0
        rationale = (
            "Below the commonly used >5 mm threshold for significant mass effect; this does not by itself suggest a high herniation-risk surrogate tier."
        )
    elif shift_mm < 7.5:
        tier, severity_rank = "warning", 1
        rationale = (
            "At or above the commonly used >5 mm significant-shift threshold, which is associated with clinically relevant mass effect and warrants caution."
        )
    else:
        tier, severity_rank = "critical", 2
        rationale = (
            "At or above the ~7-7.5 mm range linked in neurotrauma literature to effaced basal cisterns and abnormal pupillary findings, supporting a high-risk herniation surrogate tier."
        )
    return ClinicalImplicationItem(
        implication_type="herniation_risk",
        source_metric="midline_shift_mm",
        metric_value=round(shift_mm, 4),
        unit="mm",
        tier=tier,
        severity_rank=severity_rank,
        rationale=rationale,
        evidence_level="literature_anchored_surrogate",
        literature_anchor=list(_MLS_REFERENCES),
        btrads_alignment=_bt_rads_alignment_for_tier(tier),
    )


def _motor_cortex_implication(distance_mm: float) -> ClinicalImplicationItem:
    if distance_mm > 8.0:
        tier, severity_rank = "safe", 0
        rationale = (
            "Above the ~8 mm separation often used in motor-eloquence planning literature as a lower-risk zone for postoperative motor deficit."
        )
    elif distance_mm > 5.0:
        tier, severity_rank = "caution", 1
        rationale = (
            "Within an intermediate motor-eloquence corridor where resection planning usually warrants caution and functional mapping support."
        )
    else:
        tier, severity_rank = "high_risk", 2
        rationale = (
            "Within a sub-5 mm eloquent interval; this is surgically high-risk and usually treated as requiring strong mapping caution."
        )
    return ClinicalImplicationItem(
        implication_type="surgical_eloquence_risk",
        source_metric="motor_cortex_distance_mm",
        metric_value=round(distance_mm, 4),
        unit="mm",
        tier=tier,
        severity_rank=severity_rank,
        rationale=rationale,
        evidence_level="literature_anchored_planning_threshold",
        literature_anchor=list(_MOTOR_REFERENCES),
        btrads_alignment=_bt_rads_alignment_for_tier(tier),
    )


def _brainstem_implication(distance_mm: float) -> ClinicalImplicationItem:
    if distance_mm > 10.0:
        tier, severity_rank = "safe", 0
        rationale = "More than 10 mm from the brainstem, which lowers immediate concern for direct brainstem compression."
    elif distance_mm > 5.0:
        tier, severity_rank = "caution", 1
        rationale = "Within 5-10 mm of the brainstem, supporting caution for evolving mass effect or crowding."
    else:
        tier, severity_rank = "high_risk", 2
        rationale = "Within 5 mm of the brainstem, supporting a high-risk proximity tier for compression-related concern."
    return ClinicalImplicationItem(
        implication_type="brainstem_compression_risk",
        source_metric="brainstem_distance_mm",
        metric_value=round(distance_mm, 4),
        unit="mm",
        tier=tier,
        severity_rank=severity_rank,
        rationale=rationale,
        evidence_level="critical_structure_proximity_surrogate",
        literature_anchor=list(_BRAINSTEM_REFERENCES),
        btrads_alignment=_bt_rads_alignment_for_tier(tier),
    )


def _enhancing_volume_implication(enhancing_volume_mm3: float) -> ClinicalImplicationItem:
    volume_cm3 = enhancing_volume_mm3 / 1000.0
    if volume_cm3 < 15.0:
        tier, severity_rank = "limited", 0
        rationale = (
            "Enhancing burden is below the larger-volume ranges commonly discussed in glioblastoma cytoreduction literature."
        )
    elif volume_cm3 <= 30.0:
        tier, severity_rank = "notable", 1
        rationale = (
            "Enhancing burden is in a moderate range where cytoreductive benefit becomes clinically meaningful if anatomy is favorable."
        )
    else:
        tier, severity_rank = "substantial", 2
        rationale = (
            "Enhancing burden exceeds 30 cm3, which is a substantial preoperative volume range and strengthens resection-consideration language if safely accessible."
        )
    return ClinicalImplicationItem(
        implication_type="resection_consideration_tier",
        source_metric="enhancing_volume_mm3",
        metric_value=round(enhancing_volume_mm3, 4),
        unit="mm3",
        tier=tier,
        severity_rank=severity_rank,
        rationale=rationale,
        evidence_level="cohort_anchored_volumetric_surrogate",
        literature_anchor=list(_VOLUME_REFERENCES),
        btrads_alignment=_bt_rads_alignment_for_tier(tier),
    )


def _total_volume_implication(total_volume_mm3: float) -> ClinicalImplicationItem:
    volume_cm3 = total_volume_mm3 / 1000.0
    if volume_cm3 < 25.0:
        tier, severity_rank = "low", 0
        rationale = "Overall segmented burden is relatively limited on a volumetric basis."
    elif volume_cm3 <= 75.0:
        tier, severity_rank = "moderate", 1
        rationale = "Overall segmented burden is moderate and may contribute to meaningful local mass effect depending on location."
    else:
        tier, severity_rank = "high_risk", 2
        rationale = "Overall segmented burden is high, which strengthens concern for clinically important intracranial mass effect."
    return ClinicalImplicationItem(
        implication_type="tumor_burden_tier",
        source_metric="total_tumor_volume_mm3",
        metric_value=round(total_volume_mm3, 4),
        unit="mm3",
        tier=tier,
        severity_rank=severity_rank,
        rationale=rationale,
        evidence_level="cohort_anchored_burden_surrogate",
        literature_anchor=list(_VOLUME_REFERENCES),
        btrads_alignment=_bt_rads_alignment_for_tier(tier),
    )


def _cisternal_implication(seg: dict[str, Any]) -> ClinicalImplicationItem | None:
    obliteration = _mass_effect_metric(seg, "cisternal_obliteration")
    effacement = _mass_effect_metric(seg, "cisternal_effacement")
    if isinstance(obliteration, bool):
        if obliteration:
            tier, severity_rank = "critical", 2
            rationale = "Basal cisternal obliteration is a high-risk surrogate for raised ICP and transtentorial crowding."
        else:
            if isinstance(effacement, bool) and effacement:
                tier, severity_rank = "warning", 1
                rationale = "Cisternal effacement/compression is a warning-level surrogate for raised intracranial pressure."
            else:
                tier, severity_rank = "none", 0
                rationale = "No cisternal effacement or obliteration is recorded in the current structured mass-effect fields."
        return ClinicalImplicationItem(
            implication_type="icp_surrogate_tier",
            source_metric="cisternal_obliteration",
            metric_value=obliteration,
            unit=None,
            tier=tier,
            severity_rank=severity_rank,
            rationale=rationale,
            evidence_level="literature_anchored_surrogate",
            literature_anchor=list(_CISTERN_REFERENCES),
            btrads_alignment=_bt_rads_alignment_for_tier(tier),
        )
    if isinstance(effacement, bool):
        tier = "warning" if effacement else "none"
        severity_rank = 1 if effacement else 0
        rationale = (
            "Cisternal effacement/compression is present and is used as a warning-level surrogate for elevated ICP."
            if effacement
            else "No cisternal effacement is recorded in the current structured fields."
        )
        return ClinicalImplicationItem(
            implication_type="icp_surrogate_tier",
            source_metric="cisternal_effacement",
            metric_value=effacement,
            unit=None,
            tier=tier,
            severity_rank=severity_rank,
            rationale=rationale,
            evidence_level="literature_anchored_surrogate",
            literature_anchor=list(_CISTERN_REFERENCES),
            btrads_alignment=_bt_rads_alignment_for_tier(tier),
        )
    return None


def _all_implications_from_seg(seg: dict[str, Any]) -> list[ClinicalImplicationItem]:
    items: list[ClinicalImplicationItem] = []

    shift_mm = _midline_shift_mm(seg)
    if shift_mm is not None:
        items.append(_midline_shift_implication(shift_mm))

    motor_distance = _critical_distance_mm(seg, "motor_cortex")
    if motor_distance is not None:
        items.append(_motor_cortex_implication(motor_distance))

    brainstem_distance = _critical_distance_mm(seg, "brainstem")
    if brainstem_distance is not None:
        items.append(_brainstem_implication(brainstem_distance))

    enhancing_volume_mm3 = _enhancing_volume_mm3(seg)
    if enhancing_volume_mm3 is not None:
        items.append(_enhancing_volume_implication(enhancing_volume_mm3))

    total_volume_mm3 = _total_tumor_volume_mm3(seg)
    if total_volume_mm3 is not None:
        items.append(_total_volume_implication(total_volume_mm3))

    cisternal_item = _cisternal_implication(seg)
    if cisternal_item is not None:
        items.append(cisternal_item)

    return items


def _measurement_to_implication_types(intent: QAIntent) -> set[str]:
    mapping = {
        "midline_shift_mm": {"herniation_risk"},
        "distance_to_motor_cortex_mm": {"surgical_eloquence_risk"},
        "distance_to_brainstem_mm": {"brainstem_compression_risk"},
        "enhancing_volume_mm3": {"resection_consideration_tier"},
        "total_volume_mm3": {"tumor_burden_tier"},
        "cisternal_obliteration": {"icp_surrogate_tier"},
        "cisternal_effacement": {"icp_surrogate_tier"},
    }
    target_measurement = str(intent.target_measurement or "")
    if target_measurement in mapping:
        return mapping[target_measurement]
    subtype = str(intent.legacy_question_subtype or "")
    subtype_map = {
        "midline_shift": {"herniation_risk"},
        "motor_cortex": {"surgical_eloquence_risk"},
        "brainstem": {"brainstem_compression_risk"},
        "enhancing_volume": {"resection_consideration_tier"},
        "total_volume": {"tumor_burden_tier"},
        "cisternal_obliteration": {"icp_surrogate_tier"},
        "sulcal_effacement": set(),
        "ventricular_compression": set(),
    }
    return subtype_map.get(subtype, set())


def _filter_implications(items: list[ClinicalImplicationItem], intent: QAIntent) -> list[ClinicalImplicationItem]:
    if intent.intent_class == "mass_effect_query":
        allowed = {"herniation_risk", "icp_surrogate_tier"}
        if intent.legacy_question_subtype == "midline_shift":
            allowed = {"herniation_risk"}
        elif intent.legacy_question_subtype == "cisternal_obliteration":
            allowed = {"icp_surrogate_tier"}
        return [item for item in items if item.implication_type in allowed]

    if intent.intent_class == "critical_proximity_query":
        allowed = {"surgical_eloquence_risk", "brainstem_compression_risk"}
        if intent.legacy_question_subtype == "motor_cortex":
            allowed = {"surgical_eloquence_risk"}
        elif intent.legacy_question_subtype == "brainstem":
            allowed = {"brainstem_compression_risk"}
        return [item for item in items if item.implication_type in allowed]

    if intent.intent_class == "urgency_assessment":
        return [item for item in items if item.tier not in {"none", "safe", "limited", "low"}]

    if intent.intent_class == "measurement_query":
        allowed = _measurement_to_implication_types(intent)
        if not allowed:
            return []
        filtered = [item for item in items if item.implication_type in allowed]
        return [item for item in filtered if item.tier not in {"none", "safe", "limited", "low"}]

    return []


def _overall_tier(items: list[ClinicalImplicationItem]) -> str:
    if not items:
        return ""
    ranked = max(items, key=lambda item: item.severity_rank)
    return ranked.tier


def _summary(items: list[ClinicalImplicationItem]) -> str:
    if not items:
        return ""
    top = max(items, key=lambda item: item.severity_rank)
    summaries = {
        "herniation_risk": "Mass-effect metrics support a herniation-risk surrogate assessment.",
        "surgical_eloquence_risk": "Critical proximity metrics support a motor-eloquence surgical-risk assessment.",
        "brainstem_compression_risk": "Critical proximity metrics support a brainstem-compression risk assessment.",
        "resection_consideration_tier": "Enhancing burden supports a resection-consideration tier.",
        "icp_surrogate_tier": "Cisternal findings support an ICP-surrogate tier.",
        "tumor_burden_tier": "Overall segmented burden supports a volumetric burden tier.",
    }
    return summaries.get(top.implication_type, "Structured imaging metrics support a clinical implication tier.")


def build_clinical_implication(context: Any, intent: QAIntent) -> ClinicalImplication | None:
    seg = _as_dict(getattr(context, "segmentation_metrics", None))
    if not seg:
        return None
    items = _all_implications_from_seg(seg)
    filtered = _filter_implications(items, intent)
    if not filtered:
        return None
    return ClinicalImplication(
        generated=True,
        summary=_summary(filtered),
        overall_tier=_overall_tier(filtered),
        triggered_by_intent=True,
        items=filtered,
    )


def implication_should_surface_in_answer(implication: ClinicalImplication | None, intent: QAIntent) -> bool:
    if implication is None or not implication.items:
        return False
    if intent.intent_class in {"mass_effect_query", "critical_proximity_query", "urgency_assessment"}:
        return True
    if intent.intent_class == "measurement_query":
        return any(item.severity_rank >= 1 for item in implication.items)
    return False


def append_implication_to_answer(answer: str, implication: ClinicalImplication | None, intent: QAIntent) -> tuple[str, ClinicalImplication | None]:
    if not implication_should_surface_in_answer(implication, intent):
        return answer, implication
    if implication is None:
        return answer, None

    surfaced_items: list[ClinicalImplicationItem] = []
    for item in implication.items:
        surfaced = ClinicalImplicationItem(**item.to_dict())
        surfaced.surfaced_in_answer = True
        surfaced_items.append(surfaced)

    updated = ClinicalImplication(
        generated=implication.generated,
        summary=implication.summary,
        overall_tier=implication.overall_tier,
        triggered_by_intent=implication.triggered_by_intent,
        items=surfaced_items,
    )

    lead = max(surfaced_items, key=lambda item: item.severity_rank)
    implication_text = (
        f" Clinical implication: {lead.implication_type.replace('_', ' ')} tier is {lead.tier.replace('_', ' ')}. "
        f"{lead.rationale}"
    )
    return answer + implication_text, updated
