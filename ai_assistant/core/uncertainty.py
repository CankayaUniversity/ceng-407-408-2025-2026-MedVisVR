from __future__ import annotations

from typing import Any

_VALIDATED_DICE: dict[str, float] = {
    "enhancing":    0.87,   
    "nonenhancing": 0.80,   
    "edema":        0.82,   
    "lung":         0.65,   
    "liver":        0.95,   
    "liver_tumor":  0.60,   
}

_DEFAULT_DICE    = 0.80   
_MINIMUM_FLOOR   = 0.03   


def _round4(value: float) -> float:
    return round(float(value), 4)


def _dice_for_label(label_name: str) -> float:
    key = (label_name or "").lower().strip()
    return _VALIDATED_DICE.get(key, _DEFAULT_DICE)


def _uncertainty_pct(label_name: str) -> float:
    dice = _dice_for_label(label_name)
    return max(round(1.0 - dice, 4), _MINIMUM_FLOOR)


def _with_ci(value: float, error_pct: float) -> dict[str, float]:
    margin = value * error_pct
    return {
        "value": _round4(value),
        "lower": _round4(max(0.0, value - margin)),
        "upper": _round4(value + margin),
    }


def _confidence_label(total_mm3: float, tumor_count: int, mean_dice: float) -> str:
    if total_mm3 <= 0:
        return "Low"
    if mean_dice >= 0.85 and tumor_count <= 1:
        return "High"
    if mean_dice >= 0.75:
        return "Moderate"
    return "Low"


def add_confidence_intervals(
    metrics: dict[str, Any],
    measurement_error_pct: float | None = None,  
) -> dict[str, Any]:
    enriched = dict(metrics)
    total_mm3 = float(
        enriched.get("total_tumor_volume_mm3", enriched.get("volume_mm3", 0.0)) or 0.0
    )
    tumor_count = int(
        enriched.get("segmented_component_count", enriched.get("tumor_count", 0)) or 0
    )
    label_volumes: dict[str, Any] = enriched.get("label_volumes_mm3", {})
    if not isinstance(label_volumes, dict):
        label_volumes = {}

    per_label_ci: dict[str, Any] = {}
    dice_values: list[float] = []
    for label_name, vol in label_volumes.items():
        vol_f = float(vol or 0.0)
        unc   = _uncertainty_pct(label_name)
        dice  = _dice_for_label(label_name)
        dice_values.append(dice)
        per_label_ci[str(label_name)] = {
            **_with_ci(vol_f, unc),
            "uncertainty_pct": round(unc * 100.0, 1),
            "validated_dice":  dice,
        }

    mean_dice   = (sum(dice_values) / len(dice_values)) if dice_values else _DEFAULT_DICE
    overall_unc = max(round(1.0 - mean_dice, 4), _MINIMUM_FLOOR)

    max_diameter_mm = float(enriched.get("max_diameter_mm", 0.0) or 0.0)
    diameter_method = str(enriched.get("diameter_method", "bounding_box_max"))

    uncertainty: dict[str, Any] = {
        "method": "dice_validated",
        "note": (
            "Confidence intervals are derived from (1 − Dice) per tissue label "
            "using internally validated Dice scores. "
            "This reflects the expected segmentation volume error relative to "
            "ground truth. "
            "Previously a fixed ±3 % band was used, which underestimated real "
            "model error (e.g. edema Dice=0.82 → true uncertainty ~18 %)."
        ),
        "validated_dice_per_label": {
            k: v["validated_dice"] for k, v in per_label_ci.items()
        },
        "confidence_label":   _confidence_label(total_mm3, tumor_count, mean_dice),
        "mean_validated_dice": _round4(mean_dice),
        "total_tumor_volume_mm3": {
            **_with_ci(total_mm3, overall_unc),
            "uncertainty_pct": round(overall_unc * 100.0, 1),
        },
        "total_tumor_volume_cm3": {
            **_with_ci(total_mm3 / 1000.0, overall_unc),
            "uncertainty_pct": round(overall_unc * 100.0, 1),
        },
        "label_volumes_mm3": per_label_ci,
    }

    if max_diameter_mm > 0:
        diam_unc = max(overall_unc, 0.05)
        uncertainty["max_diameter_mm"] = {
            **_with_ci(max_diameter_mm, diam_unc),
            "uncertainty_pct": round(diam_unc * 100.0, 1),
            "diameter_method": diameter_method,
        }

    enriched["uncertainty"] = uncertainty
    return enriched
