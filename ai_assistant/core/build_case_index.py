from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any
import sys

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from ai_assistant.core.brats_feature_extractor import _configure_logging, _default_brain_root, _default_mask_root, extract_case_features


LOGGER = logging.getLogger(__name__)


def _default_index_root() -> Path:
    return _default_mask_root() / "case_index"


def _lobe_localization_phrase(location: dict) -> str:
    lobe       = location.get("lobe", "unknown")
    confidence = location.get("lobe_confidence", "low")
    method     = location.get("lobe_method", "")
    top_regions = location.get("atlas_top_regions", [])
    overlap    = location.get("lobe_overlap_fraction", 0.0)

    if confidence in ("high", "medium") and "atlas" in method and top_regions:
        top_region = top_regions[0]["region"]
        pct = round(float(overlap) * 100)
        return f"atlas-supported {lobe} localization ({top_region}, overlap {pct}%)"
    return f"approximate {lobe} localization (heuristic)"


def _case_narrative(case_features: dict[str, Any]) -> str:
    burden = case_features.get("tumor_burden", {})
    location = case_features.get("location", {})
    mass_effect = case_features.get("mass_effect", {})
    distortion = case_features.get("distortion_context", {})
    critical = case_features.get("critical_proximity", {})
    return (
        f"Case {case_features.get('case_id')}: MRI brain, {location.get('hemisphere', 'unknown')} "
        f"{_lobe_localization_phrase(location)} {location.get('depth_class', 'unknown')} tumor. "
        f"Volumes: enhancing {float(burden.get('enhancing_volume_mm3', 0.0)) / 1000.0:.2f}cm3, "
        f"edema {float(burden.get('edema_volume_mm3', 0.0)) / 1000.0:.2f}cm3, "
        f"necrotic {float(burden.get('necrotic_volume_mm3', 0.0)) / 1000.0:.2f}cm3. "
        f"Max diameter: {float(burden.get('max_diameter_mm', 0.0)):.2f}mm. "
        f"Estimated MLS: {float(mass_effect.get('midline_shift_mm', distortion.get('midline_shift_mm', 0.0)) or 0.0):.2f}mm. "
        f"Ventricular relation: {location.get('ventricular_relation', 'unknown')} "
        f"({float(location.get('ventricular_relation_mm', distortion.get('ventricle_proxy_distance_mm', 0.0)) or 0.0):.2f}mm). "
        f"Critical proximity: motor cortex {float(critical.get('motor_cortex_distance_mm') or 0.0):.2f}mm, "
        f"brainstem {float(critical.get('brainstem_distance_mm') or 0.0):.2f}mm."
    )


def build_case_index(
    brain_root: str | Path,
    mask_root: str | Path,
    output_root: str | Path,
    *,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    brain_root = Path(brain_root)
    mask_root = Path(mask_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    cases = sorted(path for path in brain_root.iterdir() if path.is_dir() and path.name.startswith("BraTS20_Validation_"))
    if limit is not None:
        cases = cases[:limit]

    combined: list[dict[str, Any]] = []
    for case_dir in cases:
        case_id = case_dir.name
        mask_path = mask_root / f"{case_id}.nii.gz"
        if not mask_path.exists():
            LOGGER.error("Skipping %s: mask missing at %s", case_id, mask_path)
            continue
        try:
            case_features = extract_case_features(case_dir, mask_path)
            case_features["narrative_text"] = _case_narrative(case_features)
            output_path = output_root / f"{case_id}.json"
            output_path.write_text(json.dumps(case_features, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
            combined.append(case_features)
            LOGGER.info("Indexed %s -> %s", case_id, output_path)
        except Exception as exc:
            LOGGER.exception("Failed to index %s: %s", case_id, exc)

    full_index_path = output_root / "full_index.json"
    full_index_path.write_text(json.dumps(combined, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    LOGGER.info("Wrote combined index with %d cases to %s", len(combined), full_index_path)
    return combined


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the BraTS case retrieval index JSON corpus.")
    parser.add_argument("--brain-root", default=str(_default_brain_root()))
    parser.add_argument("--mask-root", default=str(_default_mask_root()))
    parser.add_argument("--output-root", default=str(_default_index_root()))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    _configure_logging(args.verbose)
    build_case_index(args.brain_root, args.mask_root, args.output_root, limit=args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
