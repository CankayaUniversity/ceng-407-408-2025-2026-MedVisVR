from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path
from typing import Any

from ai_assistant.core.paths import workspace_root


DEFAULT_IMAGE_SHAPE_ZYX = (155, 240, 240)
ATLAS_ROOT = workspace_root() / "ai_assistant" / "assets" / "atlases"
MNI_TEMPLATE_PATH = ATLAS_ROOT / "MNI152_T1_1mm.nii.gz"
HO_CORTICAL_XML = ATLAS_ROOT / "fsl" / "data" / "atlases" / "HarvardOxford-Cortical-Lateralized.xml"
HO_SUBCORTICAL_XML = ATLAS_ROOT / "fsl" / "data" / "atlases" / "HarvardOxford-Subcortical.xml"
HO_CORTICAL_MAP = ATLAS_ROOT / "fsl" / "data" / "atlases" / "HarvardOxford" / "HarvardOxford-cortl-maxprob-thr25-1mm.nii.gz"
HO_SUBCORTICAL_MAP = ATLAS_ROOT / "fsl" / "data" / "atlases" / "HarvardOxford" / "HarvardOxford-sub-maxprob-thr25-1mm.nii.gz"
REGISTRATION_METHOD = "simpleitk_rigid_affine_histmatch_mattes_mi"


def _atlas_localization_enabled() -> bool:
    return os.getenv("AI_USE_ATLAS_LOCALIZATION", "1") == "1"


def _safe_norm(value: float, upper: float) -> float:
    if upper <= 1:
        return 0.5
    clipped = min(max(value, 0.0), upper - 1.0)
    return clipped / float(upper - 1.0)


def _normalized_xyz(x: float, y: float, z: float, image_shape_zyx: tuple[int, int, int]) -> tuple[float, float, float]:
    depth, height, width = image_shape_zyx
    return (
        _safe_norm(x, width),
        _safe_norm(y, height),
        _safe_norm(z, depth),
    )


def _map_hemisphere(nx: float) -> str:
    return "Left" if nx < 0.5 else "Right"


def _map_lobe(ny: float, nz: float) -> str:
    if ny <= 0.30:
        return "Frontal"
    if ny >= 0.75:
        return "Occipital"
    if nz <= 0.38:
        return "Temporal"
    return "Parietal"


def _map_depth(nx: float, ny: float) -> str:
    dx = abs(nx - 0.5)
    dy = abs(ny - 0.5)
    radial = (dx * dx + dy * dy) ** 0.5
    if radial <= 0.16:
        return "Periventricular"
    if radial <= 0.34:
        return "Deep white matter"
    return "Subcortical"


def _map_level(nz: float) -> str:
    if nz >= 0.67:
        return "Superior"
    if nz >= 0.33:
        return "Middle"
    return "Inferior"


def _parse_labels(xml_path: Path) -> dict[int, str]:
    labels: dict[int, str] = {}
    if not xml_path.exists():
        return labels
    root = ET.fromstring(xml_path.read_text(encoding="iso-8859-1"))
    for node in root.findall(".//label"):
        index = int(node.attrib.get("index", "0"))
        text = (node.text or "").strip()
        if text:
            labels[index + 1] = text
    return labels


@lru_cache(maxsize=1)
def _atlas_assets() -> dict[str, Any]:
    try:
        import nibabel as nib  
        import numpy as np 
    except Exception:
        return {}

    if not (MNI_TEMPLATE_PATH.exists() and HO_CORTICAL_MAP.exists() and HO_SUBCORTICAL_MAP.exists()):
        return {}

    cortical_img = nib.load(str(HO_CORTICAL_MAP))
    subcortical_img = nib.load(str(HO_SUBCORTICAL_MAP))
    mni_img = nib.load(str(MNI_TEMPLATE_PATH))
    cortical_labels = _parse_labels(HO_CORTICAL_XML)
    subcortical_labels = _parse_labels(HO_SUBCORTICAL_XML)
    return {
        "np": np,
        "mni_shape": tuple(int(v) for v in mni_img.shape[:3]),
        "mni_affine": mni_img.affine,
        "mni_affine_inv": np.linalg.inv(mni_img.affine),
        "cortical_data": np.asarray(cortical_img.dataobj),
        "cortical_affine_inv": np.linalg.inv(cortical_img.affine),
        "cortical_labels": cortical_labels,
        "subcortical_data": np.asarray(subcortical_img.dataobj),
        "subcortical_affine_inv": np.linalg.inv(subcortical_img.affine),
        "subcortical_labels": subcortical_labels,
    }


def _hemisphere_from_label(region_name: str, fallback_mni_x: float | None = None) -> str:
    low = region_name.lower().strip()
    if low.startswith("left "):
        return "Left"
    if low.startswith("right "):
        return "Right"
    if fallback_mni_x is None:
        return "Unknown"
    return "Left" if fallback_mni_x < 0 else "Right"


def _lobe_from_region(region_name: str, fallback_xyz: tuple[float, float, float] | None = None) -> str:
    low = region_name.lower()
    rules = [
        ("frontal", "Frontal"),
        ("precentral", "Frontal"),
        ("orbital", "Frontal"),
        ("paracingulate", "Frontal"),
        ("subcallosal", "Frontal"),
        ("temporal", "Temporal"),
        ("hippocamp", "Temporal"),
        ("amygdala", "Temporal"),
        ("parahippocampal", "Temporal"),
        ("heschl", "Temporal"),
        ("planum", "Temporal"),
        ("fusiform", "Temporal"),
        ("pariet", "Parietal"),
        ("postcentral", "Parietal"),
        ("supramarginal", "Parietal"),
        ("angular", "Parietal"),
        ("precune", "Parietal"),
        ("occip", "Occipital"),
        ("cune", "Occipital"),
        ("calcar", "Occipital"),
        ("lingual", "Occipital"),
    ]
    for key, label in rules:
        if key in low:
            return label
    if fallback_xyz is None:
        return "Unknown"
    _, ny, nz = fallback_xyz
    return _map_lobe(ny, nz)


def _depth_from_region(region_name: str, fallback_xyz: tuple[float, float, float] | None = None) -> str:
    low = region_name.lower()
    if any(term in low for term in ["ventricle", "ventrical"]):
        return "Periventricular"
    if any(term in low for term in [
        "white matter",
        "thalamus",
        "caudate",
        "putamen",
        "pallidum",
        "accumbens",
        "amygdala",
        "hippocampus",
        "brain-stem",
    ]):
        return "Deep white matter"
    if low and low != "background":
        return "Subcortical"
    if fallback_xyz is None:
        return "Unknown"
    nx, ny, _ = fallback_xyz
    return _map_depth(nx, ny)


def _level_from_mni_index(index_xyz: tuple[float, float, float], mni_shape: tuple[int, int, int]) -> str:
    nx, ny, nz = _normalized_xyz(index_xyz[0], index_xyz[1], index_xyz[2], (mni_shape[2], mni_shape[1], mni_shape[0]))
    return _map_level(nz)


def _sample_label(data: Any, affine_inv: Any, point_xyz_mm: tuple[float, float, float]) -> int:
    np = _atlas_assets().get("np")
    if np is None:
        return 0
    idx = np.dot(affine_inv, [point_xyz_mm[0], point_xyz_mm[1], point_xyz_mm[2], 1.0])[:3]
    xyz = tuple(int(round(v)) for v in idx.tolist())
    if min(xyz) < 0 or xyz[0] >= data.shape[0] or xyz[1] >= data.shape[1] or xyz[2] >= data.shape[2]:
        return 0
    return int(data[xyz])


def _sample_labels_for_points(points_xyz_mm: list[tuple[float, float, float]]) -> list[dict[str, str]]:
    assets = _atlas_assets()
    if not assets or not points_xyz_mm:
        return []

    results: list[dict[str, str]] = []
    for point in points_xyz_mm:
        cortical_value = _sample_label(assets["cortical_data"], assets["cortical_affine_inv"], point)
        subcortical_value = _sample_label(assets["subcortical_data"], assets["subcortical_affine_inv"], point)
        cortical_name = assets["cortical_labels"].get(cortical_value, "")
        subcortical_name = assets["subcortical_labels"].get(subcortical_value, "")
        if cortical_name:
            results.append({"region": cortical_name, "source": "cortical"})
        elif subcortical_name:
            results.append({"region": subcortical_name, "source": "subcortical"})
        else:
            results.append({"region": "Unlabeled standard-space region", "source": "background"})
    return results


def _extract_transform(registration_result: Any) -> Any | None:
    if isinstance(registration_result, dict):
        return registration_result.get("transform")
    return registration_result


def _registration_summary(registration_result: Any) -> dict[str, Any]:
    if isinstance(registration_result, dict):
        summary = registration_result.get("summary")
        if isinstance(summary, dict):
            return summary
    return {}


def _confidence_rank(label: str) -> int:
    ranks = {"low": 0, "moderate": 1, "high": 2}
    return ranks.get(str(label).lower(), 0)


def _combine_confidence(primary: str, secondary: str | None) -> str:
    if not secondary:
        return primary
    inv = {0: "low", 1: "moderate", 2: "high"}
    return inv[min(_confidence_rank(primary), _confidence_rank(secondary))]


def _region_matches_hemisphere(region_name: str, hemisphere: str) -> bool:
    if not region_name or not hemisphere or hemisphere == "Unknown":
        return False
    return _hemisphere_from_label(region_name) == hemisphere


def _region_priority(
    *,
    region: str,
    source: str,
    fraction: float,
    centroid_region: str,
    centroid_lobe: str,
    centroid_hemisphere: str,
) -> float:
    score = float(fraction)
    if source == "background":
        score -= 0.40
    if region and centroid_region and region == centroid_region:
        score += 0.20
    if region and centroid_lobe and _lobe_from_region(region) == centroid_lobe:
        score += 0.16
    if region and centroid_hemisphere and _region_matches_hemisphere(region, centroid_hemisphere):
        score += 0.08
    if "white matter" in region.lower():
        score -= 0.04
    return score


def lookup_mni_point(mni_xyz_mm: tuple[float, float, float]) -> dict[str, Any]:
    assets = _atlas_assets()
    if not assets:
        return {}

    cortical_value = _sample_label(assets["cortical_data"], assets["cortical_affine_inv"], mni_xyz_mm)
    subcortical_value = _sample_label(assets["subcortical_data"], assets["subcortical_affine_inv"], mni_xyz_mm)
    region_name = assets["cortical_labels"].get(cortical_value, "")
    region_source = "cortical"
    if not region_name:
        region_name = assets["subcortical_labels"].get(subcortical_value, "")
        region_source = "subcortical" if region_name else "background"

    mni_index = tuple(
        float(v)
        for v in assets["np"].dot(assets["mni_affine_inv"], [mni_xyz_mm[0], mni_xyz_mm[1], mni_xyz_mm[2], 1.0])[:3].tolist()
    )
    nx, ny, nz = _normalized_xyz(mni_index[0], mni_index[1], mni_index[2], (assets["mni_shape"][2], assets["mni_shape"][1], assets["mni_shape"][0]))
    hemisphere = _hemisphere_from_label(region_name, fallback_mni_x=mni_xyz_mm[0])
    lobe = _lobe_from_region(region_name, fallback_xyz=(nx, ny, nz))
    depth = _depth_from_region(region_name, fallback_xyz=(nx, ny, nz))
    level = _level_from_mni_index(mni_index, assets["mni_shape"])
    if region_name:
        description = f"{hemisphere} {lobe} lobe ({region_name}), {depth.lower()}, {level.lower()} level"
    else:
        description = f"{hemisphere} {lobe} lobe, {depth.lower()}, {level.lower()} level"

    return {
        "hemisphere": hemisphere,
        "lobe": lobe,
        "depth": depth,
        "level": level,
        "description": description,
        "atlas_region": region_name or "Unlabeled standard-space region",
        "atlas_source": region_source,
        "mni_xyz_mm": [round(float(v), 3) for v in mni_xyz_mm],
        "mni_xyz_index": [round(float(v), 3) for v in mni_index],
        "normalized_xyz": [round(nx, 4), round(ny, 4), round(nz, 4)],
        "mapping_basis": "harvard_oxford_mni152" if region_name else "mni152_fallback",
    }


def _downsample_coords(coords_zyx: Any, max_voxels: int) -> list[tuple[int, int, int]]:
    coords_list = [tuple(int(v) for v in row) for row in coords_zyx.tolist()]
    if len(coords_list) <= max_voxels:
        return coords_list
    step = max(1, len(coords_list) // max_voxels)
    sampled = coords_list[::step]
    return sampled[:max_voxels]


def _coords_to_physical_points(source_image: Any, coords_zyx: list[tuple[int, int, int]]) -> list[tuple[float, float, float]]:
    if not coords_zyx:
        return []
    origin = source_image.GetOrigin()
    spacing = source_image.GetSpacing()
    direction = source_image.GetDirection()
    matrix = [
        [direction[0], direction[1], direction[2]],
        [direction[3], direction[4], direction[5]],
        [direction[6], direction[7], direction[8]],
    ]
    points: list[tuple[float, float, float]] = []
    for z, y, x in coords_zyx:
        scaled = [x * spacing[0], y * spacing[1], z * spacing[2]]
        physical = [
            origin[row] + sum(matrix[row][col] * scaled[col] for col in range(3))
            for row in range(3)
        ]
        points.append((float(physical[0]), float(physical[1]), float(physical[2])))
    return points


def _atlas_overlap_summary(
    *,
    source_image: Any,
    transform: Any,
    registration_summary: dict[str, Any] | None,
    lesion_coords_zyx: Any,
    centroid_xyz_voxel: list[float] | None,
    max_voxels: int = 4096,
) -> dict[str, Any]:
    sampled_coords = _downsample_coords(lesion_coords_zyx, max_voxels=max_voxels)
    if not sampled_coords:
        return {}

    physical_points = _coords_to_physical_points(source_image, sampled_coords)
    inverse_transform = transform.GetInverse()
    mni_points = [tuple(float(v) for v in inverse_transform.TransformPoint(point)) for point in physical_points]
    label_hits = _sample_labels_for_points(mni_points)
    if not label_hits:
        return {}

    counts: dict[tuple[str, str], int] = {}
    unlabeled = 0
    for hit in label_hits:
        key = (hit["region"], hit["source"])
        counts[key] = counts.get(key, 0) + 1
        if hit["source"] == "background":
            unlabeled += 1

    ranked = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    total = len(label_hits)
    unlabeled_fraction = unlabeled / total if total else 1.0

    centroid_point = None
    if centroid_xyz_voxel and len(centroid_xyz_voxel) == 3:
        centroid_point = source_image.TransformContinuousIndexToPhysicalPoint(tuple(float(v) for v in centroid_xyz_voxel))
    else:
        mid_idx = sampled_coords[len(sampled_coords) // 2]
        centroid_point = source_image.TransformIndexToPhysicalPoint((int(mid_idx[2]), int(mid_idx[1]), int(mid_idx[0])))
    centroid_mni = tuple(float(v) for v in inverse_transform.TransformPoint(centroid_point))
    centroid_lookup = lookup_mni_point(centroid_mni)
    centroid_region = str(centroid_lookup.get("atlas_region") or "")
    centroid_source = str(centroid_lookup.get("atlas_source") or "")
    centroid_hemisphere = str(centroid_lookup.get("hemisphere") or "Unknown")
    centroid_lobe = str(centroid_lookup.get("lobe") or "Unknown")

    ranked_details = []
    for (region, source), count in ranked:
        fraction = count / total if total else 0.0
        ranked_details.append(
            {
                "region": region,
                "source": source,
                "count": count,
                "fraction": fraction,
                "priority": _region_priority(
                    region=region,
                    source=source,
                    fraction=fraction,
                    centroid_region=centroid_region,
                    centroid_lobe=centroid_lobe,
                    centroid_hemisphere=centroid_hemisphere,
                ),
            }
        )

    dominant = max(ranked_details, key=lambda item: item["priority"]) if ranked_details else None
    dominant_region = str((dominant or {}).get("region") or "")
    dominant_source = str((dominant or {}).get("source") or "background")
    dominant_count = int((dominant or {}).get("count") or 0)
    support_fraction = dominant_count / total if total else 0.0

    hemisphere = _hemisphere_from_label(dominant_region, fallback_mni_x=centroid_mni[0])
    lobe = _lobe_from_region(dominant_region, fallback_xyz=tuple(float(v) for v in centroid_lookup.get("normalized_xyz", [0.5, 0.5, 0.5])))
    depth = _depth_from_region(dominant_region, fallback_xyz=tuple(float(v) for v in centroid_lookup.get("normalized_xyz", [0.5, 0.5, 0.5])))
    level = centroid_lookup.get("level", "Middle")

    preferred_top_regions = [
        item for item in ranked_details
        if item["source"] != "background"
        and item["fraction"] >= 0.01
        and (
            _region_matches_hemisphere(item["region"], centroid_hemisphere)
            or _lobe_from_region(item["region"]) == centroid_lobe
            or item["region"] == centroid_region
        )
    ]
    if not preferred_top_regions:
        preferred_top_regions = [
            item for item in ranked_details
            if item["source"] != "background" and item["fraction"] >= 0.01
        ]
    top_regions = [
        {
            "region": str(item["region"]),
            "source": str(item["source"]),
            "fraction": round(float(item["fraction"]), 4),
        }
        for item in sorted(preferred_top_regions, key=lambda item: item["priority"], reverse=True)[:5]
    ]

    base_confidence = "high"
    if dominant_source == "background" or support_fraction < 0.20:
        base_confidence = "low"
    elif support_fraction < 0.45 or unlabeled_fraction > 0.35:
        base_confidence = "moderate"
    if centroid_lobe != "Unknown" and lobe != centroid_lobe:
        base_confidence = _combine_confidence(base_confidence, "moderate")
    if centroid_source and dominant_source != centroid_source and support_fraction < 0.35:
        base_confidence = _combine_confidence(base_confidence, "moderate")

    registration_flag = str((registration_summary or {}).get("quality_flag") or "").lower() or None
    confidence = _combine_confidence(base_confidence, registration_flag)

    if confidence == "low":
        description = str(centroid_lookup.get("description") or f"{hemisphere} {lobe} lobe, {depth.lower()}, {str(level).lower()} level")
        mapping_basis = "mni152_fallback"
    elif confidence == "moderate":
        anchor_lobe = centroid_lobe if centroid_lobe != "Unknown" else lobe
        description = f"{centroid_hemisphere} {anchor_lobe} lobe near {dominant_region}, {depth.lower()}, {str(level).lower()} level"
        mapping_basis = "harvard_oxford_mni152_overlap"
    else:
        description = f"{hemisphere} {lobe} lobe ({dominant_region}), {depth.lower()}, {str(level).lower()} level"
        mapping_basis = "harvard_oxford_mni152_overlap"

    result = dict(centroid_lookup)
    result.update(
        {
            "hemisphere": hemisphere,
            "lobe": lobe,
            "depth": depth,
            "level": level,
            "description": description,
            "atlas_region": dominant_region,
            "atlas_source": dominant_source,
            "atlas_centroid_region": centroid_region,
            "atlas_centroid_source": centroid_source,
            "mapping_basis": mapping_basis,
            "atlas_top_regions": top_regions,
            "atlas_support_fraction": round(support_fraction, 4),
            "atlas_unlabeled_fraction": round(unlabeled_fraction, 4),
            "atlas_confidence": confidence,
            "atlas_sampled_voxels": total,
            "registration_method": REGISTRATION_METHOD,
            "registration_quality_flag": str((registration_summary or {}).get("quality_flag") or "unknown"),
            "registration_quality_score": round(float((registration_summary or {}).get("quality_score", 0.0) or 0.0), 4),
            "registration_ncc": round(float((registration_summary or {}).get("normalized_cross_correlation", 0.0) or 0.0), 4),
            "registration_foreground_dice": round(float((registration_summary or {}).get("foreground_dice", 0.0) or 0.0), 4),
            "registration_center_distance_mm": round(float((registration_summary or {}).get("center_distance_mm", 0.0) or 0.0), 3),
            "source_physical_xyz_mm": [round(float(v), 3) for v in centroid_point],
        }
    )
    return result


def map_centroid_to_anatomy(
    x: float,
    y: float,
    z: float,
    image_shape_zyx: tuple[int, int, int] | None = None,
) -> dict[str, Any]:
    shape = image_shape_zyx or DEFAULT_IMAGE_SHAPE_ZYX
    nx, ny, nz = _normalized_xyz(x, y, z, shape)
    hemisphere = _map_hemisphere(nx)
    lobe = _map_lobe(ny, nz)
    depth = _map_depth(nx, ny)
    level = _map_level(nz)
    return {
        "hemisphere": hemisphere,
        "lobe": lobe,
        "depth": depth,
        "level": level,
        "description": f"{hemisphere} {lobe} lobe, {depth.lower()}, {level.lower()} level",
        "normalized_xyz": [round(nx, 4), round(ny, 4), round(nz, 4)],
        "mapping_basis": "heuristic_voxel_space",
    }


def _prepare_registration_image(image: Any, *, reference: Any | None = None) -> Any:
    import SimpleITK as sitk  

    out = sitk.Cast(image, sitk.sitkFloat32)
    out = sitk.Normalize(out)
    out = sitk.DiscreteGaussian(out, variance=1.0)
    if reference is not None:
        matcher = sitk.HistogramMatchingImageFilter()
        matcher.SetNumberOfHistogramLevels(128)
        matcher.SetNumberOfMatchPoints(12)
        matcher.ThresholdAtMeanIntensityOn()
        out = matcher.Execute(out, reference)
    return out


def _foreground_mask(image: Any) -> Any:
    import SimpleITK as sitk  

    mask = sitk.OtsuThreshold(image, 0, 1, 128)
    mask = sitk.BinaryMorphologicalClosing(mask, [2, 2, 2])
    mask = sitk.BinaryFillhole(mask)
    components = sitk.ConnectedComponent(mask)
    relabeled = sitk.RelabelComponent(components, sortByObjectSize=True)
    return sitk.Cast(relabeled == 1, sitk.sitkUInt8)


def _run_registration_stage(
    *,
    fixed: Any,
    moving: Any,
    fixed_mask: Any | None,
    moving_mask: Any | None,
    initial_transform: Any,
    stage_name: str,
    learning_rate: float,
    min_step: float,
    iterations: int,
) -> tuple[Any | None, dict[str, Any]]:
    import SimpleITK as sitk  

    registration = sitk.ImageRegistrationMethod()
    registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=64)
    registration.SetMetricSamplingStrategy(registration.RANDOM)
    registration.SetMetricSamplingPercentage(0.18)
    if fixed_mask is not None:
        registration.SetMetricFixedMask(fixed_mask)
    if moving_mask is not None:
        registration.SetMetricMovingMask(moving_mask)
    registration.SetInterpolator(sitk.sitkLinear)
    registration.SetOptimizerAsRegularStepGradientDescent(
        learningRate=learning_rate,
        minStep=min_step,
        numberOfIterations=iterations,
        relaxationFactor=0.5,
        gradientMagnitudeTolerance=1e-5,
    )
    registration.SetOptimizerScalesFromPhysicalShift()
    registration.SetShrinkFactorsPerLevel([8, 4, 2, 1])
    registration.SetSmoothingSigmasPerLevel([3, 2, 1, 0])
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    registration.SetInitialTransform(initial_transform, inPlace=False)

    try:
        transform = registration.Execute(fixed, moving)
    except Exception as exc:
        return None, {"stage": stage_name, "status": "failed", "error": str(exc)}

    return transform, {
        "stage": stage_name,
        "status": "ok",
        "metric_value": round(float(registration.GetMetricValue()), 6),
        "optimizer_stop_condition": registration.GetOptimizerStopConditionDescription(),
        "iterations": int(registration.GetOptimizerIteration()),
    }


def _affine_from_transform(transform: Any) -> Any:
    import SimpleITK as sitk  

    affine = sitk.AffineTransform(3)
    try:
        affine.SetCenter(transform.GetCenter())
    except Exception:
        pass
    try:
        affine.SetMatrix(transform.GetMatrix())
    except Exception:
        pass
    try:
        affine.SetTranslation(transform.GetTranslation())
    except Exception:
        pass
    return affine


def _compute_registration_quality(
    *,
    fixed: Any,
    moving: Any,
    fixed_mask: Any,
    moving_mask: Any,
    transform: Any,
) -> dict[str, Any]:
    import numpy as np 
    import SimpleITK as sitk  

    fixed_small = sitk.Shrink(fixed, [4, 4, 4])
    moving_resampled = sitk.Resample(moving, fixed_small, transform, sitk.sitkLinear, 0.0, sitk.sitkFloat32)
    fixed_mask_small = sitk.Cast(sitk.Shrink(fixed_mask, [4, 4, 4]) > 0, sitk.sitkUInt8)
    moving_mask_resampled = sitk.Cast(
        sitk.Resample(moving_mask, fixed_small, transform, sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8) > 0,
        sitk.sitkUInt8,
    )

    fixed_arr = sitk.GetArrayViewFromImage(fixed_small)
    moving_arr = sitk.GetArrayViewFromImage(moving_resampled)
    fixed_mask_arr = sitk.GetArrayViewFromImage(fixed_mask_small) > 0
    moving_mask_arr = sitk.GetArrayViewFromImage(moving_mask_resampled) > 0
    union_mask = fixed_mask_arr | moving_mask_arr
    overlap_mask = fixed_mask_arr & moving_mask_arr

    ncc = 0.0
    if int(union_mask.sum()) > 32:
        f = fixed_arr[union_mask].astype(float)
        m = moving_arr[union_mask].astype(float)
        f = f - f.mean()
        m = m - m.mean()
        denom = float(np.linalg.norm(f) * np.linalg.norm(m))
        if denom > 1e-8:
            ncc = float(np.dot(f, m) / denom)
    intersection = int(overlap_mask.sum())
    denom = int(fixed_mask_arr.sum()) + int(moving_mask_arr.sum())
    dice = (2.0 * intersection / denom) if denom else 0.0

    fixed_center_idx = tuple((size - 1) / 2.0 for size in fixed.GetSize())
    moving_center_idx = tuple((size - 1) / 2.0 for size in moving.GetSize())
    fixed_center_mm = fixed.TransformContinuousIndexToPhysicalPoint(fixed_center_idx)
    moving_center_mm = moving.TransformContinuousIndexToPhysicalPoint(moving_center_idx)
    mapped_center_mm = transform.GetInverse().TransformPoint(moving_center_mm)
    center_distance_mm = sum((float(a) - float(b)) ** 2 for a, b in zip(mapped_center_mm, fixed_center_mm)) ** 0.5

    center_score = max(0.0, 1.0 - min(center_distance_mm / 35.0, 1.0))
    quality_score = max(0.0, min(1.0, (0.45 * max(0.0, ncc)) + (0.35 * dice) + (0.20 * center_score)))
    if quality_score >= 0.72 and dice >= 0.72 and ncc >= 0.45:
        quality_flag = "high"
    elif quality_score >= 0.45 and dice >= 0.45:
        quality_flag = "moderate"
    else:
        quality_flag = "low"

    return {
        "quality_flag": quality_flag,
        "quality_score": round(quality_score, 4),
        "normalized_cross_correlation": round(float(ncc), 4),
        "foreground_dice": round(float(dice), 4),
        "center_distance_mm": round(float(center_distance_mm), 3),
    }


def _register_reference_to_mni(reference_image_path: str) -> Any | None:
    try:
        import SimpleITK as sitk  
    except Exception:
        return None
    if not MNI_TEMPLATE_PATH.exists() or not Path(reference_image_path).exists():
        return None

    fixed_raw = sitk.ReadImage(str(MNI_TEMPLATE_PATH), sitk.sitkFloat32)
    moving_raw = sitk.ReadImage(str(reference_image_path), sitk.sitkFloat32)
    fixed = _prepare_registration_image(fixed_raw)
    moving = _prepare_registration_image(moving_raw, reference=fixed)
    fixed_mask = _foreground_mask(fixed)
    moving_mask = _foreground_mask(moving)

    rigid_initial = sitk.CenteredTransformInitializer(
        fixed,
        moving,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )
    rigid_transform, rigid_stage = _run_registration_stage(
        fixed=fixed,
        moving=moving,
        fixed_mask=fixed_mask,
        moving_mask=moving_mask,
        initial_transform=rigid_initial,
        stage_name="rigid",
        learning_rate=1.5,
        min_step=0.001,
        iterations=150,
    )
    if rigid_transform is None:
        return None
    rigid_quality = _compute_registration_quality(
        fixed=fixed,
        moving=moving,
        fixed_mask=fixed_mask,
        moving_mask=moving_mask,
        transform=rigid_transform,
    )

    affine_initial = _affine_from_transform(rigid_transform)
    affine_transform, affine_stage = _run_registration_stage(
        fixed=fixed,
        moving=moving,
        fixed_mask=None,
        moving_mask=None,
        initial_transform=affine_initial,
        stage_name="affine",
        learning_rate=0.75,
        min_step=0.0005,
        iterations=120,
    )
    selected_transform = rigid_transform
    selected_stage = "rigid"
    quality = rigid_quality
    if affine_transform is not None:
        affine_quality = _compute_registration_quality(
            fixed=fixed,
            moving=moving,
            fixed_mask=fixed_mask,
            moving_mask=moving_mask,
            transform=affine_transform,
        )
        if float(affine_quality.get("quality_score", 0.0) or 0.0) >= float(rigid_quality.get("quality_score", 0.0) or 0.0) - 0.02:
            selected_transform = affine_transform
            selected_stage = "affine"
            quality = affine_quality

    summary = {
        "method": REGISTRATION_METHOD,
        "stages": [rigid_stage, affine_stage],
        "selected_stage": selected_stage,
        **quality,
    }
    return {"transform": selected_transform, "summary": summary}


def register_reference_to_mni(reference_image_path: str) -> Any | None:
    return _register_reference_to_mni(reference_image_path)


def anatomy_from_lesion(
    lesion: dict[str, Any],
    image_shape_zyx: tuple[int, int, int] | None = None,
    *,
    reference_image_path: str | None = None,
    source_image: Any | None = None,
    standard_space_transform: Any | None = None,
    lesion_coords_zyx: Any | None = None,
) -> dict[str, Any]:
    centroid_xyz = lesion.get("centroid_xyz_voxel")
    centroid_zyx = lesion.get("centroid_zyx_voxel")
    heuristic = None

    if isinstance(centroid_xyz, list) and len(centroid_xyz) == 3:
        x, y, z = (float(v or 0.0) for v in centroid_xyz)
        heuristic = map_centroid_to_anatomy(x=x, y=y, z=z, image_shape_zyx=image_shape_zyx)
    elif isinstance(centroid_zyx, list) and len(centroid_zyx) == 3:
        z, y, x = (float(v or 0.0) for v in centroid_zyx)
        heuristic = map_centroid_to_anatomy(x=x, y=y, z=z, image_shape_zyx=image_shape_zyx)

    if not _atlas_localization_enabled():
        return heuristic or {
            "hemisphere": "Unknown",
            "lobe": "Unknown",
            "depth": "Unknown",
            "level": "Unknown",
            "description": "Anatomic localization unavailable from current segmentation payload",
            "normalized_xyz": [],
            "mapping_basis": "unavailable",
        }

    registration_result = standard_space_transform or (_register_reference_to_mni(reference_image_path) if reference_image_path else None)
    transform = _extract_transform(registration_result)
    registration_summary = _registration_summary(registration_result)
    if transform is not None and source_image is not None and lesion_coords_zyx is not None:
        try:
            atlas_match = _atlas_overlap_summary(
                source_image=source_image,
                transform=transform,
                registration_summary=registration_summary,
                lesion_coords_zyx=lesion_coords_zyx,
                centroid_xyz_voxel=centroid_xyz if isinstance(centroid_xyz, list) else None,
            )
            if atlas_match:
                return atlas_match
        except Exception:
            pass
    elif transform is not None and source_image is not None and isinstance(centroid_xyz, list) and len(centroid_xyz) == 3:
        try:
            physical_point = source_image.TransformContinuousIndexToPhysicalPoint(tuple(float(v) for v in centroid_xyz))
            inverse_transform = transform.GetInverse()
            mni_point = inverse_transform.TransformPoint(physical_point)
            atlas_match = lookup_mni_point(tuple(float(v) for v in mni_point))
            if atlas_match:
                atlas_match["source_physical_xyz_mm"] = [round(float(v), 3) for v in physical_point]
                atlas_match["registration_method"] = REGISTRATION_METHOD
                atlas_match["registration_quality_flag"] = str(registration_summary.get("quality_flag") or "unknown")
                atlas_match["registration_quality_score"] = round(float(registration_summary.get("quality_score", 0.0) or 0.0), 4)
                return atlas_match
        except Exception:
            pass

    return heuristic or {
        "hemisphere": "Unknown",
        "lobe": "Unknown",
        "depth": "Unknown",
        "level": "Unknown",
        "description": "Anatomic localization unavailable from current segmentation payload",
        "normalized_xyz": [],
        "mapping_basis": "unavailable",
    }
