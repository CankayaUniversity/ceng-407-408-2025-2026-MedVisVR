from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any
import sys

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

import nibabel as nib
import numpy as np
from scipy import ndimage

from ai_assistant.core.paths import workspace_root


LOGGER = logging.getLogger(__name__)
LABEL_MAP = {1: "necrotic_core", 2: "edema", 4: "enhancing_tumor"}

_HO_REGION_TO_LOBE: dict[str, str] = {
    "Frontal Pole": "frontal",
    "Superior Frontal Gyrus": "frontal",
    "Middle Frontal Gyrus": "frontal",
    "Inferior Frontal Gyrus, pars triangularis": "frontal",
    "Inferior Frontal Gyrus, pars opercularis": "frontal",
    "Precentral Gyrus": "frontal",
    "Frontal Medial Cortex": "frontal",
    "Frontal Orbital Cortex": "frontal",
    "Frontal Operculum Cortex": "frontal",
    "Paracingulate Gyrus": "frontal",
    "Juxtapositional Lobule Cortex (formerly Supplementary Motor Cortex)": "frontal",
    "Cingulate Gyrus, anterior division": "frontal",
    "Postcentral Gyrus": "parietal",
    "Superior Parietal Lobule": "parietal",
    "Supramarginal Gyrus, anterior division": "parietal",
    "Supramarginal Gyrus, posterior division": "parietal",
    "Angular Gyrus": "parietal",
    "Parietal Operculum Cortex": "parietal",
    "Cingulate Gyrus, posterior division": "parietal",
    "Temporal Pole": "temporal",
    "Superior Temporal Gyrus, anterior division": "temporal",
    "Superior Temporal Gyrus, posterior division": "temporal",
    "Middle Temporal Gyrus, anterior division": "temporal",
    "Middle Temporal Gyrus, posterior division": "temporal",
    "Middle Temporal Gyrus, temporooccipital part": "temporal",
    "Inferior Temporal Gyrus, anterior division": "temporal",
    "Inferior Temporal Gyrus, posterior division": "temporal",
    "Inferior Temporal Gyrus, temporooccipital part": "temporal",
    "Temporal Fusiform Cortex, anterior division": "temporal",
    "Temporal Fusiform Cortex, posterior division": "temporal",
    "Temporal Occipital Fusiform Cortex": "temporal",
    "Parahippocampal Gyrus, anterior division": "temporal",
    "Parahippocampal Gyrus, posterior division": "temporal",
    "Heschl's Gyrus (includes H1 and H2)": "temporal",
    "Planum Temporale": "temporal",
    "Planum Polare": "temporal",
    "Lateral Occipital Cortex, superior division": "occipital",
    "Lateral Occipital Cortex, inferior division": "occipital",
    "Occipital Fusiform Gyrus": "occipital",
    "Occipital Pole": "occipital",
    "Cuneal Cortex": "occipital",
    "Lingual Gyrus": "occipital",
    "Intracalcarine Cortex": "occipital",
    "Supracalcarine Cortex": "occipital",
    "Insular Cortex": "insular",
    "Central Opercular Cortex": "insular",
    "Left Thalamus": "deep/subcortical",
    "Right Thalamus": "deep/subcortical",
    "Left Caudate": "deep/subcortical",
    "Right Caudate": "deep/subcortical",
    "Left Putamen": "deep/subcortical",
    "Right Putamen": "deep/subcortical",
    "Left Pallidum": "deep/subcortical",
    "Right Pallidum": "deep/subcortical",
    "Left Hippocampus": "temporal",
    "Right Hippocampus": "temporal",
    "Left Amygdala": "temporal",
    "Right Amygdala": "temporal",
    "Left Accumbens": "deep/subcortical",
    "Right Accumbens": "deep/subcortical",
    "Brain-Stem": "deep/subcortical",
    "Cerebellum": "cerebellar",
}


def _configure_logging(verbose: bool = False) -> None:
    if logging.getLogger().handlers:
        return
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def _default_brain_root() -> Path:
    return workspace_root() / "brain_data"


def _default_mask_root() -> Path:
    return workspace_root() / "output_brain"


def _load_nifti(path: Path) -> tuple[np.ndarray, tuple[float, float, float]]:
    image = nib.load(str(path))
    data = np.asarray(image.get_fdata())
    zooms = image.header.get_zooms()[:3]
    spacing = tuple(float(v) for v in zooms)
    return data, spacing


def _voxel_volume_mm3(spacing: tuple[float, float, float]) -> float:
    return float(np.prod(np.asarray(spacing, dtype=float)))


def _brain_mask_from_t1ce(t1ce: np.ndarray) -> np.ndarray:
    positive = t1ce > 0
    if not np.any(positive):
        return positive
    thresh = float(np.percentile(t1ce[positive], 5))
    mask = t1ce > thresh
    mask = ndimage.binary_fill_holes(mask)
    mask = ndimage.binary_opening(mask, iterations=1)
    return mask.astype(bool)


def _brain_mask_from_flair(flair: np.ndarray) -> np.ndarray:
    positive = flair > 0
    if not np.any(positive):
        return positive
    thresh = float(np.percentile(flair[positive], 5))
    mask = flair > thresh
    mask = ndimage.binary_fill_holes(mask)
    mask = ndimage.binary_opening(mask, iterations=1)
    return mask.astype(bool)


def _brain_mask_from_t1(t1: np.ndarray) -> np.ndarray:
    return np.asarray(t1 > 0, dtype=bool)


def _centroid(mask: np.ndarray) -> np.ndarray:
    if not np.any(mask):
        return np.array(mask.shape, dtype=float) / 2.0
    return np.asarray(ndimage.center_of_mass(mask), dtype=float)


def _clip_index(point: np.ndarray, shape: tuple[int, int, int]) -> tuple[int, int, int]:
    rounded = np.rint(point).astype(int)
    clipped = np.clip(rounded, 0, np.asarray(shape) - 1)
    return tuple(int(v) for v in clipped)


def _largest_component_diameter_mm(mask: np.ndarray, spacing: tuple[float, float, float]) -> float:
    if not np.any(mask):
        return 0.0
    labeled, count = ndimage.label(mask.astype(np.uint8))
    max_diameter = 0.0
    for component_id in range(1, count + 1):
        coords = np.argwhere(labeled == component_id)
        if coords.size == 0:
            continue
        mins = coords.min(axis=0)
        maxs = coords.max(axis=0)
        extents_mm = (maxs - mins + 1) * np.asarray(spacing)
        diameter = float(np.linalg.norm(extents_mm))
        if diameter > max_diameter:
            max_diameter = diameter
    return max_diameter


def _distance_mm_to_region(point_index: tuple[int, int, int], region_mask: np.ndarray, spacing: tuple[float, float, float]) -> float | None:
    if not np.any(region_mask):
        return None
    distance_map = ndimage.distance_transform_edt(~region_mask.astype(bool), sampling=spacing)
    return float(distance_map[point_index])


def _min_mask_distance_mm(source_mask: np.ndarray, target_mask: np.ndarray, spacing: tuple[float, float, float]) -> float | None:
    if not np.any(source_mask) or not np.any(target_mask):
        return None
    distance_map = ndimage.distance_transform_edt(~target_mask.astype(bool), sampling=spacing)
    return float(distance_map[source_mask.astype(bool)].min())


def _brain_bbox(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coords = np.argwhere(mask)
    if coords.size == 0:
        shape = np.asarray(mask.shape)
        return np.zeros(3, dtype=int), shape - 1
    return coords.min(axis=0), coords.max(axis=0)


def _estimate_hemisphere(centroid_xyz: np.ndarray, shape: tuple[int, int, int]) -> str:
    return "left" if centroid_xyz[0] < (shape[0] / 2.0) else "right"


def _normalized_centroid(centroid_xyz: np.ndarray, brain_mask: np.ndarray) -> np.ndarray:
    mins, maxs = _brain_bbox(brain_mask)
    spans = np.maximum(maxs - mins, 1)
    return np.asarray((centroid_xyz - mins) / spans, dtype=float)


def _estimate_lobe(centroid_xyz: np.ndarray, brain_mask: np.ndarray) -> tuple[str, float, str]:
    norm = _normalized_centroid(centroid_xyz, brain_mask)
    x_frac = float(norm[0])
    y_frac = float(norm[1])
    z_frac = float(norm[2])

    if z_frac < 0.25 and 0.35 <= x_frac <= 0.65:
        return "cerebellar", 0.38, "normalized_centroid_heuristic"
    if z_frac > 0.65:
        return "frontal", 0.46, "normalized_centroid_heuristic"
    if 0.35 <= z_frac <= 0.65 and y_frac > 0.5:
        return "parietal", 0.41, "normalized_centroid_heuristic"
    if 0.35 <= z_frac <= 0.65 and y_frac < 0.5:
        return "occipital", 0.38, "normalized_centroid_heuristic"
    if z_frac < 0.35 and y_frac < 0.6:
        return "temporal", 0.42, "normalized_centroid_heuristic"
    return "parietal", 0.25, "normalized_centroid_heuristic_fallback"


def _atlas_lobe_overlap(union_mask: np.ndarray, brain_mask: np.ndarray) -> dict:
    try:
        from nilearn import datasets as _nl_datasets

        tumor_voxels = int(np.count_nonzero(union_mask))
        if tumor_voxels == 0 or not np.any(brain_mask):
            return {}

        mins, maxs = _brain_bbox(brain_mask)
        spans = np.maximum(maxs - mins, 1).astype(float)

        _A_MIN = np.array([9,   0,   4], dtype=float)
        _A_MAX = np.array([173, 217, 177], dtype=float)
        _A_SHP = (182, 218, 182)
        _a_span = _A_MAX - _A_MIN

        cort = _nl_datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-1mm", symmetric_split=False)
        sub  = _nl_datasets.fetch_atlas_harvard_oxford("sub-maxprob-thr25-1mm",  symmetric_split=False)
        cort_data   = np.asarray(cort.maps.get_fdata(), dtype=np.int32)
        sub_data    = np.asarray(sub.maps.get_fdata(),  dtype=np.int32)
        cort_labels = list(cort.labels)
        sub_labels  = list(sub.labels)

        tx, ty, tz = np.where(union_mask)
        nx = (tx - mins[0]) / spans[0]
        ny = (ty - mins[1]) / spans[1]
        nz = (tz - mins[2]) / spans[2]

        ax = np.round(_A_MAX[0] - nx * _a_span[0]).astype(int)
        ay = np.round(_A_MAX[1] - ny * _a_span[1]).astype(int)
        az = np.round(_A_MIN[2] + nz * _a_span[2]).astype(int)

        valid = (
            (ax >= 0) & (ax < _A_SHP[0]) &
            (ay >= 0) & (ay < _A_SHP[1]) &
            (az >= 0) & (az < _A_SHP[2])
        )
        if not np.any(valid):
            return {}

        axv, ayv, azv = ax[valid], ay[valid], az[valid]

        _SKIP_LABELS = {
            "Left Cerebral Cortex", "Right Cerebral Cortex",
            "Left Cerebral White Matter", "Right Cerebral White Matter",
            "Left Lateral Ventricle", "Right Lateral Ventricle",
            "3rd Ventricle", "4th Ventricle", "Background",
        }

        cort_regions: list[dict] = []
        sub_regions:  list[dict] = []

        cort_vals = cort_data[axv, ayv, azv]
        for region_idx in range(1, len(cort_labels)):
            lbl = cort_labels[region_idx]
            if lbl in _SKIP_LABELS:
                continue
            overlap = int(np.sum(cort_vals == region_idx))
            if overlap > 0:
                cort_regions.append({
                    "region": lbl,
                    "overlap_fraction": round(overlap / tumor_voxels, 4),
                    "atlas": "HarvardOxford-cortical",
                })

        sub_vals = sub_data[axv, ayv, azv]
        for region_idx in range(1, len(sub_labels)):
            lbl = sub_labels[region_idx]
            if lbl in _SKIP_LABELS:
                continue
            overlap = int(np.sum(sub_vals == region_idx))
            if overlap > 0:
                sub_regions.append({
                    "region": lbl,
                    "overlap_fraction": round(overlap / tumor_voxels, 4),
                    "atlas": "HarvardOxford-subcortical",
                })

        cort_regions.sort(key=lambda r: r["overlap_fraction"], reverse=True)
        sub_regions.sort(key=lambda r: r["overlap_fraction"], reverse=True)
        all_regions = cort_regions + sub_regions

        if not all_regions:
            return {}

        all_regions.sort(key=lambda r: r["overlap_fraction"], reverse=True)
        top3 = all_regions[:3]
        top_name     = top3[0]["region"]
        top_fraction = top3[0]["overlap_fraction"]

        lobe = _HO_REGION_TO_LOBE.get(top_name)
        if lobe is None:
            for key, val in _HO_REGION_TO_LOBE.items():
                if key.lower() in top_name.lower() or top_name.lower() in key.lower():
                    lobe = val
                    break
        if lobe is None:
            lobe = "unknown"

        if top_fraction >= 0.35:
            confidence = "high"
        elif top_fraction >= 0.15:
            confidence = "medium"
        else:
            confidence = "low"

        return {
            "lobe": lobe,
            "lobe_confidence": confidence,
            "lobe_method": "harvard_oxford_atlas_overlap",
            "atlas_top_regions": top3,
            "lobe_overlap_fraction": top_fraction,
        }

    except Exception as exc:
        LOGGER.warning("Harvard-Oxford atlas overlap failed: %s — falling back to heuristic", exc)
        return {}


def _depth_class(centroid_idx: tuple[int, int, int], brain_mask: np.ndarray, spacing: tuple[float, float, float]) -> tuple[str, float]:
    if not np.any(brain_mask):
        return "unknown", 0.0
    surface_distance = ndimage.distance_transform_edt(brain_mask.astype(bool), sampling=spacing)
    depth_mm = float(surface_distance[centroid_idx])
    if depth_mm < 10.0:
        return "cortical", depth_mm
    if depth_mm <= 30.0:
        return "subcortical", depth_mm
    return "deep", depth_mm


def _central_region_proxy(brain_mask: np.ndarray, fractions: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]) -> np.ndarray:
    mins, maxs = _brain_bbox(brain_mask)
    spans = np.maximum(maxs - mins, 1)
    proxy = np.zeros_like(brain_mask, dtype=bool)
    x0 = int(mins[0] + spans[0] * fractions[0][0])
    x1 = int(mins[0] + spans[0] * fractions[0][1])
    y0 = int(mins[1] + spans[1] * fractions[1][0])
    y1 = int(mins[1] + spans[1] * fractions[1][1])
    z0 = int(mins[2] + spans[2] * fractions[2][0])
    z1 = int(mins[2] + spans[2] * fractions[2][1])
    proxy[x0 : x1 + 1, y0 : y1 + 1, z0 : z1 + 1] = True
    return proxy & brain_mask


def _ventricle_proxy(t1ce: np.ndarray, brain_mask: np.ndarray, spacing: tuple[float, float, float]) -> np.ndarray:
    if not np.any(brain_mask):
        return np.zeros_like(brain_mask, dtype=bool)
    positive = t1ce[brain_mask]
    if positive.size == 0:
        return np.zeros_like(brain_mask, dtype=bool)
    low_thresh = float(np.percentile(positive, 18))
    central_mask = _central_region_proxy(brain_mask, ((0.35, 0.65), (0.25, 0.75), (0.25, 0.75)))
    deep_interior = ndimage.distance_transform_edt(brain_mask.astype(bool), sampling=spacing) >= 8.0
    proxy = (t1ce <= low_thresh) & central_mask & deep_interior
    if np.count_nonzero(proxy) < 50:
        proxy = central_mask & deep_interior
    return ndimage.binary_opening(proxy, iterations=1)


def _max_tumor_slice_index(union_mask: np.ndarray) -> int | None:
    if not np.any(union_mask):
        return None
    slice_areas = np.count_nonzero(union_mask, axis=(0, 1))
    if slice_areas.size == 0 or int(slice_areas.max()) <= 0:
        return None
    return int(np.argmax(slice_areas))


def _midline_shift_measurement(union_mask: np.ndarray, spacing: tuple[float, float, float]) -> dict[str, Any]:
    slice_index = _max_tumor_slice_index(union_mask)
    if slice_index is None:
        return {
            "midline_shift_mm": None,
            "reference_slice_index": None,
            "method": "centroid_to_midline",
            "confidence": 0.0,
        }
    slice_mask = union_mask[:, :, slice_index]
    if not np.any(slice_mask):
        return {
            "midline_shift_mm": None,
            "reference_slice_index": slice_index,
            "method": "centroid_to_midline",
            "confidence": 0.0,
        }
    centroid_xy = np.asarray(ndimage.center_of_mass(slice_mask), dtype=float)
    image_midline_x = union_mask.shape[0] / 2.0
    shift_mm = abs(float(centroid_xy[0] - image_midline_x)) * float(spacing[0])
    return {
        "midline_shift_mm": round(shift_mm, 4),
        "reference_slice_index": slice_index,
        "method": "centroid_to_midline",
        "confidence": 0.62,
    }


def _ventricular_compression_measurement(
    edema_mask: np.ndarray,
    spacing: tuple[float, float, float],
    shape: tuple[int, int, int],
) -> dict[str, Any]:
    if not np.any(edema_mask):
        return {
            "ventricular_compression": None,
            "ventricle_proximity_mm": None,
            "method": "edema_to_centerline_proxy",
            "confidence": 0.0,
        }
    x_coords = np.argwhere(edema_mask)[:, 0].astype(float)
    image_midline_x = shape[0] / 2.0
    distances_mm = np.abs(x_coords - image_midline_x) * float(spacing[0])
    proximity_mm = float(distances_mm.min()) if distances_mm.size else None
    compression = proximity_mm is not None and proximity_mm <= 10.0
    return {
        "ventricular_compression": bool(compression),
        "ventricle_proximity_mm": round(float(proximity_mm), 4) if proximity_mm is not None else None,
        "method": "edema_to_centerline_proxy",
        "confidence": 0.5,
    }


def _sulcal_effacement_measurement(
    flair: np.ndarray,
    brain_mask: np.ndarray,
    union_mask: np.ndarray,
    spacing: tuple[float, float, float],
    hemisphere: str,
) -> dict[str, Any]:
    if not np.any(brain_mask):
        return {
            "sulcal_effacement": None,
            "hemispheric_asymmetry_index": None,
            "method": "flair_peripheral_csf_proxy",
            "confidence": 0.0,
        }

    positive = flair[brain_mask]
    if positive.size == 0:
        return {
            "sulcal_effacement": None,
            "hemispheric_asymmetry_index": None,
            "method": "flair_peripheral_csf_proxy",
            "confidence": 0.0,
        }

    low_thresh = float(np.percentile(positive, 15))
    surface_distance = ndimage.distance_transform_edt(brain_mask.astype(bool), sampling=spacing)
    peripheral_band = brain_mask & (surface_distance <= 6.0)
    csf_like = peripheral_band & (flair <= low_thresh) & (~union_mask)

    x_mid = brain_mask.shape[0] // 2
    left_mask = np.zeros_like(brain_mask, dtype=bool)
    right_mask = np.zeros_like(brain_mask, dtype=bool)
    left_mask[:x_mid, :, :] = True
    right_mask[x_mid:, :, :] = True

    ipsi_mask = left_mask if hemisphere == "left" else right_mask
    contra_mask = right_mask if hemisphere == "left" else left_mask
    ipsi_count = int(np.count_nonzero(csf_like & ipsi_mask))
    contra_count = int(np.count_nonzero(csf_like & contra_mask))

    denom = max(ipsi_count, contra_count, 1)
    asymmetry = abs(ipsi_count - contra_count) / float(denom)
    return {
        "sulcal_effacement": bool(asymmetry > 0.3),
        "hemispheric_asymmetry_index": round(float(asymmetry), 4),
        "ipsilateral_csf_like_voxels": ipsi_count,
        "contralateral_csf_like_voxels": contra_count,
        "method": "flair_peripheral_csf_proxy",
        "confidence": 0.44,
    }


def _cisternal_effacement_measurement(t1: np.ndarray, brain_mask: np.ndarray) -> dict[str, Any]:
    if not np.any(brain_mask):
        return {
            "cisternal_effacement": None,
            "inferior_csf_voxel_count": None,
            "method": "inferior_t1_csf_proxy",
            "confidence": 0.0,
        }

    positive = t1[brain_mask]
    if positive.size == 0:
        return {
            "cisternal_effacement": None,
            "inferior_csf_voxel_count": None,
            "method": "inferior_t1_csf_proxy",
            "confidence": 0.0,
        }

    z_start = int(np.floor(t1.shape[2] * 0.85))
    inferior_roi = np.zeros_like(brain_mask, dtype=bool)
    inferior_roi[:, :, z_start:] = True
    inferior_roi &= brain_mask

    low_thresh = float(np.percentile(positive, 12))
    csf_like = inferior_roi & (t1 <= low_thresh)
    count = int(np.count_nonzero(csf_like))
    return {
        "cisternal_effacement": bool(count < 50),
        "inferior_csf_voxel_count": count,
        "method": "inferior_t1_csf_proxy",
        "confidence": 0.4,
    }


def _brainstem_proxy(brain_mask: np.ndarray) -> np.ndarray:
    return _central_region_proxy(brain_mask, ((0.38, 0.62), (0.35, 0.7), (0.0, 0.22)))


def _motor_cortex_proxy(brain_mask: np.ndarray, spacing: tuple[float, float, float]) -> np.ndarray:
    surface_distance = ndimage.distance_transform_edt(brain_mask.astype(bool), sampling=spacing)
    cortical_band = surface_distance <= 10.0
    central_strip = _central_region_proxy(brain_mask, ((0.15, 0.85), (0.38, 0.62), (0.5, 0.95)))
    return cortical_band & central_strip


def _bool_or_none(value: bool | None) -> bool | None:
    if value is None:
        return None
    return bool(value)


def _descriptor_from_ventricle_distance(distance_mm: float | None) -> str:
    if distance_mm is None:
        return "ventricle proxy unavailable"
    if distance_mm < 10.0:
        return "periventricular"
    if distance_mm < 25.0:
        return "near"
    return "remote"


def extract_case_features(
    case_dir: str | Path,
    mask_path: str | Path,
    t1ce_path: str | Path | None = None,
) -> dict[str, Any]:
    case_dir = Path(case_dir)
    mask_path = Path(mask_path)
    case_id = case_dir.name

    def _find_seq(seq: str) -> Path | None:
        for pat in (f"{case_id}_{seq}.nii*", f"*_{seq}.nii.gz", f"*_{seq}.nii"):
            m = sorted(case_dir.glob(pat))
            if m:
                return m[0]
        return None

    if t1ce_path is None:
        t1ce_path = _find_seq("t1ce")
        if not t1ce_path:
            raise FileNotFoundError(f"T1ce volume not found for {case_id} in {case_dir}")
    t1ce_path = Path(t1ce_path)

    flair_path = _find_seq("flair")
    if not flair_path:
        raise FileNotFoundError(f"FLAIR volume not found for {case_id} in {case_dir}")

    t1_path = _find_seq("t1")
    if not t1_path:
        raise FileNotFoundError(f"T1 volume not found for {case_id} in {case_dir}")

    mask_data, mask_spacing = _load_nifti(mask_path)
    t1ce_data, _ = _load_nifti(t1ce_path)
    flair_data, _ = _load_nifti(flair_path)
    t1_data, _ = _load_nifti(t1_path)
    mask = np.rint(mask_data).astype(np.int16)
    union_mask = mask > 0
    voxel_volume = _voxel_volume_mm3(mask_spacing)

    label_volumes_mm3: dict[str, float] = {}
    for label_value, label_name in LABEL_MAP.items():
        voxel_count = int(np.count_nonzero(mask == label_value))
        if voxel_count:
            label_volumes_mm3[label_name] = round(voxel_count * voxel_volume, 4)

    total_tumor_volume_mm3 = round(float(np.count_nonzero(union_mask) * voxel_volume), 4)
    max_diameter_mm = round(_largest_component_diameter_mm(union_mask, mask_spacing), 4)
    centroid_xyz = _centroid(union_mask)
    centroid_idx = _clip_index(centroid_xyz, union_mask.shape)

    brain_mask = _brain_mask_from_t1(t1_data)
    if not np.any(brain_mask):
        brain_mask = _brain_mask_from_t1ce(t1ce_data)
    if not np.any(brain_mask):
        brain_mask = _brain_mask_from_flair(flair_data)
    hemisphere = _estimate_hemisphere(centroid_xyz, union_mask.shape)
    lobe, lobe_confidence_score, lobe_method = _estimate_lobe(centroid_xyz, brain_mask)
    normalized_centroid = _normalized_centroid(centroid_xyz, brain_mask) if np.any(brain_mask) else np.array([0.5, 0.5, 0.5], dtype=float)

    _atlas = _atlas_lobe_overlap(union_mask, brain_mask)
    if _atlas and _atlas.get("lobe_confidence") in ("high", "medium"):
        lobe        = _atlas["lobe"]
        lobe_method = _atlas["lobe_method"]
    atlas_top_regions    = _atlas.get("atlas_top_regions", [])
    lobe_overlap_fraction = _atlas.get("lobe_overlap_fraction", 0.0)
    lobe_confidence_final = _atlas.get("lobe_confidence", "low")
    depth_class, depth_mm = _depth_class(centroid_idx, brain_mask, mask_spacing)

    midline_measurement = _midline_shift_measurement(union_mask, mask_spacing)
    midline_shift_mm = midline_measurement["midline_shift_mm"]

    ventricle_proxy = _ventricle_proxy(t1ce_data, brain_mask, mask_spacing)
    edema_mask = mask == 2
    ventricular_measurement = _ventricular_compression_measurement(edema_mask, mask_spacing, union_mask.shape)
    ventricle_zone_centroid = _centroid(edema_mask) if np.any(edema_mask) else centroid_xyz
    image_center_xyz = np.asarray([(union_mask.shape[0] - 1) / 2.0, (union_mask.shape[1] - 1) / 2.0, (union_mask.shape[2] - 1) / 2.0], dtype=float)
    ventricle_distance_mm = float(np.linalg.norm((ventricle_zone_centroid - image_center_xyz) * np.asarray(mask_spacing, dtype=float)))
    ventricular_relation = _descriptor_from_ventricle_distance(ventricle_distance_mm)

    brainstem_proxy = _brainstem_proxy(brain_mask)
    brainstem_distance_mm = _min_mask_distance_mm(union_mask, brainstem_proxy, mask_spacing)
    motor_proxy = _motor_cortex_proxy(brain_mask, mask_spacing)
    motor_cortex_distance_mm = _min_mask_distance_mm(union_mask, motor_proxy, mask_spacing)

    sulcal_measurement = _sulcal_effacement_measurement(flair_data, brain_mask, union_mask, mask_spacing, hemisphere)
    cisternal_measurement = _cisternal_effacement_measurement(t1_data, brain_mask)

    if isinstance(midline_shift_mm, (int, float)):
        if midline_shift_mm > 5.0:
            atlas_reliability = "low"
        elif midline_shift_mm >= 2.0:
            atlas_reliability = "moderate"
        else:
            atlas_reliability = "high"
    else:
        atlas_reliability = "unknown"

    result = {
        "case_id": case_id,
        "modality": "MRI brain",
        "pathology_family": "brats_glioma_pattern",
        "tumor_burden": {
            "total_volume_mm3": total_tumor_volume_mm3,
            "enhancing_volume_mm3": float(label_volumes_mm3.get("enhancing_tumor", 0.0)),
            "edema_volume_mm3": float(label_volumes_mm3.get("edema", 0.0)),
            "necrotic_volume_mm3": float(label_volumes_mm3.get("necrotic_core", 0.0)),
            "max_diameter_mm": max_diameter_mm,
        },
        "location": {
            "hemisphere": hemisphere,
            "hemisphere_confidence": "high",
            "hemisphere_method": "centroid_to_image_midline",
            "lobe": lobe,
            "lobe_confidence": lobe_confidence_final,
            "lobe_confidence_score": round(float(lobe_overlap_fraction if lobe_overlap_fraction else lobe_confidence_score), 4),
            "lobe_method": lobe_method,
            "atlas_top_regions": atlas_top_regions,
            "lobe_overlap_fraction": round(float(lobe_overlap_fraction), 4),
            "depth_class": depth_class,
            "depth_mm": round(depth_mm, 4),
            "depth_method": "surface_distance",
            "depth_confidence": "moderate",
            "ventricular_relation": ventricular_relation,
            "ventricular_relation_mm": round(float(ventricle_distance_mm), 4),
            "ventricular_relation_method": "edema_or_tumor_centroid_to_image_center",
            "ventricular_relation_confidence": "low",
            "centroid_xyz_voxel": [round(float(v), 3) for v in centroid_xyz.tolist()],
            "normalized_centroid_xyz": [round(float(v), 4) for v in normalized_centroid.tolist()],
        },
        "mass_effect": {
            "midline_shift_mm": midline_shift_mm,
            "midline_shift": midline_measurement,
            "ventricular_compression": _bool_or_none(ventricular_measurement.get("ventricular_compression")),
            "ventricle_proximity_mm": ventricular_measurement.get("ventricle_proximity_mm"),
            "ventricular_compression_measurement": ventricular_measurement,
            "sulcal_effacement": _bool_or_none(sulcal_measurement.get("sulcal_effacement")),
            "hemispheric_asymmetry_index": sulcal_measurement.get("hemispheric_asymmetry_index"),
            "sulcal_effacement_measurement": sulcal_measurement,
            "cisternal_effacement": _bool_or_none(cisternal_measurement.get("cisternal_effacement")),
            "cisternal_obliteration": _bool_or_none(cisternal_measurement.get("cisternal_effacement")),
            "inferior_csf_voxel_count": cisternal_measurement.get("inferior_csf_voxel_count"),
            "cisternal_effacement_measurement": cisternal_measurement,
        },
        "critical_proximity": {
            "brainstem_distance_mm": round(float(brainstem_distance_mm), 4) if brainstem_distance_mm is not None else None,
            "motor_cortex_distance_mm": round(float(motor_cortex_distance_mm), 4) if motor_cortex_distance_mm is not None else None,
        },
        "distortion_context": {
            "midline_shift_mm": midline_shift_mm,
            "atlas_reliability": atlas_reliability,
            "ventricle_proxy_distance_mm": ventricular_measurement.get("ventricle_proximity_mm"),
        },
        "source_paths": {
            "case_dir": str(case_dir),
            "t1_path": str(t1_path),
            "t1ce_path": str(t1ce_path),
            "flair_path": str(flair_path),
            "mask_path": str(mask_path),
        },
        "feature_extraction": {
            "spacing_xyz_mm": [round(v, 6) for v in mask_spacing],
            "brain_mask_nonzero_voxels": int(np.count_nonzero(brain_mask)),
            "ventricle_proxy_nonzero_voxels": int(np.count_nonzero(ventricle_proxy)),
            "method": "measurement_grounded_no_registration",
        },
    }
    LOGGER.info(
        "Extracted %s | total=%.2f mm3 | enhancing=%.2f mm3 | edema=%.2f mm3 | mls=%s mm | vent_comp=%s | sulcal=%s | cisternal=%s",
        case_id,
        total_tumor_volume_mm3,
        result["tumor_burden"]["enhancing_volume_mm3"],
        result["tumor_burden"]["edema_volume_mm3"],
        f"{midline_shift_mm:.2f}" if isinstance(midline_shift_mm, (int, float)) else "null",
        result["mass_effect"]["ventricular_compression"],
        result["mass_effect"]["sulcal_effacement"],
        result["mass_effect"]["cisternal_effacement"],
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract lightweight BraTS retrieval features for one case.")
    parser.add_argument("--case-dir", required=True, help="Path to the BraTS case folder.")
    parser.add_argument("--mask-path", required=True, help="Path to the corresponding BraTS tumor mask (.nii.gz).")
    parser.add_argument("--t1ce-path", help="Optional explicit path to the t1ce image.")
    parser.add_argument("--output-json", help="Optional output JSON path.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    _configure_logging(args.verbose)
    features = extract_case_features(args.case_dir, args.mask_path, args.t1ce_path)
    rendered = json.dumps(features, indent=2, ensure_ascii=True)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
        LOGGER.info("Wrote feature JSON to %s", output_path)
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
