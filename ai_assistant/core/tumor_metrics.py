import json
import os
from pathlib import Path
from typing import Any

from ai_assistant.core.anatomic_mapper import anatomy_from_lesion, register_reference_to_mni
from ai_assistant.core.paths import to_workspace_relative
from ai_assistant.core.uncertainty import add_confidence_intervals



BRATS_LABELS = {
    1: "nonenhancing",
    2: "edema",
    3: "empty",
    4: "enhancing",
}


def _round4(value: float) -> float:
    return round(float(value), 4)


def _connected_component_count(mask: Any, sitk: Any, min_voxels: int) -> int:
    cc_img = sitk.ConnectedComponent(sitk.GetImageFromArray(mask.astype("uint8")))
    cc_arr = sitk.GetArrayFromImage(cc_img)
    component_ids = [int(x) for x in set(cc_arr.flatten().tolist()) if int(x) > 0]
    count = 0
    for component_id in component_ids:
        voxel_count = int((cc_arr == component_id).sum())
        if voxel_count >= min_voxels:
            count += 1
    return count


def compute_tumor_metrics_from_mask(
    mask_path: str,
    patient_id: str,
    study_id: str,
    reference_image_path: str | None = None,
) -> dict[str, Any]:
    try:
        import numpy as np  
        import SimpleITK as sitk  
    except Exception as exc:
        raise RuntimeError("SimpleITK and numpy are required for tumor metrics extraction") from exc

    path = Path(mask_path)
    if not path.exists():
        raise FileNotFoundError(f"mask not found: {path}")

    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img)
    spacing_xyz = img.GetSpacing()
    spacing_zyx = (spacing_xyz[2], spacing_xyz[1], spacing_xyz[0])
    voxel_volume_mm3 = float(spacing_xyz[0] * spacing_xyz[1] * spacing_xyz[2])
    image_shape_zyx = tuple(int(v) for v in arr.shape)

    min_lesion_voxels = int(os.getenv("AI_MIN_LESION_VOXELS", "20") or 20)
    standard_space_transform = register_reference_to_mni(reference_image_path) if reference_image_path else None
    lesion_list: list[dict[str, Any]] = []
    label_volumes_mm3: dict[str, float] = {}
    location_labels: list[str] = []
    lesion_id = 1

    unique_labels = sorted(int(x) for x in set(arr.flatten().tolist()))
    for label in unique_labels:
        if label in (0, 3):
            continue
        label_mask = (arr == label)
        if not label_mask.any():
            continue

        label_name = BRATS_LABELS.get(label, f"label_{label}")
        location_labels.append(label_name)
        label_voxels = int(label_mask.sum())
        label_volumes_mm3[label_name] = _round4(label_voxels * voxel_volume_mm3)

        cc_img = sitk.ConnectedComponent(sitk.GetImageFromArray(label_mask.astype("uint8")))
        cc_arr = sitk.GetArrayFromImage(cc_img)
        component_ids = [int(x) for x in set(cc_arr.flatten().tolist()) if int(x) > 0]

        for component_id in component_ids:
            component_mask = (cc_arr == component_id)
            voxel_count = int(component_mask.sum())
            if voxel_count < min_lesion_voxels:
                continue

            coords = np.argwhere(component_mask)
            mins = coords.min(axis=0)
            maxs = coords.max(axis=0)
            centroid_zyx = coords.mean(axis=0)
            centroid_xyz = [float(centroid_zyx[2]), float(centroid_zyx[1]), float(centroid_zyx[0])]


            spacing_arr = np.array(spacing_zyx, dtype=float)
            coords_mm = (coords - centroid_zyx) * spacing_arr
            diameter_method = "pca_longest_axis"
            try:
                if voxel_count >= 3:
                    cov = np.cov(coords_mm.T)
                    if cov.ndim == 2 and cov.shape == (3, 3):
                        _, eigenvectors = np.linalg.eigh(cov)
                        first_pc = eigenvectors[:, -1]          
                        projections = coords_mm @ first_pc
                        max_diameter_mm = float(projections.max() - projections.min())
                    else:
                        raise ValueError("degenerate covariance")
                else:
                    raise ValueError("too few voxels for PCA")
            except Exception:
                extents_mm = (maxs - mins + 1).astype(float) * spacing_arr
                max_diameter_mm = float(extents_mm.max())
                diameter_method = "bounding_box_max"

            lesion_payload = {
                "lesion_id": lesion_id,
                "label_value": label,
                "label_name": label_name,
                "voxel_count": voxel_count,
                "volume_mm3": _round4(voxel_count * voxel_volume_mm3),
                "max_diameter_mm": _round4(max_diameter_mm),
                "diameter_method": diameter_method,
                "centroid_zyx_voxel": [_round4(x) for x in centroid_zyx.tolist()],
                "centroid_xyz_voxel": [_round4(x) for x in centroid_xyz],
            }
            lesion_payload["anatomy"] = anatomy_from_lesion(
                lesion_payload,
                image_shape_zyx=image_shape_zyx,
                reference_image_path=reference_image_path,
                source_image=img,
                standard_space_transform=standard_space_transform,
                lesion_coords_zyx=coords,
            )

            lesion_list.append(lesion_payload)
            lesion_id += 1

    union_mask = (arr > 0) & (arr != 3)
    lesion_count = _connected_component_count(union_mask, sitk, min_lesion_voxels)
    component_count = len(lesion_list)
    total_union_voxels = int(union_mask.sum())
    total_tumor_volume_mm3 = _round4(total_union_voxels * voxel_volume_mm3)
    max_diameter_mm = _round4(max((x["max_diameter_mm"] for x in lesion_list), default=0.0))
    dominant_for_diam = max(lesion_list, key=lambda x: float(x.get("max_diameter_mm", 0.0) or 0.0), default={})
    diameter_method = dominant_for_diam.get("diameter_method", "pca_longest_axis")
    dominant = max(lesion_list, key=lambda lesion: float(lesion.get("volume_mm3", 0.0) or 0.0), default={})

    metrics = {
        "patient_id": patient_id,
        "study_id": study_id,
        "tumor_count": lesion_count,
        "segmented_component_count": component_count,
        "lesion_list": lesion_list,
        "volume_mm3": total_tumor_volume_mm3,
        "total_tumor_volume_mm3": total_tumor_volume_mm3,
        "max_diameter_mm": max_diameter_mm,
        "location_labels": sorted(set(location_labels)),
        "label_volumes_mm3": label_volumes_mm3,
        "voxel_spacing_xyz_mm": [_round4(x) for x in spacing_xyz],
        "image_shape_zyx": list(image_shape_zyx),
        "source_mask_path": to_workspace_relative(path),
        "reference_image_path": to_workspace_relative(reference_image_path) if reference_image_path else None,
        "model_version": os.getenv("IMG_MODEL_VERSION", "nnunet_brain_3d_fullres"),
        "dominant_lesion_id": dominant.get("lesion_id"),
        "dominant_anatomic_location": (dominant.get("anatomy") or {}).get("description"),
        "anatomic_mapping_basis": (dominant.get("anatomy") or {}).get("mapping_basis"),
        "dominant_atlas_confidence": (dominant.get("anatomy") or {}).get("atlas_confidence"),
        "dominant_registration_quality_flag": (dominant.get("anatomy") or {}).get("registration_quality_flag"),
        "dominant_registration_quality_score": (dominant.get("anatomy") or {}).get("registration_quality_score"),
        "registration_summary": (standard_space_transform or {}).get("summary", {}) if isinstance(standard_space_transform, dict) else {},
        "diameter_method": diameter_method,
        "audit_gate": {
            "status": "applied",
            "scope": "brain",
            "repairs": [
                "BraTS labels normalized to 1=NCR/NET, 2=ED, 4=ET",
                "total_tumor_volume_mm3 recomputed from union of valid tumor voxels",
                "workspace-relative artifact paths enforced",
            ],
            "min_lesion_voxels": min_lesion_voxels,
            "union_voxel_count": total_union_voxels,
        },
    }
    return add_confidence_intervals(metrics)


def save_tumor_metrics(metrics: dict[str, Any], out_path: str) -> None:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=True, indent=2)
