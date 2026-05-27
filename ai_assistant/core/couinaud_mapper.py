from __future__ import annotations
import numpy as np
from typing import Any

COUINAUD_SEGMENTS = {
    1: "Segment I (Caudate Lobe) — medial, posterior, around IVC",
    2: "Segment II — left lateral superior",
    3: "Segment III — left lateral inferior",
    4: "Segment IV (Medial left lobe) — between falciform and IHV",
    5: "Segment V — right anterior inferior (gallbladder fossa)",
    6: "Segment VI — right posterior inferior",
    7: "Segment VII — right posterior superior",
    8: "Segment VIII — right anterior superior",
}


def estimate_couinaud_segment(
    centroid_ijk: tuple[float, float, float],
    liver_bbox_min: tuple[float, float, float],
    liver_bbox_max: tuple[float, float, float],
    affine: Any = None,
) -> dict:

    bbox_min = np.array(liver_bbox_min, dtype=float)
    bbox_max = np.array(liver_bbox_max, dtype=float)
    span = np.maximum(bbox_max - bbox_min, 1.0)
    centroid = np.array(centroid_ijk, dtype=float)

    norm = np.clip((centroid - bbox_min) / span, 0.0, 1.0)
    nx, ny, nz = float(norm[0]), float(norm[1]), float(norm[2])

    if ny > 0.65 and 0.35 < nx < 0.65:
        seg = 1; conf = "medium"
    elif nx < 0.45:
        if nx < 0.30:
            seg = 2 if nz > 0.5 else 3; conf = "medium"
        else:
            seg = 4; conf = "medium"
    else:
        if ny > 0.50:  
            seg = 7 if nz > 0.5 else 6; conf = "medium"
        else:  
            seg = 8 if nz > 0.5 else 5; conf = "medium"

    return {
        "couinaud_segment": seg,
        "couinaud_description": COUINAUD_SEGMENTS[seg],
        "couinaud_confidence": conf,
        "couinaud_method": "centroid_bbox_heuristic",
        "normalized_position_xyz": [round(nx, 3), round(ny, 3), round(nz, 3)],
    }


def annotate_lesions_with_couinaud(metrics: dict) -> dict:
    import copy
    result = copy.deepcopy(metrics)
    lesion_list = result.get("lesion_list", [])

    liver_lesions = [
        l for l in lesion_list
        if l.get("label_value") == 1 or l.get("label_name") in ("liver", "liver_parenchyma")
    ]
    tumor_lesions = [
        l for l in lesion_list
        if l.get("label_value") == 2 or l.get("label_name") in ("liver_tumor", "tumor")
    ]

    if not liver_lesions:
        return result

    centroids = [l.get("centroid_zyx_voxel", [0, 0, 0]) for l in liver_lesions]
    if not centroids:
        return result

    arr = np.array(centroids)
    bbox_min = arr.min(axis=0).tolist()
    bbox_max = arr.max(axis=0).tolist()

    pad = 20.0
    bbox_min = [b - pad for b in bbox_min]
    bbox_max = [b + pad for b in bbox_max]

    for lesion in lesion_list:
        if lesion.get("label_value", 0) in (2,) or lesion.get("label_name") in ("liver_tumor", "tumor"):
            c = lesion.get("centroid_zyx_voxel", [0, 0, 0])
            seg_info = estimate_couinaud_segment(
                centroid_ijk=(c[2], c[1], c[0]),  
                liver_bbox_min=bbox_min,
                liver_bbox_max=bbox_max,
            )
            lesion["couinaud"] = seg_info

    result["lesion_list"] = lesion_list
    return result
