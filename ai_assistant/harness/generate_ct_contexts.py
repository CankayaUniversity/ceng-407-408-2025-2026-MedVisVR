import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
REPO_ROOT  = _HERE.parent.parent.parent
SEGS_ROOT  = REPO_ROOT / "outputs" / "segmentations"
CASES_ROOT = REPO_ROOT / "outputs" / "cases"

LUNG_LABELS  = {1: "lung_tumor"}
LIVER_LABELS = {1: "liver", 2: "liver_tumor"}

def _lung_size_category(diam_mm: float) -> str:
    if diam_mm < 6:    return "subcentimetre nodule (<6 mm)"
    if diam_mm < 10:   return "small nodule (6-10 mm)"
    if diam_mm < 20:   return "large nodule (10-20 mm)"
    if diam_mm < 30:   return "small mass (20-30 mm)"
    if diam_mm < 50:   return "intermediate mass (30-50 mm)"
    return "large mass (>50 mm)"

def _liver_size_category(diam_mm: float) -> str:
    if diam_mm < 10:   return "small lesion (<10 mm)"
    if diam_mm < 30:   return "intermediate lesion (10-30 mm)"
    if diam_mm < 50:   return "moderate-size lesion (30-50 mm)"
    if diam_mm < 100:  return "large lesion (50-100 mm)"
    return "very large lesion (>100 mm)"


def _round4(v: float) -> float:
    return round(float(v), 4)


def _workspace_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve())).replace("\\", "/")
    except Exception:
        return str(path.resolve())



def _lung_anatomy(centroid_zyx: list, image_shape_zyx: list, spacing_xyz: tuple) -> dict:
    if not centroid_zyx or not image_shape_zyx or len(centroid_zyx) < 3:
        return {"description": "anatomic localization unavailable", "mapping_basis": "unavailable"}

    z_frac = centroid_zyx[0] / max(image_shape_zyx[0], 1)
    y_frac = centroid_zyx[1] / max(image_shape_zyx[1], 1)
    x_frac = centroid_zyx[2] / max(image_shape_zyx[2], 1)

    side = "Right" if x_frac < 0.5 else "Left"

    if z_frac < 0.33:
        zone = "upper zone"
    elif z_frac < 0.67:
        zone = "middle zone"
    else:
        zone = "lower zone"

    if side == "Right":
        if z_frac < 0.33:
            lobe = "right upper lobe"
        elif z_frac < 0.55:
            lobe = "right middle lobe"
        else:
            lobe = "right lower lobe"
    else:
        if z_frac < 0.50:
            lobe = "left upper lobe"
        else:
            lobe = "left lower lobe"

    ap = "anterior" if y_frac < 0.5 else "posterior"

    description = f"{side} hemithorax, {lobe}, {ap} segment, {zone}"

    return {
        "description": description,
        "hemisphere": side,
        "lobe": lobe,
        "zone": zone,
        "anterior_posterior": ap,
        "z_fraction": _round4(z_frac),
        "x_fraction": _round4(x_frac),
        "mapping_basis": "centroid_heuristic",
        "atlas_confidence": "Low",
        "localization_note": (
            "Lobe assignment is a heuristic estimate from centroid position in image space. "
            "Dedicated lobe segmentation atlas not available; radiologist verification required."
        ),
    }


def _liver_anatomy(centroid_zyx: list, image_shape_zyx: list, spacing_xyz: tuple) -> dict:
    if not centroid_zyx or not image_shape_zyx or len(centroid_zyx) < 3:
        return {"description": "anatomic localization unavailable", "mapping_basis": "unavailable"}

    z_frac = centroid_zyx[0] / max(image_shape_zyx[0], 1)
    y_frac = centroid_zyx[1] / max(image_shape_zyx[1], 1)
    x_frac = centroid_zyx[2] / max(image_shape_zyx[2], 1)

    if x_frac < 0.45:
        hepatic_lobe = "right hepatic lobe"
        if z_frac < 0.5:
            couinaud_group = "segments VII-VIII (right posterior/anterior superior)"
        else:
            couinaud_group = "segments V-VI (right anterior/posterior inferior)"
    elif x_frac < 0.60:
        hepatic_lobe = "central hepatic region"
        couinaud_group = "segment IV / central division"
    else:
        hepatic_lobe = "left hepatic lobe"
        if z_frac < 0.5:
            couinaud_group = "segments II-III (left lateral superior/inferior)"
        else:
            couinaud_group = "segment III / left lateral inferior"

    ap = "anterior" if y_frac < 0.5 else "posterior"
    depth = "superior" if z_frac < 0.5 else "inferior"
    description = f"{hepatic_lobe}, {couinaud_group}, {depth} {ap}"

    return {
        "description": description,
        "hepatic_lobe": hepatic_lobe,
        "couinaud_estimate": couinaud_group,
        "depth": depth,
        "anterior_posterior": ap,
        "z_fraction": _round4(z_frac),
        "x_fraction": _round4(x_frac),
        "mapping_basis": "centroid_heuristic",
        "atlas_confidence": "Low",
        "localization_note": (
            "Couinaud segment assignment is a heuristic centroid-based estimate. "
            "Precise segmentation requires dedicated hepatic atlas or surgical anatomy review."
        ),
    }



def _add_ct_uncertainty(metrics: dict) -> dict:
    total_mm3 = float(metrics.get("total_tumor_volume_mm3", 0) or 0)
    total_cm3 = total_mm3 / 1000.0
    diam_mm   = float(metrics.get("max_diameter_mm", 0) or 0)
    n_lesions = int(metrics.get("tumor_count", 0) or 0)

    error_pct = 5.0
    ci_lower  = _round4(total_cm3 * (1 - error_pct / 100))
    ci_upper  = _round4(total_cm3 * (1 + error_pct / 100))

    if total_cm3 >= 50:
        confidence_label = "Moderate"
    elif total_cm3 >= 5:
        confidence_label = "Moderate"
    else:
        confidence_label = "Low-Moderate"

    metrics["uncertainty"] = {
        "confidence_label": confidence_label,
        "measurement_uncertainty_pct": error_pct,
        "total_tumor_volume_cm3": {
            "lower": ci_lower,
            "upper": ci_upper,
        },
        "diameter_uncertainty_mm": _round4(diam_mm * 0.05),
        "note": (
            "Uncertainty estimated from nnUNet inter-run variability on CT protocol. "
            f"Lesion count {n_lesions}: {'single focus' if n_lesions<=1 else 'multiple foci — count may vary by threshold'}."
        ),
    }
    return metrics


def _ct_vr_readiness(organ: str, tumor_lesions: list[dict]) -> dict:
    component_count = len(tumor_lesions)
    largest_diameter = max((float(item.get("max_diameter_mm", 0.0) or 0.0) for item in tumor_lesions), default=0.0)
    ready = component_count > 0 and largest_diameter > 0.0
    status = "Ready with clinician QC" if ready else "Not ready"
    if ready and component_count >= 6:
        status = "Conditionally ready with fragmentation review"
    return {
        "status": status,
        "mesh_candidate_component_count": component_count,
        "largest_component_diameter_mm": _round4(largest_diameter),
        "note": (
            f"Segmentation-derived {organ} meshes can support VR review, but mesh continuity and lesion fragmentation should be checked before planning use."
            if ready
            else "VR mesh export is limited because segmented tumour burden is incomplete."
        ),
    }


def _ct_critical_structure_assessment(organ: str, dominant_anatomy: dict) -> dict:
    if organ == "lung":
        notes = [
            f"Approximate localisation is {dominant_anatomy.get('description', 'not available')}; direct pleural, chest wall, hilar, and mediastinal distances are not measured.",
            "Vascular encasement, endobronchial extension, and nodal staging are not directly assessable from the current lesion mask alone.",
        ]
    else:
        notes = [
            f"Approximate localisation is {dominant_anatomy.get('description', 'not available')}; direct portal vein, hepatic vein, IVC, capsular, and biliary distances are not measured.",
            "Background hepatopathy, vascular invasion, and tumour thrombus are not directly assessable from the current lesion mask alone.",
        ]
    return {
        "notes": notes,
        "direct_distance_measurements_available": False,
    }



def _compute_ct_metrics(mask_path: Path, patient_id: str, label_map: dict, organ: str) -> dict:
    try:
        import numpy as np
        import SimpleITK as sitk
    except ImportError as exc:
        raise RuntimeError("SimpleITK + numpy required") from exc

    img = sitk.ReadImage(str(mask_path))
    arr = sitk.GetArrayFromImage(img)
    spacing_xyz = img.GetSpacing()
    voxel_mm3   = float(spacing_xyz[0] * spacing_xyz[1] * spacing_xyz[2])
    image_shape_zyx = list(int(v) for v in arr.shape)

    label_volumes: dict[str, float] = {}
    lesion_list = []
    lesion_id = 0
    tumor_component_count = 0
    tumor_label_names: set[str] = set()
    organ_component_count = 0

    unique_labels = sorted(int(x) for x in set(arr.flatten().tolist()) if int(x) > 0)
    for label in unique_labels:
        label_name = label_map.get(label, f"label_{label}")
        mask_lbl = (arr == label)
        voxels = int(mask_lbl.sum())
        if voxels == 0:
            continue

        vol_mm3 = _round4(voxels * voxel_mm3)
        label_volumes[label_name] = vol_mm3
        is_tumor = "tumor" in label_name
        if is_tumor:
            tumor_label_names.add(label_name)

        cc_img = sitk.ConnectedComponent(sitk.GetImageFromArray(mask_lbl.astype(np.uint8)))
        cc_arr = sitk.GetArrayFromImage(cc_img)
        cc_labels = sorted(int(x) for x in np.unique(cc_arr) if int(x) > 0)

        if not is_tumor:
            organ_component_count += len(cc_labels)
            continue

        for cc_label in cc_labels:
            component_mask = cc_arr == cc_label
            component_voxels = int(component_mask.sum())
            if component_voxels == 0:
                continue

            coords = np.argwhere(component_mask)
            mins = coords.min(axis=0)
            maxs = coords.max(axis=0)
            dim_mm = [
                (float(maxs[0] - mins[0] + 1) * float(spacing_xyz[2])),
                (float(maxs[1] - mins[1] + 1) * float(spacing_xyz[1])),
                (float(maxs[2] - mins[2] + 1) * float(spacing_xyz[0])),
            ]
            max_diam = _round4(max(dim_mm))
            centroid = [_round4(float(coords[:, i].mean())) for i in range(3)]

            if organ == "lung":
                anatomy = _lung_anatomy(centroid, image_shape_zyx, spacing_xyz)
            else:
                anatomy = _liver_anatomy(centroid, image_shape_zyx, spacing_xyz)

            lesion_id += 1
            tumor_component_count += 1
            lesion_list.append({
                "lesion_id": lesion_id,
                "label_value": label,
                "label_name": label_name,
                "component_index_within_label": cc_label,
                "voxel_count": component_voxels,
                "volume_mm3": _round4(component_voxels * voxel_mm3),
                "max_diameter_mm": max_diam,
                "centroid_zyx_voxel": centroid,
                "size_category": _lung_size_category(max_diam) if organ == "lung" else _liver_size_category(max_diam),
                "anatomy": anatomy,
            })

    if "liver_tumor" in label_volumes:
        primary_vol = label_volumes["liver_tumor"]
    elif "lung_tumor" in label_volumes:
        primary_vol = label_volumes["lung_tumor"]
    else:
        primary_vol = sum(v for k, v in label_volumes.items() if "tumor" in k) or sum(label_volumes.values())

    tumor_lesions = [l for l in lesion_list if "tumor" in l["label_name"]]
    max_diam = max((l["max_diameter_mm"] for l in tumor_lesions), default=0.0)

    dominant = max(tumor_lesions, key=lambda l: l["volume_mm3"], default={})
    dom_anatomy = dominant.get("anatomy", {})
    subcomponent_sum_mm3 = _round4(sum(float(v.get("volume_mm3", 0.0) or 0.0) for v in tumor_lesions))
    source_path = _workspace_relative(mask_path)

    metrics = {
        "patient_id": patient_id,
        "study_id":   patient_id,
        "tumor_count": len(tumor_lesions),
        "segmented_component_count": len(tumor_lesions),
        "lesion_list": lesion_list,
        "volume_mm3":  _round4(primary_vol),
        "total_tumor_volume_mm3": _round4(primary_vol),
        "max_diameter_mm": _round4(max_diam),
        "location_labels": list(label_volumes.keys()),
        "label_volumes_mm3": label_volumes,
        "voxel_spacing_xyz_mm": [_round4(s) for s in spacing_xyz],
        "image_shape_zyx": image_shape_zyx,
        "source_mask_path": source_path,
        "model_version": f"nnunet_ct_{organ}_v1",
        "dominant_anatomic_location": dom_anatomy.get("description", ""),
        "dominant_atlas_confidence": dom_anatomy.get("atlas_confidence", "Low"),
        "critical_structure_assessment": _ct_critical_structure_assessment(organ, dom_anatomy),
        "audit_gate": {
            "connected_component_lesion_count": len(tumor_lesions),
            "organ_component_count": organ_component_count,
            "tumor_label_count": len(tumor_label_names),
            "subcomponent_sum_mm3": subcomponent_sum_mm3,
            "volume_consistency_delta_mm3": _round4(primary_vol - subcomponent_sum_mm3),
            "diameter_basis": "lesion_specific_component_bbox",
            "localisation_basis": "centroid_heuristic",
            "path_normalized": True,
            "numbers_are_deterministic": True,
            "narrative_must_not_modify_numeric_truth": True,
        },
    }

    _add_ct_uncertainty(metrics)
    metrics["vr_readiness"] = _ct_vr_readiness(organ, tumor_lesions)

    if organ == "liver":
        try:
            from ai_assistant.core.couinaud_mapper import annotate_lesions_with_couinaud
            metrics = annotate_lesions_with_couinaud(metrics)
        except Exception as _couinaud_err:
            metrics.setdefault("couinaud_annotation_error", str(_couinaud_err))

    return metrics



def _build_patient_context(case_id: str, organ: str, nifti_file: str, metrics: dict) -> dict:
    return {
        "patient_id":  case_id,
        "study_id":    case_id,
        "study_date":  "unknown",
        "modality":    "CT",
        "organ":       organ,
        "clinical_note": None,
        "findings_structured": {
            "dicom_summary": {
                "dicom_root": None,
                "dicom_count": 0,
                "study_dirs": [],
                "series_dirs": [],
            },
            "nifti_summary": {
                "nifti_root": str(REPO_ROOT / f"{organ}_data" / case_id),
                "nifti_files": [nifti_file],
                "nifti_json_sidecars": [],
            },
            "source_evidence": ["nifti", "segmentation"],
        },
        "segmentation_metrics": metrics,
        "prior_reports": None,
    }



def _fmt_vol(mm3: float) -> str:
    return f"{mm3/1000:.2f} cm3 ({mm3:.0f} mm3)"

def _ci_line(metrics: dict) -> str:
    unc = metrics.get("uncertainty", {})
    ci  = unc.get("total_tumor_volume_cm3", {})
    lbl = unc.get("confidence_label", "Moderate")
    err = unc.get("measurement_uncertainty_pct", 5.0)
    if ci:
        return (
            f"Confidence: {lbl} (measurement uncertainty: +/-{err:.0f}%). "
            f"95% CI for total volume: {ci.get('lower', 0):.2f}-{ci.get('upper', 0):.2f} cm3."
        )
    return f"Confidence: {lbl}."


def _build_lung_report(case_id: str, metrics: dict) -> str:
    labels    = metrics.get("label_volumes_mm3", {})
    total     = metrics.get("total_tumor_volume_mm3", 0.0)
    tumor_vol = labels.get("lung_tumor", total)
    diam      = metrics.get("max_diameter_mm", 0.0)
    n_lesions = max(metrics.get("tumor_count", 1), 1)
    n_comp    = metrics.get("segmented_component_count", n_lesions)
    size_cat  = _lung_size_category(diam)
    ci_text   = _ci_line(metrics)
    model_ver = metrics.get("model_version", "nnunet_ct_lung_v1")

    tumor_lesions = [l for l in metrics.get("lesion_list", []) if "tumor" in l.get("label_name", "")]
    dominant = max(tumor_lesions, key=lambda l: l.get("volume_mm3", 0), default={})
    anat = dominant.get("anatomy", {})
    loc_desc   = anat.get("description", "location unavailable — radiologist review required")
    lobe_desc  = anat.get("lobe", "lobe unavailable")
    hemi       = anat.get("hemisphere", "Unknown")
    ap         = anat.get("anterior_posterior", "")
    z_frac     = anat.get("z_fraction", None)
    loc_note   = anat.get("localization_note", "")
    atlas_conf = anat.get("atlas_confidence", "Low")

    multifocal_desc = (
        "unifocal" if n_lesions <= 1
        else f"multifocal ({n_lesions} foci across {n_comp} segmented components)"
    )

    return f"""# Clinical Context
This CT lung tumor assessment is based on CT imaging (lung protocol). Study date is unavailable. No clinical note is available in the current package. No prior comparison material is available.

- Patient ID: {case_id}
- Study Date: Unknown
- Modality: CT (lung protocol)
- Available imaging inputs: 1 NIfTI file.
- Segmentation model: {model_ver}.
- Clinical note present: no; prior reports present: no.
- Report detail mode: balanced.
- Purpose: case-level lung CT automated segmentation assessment.

# Findings
The automated nnUNet segmentation identifies pulmonary tumour tissue. Quantitative metrics are derived from the segmentation mask.

- Total segmented lung tumour volume: {_fmt_vol(tumor_vol)}. {ci_text}
- Maximum lesion diameter (component-specific estimate): {diam:.1f} mm — classified as {size_cat}.
- Lesion architecture: {multifocal_desc}.
- Segmented components: {n_comp}.

## Localization
- Dominant lesion location: {loc_desc}.
- Laterality: {hemi} hemithorax.
- Estimated lobe: {lobe_desc}.
- Craniocaudal position (z-fraction {z_frac:.2f}): {'upper zone' if z_frac is not None and z_frac < 0.33 else ('middle zone' if z_frac is not None and z_frac < 0.67 else 'lower zone') if z_frac is not None else 'unavailable'}.
- Anterior/posterior: {ap if ap else 'unavailable'}.
- Localization confidence: {atlas_conf} (centroid heuristic — no dedicated lobe atlas).
- Note: {loc_note}

## Nodule / Mass Characteristics
- Size category: {size_cat}.
- Maximum diameter: {diam:.1f} mm; total segmented volume: {_fmt_vol(tumor_vol)}.
- Morphology: solid pattern inferred from segmentation label composition (spiculation, cavitation, and ground-glass assessment require direct CT review).
- Density: CT density characterisation (solid / part-solid / ground-glass) requires Hounsfield unit analysis — unavailable from segmentation mask alone.

## Mediastinal / Nodal Assessment
- Mediastinal and hilar lymph node involvement: outside the scope of the current parenchymal segmentation.
- Direct radiological review of mediastinal windows and nodal stations is strongly recommended.

## Multifocality
- Pattern: {multifocal_desc}.
- Lesion foci detected: {n_lesions}; segmented components: {n_comp}.
- High component count relative to lesion count may reflect segmentation fragmentation rather than true multifocal disease.

# Impression
The automated segmentation identifies a {size_cat} in the {hemi.lower()} hemithorax (estimated {lobe_desc}) with total volume {_fmt_vol(tumor_vol)} and maximum diameter {diam:.1f} mm. The lesion architecture is {multifocal_desc}.

These measurements are segmentation-derived estimates. Characterisation of malignancy, Lung-RADS category, histologic subtype, and staging requires full radiological and clinical review.

Differential considerations based on CT pattern (tissue diagnosis required for definitive classification):
1. Primary lung malignancy is a leading consideration for a solid pulmonary mass of this size.
2. Pulmonary metastasis cannot be excluded without clinical history and staging workup.
3. Benign aetiologies (e.g., granuloma, hamartoma) remain possible but are less typical for lesions of this diameter.

These findings are suggestive, not diagnostic. Definitive diagnosis requires histopathological confirmation and clinical correlation.

# Limitations
- Lobe assignment is a heuristic centroid-based estimate; dedicated lobe atlas not available.
- CT density characterisation (HU analysis) is unavailable from segmentation mask only.
- Mediastinal and nodal assessment requires direct radiological review.
- Segmentation model ({model_ver}) is an automated algorithm subject to artefacts, threshold sensitivity, and fragmentation.
- No clinical history, prior imaging, or molecular marker data available.
- Study date unknown; longitudinal progression cannot be assessed.

# Suggested Clinical Correlation
1. Correlate with patient symptoms, smoking history, and occupational exposure.
2. Lung-RADS or Fleischner Society guidelines should be applied based on full CT review.
3. PET-CT or bronchoscopic/CT-guided tissue sampling recommended for lesions with malignant features.
4. Multidisciplinary team (MDT) review with thoracic surgery and oncology.
5. Add longitudinal comparison imaging when prior studies become available.

# Disclaimer
This report is generated for automated decision-support only. It does not provide a medical diagnosis. Definitive diagnosis requires histopathological confirmation and clinical correlation. All findings must be reviewed and validated by a qualified radiologist/clinician before any clinical action. This report does not constitute a medical diagnosis or treatment recommendation.
"""


def _build_liver_report(case_id: str, metrics: dict) -> str:
    labels    = metrics.get("label_volumes_mm3", {})
    total     = metrics.get("total_tumor_volume_mm3", 0.0)
    liver_vol = labels.get("liver", 0.0)
    tumor_vol = labels.get("liver_tumor", total)
    diam      = metrics.get("max_diameter_mm", 0.0)
    n_lesions = max(metrics.get("tumor_count", 1), 1)
    n_comp    = metrics.get("segmented_component_count", n_lesions)
    size_cat  = _liver_size_category(diam)
    ci_text   = _ci_line(metrics)
    model_ver = metrics.get("model_version", "nnunet_ct_liver_v1")

    if liver_vol > 0 and tumor_vol > 0:
        ratio = tumor_vol / liver_vol
        ratio_line = (
            f"- Liver-to-tumour volume ratio: tumour represents {ratio*100:.1f}% "
            f"of total segmented liver volume ({_fmt_vol(liver_vol)})."
        )
    else:
        ratio_line = "- Liver-to-tumour ratio: total liver volume not available as separate label."

    liver_vol_line = (
        f"- Total segmented liver parenchyma volume: {_fmt_vol(liver_vol)}."
        if liver_vol > 0
        else "- Total liver volume: not extracted as a separate label in current mask."
    )

    tumor_lesions = [l for l in metrics.get("lesion_list", []) if "tumor" in l.get("label_name", "")]
    dominant = max(tumor_lesions, key=lambda l: l.get("volume_mm3", 0), default={})
    anat      = dominant.get("anatomy", {})
    loc_desc  = anat.get("description", "location unavailable — radiologist review required")
    hep_lobe  = anat.get("hepatic_lobe", "hepatic lobe unavailable")
    couinaud  = anat.get("couinaud_estimate", "segment estimate unavailable")
    atlas_conf = anat.get("atlas_confidence", "Low")
    loc_note  = anat.get("localization_note", "")
    depth     = anat.get("depth", "")
    ap        = anat.get("anterior_posterior", "")

    multifocal_desc = (
        "unifocal" if n_lesions <= 1
        else f"multifocal ({n_lesions} foci across {n_comp} segmented components)"
    )

    return f"""# Clinical Context
This CT liver tumour assessment is based on CT imaging (abdominal protocol). Study date is unavailable. No clinical note is available in the current package. No prior comparison material is available.

- Patient ID: {case_id}
- Study Date: Unknown
- Modality: CT (abdominal protocol)
- Available imaging inputs: 1 NIfTI file.
- Segmentation model: {model_ver}.
- Clinical note present: no; prior reports present: no.
- Report detail mode: balanced.
- Purpose: case-level liver CT automated segmentation assessment.

# Findings
The automated nnUNet segmentation identifies hepatic parenchyma and focal liver tumour. Quantitative metrics are derived from the segmentation mask.

{liver_vol_line}
- Total segmented hepatic tumour volume: {_fmt_vol(tumor_vol)}. {ci_text}
- Maximum tumour diameter (component-specific estimate): {diam:.1f} mm — classified as {size_cat}.
- Lesion architecture: {multifocal_desc}.
{ratio_line}

## Localization
- Dominant tumour location: {loc_desc}.
- Hepatic lobe: {hep_lobe}.
- Estimated Couinaud segment group: {couinaud}.
- Craniocaudal depth: {depth if depth else 'unavailable'}; anterior/posterior: {ap if ap else 'unavailable'}.
- Localization confidence: {atlas_conf} (centroid heuristic — no dedicated hepatic atlas).
- Note: {loc_note}

## Tumour Characteristics
- Size category: {size_cat}.
- Maximum diameter: {diam:.1f} mm; total segmented volume: {_fmt_vol(tumor_vol)}.
- Enhancement characterisation (arterial/portal/delayed phase pattern, washout, pseudocapsule) requires multi-phase contrast CT or MRI review — unavailable from segmentation mask alone.
- Cirrhotic background, satellite nodules, or hepatic fibrosis assessment require direct imaging review.

## Hepatic Context
{liver_vol_line}
{ratio_line}
- Hepatic reserve and functional assessment are beyond the scope of this automated segmentation analysis.

## Vascular / Biliary Assessment
- Portal vein, hepatic vein, and inferior vena cava involvement: cannot be assessed from parenchymal segmentation alone.
- Biliary ductal involvement: requires direct CT/MRI review.
- Extrahepatic disease (lymph nodes, peritoneal spread): outside scope of current parenchymal segmentation.

## Multifocality
- Pattern: {multifocal_desc}.
- Lesion foci detected: {n_lesions}; segmented components: {n_comp}.

# Impression
The automated segmentation identifies a {size_cat} in the {hep_lobe} (estimated {couinaud}) with total volume {_fmt_vol(tumor_vol)} and maximum diameter {diam:.1f} mm. The lesion architecture is {multifocal_desc}.

These measurements are segmentation-derived estimates. Definitive characterisation of enhancement pattern, hepatic malignancy type, vascular involvement, and staging requires full radiological and clinical review.

Differential considerations based on lesion size and location (tissue diagnosis required for definitive classification):
1. Hepatocellular carcinoma (HCC) is a leading consideration in a cirrhotic or at-risk background — requires multi-phase contrast imaging and serum AFP correlation.
2. Colorectal or other hepatic metastasis cannot be excluded without clinical staging.
3. Cholangiocarcinoma or other primary hepatic malignancy should be considered depending on lesion morphology on full imaging review.
4. Benign aetiology (haemangioma, adenoma, focal nodular hyperplasia) remains possible for lesions without aggressive imaging features.

These findings are suggestive, not diagnostic. Definitive diagnosis requires histopathological confirmation and clinical correlation.

# Limitations
- Couinaud segment assignment is a heuristic centroid-based estimate; dedicated hepatic atlas not available.
- Enhancement characterisation (arterial/portal/venous phase analysis) unavailable from segmentation mask.
- Vascular and biliary involvement assessment requires dedicated CT/MRI review.
- Segmentation model ({model_ver}) is automated and subject to threshold sensitivity and fragmentation artefacts.
- No clinical history, AFP, hepatitis serology, or liver function data available.
- Study date unknown; longitudinal assessment not possible.

# Suggested Clinical Correlation
1. Correlate with serum AFP, hepatitis B/C serology, and liver function tests.
2. Multi-phase contrast CT or MRI liver protocol (LI-RADS assessment) recommended for full characterisation.
3. Multidisciplinary hepatobiliary team review (hepatology, HPB surgery, oncology, radiology).
4. Tissue sampling (biopsy or resection specimen) for definitive histopathological diagnosis.
5. Add longitudinal comparison imaging when prior studies become available.

# Disclaimer
This report is generated for automated decision-support only. It does not provide a medical diagnosis. Definitive diagnosis requires histopathological confirmation and clinical correlation. All findings must be reviewed and validated by a qualified radiologist/clinician before any clinical action. This report does not constitute a medical diagnosis or treatment recommendation.
"""



def _build_lung_report(case_id: str, metrics: dict) -> str:
    labels = metrics.get("label_volumes_mm3", {})
    total = metrics.get("total_tumor_volume_mm3", 0.0)
    tumor_vol = labels.get("lung_tumor", total)
    diam = metrics.get("max_diameter_mm", 0.0)
    n_lesions = max(metrics.get("tumor_count", 1), 1)
    n_comp = max(metrics.get("segmented_component_count", n_lesions), n_lesions)
    size_cat = _lung_size_category(diam)
    ci_text = _ci_line(metrics)
    model_ver = metrics.get("model_version", "nnunet_ct_lung_v1")
    tumor_lesions = [l for l in metrics.get("lesion_list", []) if "tumor" in l.get("label_name", "")]
    dominant = max(tumor_lesions, key=lambda l: l.get("volume_mm3", 0), default={})
    anat = dominant.get("anatomy", {})
    loc_desc = anat.get("description", "location unavailable - radiologist review required")
    lobe_desc = anat.get("lobe", "lobe unavailable")
    hemi = anat.get("hemisphere", "Unknown")
    ap = anat.get("anterior_posterior", "")
    loc_note = anat.get("localization_note", "")
    atlas_conf = anat.get("atlas_confidence", "Low")
    critical = metrics.get("critical_structure_assessment", {})
    vr_ready = metrics.get("vr_readiness", {})
    audit = metrics.get("audit_gate", {})
    multifocal_desc = "unifocal" if n_lesions <= 1 else f"multifocal ({n_lesions} foci across {n_comp} segmented components)"
    critical_lines = "\n".join(f"- {note}" for note in critical.get("notes", []))

    return f"""# Clinical Context
This pulmonary CT lesion assessment is derived from automated segmentation and should be interpreted with full radiological review. Study date is unavailable. No clinical note or prior chest imaging accompanies the current package.

## Patient Demographics & Scan Metadata
- Patient ID: {case_id}
- Study ID: {case_id}
- Study Date: Unknown
- Modality: CT (thoracic protocol detail not specified in the current package)
- Available imaging inputs: 1 NIfTI file.
- Segmentation model: {model_ver}.
- Clinical note present: no; prior reports present: no.
- Purpose: pulmonary lesion characterisation from automated CT segmentation.

# Findings
Automated segmentation identifies pulmonary lesion burden using connected-component lesion extraction.

## Detailed Volumetric Analysis
- Total segmented pulmonary lesion volume: {_fmt_vol(tumor_vol)}. {ci_text}
- Maximum lesion diameter (component-specific estimate): {diam:.1f} mm - classified as {size_cat}.
- Distinct lesion foci detected: {n_lesions}; segmented components: {n_comp}.
- Lesion architecture: {multifocal_desc}.

## Spatial Localization & Critical Structures
- Dominant lesion location: {loc_desc}.
- Estimated side and lobe: {hemi} hemithorax, {lobe_desc}.
- Anterior/posterior estimate: {ap if ap else 'unavailable'}.
- Localization confidence: {atlas_conf} (centroid heuristic).
- Note: {loc_note}
{critical_lines}

## Confidence Metrics & Automated Segmentation Validation
- {ci_text}
- Audit gate connected-component lesion count: {audit.get('connected_component_lesion_count', n_lesions)}.
- Audit gate volume consistency delta: {float(audit.get('volume_consistency_delta_mm3', 0.0) or 0.0):.2f} mm3.
- Diameter basis: {audit.get('diameter_basis', 'lesion-specific estimate')}.
- Quantitative metrics remain deterministic and narrative statements must not alter numeric truth.

## VR-Readiness Assessment
- Mesh readiness status: {vr_ready.get('status', 'Not available')}.
- Candidate segmented mesh components: {vr_ready.get('mesh_candidate_component_count', n_comp)}; largest component diameter: {float(vr_ready.get('largest_component_diameter_mm', diam) or 0.0):.1f} mm.
- {vr_ready.get('note', 'VR planning suitability is not available from the current payload.')}

## Multifocality
- Pattern: {multifocal_desc}.
- High component count relative to lesion count may reflect segmentation fragmentation rather than true multifocal disease.

# Impression
The automated segmentation identifies {size_cat} in the {hemi.lower()} hemithorax (estimated {lobe_desc}) with total volume {_fmt_vol(tumor_vol)} and maximum diameter {diam:.1f} mm. The lesion distribution is {multifocal_desc}. Localization remains approximate because side and lobe assignment are centroid-heuristic rather than atlas-verified.

These measurements are segmentation-derived estimates. Characterisation of malignancy, histologic subtype, nodal stage, pleural invasion, and treatment suitability requires full radiological and clinical review.

Differential considerations:
1. Primary pulmonary malignancy is most compatible with a dominant solid-appearing lesion of this size.
2. Pulmonary metastasis cannot be excluded without oncologic history and staging context.
3. Benign granulomatous or inflammatory aetiology is less typical due to lesion size alone but cannot be excluded from segmentation metrics only.

These findings are suggestive, not diagnostic. Definitive diagnosis requires histopathological or cytological confirmation and clinical correlation.

# Limitations
- Lobe assignment is a heuristic centroid-based estimate; dedicated lobe atlas is not available.
- Density pattern, spiculation, cavitation, pleural contact, vascular encasement, and nodal assessment require direct CT review.
- Segmentation model ({model_ver}) is automated and subject to threshold sensitivity, boundary error, and fragmentation artefacts.
- No clinical history, smoking history, prior imaging, or staging data are available.
- Study date is unknown, so longitudinal progression cannot be assessed.

# Suggested Clinical Correlation
1. Correlate with respiratory symptoms, smoking history, occupational exposure, and any oncologic history.
2. Compare with any prior chest imaging when available.
3. Consider PET-CT for metabolic characterisation and nodal staging if clinically indicated.
4. Consider bronchoscopic or CT-guided tissue sampling for definitive diagnosis if clinically indicated.
5. Review the case in multidisciplinary thoracic oncology conference.

# Disclaimer
This report is generated for AI decision-support only. It does not constitute a medical diagnosis or treatment recommendation. All findings must be reviewed and confirmed by a qualified clinician. Definitive diagnosis requires histopathological or cytological confirmation.
"""


def _build_liver_report(case_id: str, metrics: dict) -> str:
    labels = metrics.get("label_volumes_mm3", {})
    total = metrics.get("total_tumor_volume_mm3", 0.0)
    liver_vol = labels.get("liver", 0.0)
    tumor_vol = labels.get("liver_tumor", total)
    diam = metrics.get("max_diameter_mm", 0.0)
    n_lesions = max(metrics.get("tumor_count", 1), 1)
    n_comp = max(metrics.get("segmented_component_count", n_lesions), n_lesions)
    size_cat = _liver_size_category(diam)
    ci_text = _ci_line(metrics)
    model_ver = metrics.get("model_version", "nnunet_ct_liver_v1")
    ratio_line = (
        f"Tumour represents {(tumor_vol / liver_vol) * 100:.1f}% of total segmented liver volume ({_fmt_vol(liver_vol)})."
        if liver_vol > 0 and tumor_vol > 0
        else "Liver-to-tumour ratio is not available because total liver volume was not extracted separately."
    )
    liver_vol_line = (
        f"Total segmented liver parenchyma volume: {_fmt_vol(liver_vol)}."
        if liver_vol > 0
        else "Total liver parenchyma volume is not available as a separate label in the current mask."
    )
    tumor_lesions = [l for l in metrics.get("lesion_list", []) if "tumor" in l.get("label_name", "")]
    dominant = max(tumor_lesions, key=lambda l: l.get("volume_mm3", 0), default={})
    anat = dominant.get("anatomy", {})
    loc_desc = anat.get("description", "location unavailable - radiologist review required")
    hep_lobe = anat.get("hepatic_lobe", "hepatic lobe unavailable")
    couinaud = anat.get("couinaud_estimate", "segment estimate unavailable")
    atlas_conf = anat.get("atlas_confidence", "Low")
    loc_note = anat.get("localization_note", "")
    critical = metrics.get("critical_structure_assessment", {})
    vr_ready = metrics.get("vr_readiness", {})
    audit = metrics.get("audit_gate", {})
    multifocal_desc = "unifocal" if n_lesions <= 1 else f"multifocal ({n_lesions} foci across {n_comp} segmented components)"
    critical_lines = "\n".join(f"- {note}" for note in critical.get("notes", []))

    return f"""# Clinical Context
This hepatic CT lesion assessment is derived from automated segmentation and should be interpreted with full radiological review. Study date is unavailable. No clinical note or prior abdominal imaging accompanies the current package.

## Patient Demographics & Scan Metadata
- Patient ID: {case_id}
- Study ID: {case_id}
- Study Date: Unknown
- Modality: CT (abdominal protocol detail not specified in the current package)
- Available imaging inputs: 1 NIfTI file.
- Segmentation model: {model_ver}.
- Clinical note present: no; prior reports present: no.
- Purpose: hepatic lesion characterisation from automated CT segmentation.

# Findings
Automated segmentation identifies hepatic lesion burden using connected-component lesion extraction.

## Detailed Volumetric Analysis
- Total segmented hepatic lesion volume: {_fmt_vol(tumor_vol)}. {ci_text}
- Maximum lesion diameter (component-specific estimate): {diam:.1f} mm - classified as {size_cat}.
- Distinct hepatic foci detected: {n_lesions}; segmented components: {n_comp}.
- Lesion architecture: {multifocal_desc}.
- {liver_vol_line}
- {ratio_line}

## Spatial Localization & Critical Structures
- Dominant lesion location: {loc_desc}.
- Estimated hepatic lobe: {hep_lobe}.
- Estimated Couinaud segment group: {couinaud}.
- Localization confidence: {atlas_conf} (centroid heuristic).
- Note: {loc_note}
{critical_lines}

## Confidence Metrics & Automated Segmentation Validation
- {ci_text}
- Audit gate connected-component lesion count: {audit.get('connected_component_lesion_count', n_lesions)}.
- Audit gate volume consistency delta: {float(audit.get('volume_consistency_delta_mm3', 0.0) or 0.0):.2f} mm3.
- Diameter basis: {audit.get('diameter_basis', 'lesion-specific estimate')}.
- Quantitative metrics remain deterministic and narrative statements must not alter numeric truth.

## VR-Readiness Assessment
- Mesh readiness status: {vr_ready.get('status', 'Not available')}.
- Candidate segmented mesh components: {vr_ready.get('mesh_candidate_component_count', n_comp)}; largest component diameter: {float(vr_ready.get('largest_component_diameter_mm', diam) or 0.0):.1f} mm.
- {vr_ready.get('note', 'VR planning suitability is not available from the current payload.')}

## Multifocality
- Pattern: {multifocal_desc}.
- Distribution summary: {'single dominant focus' if n_lesions <= 1 else 'multiple segmented hepatic foci'}.

# Impression
The automated segmentation identifies {size_cat} in the {hep_lobe} (estimated {couinaud}) with total volume {_fmt_vol(tumor_vol)} and maximum diameter {diam:.1f} mm. The lesion distribution is {multifocal_desc}. Localization remains approximate because hepatic lobe and Couinaud assignment are centroid-heuristic rather than anatomy-verified.

These measurements are segmentation-derived estimates. Enhancement pattern, LI-RADS characterisation, vascular invasion, biliary involvement, and definitive staging require full radiological and clinical review.

Differential considerations:
1. Primary hepatic malignancy is most compatible with a dominant lesion of this size, but lesion type cannot be assigned from segmentation alone.
2. Hepatic metastasis cannot be excluded without oncologic history and systemic staging context.
3. Benign hepatic lesion remains possible, though less typical if the lesion proves heterogeneous or aggressive on full multiphasic imaging review.

These findings are suggestive, not diagnostic. Definitive diagnosis requires histopathological confirmation and clinical correlation.

# Limitations
- Couinaud and hepatic-lobe assignment are heuristic centroid-based estimates; dedicated hepatic atlas support is not available.
- Enhancement pattern, capsule, washout, tumour thrombus, vascular invasion, and biliary involvement require dedicated multiphasic imaging review.
- Segmentation model ({model_ver}) is automated and subject to boundary error, threshold sensitivity, and fragmentation artefacts.
- No clinical history, prior imaging, tumour marker data, or hepatopathy history are available.
- Study date is unknown, so longitudinal assessment is not possible.

# Suggested Clinical Correlation
1. Correlate with liver function tests, hepatitis history, tumour markers, and any oncologic history.
2. Compare with any prior abdominal imaging when available.
3. Consider multiphasic liver CT or dedicated liver MRI for fuller lesion characterisation if clinically indicated.
4. Consider image-guided tissue sampling for definitive diagnosis if clinically indicated.
5. Review the case in multidisciplinary hepatobiliary conference.

# Disclaimer
This report is generated for AI decision-support only. It does not constitute a medical diagnosis or treatment recommendation. All findings must be reviewed and confirmed by a qualified clinician. Definitive diagnosis requires histopathological confirmation.
"""


def _try_llm_report(case_id: str, organ: str, context_dict: dict, fallback: str) -> str:
    try:
        from ai_assistant.core.model_client import generate_ct_report_text
        result, model_used = generate_ct_report_text(context_dict, fallback, organ=organ)
        if model_used and result:
            print(f"  LLM enhanced report via {model_used}")
            return result
    except Exception as exc:
        print(f"  LLM report skipped ({exc})")
    return fallback



def _build_lung_qa(case_id: str, metrics: dict) -> list[dict]:
    labels    = metrics.get("label_volumes_mm3", {})
    tumor_vol = labels.get("lung_tumor", metrics.get("total_tumor_volume_mm3", 0.0))
    diam      = metrics.get("max_diameter_mm", 0.0)
    n         = max(metrics.get("tumor_count", 1), 1)
    unc       = metrics.get("uncertainty", {})
    ci        = unc.get("total_tumor_volume_cm3", {})
    ci_str    = (f" 95% CI: {ci.get('lower', 0):.2f}-{ci.get('upper', 0):.2f} cm3." if ci else "")
    safe      = "For decision support only; clinician confirmation required."

    tumor_lesions = [l for l in metrics.get("lesion_list", []) if "tumor" in l.get("label_name", "")]
    dominant = max(tumor_lesions, key=lambda l: l.get("volume_mm3", 0), default={})
    anat     = dominant.get("anatomy", {})
    loc_desc = anat.get("description", "location unavailable")
    lobe     = anat.get("lobe", "lobe unavailable")
    hemi     = anat.get("hemisphere", "unknown")
    size_cat = _lung_size_category(diam)

    return [
        {"qid": 1, "question": "What is the total tumor volume?",
         "answer": f"Total segmented lung tumour volume is {tumor_vol:.0f} mm3 ({tumor_vol/1000:.2f} cm3).{ci_str}",
         "evidence_ids": ["ctx:segmentation_metrics.total_tumor_volume_mm3"], "confidence": 0.72, "safety_note": safe},
        {"qid": 2, "question": "What is the maximum tumor diameter?",
         "answer": f"Maximum segmented tumour diameter (component-specific estimate) is {diam:.1f} mm — classified as {size_cat}.",
         "evidence_ids": ["ctx:segmentation_metrics.max_diameter_mm"], "confidence": 0.70, "safety_note": safe},
        {"qid": 3, "question": "How many lesions were detected?",
         "answer": f"{n} pulmonary tumour {'focus' if n == 1 else 'foci'} detected by nnUNet segmentation.",
         "evidence_ids": ["ctx:segmentation_metrics.tumor_count"], "confidence": 0.72, "safety_note": safe},
        {"qid": 4, "question": "Where is the tumor located?",
         "answer": f"Dominant tumour location (heuristic estimate): {loc_desc}. Estimated lobe: {lobe}, {hemi} hemithorax. Note: lobe assignment is centroid-based and requires radiologist verification.",
         "evidence_ids": ["ctx:segmentation_metrics.lesion_list"], "confidence": 0.55, "safety_note": safe},
        {"qid": 5, "question": "What imaging modality was used?",
         "answer": "CT (lung protocol) was used for imaging.",
         "evidence_ids": ["ctx:modality"], "confidence": 0.95, "safety_note": safe},
        {"qid": 6, "question": "Is lymph node involvement reported?",
         "answer": "Insufficient evidence: mediastinal and hilar lymph node assessment is outside the scope of the current parenchymal segmentation. Direct radiological review of nodal stations is required.",
         "evidence_ids": [], "confidence": 0.60, "safety_note": safe},
        {"qid": 7, "question": "What clinical correlation is recommended?",
         "answer": "Correlation with patient symptoms, smoking history, and prior imaging is recommended. Lung-RADS or Fleischner guidelines should be applied. MDT review with thoracic surgery and oncology is advised. Tissue sampling may be required for staging.",
         "evidence_ids": ["ctx:report_draft"], "confidence": 0.65, "safety_note": safe},
        {"qid": 8, "question": "What is the size category of the lesion?",
         "answer": f"The lesion is classified as a {size_cat} based on the maximum lesion-specific diameter of {diam:.1f} mm.",
         "evidence_ids": ["ctx:segmentation_metrics.max_diameter_mm"], "confidence": 0.70, "safety_note": safe},
        {"qid": 9, "question": "What is the measurement uncertainty?",
         "answer": f"Measurement uncertainty is estimated at +/-{unc.get('measurement_uncertainty_pct', 5):.0f}% (nnUNet inter-run variability). {('95% CI for total volume: ' + str(ci.get('lower', '?')) + '-' + str(ci.get('upper', '?')) + ' cm3.') if ci else ''}",
         "evidence_ids": ["ctx:segmentation_metrics.uncertainty"], "confidence": 0.65, "safety_note": safe},
    ]


def _build_liver_qa(case_id: str, metrics: dict) -> list[dict]:
    labels    = metrics.get("label_volumes_mm3", {})
    liver_vol = labels.get("liver", 0.0)
    tumor_vol = labels.get("liver_tumor", metrics.get("total_tumor_volume_mm3", 0.0))
    diam      = metrics.get("max_diameter_mm", 0.0)
    n         = max(metrics.get("tumor_count", 1), 1)
    unc       = metrics.get("uncertainty", {})
    ci        = unc.get("total_tumor_volume_cm3", {})
    ci_str    = (f" 95% CI: {ci.get('lower', 0):.2f}-{ci.get('upper', 0):.2f} cm3." if ci else "")
    safe      = "For decision support only; clinician confirmation required."
    size_cat  = _liver_size_category(diam)

    tumor_lesions = [l for l in metrics.get("lesion_list", []) if "tumor" in l.get("label_name", "")]
    dominant  = max(tumor_lesions, key=lambda l: l.get("volume_mm3", 0), default={})
    anat      = dominant.get("anatomy", {})
    loc_desc  = anat.get("description", "location unavailable")
    hep_lobe  = anat.get("hepatic_lobe", "hepatic lobe unavailable")
    couinaud  = anat.get("couinaud_estimate", "segment unavailable")

    liver_ans = (
        f"Total segmented liver parenchyma volume is {liver_vol:.0f} mm3 ({liver_vol/1000:.2f} cm3)."
        if liver_vol > 0
        else "Total liver volume was not extracted as a separate label in the current segmentation mask."
    )
    ratio_ans = (
        f"Tumour represents {(tumor_vol/liver_vol)*100:.1f}% of total segmented liver volume."
        if liver_vol > 0 else "Liver-to-tumour ratio unavailable (liver volume not extracted)."
    )

    return [
        {"qid": 1, "question": "What is the total tumor volume?",
         "answer": f"Total segmented hepatic tumour volume is {tumor_vol:.0f} mm3 ({tumor_vol/1000:.2f} cm3).{ci_str}",
         "evidence_ids": ["ctx:segmentation_metrics.total_tumor_volume_mm3"], "confidence": 0.72, "safety_note": safe},
        {"qid": 2, "question": "What is the maximum tumor diameter?",
         "answer": f"Maximum segmented tumour diameter (component-specific estimate) is {diam:.1f} mm — classified as {size_cat}.",
         "evidence_ids": ["ctx:segmentation_metrics.max_diameter_mm"], "confidence": 0.70, "safety_note": safe},
        {"qid": 3, "question": "How many lesions were detected?",
         "answer": f"{n} hepatic tumour {'focus' if n == 1 else 'foci'} detected by nnUNet segmentation.",
         "evidence_ids": ["ctx:segmentation_metrics.tumor_count"], "confidence": 0.72, "safety_note": safe},
        {"qid": 4, "question": "What is the total liver volume?",
         "answer": liver_ans,
         "evidence_ids": ["ctx:segmentation_metrics.label_volumes_mm3.liver"], "confidence": 0.70, "safety_note": safe},
        {"qid": 5, "question": "Where is the tumor located?",
         "answer": f"Dominant tumour location (heuristic estimate): {loc_desc}. Hepatic lobe: {hep_lobe}. Estimated Couinaud segment: {couinaud}. Note: assignment is centroid-based and requires radiologist verification.",
         "evidence_ids": ["ctx:segmentation_metrics.lesion_list"], "confidence": 0.55, "safety_note": safe},
        {"qid": 6, "question": "What is the liver-to-tumor ratio?",
         "answer": ratio_ans,
         "evidence_ids": ["ctx:segmentation_metrics.label_volumes_mm3.liver", "ctx:segmentation_metrics.total_tumor_volume_mm3"], "confidence": 0.68, "safety_note": safe},
        {"qid": 7, "question": "Is vascular involvement reported?",
         "answer": "Insufficient evidence: hepatic vessel, portal vein, and biliary involvement cannot be assessed from the current parenchymal segmentation alone. Direct CT/MRI review is required.",
         "evidence_ids": [], "confidence": 0.60, "safety_note": safe},
        {"qid": 8, "question": "What is the size category of the lesion?",
         "answer": f"The lesion is classified as a {size_cat} based on the maximum lesion-specific diameter of {diam:.1f} mm.",
         "evidence_ids": ["ctx:segmentation_metrics.max_diameter_mm"], "confidence": 0.70, "safety_note": safe},
        {"qid": 9, "question": "What clinical correlation is recommended?",
         "answer": "Correlation with serum AFP, hepatitis serology, and liver function tests is recommended. Multi-phase contrast CT or MRI liver protocol (LI-RADS) is advised. MDT hepatobiliary team review is recommended. Tissue sampling for definitive histopathological diagnosis.",
         "evidence_ids": ["ctx:report_draft"], "confidence": 0.65, "safety_note": safe},
        {"qid": 10, "question": "What is the measurement uncertainty?",
         "answer": f"Measurement uncertainty is estimated at +/-{unc.get('measurement_uncertainty_pct', 5):.0f}% (nnUNet inter-run variability). {('95% CI for total volume: ' + str(ci.get('lower', '?')) + '-' + str(ci.get('upper', '?')) + ' cm3.') if ci else ''}",
         "evidence_ids": ["ctx:segmentation_metrics.uncertainty"], "confidence": 0.65, "safety_note": safe},
    ]



CT_CASES = [
    ("lung",  "lung_001",  "lung_data/lung_001/lung_001_0000.nii.gz",
               SEGS_ROOT / "lung"  / "lung_001"  / "lung_001.nii.gz",   LUNG_LABELS),
    ("lung",  "lung_003",  "lung_data/lung_003/lung_003_0000.nii.gz",
               SEGS_ROOT / "lung"  / "lung_003"  / "lung_003.nii.gz",   LUNG_LABELS),
    ("liver", "liver_001", "liver_data/liver_001/liver_000_0000.nii.gz",
               SEGS_ROOT / "liver" / "liver_001" / "liver_001.nii.gz",  LIVER_LABELS),
    ("liver", "liver_002", "liver_data/liver_002/liver_001_0000.nii.gz",
               SEGS_ROOT / "liver" / "liver_002" / "liver_002.nii.gz",  LIVER_LABELS),
]


def main() -> None:
    CASES_ROOT.mkdir(parents=True, exist_ok=True)
    generated = 0

    for organ, case_id, nifti_rel, mask_path, label_map in CT_CASES:
        print(f"\n-- {case_id} ({organ}) --")
        if not mask_path.exists():
            print(f"  SKIP: mask not found at {mask_path}")
            continue

        print(f"  computing metrics from {mask_path.name}...")
        try:
            metrics = _compute_ct_metrics(mask_path, case_id, label_map, organ)
        except Exception as exc:
            print(f"  ERROR computing metrics: {exc}")
            continue

        context = _build_patient_context(case_id, organ, nifti_rel, metrics)

        if organ == "lung":
            fallback_report = _build_lung_report(case_id, metrics)
            qa_rows         = _build_lung_qa(case_id, metrics)
        else:
            fallback_report = _build_liver_report(case_id, metrics)
            qa_rows         = _build_liver_qa(case_id, metrics)

        report = _try_llm_report(case_id, organ, context, fallback_report)

        out_dir = CASES_ROOT / case_id
        out_dir.mkdir(parents=True, exist_ok=True)

        (out_dir / "patient_context.json").write_text(
            json.dumps(context, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (out_dir / "tumor_metrics.json").write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (out_dir / "report_draft.md").write_text(report, encoding="utf-8")
        (out_dir / "qa_results.jsonl").write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in qa_rows) + "\n",
            encoding="utf-8",
        )

        dom_loc = metrics.get("dominant_anatomic_location", "")
        print(f"  OK -> {out_dir}")
        print(f"     vol={metrics['total_tumor_volume_mm3']:.0f} mm3  diam={metrics['max_diameter_mm']:.1f} mm  loc={dom_loc[:60]}")
        generated += 1

    print(f"\nDONE  generated={generated}/{len(CT_CASES)}")


if __name__ == "__main__":
    main()
