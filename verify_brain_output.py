import numpy as np
import nibabel as nib
from pathlib import Path

ROOT       = Path(__file__).parent
BRAIN_OUT  = ROOT / "outputs" / "_inference_temp" / "brain_output"
BRAIN_GT   = ROOT / "outputs" / "_inference_temp" / "brain_gt"

print("=" * 55)
print("BRAIN INFERENCE OUTPUT CHECK")
print("=" * 55)

pred_files = sorted(BRAIN_OUT.glob("*.nii.gz"))
if not pred_files:
    print("ERROR: No .nii.gz files found in brain_output/")
    exit(1)

all_ok = True
for pf in pred_files:
    nii   = nib.load(str(pf))
    arr   = nii.get_fdata(dtype=np.float32).astype(np.int32)
    shape = arr.shape
    unique, counts = np.unique(arr, return_counts=True)
    total_vox  = arr.size
    tumour_vox = total_vox - counts[unique == 0][0] if 0 in unique else total_vox
    tumour_pct = 100.0 * tumour_vox / total_vox

    has_labels = any(u != 0 for u in unique)

    gt_path = BRAIN_GT / (pf.stem.replace(".nii", "") + ".nii")
    if not gt_path.exists():
        gt_path = BRAIN_GT / (pf.name.replace(".gz", ""))
    gt_str = "GT NOT FOUND"
    if gt_path.exists():
        gt_nii  = nib.load(str(gt_path))
        gt_arr  = gt_nii.get_fdata(dtype=np.float32).astype(np.int32)
        gt_unique, gt_counts = np.unique(gt_arr, return_counts=True)
        gt_tumour = gt_arr.size - gt_counts[gt_unique == 0][0] if 0 in gt_unique else gt_arr.size
        gt_str = f"GT tumour: {gt_tumour:,} vox  labels={sorted(gt_unique.tolist())}"

    status = "OK" if has_labels else "WARNING: Background only (0)!"
    print(f"\n{pf.name}  ({pf.stat().st_size // 1024} KB)")
    print(f"  Shape   : {shape}")
    print(f"  Labels  : {sorted(unique.tolist())}  counts={counts.tolist()}")
    print(f"  Tumour  : {tumour_vox:,} vox  ({tumour_pct:.2f}%)")
    print(f"  {gt_str}")
    print(f"  => {status}")
    if not has_labels:
        all_ok = False

print("\n" + "=" * 55)
if all_ok:
    print("RESULT: All predictions contain tumour labels — inference SUCCESSFUL")
else:
    print("WARNING: Some predictions are entirely background.")
    print("  Possible cause: model failed to read brain_input files")
    print("  correctly, or model/data mismatch.")
print("=" * 55)