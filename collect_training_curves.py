import shutil
from pathlib import Path

ROOT     = Path(__file__).parent
OUT_DIR  = ROOT / "outputs" / "evaluation_real" / "training_curves"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BRAIN_ROOT = ROOT / "nnunet_brain"
LIVER_ROOT = ROOT / "nnunet_liver" / "nnUNetTrainerV2__nnUNetPlansv2.1"
LUNG_ROOT  = Path(r"C:\Users\User\OneDrive\Desktop\nnUNet_data\nnUNet_results\Dataset601_LungTumor\nnUNetTrainer_250epochs__nnUNetPlans__3d_fullres")

copied = []

print("=" * 55)
print("Collecting training curves → evaluation_real/training_curves/")
print("=" * 55)

print("\n[BRAIN]")
for fold in range(5):
    src = BRAIN_ROOT / f"fold_{fold}" / "progress.png"
    if src.exists():
        dst = OUT_DIR / f"brain_fold{fold}_progress.png"
        shutil.copy2(src, dst)
        print(f"  OK  fold_{fold} → {dst.name}  ({src.stat().st_size // 1024} KB)")
        copied.append(dst.name)
    else:
        print(f"  MISS  fold_{fold}: {src}")

print("\n[LIVER]")
for fold in range(5):
    src = LIVER_ROOT / f"fold_{fold}" / "progress.png"
    if src.exists():
        dst = OUT_DIR / f"liver_fold{fold}_progress.png"
        shutil.copy2(src, dst)
        print(f"  OK  fold_{fold} → {dst.name}  ({src.stat().st_size // 1024} KB)")
        copied.append(dst.name)
    else:
        print(f"  MISS  fold_{fold}: {src}")

print("\n[LUNG]")
src = LUNG_ROOT / "fold_0" / "progress.png"
if src.exists():
    dst = OUT_DIR / "lung_fold0_progress.png"
    shutil.copy2(src, dst)
    print(f"  OK  fold_0 → {dst.name}  ({src.stat().st_size // 1024} KB)")
    print(f"  NOTE: Lung model trained 250 epochs, fold_0 only (single fold)")
    copied.append(dst.name)
else:
    print(f"  MISS: {src}")

print(f"\nTotal: {len(copied)} file(s) copied → {OUT_DIR}")
print("\nContents:")
for f in sorted(OUT_DIR.iterdir()):
    print(f"  {f.name}  ({f.stat().st_size // 1024} KB)")
