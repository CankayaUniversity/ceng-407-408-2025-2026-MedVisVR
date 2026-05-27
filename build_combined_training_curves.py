from pathlib import Path
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

ROOT    = Path(__file__).parent
OUT_DIR = ROOT / "outputs" / "evaluation_real" / "training_curves"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BRAIN_ROOT = ROOT / "nnunet_brain"
LIVER_ROOT = ROOT / "nnunet_liver" / "nnUNetTrainerV2__nnUNetPlansv2.1"
LUNG_ROOT  = Path(r"C:\Users\User\OneDrive\Desktop\nnUNet_data\nnUNet_results\Dataset601_LungTumor\nnUNetTrainer_250epochs__nnUNetPlans__3d_fullres")

ORGANS = [
    {
        "name":    "Brain MRI",
        "key":     "brain",
        "color":   "#2563eb",
        "model":   "nnUNetv2  |  1000 epochs  |  5-fold CV",
        "folds":   [BRAIN_ROOT / f"fold_{i}" / "progress.png" for i in range(5)],
        "out":     OUT_DIR / "brain_training_curves.png",
    },
    {
        "name":    "Liver CT",
        "key":     "liver",
        "color":   "#16a34a",
        "model":   "nnUNetv1 TrainerV2  |  1000 epochs  |  5-fold CV",
        "folds":   [LIVER_ROOT / f"fold_{i}" / "progress.png" for i in range(5)],
        "out":     OUT_DIR / "liver_training_curves.png",
    },
    {
        "name":    "Lung CT",
        "key":     "lung",
        "color":   "#b45309",
        "model":   "nnUNetv2  |  250 epochs  |  fold_0 only",
        "folds":   [LUNG_ROOT / "fold_0" / "progress.png"],
        "out":     OUT_DIR / "lung_training_curves.png",
    },
]

print("=" * 55)
print("Building combined training curves")
print("=" * 55)

for organ in ORGANS:
    folds    = organ["folds"]
    n_folds  = len(folds)
    existing = [p for p in folds if p.exists()]

    if not existing:
        print(f"\n[{organ['name']}] SKIP — no progress.png found")
        continue

    print(f"\n[{organ['name']}]  {len(existing)}/{n_folds} folds found")

    ncols = len(existing)
    nrows = 1

    fig = plt.figure(figsize=(6 * ncols, 5.5))

    fig.suptitle(
        f"{organ['name']} — Training Curves\n{organ['model']}",
        fontsize=14, fontweight="bold", color=organ["color"], y=1.02
    )

    for idx, p in enumerate(existing):
        ax = fig.add_subplot(nrows, ncols, idx + 1)
        img = mpimg.imread(str(p))
        ax.imshow(img)
        ax.axis("off")
        fold_num = p.parent.name
        ax.set_title(fold_num, fontsize=11, fontweight="bold",
                     color=organ["color"], pad=4)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(str(organ["out"]), dpi=130,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)

    kb = organ["out"].stat().st_size // 1024
    print(f"  → {organ['out'].name}  ({kb} KB)")

print(f"\nDone. Output: {OUT_DIR}")
print("\nFolder contents:")
for f in sorted(OUT_DIR.iterdir()):
    print(f"  {f.name}  ({f.stat().st_size // 1024} KB)")
