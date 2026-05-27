from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as mpatches
import numpy as np

ROOT = Path(__file__).parent
OUT  = ROOT / "outputs" / "evaluation"
OUT.mkdir(parents=True, exist_ok=True)

BRAIN_DIR = ROOT / "nnunet_brain"
LIVER_DIR = ROOT / "nnunet_liver" / "nnUNetTrainerV2__nnUNetPlansv2.1"
LUNG_DIR  = ROOT / "nnunet_lung"

def get_folds(base_dir):
    folds = []
    for d in sorted(base_dir.iterdir()):
        if d.is_dir() and d.name.startswith("fold_"):
            p = d / "progress.png"
            if p.exists():
                folds.append((d.name, p))
    return folds

brain_folds = get_folds(BRAIN_DIR)
liver_folds = get_folds(LIVER_DIR)
lung_folds  = get_folds(LUNG_DIR) if LUNG_DIR.exists() else []

print(f"Brain folds with progress.png: {len(brain_folds)}")
print(f"Liver folds with progress.png: {len(liver_folds)}")
print(f"Lung  folds with progress.png: {len(lung_folds)}")

if brain_folds:
    n = len(brain_folds)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.5, rows * 7))
    axes = np.array(axes).flatten()

    for i, (fold_name, png_path) in enumerate(brain_folds):
        img = mpimg.imread(str(png_path))
        axes[i].imshow(img)
        axes[i].set_title(f"Brain MRI — {fold_name.replace('_',' ').title()}",
                          fontsize=11, fontweight="bold", pad=8)
        axes[i].axis("off")

    for j in range(i+1, len(axes)):
        axes[j].axis("off")

    fig.suptitle(
        "nnUNet Brain MRI Segmentation — Training Curves (5-Fold Cross-Validation)\n"
        "Model: nnUNet v2  |  Dataset: BraTS2020  |  1000 Epochs  |  "
        "Blue=Train Loss  Red=Val Loss  Green=Pseudo Dice",
        fontsize=11, fontweight="bold", y=1.01
    )
    fig.patch.set_facecolor("white")
    fig.tight_layout()
    p = OUT / "training_brain_all_folds.png"
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved: {p}")
    plt.close(fig)

if liver_folds:
    n = len(liver_folds)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.5, rows * 5))
    axes = np.array(axes).flatten()

    for i, (fold_name, png_path) in enumerate(liver_folds):
        img = mpimg.imread(str(png_path))
        axes[i].imshow(img)
        axes[i].set_title(f"Liver CT — {fold_name.replace('_',' ').title()}",
                          fontsize=11, fontweight="bold", pad=8)
        axes[i].axis("off")

    for j in range(i+1, len(axes)):
        axes[j].axis("off")

    fig.suptitle(
        "nnUNet Liver CT Segmentation — Training Curves (5-Fold Cross-Validation)\n"
        "Model: nnUNet v1 (TrainerV2)  |  Dataset: MSD Task03_Liver  |  1000 Epochs  |  "
        "Blue=Train Loss  Red=Val Loss  Green=Dice",
        fontsize=11, fontweight="bold", y=1.01
    )
    fig.patch.set_facecolor("white")
    fig.tight_layout()
    p = OUT / "training_liver_all_folds.png"
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved: {p}")
    plt.close(fig)

fig = plt.figure(figsize=(17, 8))
fig.patch.set_facecolor("white")

gs = fig.add_gridspec(2, 3,
                      width_ratios=[2.2, 2.2, 1.2],
                      height_ratios=[1, 1],
                      hspace=0.35, wspace=0.25)

ax_brain = fig.add_subplot(gs[:, 0])
if brain_folds:
    img = mpimg.imread(str(brain_folds[0][1]))
    ax_brain.imshow(img)
ax_brain.axis("off")
ax_brain.set_title("Brain MRI — Fold 0\n(nnUNet v2, BraTS2020)", fontsize=12,
                   fontweight="bold", color="#1d4ed8", pad=10)

ax_liver = fig.add_subplot(gs[:, 1])
if liver_folds:
    img = mpimg.imread(str(liver_folds[0][1]))
    ax_liver.imshow(img)
ax_liver.axis("off")
ax_liver.set_title("Liver CT — Fold 0\n(nnUNet v1 TrainerV2, Task03_Liver)", fontsize=12,
                   fontweight="bold", color="#15803d", pad=10)

ax_info_top = fig.add_subplot(gs[0, 2])
ax_info_top.axis("off")

specs = [
    ["",              "Brain MRI",          "Liver CT",           "Lung CT"],
    ["Framework",     "nnUNet v2",          "nnUNet v1",          "nnUNet v2"],
    ["Architecture",  "3D U-Net",           "3D U-Net",           "3D U-Net"],
    ["Dataset",       "BraTS2020",          "MSD Task03",         "MSD Task06"],
    ["Classes",       "3 sub-regions",      "Parenchyma+Tumour",  "Lung nodule"],
    ["Epochs",        "1000",               "1000",               "—"],
    ["Folds",         "5-fold CV",          "5-fold CV",          "fold 0 only"],
    ["Optimizer",     "SGD + poly LR",      "SGD + poly LR",      "SGD + poly LR"],
    ["Patch size",    "128³",               "128³",               "—"],
    ["Training log",  "Available",          "Available",          "Not available"],
]

tbl = ax_info_top.table(
    cellText=[r[1:] for r in specs[1:]],
    rowLabels=[r[0] for r in specs[1:]],
    colLabels=specs[0][1:],
    loc="center", cellLoc="center",
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(8)
tbl.scale(1, 1.55)
for (r, c), cell in tbl.get_celld().items():
    cell.set_edgecolor("#e5e7eb")
    if r == 0:
        if c == 0:
            cell.set_facecolor("#1a1a2e")
            cell.set_text_props(color="white", fontweight="bold")
        elif c == 1:
            cell.set_facecolor("#1d4ed8")
            cell.set_text_props(color="white", fontweight="bold")
        elif c == 2:
            cell.set_facecolor("#15803d")
            cell.set_text_props(color="white", fontweight="bold")
        elif c == 3:
            cell.set_facecolor("#b45309")
            cell.set_text_props(color="white", fontweight="bold")
    elif c == -1:
        cell.set_facecolor("#f8fafc")
        cell.set_text_props(fontweight="600", fontsize=7.5)
    if c == 2 and r > 0:
        txt = cell.get_text().get_text()
        if txt in ("—", "Not available", "fold 0 only"):
            cell.set_facecolor("#fef9c3")

ax_info_top.set_title("Model Specifications", fontsize=10, fontweight="bold", pad=10)

ax_info_bot = fig.add_subplot(gs[1, 2])
ax_info_bot.axis("off")
note = (
    "Lung CT Model\n"
    "─────────────────\n"
    "Framework : nnUNet v2\n"
    "Checkpoint: fold_0 final\n"
    "Training log: Not saved\n"
    "(pre-trained model\n"
    " downloaded & used\n"
    " for inference only)\n\n"
    "Inference results:\n"
    "  lung_001, lung_003\n"
    "  Segmentation maps\n"
    "  generated & saved"
)
ax_info_bot.text(0.05, 0.95, note, transform=ax_info_bot.transAxes,
                 fontsize=8.5, va="top", ha="left",
                 family="monospace",
                 bbox=dict(boxstyle="round,pad=0.6", facecolor="#fef9c3",
                           edgecolor="#ca8a04", linewidth=1.5))

fig.suptitle(
    "CARVIS — Segmentation Model Training Overview\n"
    "nnUNet Self-Configuring Framework | 1000 Epochs | 5-Fold Cross-Validation",
    fontsize=13, fontweight="bold", y=1.01
)
p = OUT / "training_combined.png"
fig.savefig(p, dpi=160, bbox_inches="tight", facecolor="white")
print(f"Saved: {p}")
plt.close(fig)

print("\nDone! eval_10_training_curves.py complete.")
print(f"\nOutputs in: {OUT}")
print("  training_brain_all_folds.png   -- Brain 5-fold training grid")
print("  training_liver_all_folds.png   -- Liver 5-fold training grid")
print("  training_combined.png          -- Poster main (Brain + Liver + specs table)")
