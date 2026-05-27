from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

ROOT = Path(__file__).parent
OUT  = ROOT / "outputs" / "evaluation_v2"
OUT.mkdir(parents=True, exist_ok=True)

BRAIN_DIR = ROOT / "nnunet_brain"
LIVER_DIR = ROOT / "nnunet_liver" / "nnUNetTrainerV2__nnUNetPlansv2.1"
LUNG_DIR  = ROOT / "nnunet_lung"

def get_folds(base_dir):
    folds = []
    if not base_dir.exists(): return folds
    for d in sorted(base_dir.iterdir()):
        if d.is_dir() and d.name.startswith("fold_"):
            p = d / "progress.png"
            if p.exists(): folds.append((d.name, p))
    return folds

brain_folds = get_folds(BRAIN_DIR)
liver_folds = get_folds(LIVER_DIR)
print(f"Brain folds: {len(brain_folds)}  Liver folds: {len(liver_folds)}")

if brain_folds:
    n = len(brain_folds); cols = min(n,3); rows = (n+cols-1)//cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols*5.5, rows*7))
    axes = np.array(axes).flatten()
    for i,(fold_name,png) in enumerate(brain_folds):
        axes[i].imshow(mpimg.imread(str(png)))
        axes[i].set_title(f"Brain MRI \u2014 {fold_name.replace('_',' ').title()}",
                          fontsize=11, fontweight="bold", pad=8)
        axes[i].axis("off")
    for j in range(i+1, len(axes)): axes[j].axis("off")
    fig.suptitle(
        "nnUNet Brain MRI Segmentation \u2014 Training Curves (5-Fold CV)\n"
        "Model: nnUNet v2  |  Dataset: BraTS-style multi-sequence MRI (BraTS2020)  |  "
        "1000 Epochs  |  Blue=Train Loss  Red=Val Loss  Green=Pseudo Dice",
        fontsize=11, fontweight="bold", y=1.01)
    fig.patch.set_facecolor("white")
    fig.tight_layout()
    p = OUT / "training_brain_all_folds_v2.png"
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved: {p}"); plt.close(fig)

if liver_folds:
    n = len(liver_folds); cols = min(n,3); rows = (n+cols-1)//cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols*5.5, rows*5))
    axes = np.array(axes).flatten()
    for i,(fold_name,png) in enumerate(liver_folds):
        axes[i].imshow(mpimg.imread(str(png)))
        axes[i].set_title(f"Liver CT \u2014 {fold_name.replace('_',' ').title()}",
                          fontsize=11, fontweight="bold", pad=8)
        axes[i].axis("off")
    for j in range(i+1, len(axes)): axes[j].axis("off")
    fig.suptitle(
        "nnUNet Liver CT Segmentation \u2014 Training Curves (5-Fold CV)\n"
        "Model: nnUNet v1 (TrainerV2)  |  Dataset: MSD Task03_Liver  |  "
        "1000 Epochs  |  Blue=Train Loss  Red=Val Loss  Green=Dice",
        fontsize=11, fontweight="bold", y=1.01)
    fig.patch.set_facecolor("white")
    fig.tight_layout()
    p = OUT / "training_liver_all_folds_v2.png"
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved: {p}"); plt.close(fig)

fig = plt.figure(figsize=(17, 8))
fig.patch.set_facecolor("white")
gs = fig.add_gridspec(2, 3, width_ratios=[2.2,2.2,1.2],
                      height_ratios=[1,1], hspace=0.35, wspace=0.25)

ax_brain = fig.add_subplot(gs[:,0])
if brain_folds:
    ax_brain.imshow(mpimg.imread(str(brain_folds[0][1])))
ax_brain.axis("off")
ax_brain.set_title("Brain MRI \u2014 Fold 0\n"
                   "(nnUNet v2, BraTS-style multi-sequence MRI)",
                   fontsize=12, fontweight="bold", color="#1d4ed8", pad=10)

ax_liver = fig.add_subplot(gs[:,1])
if liver_folds:
    ax_liver.imshow(mpimg.imread(str(liver_folds[0][1])))
ax_liver.axis("off")
ax_liver.set_title("Liver CT \u2014 Fold 0\n"
                   "(nnUNet v1 TrainerV2, MSD Task03_Liver)",
                   fontsize=12, fontweight="bold", color="#15803d", pad=10)

ax_info = fig.add_subplot(gs[0,2])
ax_info.axis("off")
specs = [
    ["",               "Brain MRI",             "Liver CT",             "Lung CT"],
    ["Framework",      "nnUNet v2",              "nnUNet v1",            "nnUNet v2"],
    ["Architecture",   "3D U-Net",               "3D U-Net",             "3D U-Net"],
    ["Dataset",        "BraTS2020",              "MSD Task03",           "MSD Task06"],
    ["Task",           "Tumour segm.",            "Liver + tumour",       "Lung tumour (NSCLC)"],
    ["Epochs",         "1000",                   "1000",                 "\u2014"],
    ["Folds",          "5-fold CV",              "5-fold CV",            "fold 0 only"],
    ["Anat. labelling","Anatomical regions",     "Anatomical regions",   "\u2014"],
    ["Training log",   "Available",              "Available",            "Not available"],
]
tbl = ax_info.table(
    cellText=[r[1:] for r in specs[1:]],
    rowLabels=[r[0] for r in specs[1:]],
    colLabels=specs[0][1:],
    loc="center", cellLoc="center",
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(7.5)
tbl.scale(1, 1.55)
for (r,c), cell in tbl.get_celld().items():
    cell.set_edgecolor("#e5e7eb")
    if r == 0:
        colors_ = {0:"#1a1a2e", 1:"#1d4ed8", 2:"#15803d", 3:"#b45309"}
        cell.set_facecolor(colors_.get(c, "#1a1a2e"))
        cell.set_text_props(color="white", fontweight="bold")
    elif c == -1:
        cell.set_facecolor("#f8fafc")
        cell.set_text_props(fontweight="600", fontsize=7)
    if c == 2 and r > 0:
        if cell.get_text().get_text() in ("\u2014","Not available","fold 0 only"):
            cell.set_facecolor("#fef9c3")
ax_info.set_title("Model Specifications", fontsize=10, fontweight="bold", pad=10)

ax_note = fig.add_subplot(gs[1,2])
ax_note.axis("off")
note = (
    "Lung CT Model\n"
    "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
    "Framework : nnUNet v2\n"
    "Checkpoint: fold_0 final\n"
    "Training log: Not saved\n"
    "(Pre-trained model used\n"
    " for inference only)\n\n"
    "Task: Lung tumour\n"
    "(NSCLC, MSD Task06)\n\n"
    "Inference cases:\n"
    "  lung_001, lung_003\n"
    "  No GT available"
)
ax_note.text(0.05, 0.95, note, transform=ax_note.transAxes,
             fontsize=8, va="top", ha="left", family="monospace",
             bbox=dict(boxstyle="round,pad=0.6", facecolor="#fef9c3",
                       edgecolor="#ca8a04", linewidth=1.5))

fig.suptitle(
    "CARVIS \u2014 Segmentation Model Training Overview (v2)\n"
    "nnUNet Self-Configuring Framework | 1000 Epochs | 5-Fold Cross-Validation",
    fontsize=13, fontweight="bold", y=1.01)
p = OUT / "training_combined_v2.png"
fig.savefig(p, dpi=160, bbox_inches="tight", facecolor="white")
print(f"Saved: {p}"); plt.close(fig)

print("\nDone. eval_v2_10_training.py complete.")