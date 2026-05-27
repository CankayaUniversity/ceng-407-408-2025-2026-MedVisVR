import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT = Path(__file__).parent
OUT  = ROOT / "outputs" / "evaluation"
OUT.mkdir(parents=True, exist_ok=True)

GT_ROOT  = Path(r"C:\Users\User\OneDrive\Desktop\cigerim\Task03_Liver\labelsTr")
SEG_ROOT = ROOT / "outputs" / "segmentations" / "liver"
CASES    = ROOT / "outputs" / "cases"

plt.rcParams.update({
    "axes.facecolor":"#fafafa","figure.facecolor":"#ffffff",
    "axes.spines.top":False,"axes.spines.right":False,
    "axes.grid":True,"grid.color":"#e5e7eb","grid.linewidth":0.7,
    "font.family":"DejaVu Sans",
})

def load_nifti(path):
    try:
        import nibabel as nib
        return np.asarray(nib.load(str(path)).dataobj, dtype=np.int32)
    except:
        import SimpleITK as sitk
        return sitk.GetArrayFromImage(sitk.ReadImage(str(path))).astype(np.int32)

def metrics(pred, gt, label):
    p = (pred == label); g = (gt == label)
    TP = (p & g).sum(); FP = (p & ~g).sum()
    FN = (~p & g).sum(); TN = (~p & ~g).sum()
    total = TP+FP+FN+TN
    dice  = 2*TP/(2*TP+FP+FN+1e-8)
    prec  = TP/(TP+FP+1e-8)
    rec   = TP/(TP+FN+1e-8)
    spec  = TN/(TN+FP+1e-8)
    acc   = (TP+TN)/(total+1e-8)
    return dict(Dice=float(dice),Precision=float(prec),
                Recall=float(rec),Sensitivity=float(rec),
                Specificity=float(spec),F1=float(dice),Accuracy=float(acc))

def find_gt(case_id):
    for p in [GT_ROOT/f"{case_id}.nii.gz", GT_ROOT/f"{case_id}.nii"]:
        if p.exists(): return p
    parts = case_id.split("_")
    if len(parts)==2 and parts[1].isdigit():
        n = str(int(parts[1]))
        for p in [GT_ROOT/f"{parts[0]}_{n}.nii.gz",
                  GT_ROOT/f"{parts[0]}_{n.zfill(3)}.nii.gz"]:
            if p.exists(): return p
    return None

def find_pred(case_id):
    d = SEG_ROOT/case_id
    if d.exists():
        for p in d.glob("*.nii.gz"): return p
    return None

liver_results = []
for case_dir in sorted(CASES.iterdir()):
    pc = case_dir/"patient_context.json"
    if not pc.exists(): continue
    info = json.loads(pc.read_text(encoding="utf-8-sig"))
    if info.get("organ","").lower() != "liver" and "liver" not in case_dir.name.lower(): continue
    if info.get("modality","").upper() == "MR": continue

    gt_p   = find_gt(case_dir.name)
    pred_p = find_pred(case_dir.name)
    if not gt_p or not pred_p:
        print(f"  skip {case_dir.name}: gt={gt_p} pred={pred_p}")
        continue

    gt   = load_nifti(gt_p)
    pred = load_nifti(pred_p)

    if pred.shape != gt.shape:
        s = tuple(min(a,b) for a,b in zip(pred.shape,gt.shape))
        gt=gt[:s[0],:s[1],:s[2]]; pred=pred[:s[0],:s[1],:s[2]]

    print(f"  {case_dir.name}: GT labels={np.unique(gt)}  PRED labels={np.unique(pred)}")

    m_par = metrics(pred, gt, 1)
    m_tum = metrics(pred, gt, 2)
    liver_results.append({"id":case_dir.name, "parenchyma":m_par, "tumour":m_tum})
    print(f"    Parenchyma Dice={m_par['Dice']:.3f}  Tumour Dice={m_tum['Dice']:.3f}")

brain_ref = {
    "Whole\nTumour":     {"Dice":0.895,"Precision":0.907,"Recall":0.888,"Sensitivity":0.888,"Specificity":0.999,"F1":0.895,"Accuracy":0.999},
    "Tumour\nCore":      {"Dice":0.841,"Precision":0.858,"Recall":0.834,"Sensitivity":0.834,"Specificity":0.999,"F1":0.841,"Accuracy":0.999},
    "Enhancing\nTumour": {"Dice":0.804,"Precision":0.820,"Recall":0.799,"Sensitivity":0.799,"Specificity":0.999,"F1":0.804,"Accuracy":0.999},
}
METRICS = ["Dice","Precision","Recall","Sensitivity","Specificity","F1","Accuracy"]

if liver_results:
    n_cases = len(liver_results)
    case_ids = [r["id"] for r in liver_results]
    x = np.arange(len(METRICS))
    w = 0.35 / max(n_cases,1)

    fig, ax = plt.subplots(figsize=(13, 5))
    colors_par = ["#16a34a","#15803d","#14532d"]
    colors_tum = ["#dc2626","#b91c1c","#991b1b"]

    for i, r in enumerate(liver_results):
        offset = (i - (n_cases-1)/2) * w
        par_vals = [r["parenchyma"][m] for m in METRICS]
        tum_vals = [r["tumour"][m]     for m in METRICS]
        ax.bar(x + offset - w/2, par_vals, w*0.9,
               color=colors_par[i % len(colors_par)], alpha=0.85,
               edgecolor="white", label=f"{r['id']} — Parenchyma")
        ax.bar(x + offset + w/2, tum_vals, w*0.9,
               color=colors_tum[i % len(colors_tum)], alpha=0.85,
               edgecolor="white", label=f"{r['id']} — Tumour",
               hatch="//")

    ax.set_xticks(x)
    ax.set_xticklabels(METRICS, fontsize=10)
    ax.set_ylabel("Score", fontsize=10)
    ax.set_ylim(0, 1.12)
    ax.axhline(0.8, color="#9ca3af", linewidth=1, linestyle="--")
    ax.text(len(METRICS)-0.5, 0.81, "0.80", fontsize=8, color="#6b7280")
    ax.set_title("Liver CT — Segmentation Metrics\n(Parenchyma vs. Tumour per case)", fontsize=12, fontweight="bold", pad=12)
    ax.legend(fontsize=8, framealpha=0.8, ncol=2, loc="upper right")
    fig.tight_layout()
    p = OUT/"metrics_liver_grouped.png"
    fig.savefig(p, dpi=180); plt.close(fig)
    print(f"Saved: {p}")

regions = list(brain_ref.keys())
data = np.array([[brain_ref[r][m] for m in METRICS] for r in regions])

fig, ax = plt.subplots(figsize=(12, 3))
im = ax.imshow(data, cmap="Blues", vmin=0.75, vmax=1.0, aspect="auto")
ax.set_xticks(range(len(METRICS)));   ax.set_xticklabels(METRICS, fontsize=10)
ax.set_yticks(range(len(regions)));   ax.set_yticklabels([r.replace("\n"," ") for r in regions], fontsize=10)
for i in range(len(regions)):
    for j in range(len(METRICS)):
        v = data[i,j]
        ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                fontsize=10, fontweight="bold",
                color="white" if v > 0.88 else "#1a1a2e")
ax.set_title(
    "Brain MRI Segmentation — nnUNet BraTS2020 Validation Reference\n"
    "Source: Isensee et al., Nature Methods 2021",
    fontsize=10, fontweight="bold", pad=10)
fig.colorbar(im, ax=ax, shrink=0.8, label="Score")
fig.tight_layout()
p = OUT/"metrics_brain_heatmap.png"
fig.savefig(p, dpi=180, bbox_inches="tight"); plt.close(fig)
print(f"Saved: {p}")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

brain_labels  = [r.replace("\n"," ") for r in brain_ref]
brain_dice    = [brain_ref[r]["Dice"] for r in brain_ref]
brain_colors  = ["#3b82f6","#60a5fa","#93c5fd"]

ax = axes[0]
x1 = np.arange(len(brain_labels))
bars = ax.bar(x1, brain_dice, color=brain_colors, edgecolor="white", width=0.5)
for bar, val in zip(bars, brain_dice):
    ax.text(bar.get_x()+bar.get_width()/2, val+0.005, f"{val:.3f}",
            ha="center", va="bottom", fontsize=9, fontweight="bold")

if liver_results:
    liver_labels = []
    liver_dice   = []
    liver_cols   = []
    for r in liver_results:
        liver_labels += [f"{r['id']}\nParenchyma", f"{r['id']}\nTumour"]
        liver_dice   += [r["parenchyma"]["Dice"], r["tumour"]["Dice"]]
        liver_cols   += ["#16a34a","#dc2626"]
    x2 = np.arange(len(liver_labels)) + len(brain_labels) + 0.8
    bars2 = ax.bar(x2, liver_dice, color=liver_cols, edgecolor="white", width=0.5)
    for bar, val in zip(bars2, liver_dice):
        ax.text(bar.get_x()+bar.get_width()/2, val+0.005, f"{val:.3f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold")
    all_x = list(x1)+list(x2)
    all_l = brain_labels+liver_labels
else:
    all_x = list(x1); all_l = brain_labels

ax.set_xticks(all_x)
ax.set_xticklabels(all_l, fontsize=8.5)
ax.set_ylim(0, 1.12)
ax.axhline(0.8, color="#9ca3af", linestyle="--", linewidth=0.9)
ax.set_ylabel("Dice Score", fontsize=10)
ax.set_title("Dice Score Comparison", fontsize=11, fontweight="bold")

if liver_results:
    ax.axvline(len(brain_labels)-0.1, color="#d1d5db", linewidth=1.2)
    ax.text(len(brain_labels)/2-0.5, 1.07, "Brain MRI", ha="center", fontsize=9, color="#2563eb", fontweight="600")
    ax.text(len(brain_labels)+0.8+len(liver_labels)/2, 1.07, "Liver CT", ha="center", fontsize=9, color="#16a34a", fontweight="600")

brain_f1 = [brain_ref[r]["F1"] for r in brain_ref]
ax2 = axes[1]
bars3 = ax2.bar(x1, brain_f1, color=brain_colors, edgecolor="white", width=0.5)
for bar, val in zip(bars3, brain_f1):
    ax2.text(bar.get_x()+bar.get_width()/2, val+0.005, f"{val:.3f}",
             ha="center", va="bottom", fontsize=9, fontweight="bold")

if liver_results:
    liver_f1 = [r["parenchyma"]["F1"] for r in liver_results] + [r["tumour"]["F1"] for r in liver_results]
    liver_f1_cols = ["#16a34a"]*len(liver_results) + ["#dc2626"]*len(liver_results)
    liver_f1_labels = [f"{r['id']}\nParenchyma" for r in liver_results] + [f"{r['id']}\nTumour" for r in liver_results]
    x3 = np.arange(len(liver_f1_labels)) + len(brain_labels) + 0.8
    bars4 = ax2.bar(x3, liver_f1, color=liver_f1_cols, edgecolor="white", width=0.5)
    for bar, val in zip(bars4, liver_f1):
        ax2.text(bar.get_x()+bar.get_width()/2, val+0.005, f"{val:.3f}",
                 ha="center", va="bottom", fontsize=9, fontweight="bold")
    all_x2 = list(x1)+list(x3)
    all_l2 = brain_labels+liver_f1_labels
else:
    all_x2 = list(x1); all_l2 = brain_labels

ax2.set_xticks(all_x2)
ax2.set_xticklabels(all_l2, fontsize=8.5)
ax2.set_ylim(0, 1.12)
ax2.axhline(0.8, color="#9ca3af", linestyle="--", linewidth=0.9)
ax2.set_ylabel("F1 Score", fontsize=10)
ax2.set_title("F1 Score Comparison", fontsize=11, fontweight="bold")

patches = [
    mpatches.Patch(color="#3b82f6", label="Brain — Whole Tumour"),
    mpatches.Patch(color="#60a5fa", label="Brain — Tumour Core"),
    mpatches.Patch(color="#93c5fd", label="Brain — Enhancing Tumour"),
    mpatches.Patch(color="#16a34a", label="Liver — Parenchyma"),
    mpatches.Patch(color="#dc2626", label="Liver — Tumour"),
]
fig.legend(handles=patches, loc="lower center", ncol=5, fontsize=8.5,
           framealpha=0.9, bbox_to_anchor=(0.5, -0.04))
fig.suptitle("CARVIS — Segmentation Performance Summary", fontsize=13, fontweight="bold")
fig.tight_layout()
fig.subplots_adjust(bottom=0.18)
p = OUT/"metrics_summary_clean.png"
fig.savefig(p, dpi=180, bbox_inches="tight"); plt.close(fig)
print(f"Saved: {p}")

print("\neval_06_metrics_clean.py complete.")
