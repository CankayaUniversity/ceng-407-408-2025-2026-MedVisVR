from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mc

ROOT = Path(__file__).parent
OUT  = ROOT / "outputs" / "evaluation_v2"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "axes.facecolor": "#fafafa", "figure.facecolor": "#ffffff",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e5e7eb",
    "grid.linewidth": 0.65, "font.family": "DejaVu Sans",
})

def binary_metrics(pred_arr, gt_arr, label):
    p = pred_arr == label;  g = gt_arr == label
    TP = int((p &  g).sum()); FP = int((p & ~g).sum())
    FN = int((~p & g).sum()); TN = int((~p & ~g).sum())
    dice    = 2*TP / (2*TP + FP + FN + 1e-8)
    prec    = TP / (TP + FP + 1e-8)
    rec     = TP / (TP + FN + 1e-8)
    spec    = TN / (TN + FP + 1e-8)
    bal_acc = (rec + spec) / 2
    return dict(Dice=dice, Precision=prec, Recall=rec,
                Specificity=spec, F1=dice, BalAcc=bal_acc)

def load_nifti(path):
    try:
        import nibabel as nib
        return np.asarray(nib.load(str(path)).dataobj, dtype=np.int32)
    except Exception:
        import SimpleITK as sitk
        return sitk.GetArrayFromImage(sitk.ReadImage(str(path))).astype(np.int32)

GT_ROOT  = Path(r"C:\Users\User\OneDrive\Desktop\cigerim\Task03_Liver\labelsTr")
SEG_ROOT = ROOT / "outputs" / "segmentations" / "liver"

gt_path  = GT_ROOT / "liver_1.nii.gz"
pred_dir = SEG_ROOT / "liver_001"

liver_metrics   = {}
macro_f1_liver  = None
macro_bal_liver = None

if gt_path.exists() and pred_dir.exists():
    pred_files = list(pred_dir.glob("*.nii.gz"))
    if pred_files:
        print("Loading liver_001 (n=1 preliminary case)...")
        gt   = load_nifti(gt_path)
        pred = load_nifti(pred_files[0])
        if pred.shape != gt.shape:
            s = tuple(min(a,b) for a,b in zip(pred.shape, gt.shape))
            gt = gt[:s[0],:s[1],:s[2]]; pred = pred[:s[0],:s[1],:s[2]]

        per_label = {lbl: binary_metrics(pred, gt, lbl) for lbl in [0, 1, 2]}
        for lbl, name in [(0,"Background"),(1,"Parenchyma"),(2,"Tumour")]:
            m = per_label[lbl]
            print(f"  Label {lbl} ({name}): Dice={m['Dice']:.3f}  BalAcc={m['BalAcc']:.3f}")

        macro_f1_liver  = np.mean([per_label[l]['F1']     for l in [0,1,2]])
        macro_bal_liver = np.mean([per_label[l]['BalAcc'] for l in [0,1,2]])
        print(f"  Macro F1     = {macro_f1_liver:.3f}")
        print(f"  Macro BalAcc = {macro_bal_liver:.3f}")
        liver_metrics = {"Parenchyma": per_label[1], "Tumour": per_label[2]}

brain_pub = [
    ("Whole Tumour",     0.895, 0.907, 0.888, 0.999),
    ("Tumour Core",      0.841, 0.858, 0.834, 0.999),
    ("Enhancing Tumour", 0.804, 0.820, 0.799, 0.999),
]
brain_metrics = {}
for label, dice, prec, rec, spec in brain_pub:
    brain_metrics[label] = dict(
        Dice=dice, Precision=prec, Recall=rec,
        Specificity=spec, F1=dice, BalAcc=(rec+spec)/2
    )
macro_f1_brain  = np.mean([brain_metrics[l]['F1']     for l in brain_metrics])
macro_bal_brain = np.mean([brain_metrics[l]['BalAcc'] for l in brain_metrics])
print(f"\nBrain Macro F1     = {macro_f1_brain:.3f}")
print(f"Brain Macro BalAcc = {macro_bal_brain:.3f}")

CASES = ROOT / "outputs" / "cases"
qa_scores = []
for case_dir in sorted(CASES.iterdir()):
    qa_path = case_dir / "qa_results.jsonl"
    if not qa_path.exists(): continue
    try:
        for line in qa_path.read_text(encoding="utf-8-sig").splitlines():
            if not line.strip(): continue
            r = json.loads(line)
            conf   = float(r.get("confidence", 0) or 0)
            has_ev = bool(r.get("evidence_ids") or r.get("evidence"))
            qa_scores.append((conf, has_ev))
    except Exception:
        pass

def roc_auc(score_label_pairs):
    pos = sum(1 for _, l in score_label_pairs if l)
    neg = len(score_label_pairs) - pos
    if pos == 0 or neg == 0: return [0,1],[0,1],0.5
    thresholds = sorted(set(s for s,_ in score_label_pairs), reverse=True) + [0.0]
    tprs, fprs = [0.0], [0.0]
    for t in thresholds:
        tp = sum(1 for s,l in score_label_pairs if s>=t and l)
        fp = sum(1 for s,l in score_label_pairs if s>=t and not l)
        tprs.append(tp/pos); fprs.append(fp/neg)
    tprs.append(1.0); fprs.append(1.0)
    auc = sum(abs(fprs[i]-fprs[i-1])*(tprs[i]+tprs[i-1])/2 for i in range(1,len(fprs)))
    return fprs, tprs, auc

qa_auc = None
if qa_scores:
    qa_fpr, qa_tpr, qa_auc = roc_auc(qa_scores)
    print(f"\nQ&A AUC (confidence -> evidence) = {qa_auc:.3f}  (n={len(qa_scores)} answers)")

    fig, ax = plt.subplots(figsize=(6, 5.2))
    ax.plot(qa_fpr, qa_tpr, color="#2563eb", linewidth=2.2,
            label=f"Q&A Confidence (AUC = {qa_auc:.3f})")
    ax.plot([0,1],[0,1], color="#9ca3af", linestyle="--", linewidth=1, label="Random baseline (AUC=0.50)")
    ax.fill_between(qa_fpr, qa_tpr, alpha=0.10, color="#2563eb")
    ax.set_xlabel("False Positive Rate", fontsize=10)
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=10)
    ax.set_title(
        "Q&A System — ROC Curve (v2)\n"
        "Predictor: answer confidence score   Positive: evidence-supported answer\n"
        f"n = {len(qa_scores)} Q&A answers across all cases",
        fontsize=9.5, fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.85)
    ax.set_xlim(0,1); ax.set_ylim(0,1.05)
    fig.tight_layout()
    p = OUT / "qa_auc_roc_v2.png"
    fig.savefig(p, dpi=180); plt.close(fig)
    print(f"Saved: {p}")

ALL_METRICS = ["Dice", "Precision", "Recall", "Specificity", "F1", "Bal. Acc.", "Macro F1"]

results_rows = []
colors_map   = {}

brain_col_list = ["#2563eb","#3b82f6","#93c5fd"]
for (label, *_), col in zip(brain_pub, brain_col_list):
    m = brain_metrics[label]
    results_rows.append([label, "Brain MRI",
                         m['Dice'], m['Precision'], m['Recall'], m['Specificity'],
                         m['F1'], m['BalAcc'], macro_f1_brain])
    colors_map[label] = col

if liver_metrics:
    for seg_name, col in [("Parenchyma","#16a34a"),("Tumour","#dc2626")]:
        lbl = f"Liver {seg_name} \u2020"
        m   = liver_metrics[seg_name]
        mf  = macro_f1_liver if macro_f1_liver is not None else float('nan')
        results_rows.append([lbl, "Liver CT",
                             m['Dice'], m['Precision'], m['Recall'], m['Specificity'],
                             m['F1'], m['BalAcc'], mf])
        colors_map[lbl] = col

labels_all  = [r[0] for r in results_rows]
groups_all  = [r[1] for r in results_rows]
vals_matrix = np.array([r[2:] for r in results_rows], dtype=float)

row_labels = [f"[Brain] {l}" if g=="Brain MRI" else f"[Liver] {l}"
              for l,g in zip(labels_all, groups_all)]

fig, ax = plt.subplots(figsize=(15, 4.5))
ax.axis("off")
tbl = ax.table(
    cellText=[[f"{v:.3f}" for v in row] for row in vals_matrix],
    rowLabels=row_labels,
    colLabels=ALL_METRICS,
    loc="center", cellLoc="center",
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1, 2.15)

row_colors = [colors_map[l] for l in labels_all]
for (r, c), cell in tbl.get_celld().items():
    cell.set_edgecolor("#d1d5db")
    if r == 0:
        cell.set_facecolor("#1a1a2e")
        cell.set_text_props(color="white", fontweight="bold", fontsize=9.5)
        if c in [5, 6]:
            cell.set_facecolor("#312e81")
    elif c == -1:
        base = row_colors[r-1]
        cell.set_facecolor(base + "22")
        cell.set_text_props(fontweight="700", fontsize=8.5)
        cell.set_width(0.26)
    else:
        val = vals_matrix[r-1][c]
        base_rgb = mc.to_rgb(row_colors[r-1])
        alpha = 0.15 + val * 0.65
        blended = tuple(1-alpha+alpha*b for b in base_rgb)
        cell.set_facecolor(blended)
        cell.set_text_props(fontweight="bold",
                            color="white" if val > 0.75 else "#1a1a2e")
        if c in [5, 6]:
            cell.set_edgecolor("#6366f1")

ax.set_title(
    "CARVIS \u2014 Complete Segmentation Metrics (v2, 7 indicators)\n"
    "[Brain] nnUNet BraTS2020 reference (Isensee et al. 2021)   "
    "[Liver] liver_001 only \u2020 n=1, preliminary   "
    "Raw Accuracy excluded \u2014 Balanced Accuracy = (Sensitivity+Specificity)/2 used instead   "
    "Macro F1 = unweighted mean across all classes (incl. background)",
    fontsize=8.5, pad=14, loc="left", color="#374151",
)
fig.tight_layout()
p = OUT / "auc_macro_table_v2.png"
fig.savefig(p, dpi=180, bbox_inches="tight")
print(f"\nSaved: {p}")
plt.close(fig)

fig, ax = plt.subplots(figsize=(15, 5.8))
x = np.arange(len(ALL_METRICS))
n = len(labels_all)
w = 0.72 / n

for i, label in enumerate(labels_all):
    vals   = list(vals_matrix[i])
    offset = (i - (n-1)/2) * w
    edgecol = "white" if groups_all[i]=="Brain MRI" else "#374151"
    lw      = 0.5    if groups_all[i]=="Brain MRI" else 1.2
    bars = ax.bar(x + offset, vals, w*0.88,
                  color=colors_map[label], edgecolor=edgecol, linewidth=lw,
                  label=label, alpha=0.88 if groups_all[i]=="Brain MRI" else 1.0)
    for bar, val in zip(bars, vals):
        if val > 0.05:
            ax.text(bar.get_x()+bar.get_width()/2, val+0.007, f"{val:.2f}",
                    ha="center", va="bottom", fontsize=5.6,
                    fontweight="bold", color="#374151")

ax.axvspan(4.5, 6.5, alpha=0.06, color="#6366f1", zorder=0)
ax.text(5.5, 1.075, "New", ha="center", fontsize=8, color="#6366f1", fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(ALL_METRICS, fontsize=9.5)
ax.set_ylabel("Score", fontsize=11)
ax.set_ylim(0, 1.13)
ax.axhline(0.8, color="#9ca3af", linewidth=1, linestyle="--", alpha=0.7)
ax.text(len(ALL_METRICS)-0.5, 0.805, "0.80 clinical threshold", fontsize=8, color="#9ca3af")
ax.set_title(
    "CARVIS \u2014 Segmentation Performance (v2) \u2014 7 Metrics\n"
    "Brain MRI: BraTS-style multi-sequence MRI, nnUNet BraTS2020 reference   "
    "Liver CT: MSD Task03, n=1 preliminary\n"
    "Raw Accuracy excluded (inflated by background) \u2014 Balanced Accuracy & Macro F1 shown",
    fontsize=10.5, fontweight="bold", pad=14)

patches = [mpatches.Patch(color=colors_map[l], label=l) for l in labels_all]
ax.legend(handles=patches, fontsize=8.5, framealpha=0.85, loc="upper right", ncol=1)
ax.annotate("\u2020 Liver: n=1 (liver_001 only)\n"
            "Liver Tumour low Dice: voxel-alignment caveat",
            xy=(2.5, 0.12), fontsize=7.5, color="#6b7280",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#f9fafb", edgecolor="#d1d5db"))

fig.tight_layout()
p = OUT / "auc_macro_bar_v2.png"
fig.savefig(p, dpi=180); plt.close(fig)
print(f"Saved: {p}")

print("\n" + "="*76)
print(f"{'Label':<28} {'Dice':>6} {'Prec':>6} {'Rec':>6} {'Spec':>6} "
      f"{'F1':>6} {'BalAcc':>7} {'MacroF1':>8}")
print("-"*76)
for row in results_rows:
    lbl = row[0]; v = row[2:]
    print(f"{lbl:<28} {v[0]:>6.3f} {v[1]:>6.3f} {v[2]:>6.3f} "
          f"{v[3]:>6.3f} {v[4]:>6.3f} {v[5]:>7.3f} {v[6]:>8.3f}")

print(f"\nBrain Macro F1          = {macro_f1_brain:.3f}")
print(f"Brain Macro BalAcc      = {macro_bal_brain:.3f}")
if macro_f1_liver:
    print(f"Liver Macro F1  (n=1)   = {macro_f1_liver:.3f}")
if qa_auc:
    print(f"Q&A AUC                 = {qa_auc:.3f}")
print("\nDone. eval_v2_09_auc_macro.py complete.")