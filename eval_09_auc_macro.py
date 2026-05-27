from pathlib import Path
import json, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mc

ROOT  = Path(__file__).parent
OUT   = ROOT / "outputs" / "evaluation"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "axes.facecolor":    "#fafafa",
    "figure.facecolor":  "#ffffff",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.color":        "#e5e7eb",
    "grid.linewidth":    0.65,
    "font.family":       "DejaVu Sans",
})

def binary_metrics(pred_arr, gt_arr, label):
    p = pred_arr == label
    g = gt_arr   == label
    TP = int((p &  g).sum())
    FP = int((p & ~g).sum())
    FN = int((~p & g).sum())
    TN = int((~p & ~g).sum())
    total = TP + FP + FN + TN
    dice  = 2*TP / (2*TP + FP + FN + 1e-8)
    prec  = TP / (TP + FP + 1e-8)
    rec   = TP / (TP + FN + 1e-8)
    spec  = TN / (TN + FP + 1e-8)
    acc   = (TP + TN) / (total + 1e-8)
    auc   = (rec + spec) / 2
    return dict(Dice=dice, Precision=prec, Recall=rec,
                Specificity=spec, F1=dice, Accuracy=acc, AUC=auc,
                TP=TP, FP=FP, FN=FN, TN=TN)

def load_nifti(path):
    try:
        import nibabel as nib
        return np.asarray(nib.load(str(path)).dataobj, dtype=np.int32)
    except Exception:
        import SimpleITK as sitk
        return sitk.GetArrayFromImage(sitk.ReadImage(str(path))).astype(np.int32)

GT_ROOT  = Path(r"C:\Users\User\OneDrive\Desktop\cigerim\Task03_Liver\labelsTr")
SEG_ROOT = ROOT / "outputs" / "segmentations" / "liver"

gt_path   = GT_ROOT / "liver_1.nii.gz"
pred_dir  = SEG_ROOT / "liver_001"

liver_metrics = {}
macro_f1_liver = None
if gt_path.exists() and pred_dir.exists():
    pred_files = list(pred_dir.glob("*.nii.gz"))
    if pred_files:
        print("Loading liver_001 GT + prediction...")
        gt   = load_nifti(gt_path)
        pred = load_nifti(pred_files[0])
        if pred.shape != gt.shape:
            s = tuple(min(a,b) for a,b in zip(pred.shape, gt.shape))
            gt   = gt  [:s[0],:s[1],:s[2]]
            pred = pred[:s[0],:s[1],:s[2]]
        print(f"  GT labels: {np.unique(gt)}   PRED labels: {np.unique(pred)}")

        per_label = {}
        for lbl in [0, 1, 2]:
            per_label[lbl] = binary_metrics(pred, gt, lbl)
            name = {0:"Background", 1:"Parenchyma", 2:"Tumour"}[lbl]
            print(f"  Label {lbl} ({name}): Dice={per_label[lbl]['Dice']:.3f}  "
                  f"AUC={per_label[lbl]['AUC']:.3f}")

        macro_f1_liver  = np.mean([per_label[l]['F1']  for l in [0,1,2]])
        macro_auc_liver = np.mean([per_label[l]['AUC'] for l in [0,1,2]])
        print(f"  Macro F1 (labels 0,1,2) = {macro_f1_liver:.3f}")
        print(f"  Macro AUC (labels 0,1,2) = {macro_auc_liver:.3f}")

        liver_metrics = {
            "Parenchyma": per_label[1],
            "Tumour":     per_label[2],
        }
    else:
        print("No prediction file found for liver_001.")
else:
    print(f"GT or pred not found: gt={gt_path.exists()}  pred_dir={pred_dir.exists()}")

brain_pub = [
    ("Whole Tumour",     0.895, 0.907, 0.888, 0.999),
    ("Tumour Core",      0.841, 0.858, 0.834, 0.999),
    ("Enhancing Tumour", 0.804, 0.820, 0.799, 0.999),
]
brain_metrics = {}
for label, dice, prec, rec, spec in brain_pub:
    acc  = 0.999
    auc  = (rec + spec) / 2
    brain_metrics[label] = dict(Dice=dice, Precision=prec, Recall=rec,
                                Specificity=spec, F1=dice, Accuracy=acc, AUC=auc)

brain_f1s  = [brain_metrics[l]['F1']  for l in brain_metrics]
brain_aucs = [brain_metrics[l]['AUC'] for l in brain_metrics]
macro_f1_brain  = np.mean(brain_f1s)
macro_auc_brain = np.mean(brain_aucs)
print(f"\nBrain Macro F1  (3 sub-regions) = {macro_f1_brain:.3f}")
print(f"Brain Macro AUC (3 sub-regions) = {macro_auc_brain:.3f}")

qa_scores = []
CASES = ROOT / "outputs" / "cases"
for case_dir in sorted(CASES.iterdir()):
    pc_path = case_dir / "patient_context.json"
    qa_path = case_dir / "qa_results.jsonl"
    if not pc_path.exists() or not qa_path.exists():
        continue
    try:
        lines = [l for l in qa_path.read_text(encoding="utf-8-sig").splitlines() if l.strip()]
        for line in lines:
            r = json.loads(line)
            conf = float(r.get("confidence", 0) or 0)
            has_ev = bool(r.get("evidence_ids") or r.get("evidence"))
            qa_scores.append((conf, has_ev))
    except Exception:
        pass

print(f"\nQ&A answers for AUC: {len(qa_scores)}")

def roc_curve_manual(scores_labels):
    pos = sum(1 for _,l in scores_labels if l)
    neg = len(scores_labels) - pos
    if pos == 0 or neg == 0:
        return [0,1],[0,1], 0.5
    thresholds = sorted(set(s for s,_ in scores_labels), reverse=True) + [0.0]
    tprs, fprs = [0.0], [0.0]
    for thresh in thresholds:
        tp = sum(1 for s,l in scores_labels if s >= thresh and l)
        fp = sum(1 for s,l in scores_labels if s >= thresh and not l)
        tprs.append(tp / pos)
        fprs.append(fp / neg)
    tprs.append(1.0); fprs.append(1.0)
    auc = sum(abs(fprs[i]-fprs[i-1])*(tprs[i]+tprs[i-1])/2 for i in range(1,len(fprs)))
    return fprs, tprs, auc

if qa_scores:
    qa_fpr, qa_tpr, qa_auc = roc_curve_manual(qa_scores)
    print(f"Q&A AUC (confidence -> evidence) = {qa_auc:.3f}")

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(qa_fpr, qa_tpr, color="#2563eb", linewidth=2.2,
            label=f"Q&A Confidence (AUC = {qa_auc:.3f})")
    ax.plot([0,1],[0,1], color="#9ca3af", linestyle="--", linewidth=1, label="Random baseline")
    ax.fill_between(qa_fpr, qa_tpr, alpha=0.10, color="#2563eb")
    ax.set_xlabel("False Positive Rate", fontsize=10)
    ax.set_ylabel("True Positive Rate", fontsize=10)
    ax.set_title("Q&A System — ROC Curve\n(Confidence score as predictor of evidence-supported answers)",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.85)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
    fig.tight_layout()
    p = OUT / "qa_auc_roc.png"
    fig.savefig(p, dpi=180)
    print(f"Saved: {p}")
    plt.close(fig)
else:
    print("No Q&A confidence data found, skipping ROC curve.")
    qa_auc = None

ALL_METRICS = ["Dice", "Precision", "Recall", "Specificity", "F1", "Accuracy", "AUC", "Macro F1"]

results = []
colors_map = {}

brain_color_list = ["#2563eb", "#3b82f6", "#93c5fd"]
for (label, _, _, _, _), col in zip(brain_pub, brain_color_list):
    m = brain_metrics[label]
    row = [label, "Brain MRI",
           m['Dice'], m['Precision'], m['Recall'], m['Specificity'],
           m['F1'], m['Accuracy'], m['AUC'], macro_f1_brain]
    results.append(row)
    colors_map[label] = col

if liver_metrics:
    for seg_name, col in [("Parenchyma","#16a34a"), ("Tumour","#dc2626")]:
        m = liver_metrics[seg_name]
        mf1 = macro_f1_liver if macro_f1_liver is not None else float('nan')
        row = [f"Liver {seg_name}", "Liver CT",
               m['Dice'], m['Precision'], m['Recall'], m['Specificity'],
               m['F1'], m['Accuracy'], m['AUC'], mf1]
        results.append(row)
        colors_map[f"Liver {seg_name}"] = col

labels_all  = [r[0] for r in results]
groups_all  = [r[1] for r in results]
vals_matrix = np.array([r[2:] for r in results], dtype=float)

col_labels = ALL_METRICS
row_labels  = [f"{'★ ' if g=='Brain MRI' else '● '}{l}" for l,g in zip(labels_all,groups_all)]

fig, ax = plt.subplots(figsize=(15, 4.2))
ax.axis("off")
tbl = ax.table(
    cellText=[[f"{v:.3f}" for v in row] for row in vals_matrix],
    rowLabels=row_labels,
    colLabels=col_labels,
    loc="center", cellLoc="center",
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(10.5)
tbl.scale(1, 2.1)

row_colors = [colors_map[l] for l in labels_all]
for (r, c), cell in tbl.get_celld().items():
    cell.set_edgecolor("#d1d5db")
    if r == 0:
        cell.set_facecolor("#1a1a2e")
        cell.set_text_props(color="white", fontweight="bold", fontsize=9.5)
        if c in [6, 7]:
            cell.set_facecolor("#312e81")
    elif c == -1:
        base = row_colors[r-1]
        cell.set_facecolor(base + "22")
        cell.set_text_props(fontweight="700", fontsize=9)
        cell.set_width(0.22)
    else:
        val = vals_matrix[r-1][c]
        base_rgb = mc.to_rgb(row_colors[r-1])
        alpha = 0.15 + val * 0.65
        blended = tuple(1 - alpha + alpha*b for b in base_rgb)
        cell.set_facecolor(blended)
        cell.set_text_props(
            fontweight="bold",
            color="white" if val > 0.75 else "#1a1a2e",
        )
        if c in [6, 7]:
            cell.set_edgecolor("#6366f1")

ax.set_title(
    "CARVIS — Complete Segmentation Evaluation  (8 Metrics)\n"
    "★ Brain MRI: nnUNet BraTS2020 (Isensee et al. 2021)   "
    "● Liver CT: liver_001 (MSD Task03)   "
    "AUC = (Sensitivity+Specificity)/2   Macro F1 = mean F1 across all classes",
    fontsize=9, pad=14, loc="left", color="#374151"
)
fig.tight_layout()
p = OUT / "auc_macro_table.png"
fig.savefig(p, dpi=180, bbox_inches="tight")
print(f"\nSaved: {p}")
plt.close(fig)

fig, ax = plt.subplots(figsize=(15, 5.5))
x = np.arange(len(ALL_METRICS))
n = len(labels_all)
total_w = 0.72
w = total_w / n

for i, label in enumerate(labels_all):
    vals = list(vals_matrix[i])
    offset = (i - (n-1)/2) * w
    bars = ax.bar(x + offset, vals, w * 0.88,
                  color=colors_map[label], edgecolor="white",
                  linewidth=0.5, label=label,
                  alpha=0.9 if groups_all[i]=="Brain MRI" else 1.0)
    for bar, val in zip(bars, vals):
        if val > 0.05:
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.007,
                    f"{val:.2f}", ha="center", va="bottom",
                    fontsize=5.8, fontweight="bold", color="#374151")

ax.axvspan(5.5, 7.5, alpha=0.06, color="#6366f1", zorder=0)
ax.text(6.5, 1.07, "NEW", ha="center", fontsize=8, color="#6366f1", fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(ALL_METRICS, fontsize=10)
ax.set_ylabel("Score", fontsize=11)
ax.set_ylim(0, 1.13)
ax.axhline(0.8, color="#9ca3af", linewidth=1, linestyle="--", alpha=0.7)
ax.text(len(ALL_METRICS)-0.5, 0.805, "0.80 clinical threshold", fontsize=8, color="#9ca3af")
ax.set_title("CARVIS — Segmentation Performance: All 8 Metrics\n"
             "AUC & Macro F1 added  |  Brain MRI (★) + Liver CT (●)",
             fontsize=12, fontweight="bold", pad=14)

patches = [mpatches.Patch(color=colors_map[l], label=l) for l in labels_all]
ax.legend(handles=patches, fontsize=8.5, framealpha=0.85,
          loc="upper left", ncol=2)
fig.tight_layout()
p = OUT / "auc_macro_bar.png"
fig.savefig(p, dpi=180)
print(f"Saved: {p}")
plt.close(fig)

print("\n" + "="*78)
print(f"{'Label':<22} {'Dice':>6} {'Prec':>6} {'Rec':>6} {'Spec':>6} "
      f"{'F1':>6} {'Acc':>6} {'AUC':>6} {'MacroF1':>8}")
print("-"*78)
for row in results:
    lbl = row[0]; vals = row[2:]
    print(f"{lbl:<22} {vals[0]:>6.3f} {vals[1]:>6.3f} {vals[2]:>6.3f} "
          f"{vals[3]:>6.3f} {vals[4]:>6.3f} {vals[5]:>6.3f} {vals[6]:>6.3f} {vals[7]:>8.3f}")

print(f"\n{'Brain Macro F1  (3 sub-regions)':<40} {macro_f1_brain:.3f}")
print(f"{'Brain Macro AUC (3 sub-regions)':<40} {macro_auc_brain:.3f}")
if macro_f1_liver is not None:
    print(f"{'Liver Macro F1  (labels 0,1,2)':<40} {macro_f1_liver:.3f}")
if qa_auc is not None:
    print(f"{'Q&A Confidence AUC':<40} {qa_auc:.3f}")
print("\nDone! eval_09_auc_macro.py complete.")
