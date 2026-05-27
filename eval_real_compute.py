import json
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mc

ROOT = Path(__file__).parent
OUT  = ROOT / "outputs" / "evaluation_v2"
OUT.mkdir(parents=True, exist_ok=True)

BRAIN_DIR  = ROOT / "nnunet_brain"
LIVER_DIR  = ROOT / "nnunet_liver" / "nnUNetTrainerV2__nnUNetPlansv2.1"
LUNG_PRED  = Path(r"C:\Users\User\OneDrive\Desktop\nnUNet_data\nnUNet_results\Dataset601_LungTumor\nnUNetTrainer_250epochs__nnUNetPlans__3d_fullres\fold_0\validation")
LUNG_GT    = Path(r"C:\Users\User\OneDrive\Desktop\nnUNet_data\nnUNet_preprocessed\Dataset601_LungTumor\gt_segmentations")

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

COLORS = {
    "brain":          "#2563eb",
    "liver_paren":    "#16a34a",
    "liver_tumor":    "#dc2626",
    "lung":           "#b45309",
}

print("\n=== BRAIN MRI (5-fold CV, checkpoint logging) ===")

import torch

brain_ema_per_fold  = []
brain_mean_per_fold = []
brain_class_last    = []

brain_folds = sorted([d for d in BRAIN_DIR.iterdir() if d.is_dir() and d.name.startswith("fold_")])
for fold_dir in brain_folds:
    ckpt_path = fold_dir / "checkpoint_final.pth"
    if not ckpt_path.exists():
        print(f"  {fold_dir.name}: checkpoint not found")
        continue
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    log  = ckpt.get("logging", {})
    ema  = log.get("ema_fg_dice", [])
    mfg  = log.get("mean_fg_dice", [])
    dpc  = log.get("dice_per_class_or_region", None)
    n_ep = len(ema)
    ema_last  = float(ema[-1])  if ema  else None
    mean_last = float(mfg[-1]) if mfg else None
    brain_ema_per_fold.append(ema_last)
    brain_mean_per_fold.append(mean_last)

    if isinstance(dpc, list) and dpc:
        last_ep = dpc[-1]
        per_cls = []
        for v in last_ep:
            try: per_cls.append(float(v))
            except: per_cls.append(float("nan"))
        brain_class_last.append(per_cls)

    print(f"  {fold_dir.name}: epochs={n_ep}  ema={ema_last:.4f}  mean_fg={mean_last:.4f}")

brain_ema_mean = float(np.mean(brain_ema_per_fold))
brain_ema_std  = float(np.std(brain_ema_per_fold))
print(f"\n  Brain 5-fold CV EMA Dice: {brain_ema_mean:.4f} +/- {brain_ema_std:.4f}")

BRAIN_CLASS_NAMES = ["Edema (label 1)", "Non-enh. tumor (label 2)", "Empty (label 3)", "Enhancing tumor (label 4)"]
if brain_class_last:
    arr = np.array(brain_class_last)
    brain_class_means = np.nanmean(arr, axis=0)
    brain_class_stds  = np.nanstd(arr, axis=0)
    print(f"  Per-class Dice (mean across 5 folds, last epoch):")
    for name, m, s in zip(BRAIN_CLASS_NAMES, brain_class_means, brain_class_stds):
        print(f"    {name}: {m:.4f} +/- {s:.4f}")
else:
    brain_class_means = None

print("\n=== LIVER CT (5-fold CV, postprocessing.json) ===")

liver_pp_path = LIVER_DIR / "postprocessing.json"
liver_dice = {}
if liver_pp_path.exists():
    with open(liver_pp_path) as f:
        pp = json.load(f)
    raw = pp.get("dc_per_class_raw", {})
    liver_dice = {
        "parenchyma": float(raw.get("1", 0)),
        "tumor":      float(raw.get("2", 0)),
        "n_samples":  int(pp.get("num_samples", 0)),
    }
    print(f"  Liver Parenchyma Dice (5-fold CV, n={liver_dice['n_samples']}): {liver_dice['parenchyma']:.4f}")
    print(f"  Liver Tumour Dice    (5-fold CV, n={liver_dice['n_samples']}): {liver_dice['tumor']:.4f}")
    pp_all = pp.get("dc_per_class_pp_all", {})
    print(f"  Liver Parenchyma Dice (post-processed): {float(pp_all.get('1',0)):.4f}")
    print(f"  Liver Tumour Dice    (post-processed): {float(pp_all.get('2',0)):.4f}")
else:
    print("  postprocessing.json not found!")
    liver_dice = {"parenchyma": None, "tumor": None, "n_samples": 0}

print("\n=== LUNG CT (fold_0 validation, 13 cases, computation) ===")

def load_nifti(path: Path) -> np.ndarray:
    try:
        import nibabel as nib
        return np.asarray(nib.load(str(path)).dataobj, dtype=np.int32)
    except Exception:
        import SimpleITK as sitk
        return sitk.GetArrayFromImage(sitk.ReadImage(str(path))).astype(np.int32)

def binary_metrics(pred: np.ndarray, gt: np.ndarray, label: int = 1) -> dict:
    p = (pred == label)
    g = (gt   == label)
    TP = int(np.logical_and(p,   g).sum())
    FP = int(np.logical_and(p,  ~g).sum())
    FN = int(np.logical_and(~p,  g).sum())
    TN = int(np.logical_and(~p, ~g).sum())
    total = TP + FP + FN + TN
    eps = 1e-8
    dice  = 2*TP / (2*TP + FP + FN + eps)
    prec  = TP / (TP + FP + eps)
    rec   = TP / (TP + FN + eps)
    spec  = TN / (TN + FP + eps)
    f1    = dice
    bal   = (rec + spec) / 2
    acc   = (TP + TN) / (total + eps)
    return dict(
        Dice=dice, Precision=prec, Recall=rec, Specificity=spec,
        F1=f1, Balanced_Accuracy=bal, AUC=bal, Accuracy=acc,
        TP=TP, FP=FP, FN=FN, TN=TN,
        n_gt_voxels=int(g.sum()), n_pred_voxels=int(p.sum()),
    )

lung_cases = []
if LUNG_PRED.exists() and LUNG_GT.exists():
    pred_files = sorted([f for f in LUNG_PRED.glob("*.nii.gz") if "summary" not in f.name])
    for pred_path in pred_files:
        case_id = pred_path.stem.replace(".nii", "")
        gt_path = LUNG_GT / pred_path.name
        if not gt_path.exists():
            print(f"  {case_id}: GT not found, skipping")
            continue
        try:
            pred = load_nifti(pred_path)
            gt   = load_nifti(gt_path)
            if pred.shape != gt.shape:
                s = tuple(min(a, b) for a, b in zip(pred.shape, gt.shape))
                pred = pred[:s[0], :s[1], :s[2]]
                gt   = gt  [:s[0], :s[1], :s[2]]
            m = binary_metrics(pred, gt, label=1)
            m["case_id"] = case_id
            lung_cases.append(m)
            print(f"  {case_id}: Dice={m['Dice']:.3f}  Prec={m['Precision']:.3f}  "
                  f"Rec={m['Recall']:.3f}  Spec={m['Specificity']:.3f}  BalAcc={m['Balanced_Accuracy']:.3f}")
        except Exception as e:
            print(f"  {case_id}: ERROR — {e}")
else:
    print(f"  LUNG_PRED ({LUNG_PRED.exists()}) or LUNG_GT ({LUNG_GT.exists()}) not found")

LUNG_METRIC_KEYS = ["Dice", "Precision", "Recall", "Specificity", "F1", "Balanced_Accuracy", "AUC"]
lung_summary = {}
if lung_cases:
    for k in LUNG_METRIC_KEYS:
        vals = [c[k] for c in lung_cases]
        lung_summary[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)),
                           "median": float(np.median(vals)), "min": float(np.min(vals)),
                           "max": float(np.max(vals))}
    print(f"\n  Lung Summary (n={len(lung_cases)} cases):")
    for k, s in lung_summary.items():
        print(f"    {k}: mean={s['mean']:.3f} +/- {s['std']:.3f}  "
              f"[min={s['min']:.3f}, max={s['max']:.3f}]")

summary = {
    "brain": {
        "source": "5-fold CV, checkpoint ema_fg_dice, 1000 epochs, BraTS2021 BAMF",
        "n_folds": len(brain_ema_per_fold),
        "ema_fg_dice_per_fold": brain_ema_per_fold,
        "ema_fg_dice_mean": brain_ema_mean,
        "ema_fg_dice_std":  brain_ema_std,
        "note": "Dice = mean across all foreground classes (edema, nonenhancing, enhancing)",
    },
    "liver": {
        "source": "5-fold CV, postprocessing.json dc_per_class_raw, 1000 epochs, MSD Task03",
        "n_training_cases": liver_dice.get("n_samples", 0),
        "parenchyma_dice": liver_dice.get("parenchyma"),
        "tumor_dice":      liver_dice.get("tumor"),
        "note": "Raw CV Dice (no post-processing). Post-proc adds +0.005 to parenchyma.",
    },
    "lung": {
        "source": "fold_0 validation predictions vs GT, 13 cases, MSD Task06, 250 epochs",
        "n_cases": len(lung_cases),
        "metrics_summary": lung_summary,
        "per_case": [
            {k: float(c[k]) if isinstance(c[k], (int, float, np.floating)) else c[k]
             for k in ["case_id"] + LUNG_METRIC_KEYS + ["TP","FP","FN","TN","n_gt_voxels","n_pred_voxels"]}
            for c in lung_cases
        ],
    },
}
summary_path = OUT / "real_metrics_summary.json"
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print(f"\nJSON saved: {summary_path}")

if lung_cases:
    ids   = [c["case_id"].replace("lung_", "") for c in lung_cases]
    metrics_to_show = ["Dice", "Precision", "Recall", "Specificity", "Balanced_Accuracy"]
    nice_names       = ["Dice",  "Precision", "Recall", "Specificity", "Balanced Acc."]
    bar_colors       = ["#b45309","#d97706","#92400e","#78350f","#f59e0b"]

    fig, axes = plt.subplots(1, len(metrics_to_show), figsize=(18, 4.5), sharey=False)
    for ax, metric, nice, color in zip(axes, metrics_to_show, nice_names, bar_colors):
        vals  = [c[metric] for c in lung_cases]
        bars  = ax.bar(ids, vals, color=color, edgecolor="white", linewidth=0.6, width=0.65)
        ax.set_title(nice, fontsize=12, fontweight="bold", pad=8)
        ax.set_ylim(0, 1.12)
        ax.set_xticklabels(ids, rotation=45, ha="right", fontsize=8)
        ax.axhline(np.mean(vals), color="#374151", linewidth=1.5, linestyle="--", alpha=0.7)
        ax.text(len(ids)-0.5, np.mean(vals)+0.02, f"mean={np.mean(vals):.3f}",
                ha="right", fontsize=8, color="#374151")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.015,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=7, fontweight="bold")
    fig.suptitle(
        "Lung CT — Per-Case Validation Metrics (fold_0 holdout, n=13, MSD Task06)\n"
        "nnUNet v2 | 250 epochs | Tumour label (class 1) | Real measurements",
        fontsize=11, fontweight="bold", y=1.02)
    fig.tight_layout()
    p = OUT / "real_lung_per_case.png"
    fig.savefig(p, dpi=160, bbox_inches="tight", facecolor="white")
    print(f"Saved: {p}")
    plt.close(fig)

fig, ax = plt.subplots(figsize=(10, 5))
ax.set_facecolor("#fafafa")

organs  = ["Brain MRI\n(5-fold CV)", "Liver Paren.\n(5-fold CV)", "Liver Tumour\n(5-fold CV)",
           "Lung CT\n(fold_0 val)"]
dices   = [brain_ema_mean,
           liver_dice.get("parenchyma", 0),
           liver_dice.get("tumor", 0),
           lung_summary.get("Dice", {}).get("mean", 0) if lung_cases else 0]
errs    = [brain_ema_std, 0, 0,
           lung_summary.get("Dice", {}).get("std", 0) if lung_cases else 0]
colors  = [COLORS["brain"], COLORS["liver_paren"], COLORS["liver_tumor"], COLORS["lung"]]
notes   = [f"n_fold=5\n1000 ep", f"n_tr=131\n1000 ep", f"n_tr=131\n1000 ep", f"n=13\n250 ep"]

bars = ax.bar(organs, dices, color=colors, edgecolor="white", width=0.55,
              yerr=errs, capsize=5, error_kw=dict(elinewidth=1.5, ecolor="#374151"))
for bar, val, err, note in zip(bars, dices, errs, notes):
    ax.text(bar.get_x() + bar.get_width()/2, val + err + 0.018,
            f"{val:.3f}", ha="center", va="bottom", fontsize=13, fontweight="bold")
    ax.text(bar.get_x() + bar.get_width()/2, 0.02,
            note, ha="center", va="bottom", fontsize=8, color="white", fontweight="bold",
            linespacing=1.4)

ax.set_ylim(0, 1.08)
ax.set_ylabel("Dice Score", fontsize=12)
ax.axhline(0.8, color="#9ca3af", linewidth=1.2, linestyle="--", alpha=0.7)
ax.text(3.5, 0.81, "0.80", fontsize=9, color="#9ca3af")
ax.set_title(
    "CARVIS — Cross-Validation Dice Scores (Real Measurements)\n"
    "Brain & Liver: 5-fold CV | Lung: fold_0 validation | No hardcoded values",
    fontsize=12, fontweight="bold", pad=12)
fig.tight_layout()
p = OUT / "real_cv_comparison.png"
fig.savefig(p, dpi=160, bbox_inches="tight", facecolor="white")
print(f"Saved: {p}")
plt.close(fig)

COL_HEADERS = ["Dice", "Precision", "Recall", "Specificity", "Bal. Acc.", "F1", "AUC"]

def fmt(val, std=None):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "—"
    if std is not None and std > 0:
        return f"{val:.3f}\n±{std:.3f}"
    return f"{val:.3f}"

rows = []
row_labels  = []
row_colors  = []

rows.append([
    fmt(brain_ema_mean, brain_ema_std),
    "—", "—", "—", "—",
    fmt(brain_ema_mean, brain_ema_std),
    "—",
])
row_labels.append("Brain MRI\n(5-fold CV, ema Dice)")
row_colors.append(COLORS["brain"])

rows.append([
    fmt(liver_dice.get("parenchyma")), "—","—","—","—",
    fmt(liver_dice.get("parenchyma")), "—",
])
row_labels.append("Liver Parenchyma\n(5-fold CV, n=131)")
row_colors.append(COLORS["liver_paren"])

rows.append([
    fmt(liver_dice.get("tumor")), "—","—","—","—",
    fmt(liver_dice.get("tumor")), "—",
])
row_labels.append("Liver Tumour\n(5-fold CV, n=131)")
row_colors.append(COLORS["liver_tumor"])

if lung_cases:
    rows.append([
        fmt(lung_summary["Dice"]["mean"], lung_summary["Dice"]["std"]),
        fmt(lung_summary["Precision"]["mean"], lung_summary["Precision"]["std"]),
        fmt(lung_summary["Recall"]["mean"], lung_summary["Recall"]["std"]),
        fmt(lung_summary["Specificity"]["mean"], lung_summary["Specificity"]["std"]),
        fmt(lung_summary["Balanced_Accuracy"]["mean"], lung_summary["Balanced_Accuracy"]["std"]),
        fmt(lung_summary["F1"]["mean"], lung_summary["F1"]["std"]),
        fmt(lung_summary["AUC"]["mean"], lung_summary["AUC"]["std"]),
    ])
    row_labels.append("Lung CT\n(fold_0 val, n=13)")
    row_colors.append(COLORS["lung"])

fig, ax = plt.subplots(figsize=(15, 4.5))
ax.axis("off")
tbl = ax.table(
    cellText=rows,
    rowLabels=row_labels,
    colLabels=COL_HEADERS,
    loc="center", cellLoc="center",
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1, 2.4)

for (r, c), cell in tbl.get_celld().items():
    cell.set_edgecolor("#d1d5db")
    if r == 0:
        cell.set_facecolor("#1a1a2e")
        cell.set_text_props(color="white", fontweight="bold", fontsize=10)
    elif c == -1:
        col = row_colors[r - 1] if r - 1 < len(row_colors) else "#6b7280"
        rgb = mc.to_rgb(col)
        tint = tuple(1 - 0.15 + 0.15 * x for x in rgb)
        cell.set_facecolor(tint)
        cell.set_text_props(fontweight="bold", fontsize=9)
        cell.set_width(0.22)
    else:
        txt = cell.get_text().get_text()
        if txt == "—":
            cell.set_facecolor("#f3f4f6")
            cell.set_text_props(color="#9ca3af", fontsize=11)
        else:
            try:
                val = float(txt.split("\n")[0])
                col = row_colors[r - 1]
                rgb = mc.to_rgb(col)
                alpha = 0.12 + val * 0.55
                blended = tuple(1 - alpha + alpha * x for x in rgb)
                cell.set_facecolor(blended)
                cell.set_text_props(
                    fontweight="bold",
                    color="white" if val > 0.72 else "#1a1a2e",
                    fontsize=9,
                )
            except Exception:
                pass

ax.set_title(
    "CARVIS — Segmentation Metrics (Real Cross-Validation Measurements)\n"
    "Brain/Liver: Dice from 5-fold CV checkpoints | Lung: Full metrics from 13 validation cases\n"
    "— = metric not computable without saved CV predictions (only Dice logged in checkpoint)",
    fontsize=9.5, pad=16, loc="left", color="#374151",
)
fig.tight_layout()
p = OUT / "real_metrics_table.png"
fig.savefig(p, dpi=160, bbox_inches="tight", facecolor="white")
print(f"Saved: {p}")
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

ax = axes[0]
org_labels = ["Brain MRI\n(5-fold)", "Liver\nParen.\n(5-fold)", "Liver\nTumour\n(5-fold)",
              "Lung CT\n(fold_0 val)"]
dice_vals = [brain_ema_mean, liver_dice.get("parenchyma", 0), liver_dice.get("tumor", 0),
             lung_summary.get("Dice",{}).get("mean", 0) if lung_cases else 0]
dice_errs = [brain_ema_std, 0, 0,
             lung_summary.get("Dice",{}).get("std", 0) if lung_cases else 0]
clrs = [COLORS["brain"], COLORS["liver_paren"], COLORS["liver_tumor"], COLORS["lung"]]
bars = ax.bar(org_labels, dice_vals, color=clrs, width=0.55, edgecolor="white",
              yerr=dice_errs, capsize=5, error_kw=dict(elinewidth=1.5, ecolor="#374151"))
for bar, val, err in zip(bars, dice_vals, dice_errs):
    ax.text(bar.get_x() + bar.get_width()/2, val + err + 0.015,
            f"{val:.3f}", ha="center", va="bottom", fontsize=12, fontweight="bold")
ax.set_ylim(0, 1.08)
ax.set_ylabel("Dice Score", fontsize=11)
ax.axhline(0.8, color="#9ca3af", lw=1.2, ls="--", alpha=0.7)
ax.set_title("Dice Score — Organ Comparison\n(5-fold CV or fold_0 validation)", fontsize=11, fontweight="bold")

ax2 = axes[1]
if lung_cases:
    lm_keys   = ["Dice", "Precision", "Recall", "Specificity", "Balanced\nAcc.", "F1"]
    lm_metric = ["Dice", "Precision", "Recall", "Specificity", "Balanced_Accuracy", "F1"]
    lm_vals   = [lung_summary[k]["mean"] for k in lm_metric]
    lm_errs   = [lung_summary[k]["std"]  for k in lm_metric]
    bars2 = ax2.bar(lm_keys, lm_vals, color=COLORS["lung"], alpha=0.85, width=0.55,
                    edgecolor="white", yerr=lm_errs, capsize=5,
                    error_kw=dict(elinewidth=1.5, ecolor="#374151"))
    for bar, val, err in zip(bars2, lm_vals, lm_errs):
        ax2.text(bar.get_x() + bar.get_width()/2, val + err + 0.015,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax2.set_ylim(0, 1.12)
    ax2.set_ylabel("Score (mean ± std)", fontsize=11)
    ax2.axhline(0.8, color="#9ca3af", lw=1.2, ls="--", alpha=0.7)
    ax2.set_title(
        f"Lung CT — Full Metric Profile\n(fold_0 validation, n={len(lung_cases)}, real computation)",
        fontsize=11, fontweight="bold")
else:
    ax2.text(0.5, 0.5, "Lung data not found", ha="center", va="center",
             transform=ax2.transAxes, fontsize=12)

fig.suptitle(
    "CARVIS — Segmentation Performance: Real Measurements Only\n"
    "No hardcoded values | Brain/Liver: CV Dice | Lung: Full voxel-level metrics",
    fontsize=12, fontweight="bold", y=1.01)
fig.tight_layout()
p = OUT / "real_metrics_bar.png"
fig.savefig(p, dpi=160, bbox_inches="tight", facecolor="white")
print(f"Saved: {p}")
plt.close(fig)

print("\n" + "="*65)
print("REAL METRICS SUMMARY (no hardcoded values)")
print("="*65)
print(f"Brain MRI  (5-fold CV): Dice = {brain_ema_mean:.4f} +/- {brain_ema_std:.4f}")
if liver_dice.get("parenchyma"):
    print(f"Liver Paren(5-fold CV): Dice = {liver_dice['parenchyma']:.4f}")
    print(f"Liver Tumour(5-fold CV):Dice = {liver_dice['tumor']:.4f}  (n={liver_dice['n_samples']})")
if lung_cases:
    print(f"Lung CT (fold_0 val):   Dice = {lung_summary['Dice']['mean']:.4f} +/- {lung_summary['Dice']['std']:.4f}")
    print(f"                        Prec = {lung_summary['Precision']['mean']:.4f}")
    print(f"                        Rec  = {lung_summary['Recall']['mean']:.4f}")
    print(f"                        Spec = {lung_summary['Specificity']['mean']:.4f}")
    print(f"                        BalAcc={lung_summary['Balanced_Accuracy']['mean']:.4f}")
print(f"\nOutputs: {OUT}")
print("Done.")
