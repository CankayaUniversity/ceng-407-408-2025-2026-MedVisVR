import json
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mc

ROOT  = Path(__file__).parent
OUT   = ROOT / "outputs" / "evaluation_v2"
OUT.mkdir(parents=True, exist_ok=True)

SUMMARY_PATH = OUT / "real_metrics_summary.json"
if not SUMMARY_PATH.exists():
    print("ERROR: real_metrics_summary.json not found. Run eval_real_compute.py first.")
    exit(1)

with open(SUMMARY_PATH, encoding="utf-8") as f:
    S = json.load(f)

brain = S["brain"]
liver = S["liver"]
lung  = S["lung"]

C_BRAIN  = "#2563eb"
C_LPAREN = "#16a34a"
C_LTUMOR = "#dc2626"
C_LUNG   = "#b45309"

plt.rcParams.update({
    "axes.facecolor":    "#fafafa",
    "figure.facecolor":  "#ffffff",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.color":        "#e5e7eb",
    "grid.linewidth":    0.6,
    "font.family":       "DejaVu Sans",
    "font.size":         11,
})

def lm(key):
    return lung["metrics_summary"].get(key, {})


COL_H = ["Dice", "Precision", "Recall", "Specificity", "Balanced\nAccuracy", "F1"]

def lung_cell(key):
    m = lm(key)
    if not m:
        return "—"
    mean, std = m["mean"], m["std"]
    return f"{mean:.3f}\n±{std:.3f}"

rows = [
    [f"{brain['ema_fg_dice_mean']:.3f}\n±{brain['ema_fg_dice_std']:.3f}",
     "—", "—", "—", "—",
     f"{brain['ema_fg_dice_mean']:.3f}"],
    [f"{liver['parenchyma_dice']:.3f}", "—", "—", "—", "—",
     f"{liver['parenchyma_dice']:.3f}"],
    [f"{liver['tumor_dice']:.3f}",      "—", "—", "—", "—",
     f"{liver['tumor_dice']:.3f}"],
    [lung_cell("Dice"), lung_cell("Precision"), lung_cell("Recall"),
     lung_cell("Specificity"), lung_cell("Balanced_Accuracy"), lung_cell("F1")],
]
row_labels = [
    f"Brain MRI\n(5-fold CV, n_epochs=1000)",
    f"Liver Parenchyma\n(5-fold CV, n={liver['n_training_cases']})",
    f"Liver Tumour\n(5-fold CV, n={liver['n_training_cases']})",
    f"Lung CT\n(fold_0 val, n={lung['n_cases']})",
]
row_colors = [C_BRAIN, C_LPAREN, C_LTUMOR, C_LUNG]

fig, ax = plt.subplots(figsize=(14, 5))
ax.axis("off")
tbl = ax.table(
    cellText=rows,
    rowLabels=row_labels,
    colLabels=COL_H,
    loc="center", cellLoc="center",
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1, 2.6)

for (r, c), cell in tbl.get_celld().items():
    cell.set_edgecolor("#d1d5db")
    if r == 0:
        cell.set_facecolor("#1a1a2e")
        cell.set_text_props(color="white", fontweight="bold", fontsize=10.5)
    elif c == -1:
        col = row_colors[r - 1] if r - 1 < len(row_colors) else "#6b7280"
        rgb  = mc.to_rgb(col)
        tint = tuple(0.88 + 0.12*x for x in rgb)
        cell.set_facecolor(tint)
        cell.set_text_props(fontweight="bold", fontsize=9, linespacing=1.3)
        cell.set_width(0.24)
    else:
        txt = cell.get_text().get_text()
        if txt == "—":
            cell.set_facecolor("#f3f4f6")
            cell.set_text_props(color="#9ca3af", fontsize=12)
        else:
            try:
                val = float(txt.split("\n")[0])
                col = row_colors[r - 1]
                rgb = mc.to_rgb(col)
                alpha = 0.12 + val * 0.55
                blended = tuple(1 - alpha + alpha*x for x in rgb)
                cell.set_facecolor(blended)
                cell.set_text_props(
                    fontweight="bold",
                    color="white" if val > 0.72 else "#1a1a2e",
                    fontsize=9, linespacing=1.3,
                )
            except Exception:
                pass

ax.set_title(
    "CARVIS — Segmentation Metrics (Real Cross-Validation Measurements)\n"
    "Brain/Liver: Dice from 5-fold CV checkpoint logs  |  "
    "Lung: Voxel-level computation from 13 held-out validation cases\n"
    "— = metric not available (CV predictions not stored, only Dice logged in checkpoint)",
    fontsize=9.5, pad=14, loc="left", color="#374151",
)
fig.tight_layout()
p = OUT / "poster_real_metrics_table.png"
fig.savefig(p, dpi=160, bbox_inches="tight", facecolor="white")
print(f"Saved: {p}")
plt.close(fig)


fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(15, 5.5))

org_labels = ["Brain MRI\n(5-fold CV)", "Liver\nParen.\n(5-fold)", "Liver\nTumour\n(5-fold)",
              "Lung CT\n(fold_0, n=13)"]
dice_vals  = [brain["ema_fg_dice_mean"],
              liver["parenchyma_dice"],
              liver["tumor_dice"],
              lm("Dice").get("mean", 0)]
dice_errs  = [brain["ema_fg_dice_std"], 0, 0,
              lm("Dice").get("std", 0)]
clrs = [C_BRAIN, C_LPAREN, C_LTUMOR, C_LUNG]

bars = ax_left.bar(org_labels, dice_vals, color=clrs, width=0.52, edgecolor="white",
                   linewidth=0.8, yerr=dice_errs, capsize=6,
                   error_kw=dict(elinewidth=2, ecolor="#374151"))
for bar, val, err in zip(bars, dice_vals, dice_errs):
    ax_left.text(bar.get_x() + bar.get_width()/2, val + err + 0.017,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=13, fontweight="bold")

ax_left.set_ylim(0, 1.10)
ax_left.set_ylabel("Dice Score", fontsize=12, fontweight="bold")
ax_left.axhline(0.8, color="#9ca3af", lw=1.2, ls="--", alpha=0.8)
ax_left.text(3.3, 0.815, "0.80", fontsize=9, color="#9ca3af", ha="right")
ax_left.set_title("Dice Score — Cross-Validation\n(Real measurements from checkpoints)",
                  fontsize=11.5, fontweight="bold", pad=10)
ax_left.tick_params(axis="x", labelsize=10)
ax_left.grid(axis="y", linestyle="--", alpha=0.5)

metric_keys  = ["Dice", "Precision", "Recall", "Balanced_Accuracy", "F1"]
metric_names = ["Dice", "Precision", "Recall", "Balanced\nAccuracy", "F1"]
m_vals = [lm(k).get("mean", 0) for k in metric_keys]
m_errs = [lm(k).get("std",  0) for k in metric_keys]

bars2 = ax_right.bar(metric_names, m_vals, color=C_LUNG, alpha=0.85, width=0.52,
                     edgecolor="white", linewidth=0.8,
                     yerr=m_errs, capsize=6, error_kw=dict(elinewidth=2, ecolor="#374151"))
for bar, val, err in zip(bars2, m_vals, m_errs):
    ax_right.text(bar.get_x() + bar.get_width()/2, val + err + 0.018,
                  f"{val:.3f}", ha="center", va="bottom", fontsize=12, fontweight="bold")

ax_right.set_ylim(0, 1.12)
ax_right.set_ylabel("Score (mean ± std)", fontsize=12, fontweight="bold")
ax_right.axhline(0.8, color="#9ca3af", lw=1.2, ls="--", alpha=0.8)
ax_right.set_title(
    f"Lung CT — Full Metric Profile\n(fold_0 val, n={lung['n_cases']}, voxel-level computation)",
    fontsize=11.5, fontweight="bold", pad=10)
ax_right.tick_params(axis="x", labelsize=10)
ax_right.grid(axis="y", linestyle="--", alpha=0.5)

spec_val = lm("Specificity").get("mean", 0)
ax_right.text(0.97, 0.03,
    f"Note: Specificity = {spec_val:.3f}\n(tumour << total CT volume;\nuse Balanced Acc. instead)",
    transform=ax_right.transAxes, fontsize=7.5, ha="right", va="bottom",
    color="#6b7280", linespacing=1.4,
    bbox=dict(boxstyle="round,pad=0.4", facecolor="#fef9c3", edgecolor="#d97706", alpha=0.8))

fig.suptitle(
    "CARVIS — Segmentation Performance: Real Measurements Only | No Hardcoded Values\n"
    "Brain & Liver: 5-fold Cross-Validation Dice  |  "
    "Lung CT: Full voxel-level metrics (13 held-out cases)",
    fontsize=12, fontweight="bold", y=1.01)
fig.tight_layout()
p = OUT / "poster_real_metrics_bar.png"
fig.savefig(p, dpi=160, bbox_inches="tight", facecolor="white")
print(f"Saved: {p}")
plt.close(fig)


per_case = lung.get("per_case", [])
if per_case:
    ids = [c["case_id"].replace("lung_", "L") for c in per_case]
    metrics3 = ["Dice", "Balanced_Accuracy", "Precision", "Recall"]
    nice3    = ["Dice", "Balanced Accuracy", "Precision", "Recall"]
    pal      = [C_LUNG, "#d97706", "#92400e", "#78350f"]

    fig, axes = plt.subplots(1, 4, figsize=(17, 4.5), sharey=False)
    for ax, mk, nm, clr in zip(axes, metrics3, nice3, pal):
        vals = [c[mk] for c in per_case]
        mean_v = float(np.mean(vals))
        bars = ax.bar(ids, vals, color=clr, width=0.65, edgecolor="white", linewidth=0.5)
        ax.axhline(mean_v, color="#1a1a2e", lw=2, ls="--", alpha=0.85)
        ax.text(len(ids) - 0.5, mean_v + 0.025, f"mean\n{mean_v:.3f}",
                ha="right", va="bottom", fontsize=8.5, fontweight="bold", color="#1a1a2e")
        ax.set_ylim(0, 1.15)
        ax.set_title(nm, fontsize=12, fontweight="bold", pad=8)
        ax.tick_params(axis="x", labelsize=8, rotation=45)
        for bar, val in zip(bars, vals):
            if val > 0.05:
                ax.text(bar.get_x() + bar.get_width()/2, val + 0.018,
                        f"{val:.2f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.set_facecolor("#fafafa")

    axes[0].text(ids.index("L033"), 0.05, "missed\nlesion",
                 ha="center", va="bottom", fontsize=7, color="white", fontweight="bold")

    fig.suptitle(
        f"Lung CT — Per-Case Validation Metrics (n={len(per_case)}, fold_0 holdout, MSD Task06)\n"
        "nnUNet v2 | 250 epochs | Tumour (class 1) | Voxel-level computation",
        fontsize=11.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    p = OUT / "poster_real_lung_detail.png"
    fig.savefig(p, dpi=160, bbox_inches="tight", facecolor="white")
    print(f"Saved: {p}")
    plt.close(fig)


import torch
BRAIN_DIR = ROOT / "nnunet_brain"
class_names_short = ["Edema\n(label 1)", "Non-enh.\n(label 2)", "Enhancing\n(label 4)"]
fold_per_class = []

for fold_dir in sorted([d for d in BRAIN_DIR.iterdir() if d.is_dir() and d.name.startswith("fold_")]):
    ckpt = torch.load(str(fold_dir / "checkpoint_final.pth"), map_location="cpu", weights_only=False)
    dpc = ckpt.get("logging", {}).get("dice_per_class_or_region", None)
    if dpc and isinstance(dpc, list):
        last = dpc[-1]
        try:
            vals = [float(last[0]), float(last[1]), float(last[3])]
            fold_per_class.append(vals)
        except Exception:
            pass

if fold_per_class:
    arr = np.array(fold_per_class)
    means = np.nanmean(arr, axis=0)
    stds  = np.nanstd(arr, axis=0)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(class_names_short))
    w = 0.12
    colors_fold = ["#1d4ed8","#2563eb","#3b82f6","#60a5fa","#93c5fd"]

    for i, (fold_vals, col) in enumerate(zip(fold_per_class, colors_fold)):
        ax.bar(x + (i - 2)*w, fold_vals, w*0.85, color=col, alpha=0.75, edgecolor="white",
               label=f"fold_{i}")

    ax.bar(x + 3*w, means, w*0.85, color="#1e3a5f", edgecolor="white", label="Mean (5-fold)",
           yerr=stds, capsize=5, error_kw=dict(elinewidth=2, ecolor="#374151"), zorder=5)
    for xi, (m, s) in enumerate(zip(means, stds)):
        ax.text(xi + 3*w, m + s + 0.018, f"{m:.3f}", ha="center", va="bottom",
                fontsize=11, fontweight="bold")

    ax.set_xticks(x + 0.5*w)
    ax.set_xticklabels(class_names_short, fontsize=11)
    ax.set_ylim(0, 1.10)
    ax.set_ylabel("Dice Score", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right", framealpha=0.85)
    ax.axhline(0.8, color="#9ca3af", lw=1.2, ls="--", alpha=0.8)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.set_title(
        "Brain MRI — Per-Class Dice (5-Fold Cross-Validation)\n"
        "nnUNet v2 | 1000 epochs | BraTS2021 BAMF schema | Real checkpoint data",
        fontsize=11.5, fontweight="bold", pad=10)
    fig.tight_layout()
    p = OUT / "poster_real_brain_classes.png"
    fig.savefig(p, dpi=160, bbox_inches="tight", facecolor="white")
    print(f"Saved: {p}")
    plt.close(fig)

print("\nAll poster figures generated.")
print(f"Output directory: {OUT}")
print("\nSummary (real values):")
print(f"  Brain  5-fold EMA Dice = {brain['ema_fg_dice_mean']:.4f} +/- {brain['ema_fg_dice_std']:.4f}")
print(f"  Liver  Paren  CV Dice  = {liver['parenchyma_dice']:.4f}  (n={liver['n_training_cases']} training cases)")
print(f"  Liver  Tumour CV Dice  = {liver['tumor_dice']:.4f}")
print(f"  Lung   fold_0 val Dice = {lm('Dice').get('mean',0):.4f} +/- {lm('Dice').get('std',0):.4f}  (n={lung['n_cases']})")
print(f"  Lung   BalAcc          = {lm('Balanced_Accuracy').get('mean',0):.4f} +/- {lm('Balanced_Accuracy').get('std',0):.4f}")
