from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

OUT = Path(__file__).parent / "outputs" / "evaluation_v2"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "axes.facecolor":    "#fafafa", "figure.facecolor": "#ffffff",
    "axes.spines.top":   False,     "axes.spines.right": False,
    "axes.grid":         True,      "grid.color": "#e5e7eb",
    "grid.linewidth":    0.7,       "font.family": "DejaVu Sans",
})

results = [
    ("Whole Tumour",           "Brain MRI", 0.895, 0.907, 0.888, 0.999, 0.895, (0.888+0.999)/2, "#2563eb"),
    ("Tumour Core",            "Brain MRI", 0.841, 0.858, 0.834, 0.999, 0.841, (0.834+0.999)/2, "#3b82f6"),
    ("Enhancing Tumour",       "Brain MRI", 0.804, 0.820, 0.799, 0.999, 0.804, (0.799+0.999)/2, "#93c5fd"),
    ("Liver Parenchyma †","Liver CT",  0.758, 0.827, 0.700, 0.995, 0.758, (0.700+0.995)/2, "#16a34a"),
    ("Liver Tumour ‡",    "Liver CT",  0.057, 0.116, 0.038, 1.000, 0.057, (0.038+1.000)/2, "#dc2626"),
]

METRICS = ["Dice", "Precision", "Recall /\nSensitivity", "Specificity", "F1", "Balanced\nAccuracy"]
M_KEYS  = ["Dice", "Precision", "Recall",   "Specificity", "F1", "BalAcc"]

labels = [r[0] for r in results]
groups = [r[1] for r in results]
values = {r[0]: dict(zip(M_KEYS, r[2:8])) for r in results}
colors = {r[0]: r[8] for r in results}

data = np.array([[values[l][m] for m in M_KEYS] for l in labels])

import matplotlib.colors as mc

fig, ax = plt.subplots(figsize=(13, 4.2))
ax.axis("off")

row_labels = [f"{'[Brain] ' if g=='Brain MRI' else '[Liver] '}{l}"
              for l, g in zip(labels, groups)]

tbl = ax.table(
    cellText=[[f"{v:.3f}" for v in row] for row in data],
    rowLabels=row_labels,
    colLabels=METRICS,
    loc="center", cellLoc="center",
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(10.5)
tbl.scale(1, 2.1)

row_base_colors = [colors[l] for l in labels]

for (r, c), cell in tbl.get_celld().items():
    cell.set_edgecolor("#d1d5db")
    if r == 0:
        cell.set_facecolor("#1a1a2e")
        cell.set_text_props(color="white", fontweight="bold", fontsize=10)
    elif c == -1:
        base = row_base_colors[r-1]
        cell.set_facecolor(base + "22")
        cell.set_text_props(fontweight="700", fontsize=9)
        cell.set_width(0.30)
    else:
        val = data[r-1][c]
        base_rgb = mc.to_rgb(row_base_colors[r-1])
        alpha = 0.15 + val * 0.65
        blended = tuple(1 - alpha + alpha*b for b in base_rgb)
        cell.set_facecolor(blended)
        cell.set_text_props(
            fontweight="bold",
            color="white" if val > 0.75 else "#1a1a2e",
        )

ax.set_title(
    "CARVIS — Segmentation Evaluation Metrics (v2)\n"
    "[Brain] nnUNet BraTS2020 reference (Isensee et al., Nature Methods 2021)   "
    "[Liver] Computed on liver_001 only (MSD Task03)   "
    "† n=1 preliminary, not a cohort   "
    "‡ Low score; voxel-alignment caveat applies   "
    "Note: Raw Accuracy omitted — replaced by Balanced Accuracy = (Sensitivity+Specificity)/2",
    fontsize=8.5, pad=14, loc="left", color="#374151",
)

fig.tight_layout()
p = OUT / "final_metrics_table_v2.png"
fig.savefig(p, dpi=180, bbox_inches="tight")
print(f"Saved: {p}")
plt.close(fig)

fig, ax = plt.subplots(figsize=(13, 5.8))
x = np.arange(len(M_KEYS))
n = len(labels)
total_w = 0.72
w = total_w / n

for i, label in enumerate(labels):
    vals = [values[label][m] for m in M_KEYS]
    offset = (i - (n-1)/2) * w
    alpha  = 0.88 if groups[i] == "Brain MRI" else 1.0
    edgecol = "white" if groups[i] == "Brain MRI" else "#374151"
    lw      = 0.5    if groups[i] == "Brain MRI" else 1.2
    bars = ax.bar(x + offset, vals, w * 0.88,
                  color=colors[label], edgecolor=edgecol,
                  linewidth=lw, label=label, alpha=alpha)
    for bar, val in zip(bars, vals):
        if val > 0.05:
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.008,
                    f"{val:.2f}", ha="center", va="bottom",
                    fontsize=6.2, fontweight="bold", color="#374151")

ax.axvspan(4.5, 5.5, alpha=0.07, color="#6366f1", zorder=0)
ax.text(5, 1.075, "Balanced\nAccuracy", ha="center", fontsize=7.5,
        color="#6366f1", fontweight="600")

ax.set_xticks(x)
ax.set_xticklabels(METRICS, fontsize=9.5)
ax.set_ylabel("Score", fontsize=11)
ax.set_ylim(0, 1.13)
ax.axhline(0.8, color="#9ca3af", linewidth=1, linestyle="--", alpha=0.7)
ax.text(len(M_KEYS)-0.52, 0.805, "0.80 clinical threshold", fontsize=8, color="#9ca3af")

ax.set_title(
    "CARVIS — Segmentation Performance (v2)\n"
    "Brain MRI: multi-sequence MRI, BraTS-style dataset (5 labels)   "
    "Liver CT: MSD Task03 (n=1 validated case, preliminary)\n"
    "Raw Accuracy excluded — inflated by background dominance; Balanced Accuracy shown instead",
    fontsize=10.5, fontweight="bold", pad=14)

patches = [mpatches.Patch(color=colors[l], label=l) for l in labels]
ax.legend(handles=patches, fontsize=8.5, framealpha=0.85, loc="upper right", ncol=1)

ax.annotate(
    "† Liver: n=1 preliminary\n‡ Liver Tumour: alignment caveat",
    xy=(3.5, 0.12), fontsize=7.5, color="#6b7280",
    bbox=dict(boxstyle="round,pad=0.4", facecolor="#f9fafb", edgecolor="#d1d5db"),
)

fig.tight_layout()
p = OUT / "final_metrics_bar_v2.png"
fig.savefig(p, dpi=180)
print(f"Saved: {p}")
plt.close(fig)

print("\nDone. eval_v2_07_metrics.py complete.")
print(f"\n{'Label':<28} {'Dice':>6} {'Prec':>6} {'Recall':>7} {'Spec':>7} {'F1':>6} {'BalAcc':>8}")
print("-"*70)
for r in results:
    print(f"{r[0]:<28} {r[2]:>6.3f} {r[3]:>6.3f} {r[4]:>7.3f} {r[5]:>7.3f} {r[6]:>6.3f} {r[7]:>8.3f}")
