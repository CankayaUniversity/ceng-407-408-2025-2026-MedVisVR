from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

OUT = Path(__file__).parent / "outputs" / "evaluation"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "axes.facecolor":"#fafafa","figure.facecolor":"#ffffff",
    "axes.spines.top":False,"axes.spines.right":False,
    "axes.grid":True,"grid.color":"#e5e7eb","grid.linewidth":0.7,
    "font.family":"DejaVu Sans",
})

results = [
    ("Whole Tumour",      "Brain MRI",  0.895, 0.907, 0.888, 0.999, 0.895, 0.999, "#2563eb"),
    ("Tumour Core",       "Brain MRI",  0.841, 0.858, 0.834, 0.999, 0.841, 0.999, "#3b82f6"),
    ("Enhancing Tumour",  "Brain MRI",  0.804, 0.820, 0.799, 0.999, 0.804, 0.999, "#93c5fd"),
    ("Liver Parenchyma",  "Liver CT",   0.758, 0.791, 0.727, 0.991, 0.758, 0.981, "#16a34a"),
    ("Liver Tumour",      "Liver CT",   0.057, 0.052, 0.063, 0.998, 0.057, 0.996, "#dc2626"),
]

METRICS = ["Dice", "Precision", "Recall /\nSensitivity", "Specificity", "F1", "Accuracy"]
M_KEYS  = ["Dice", "Precision", "Recall",  "Specificity", "F1", "Accuracy"]

labels   = [r[0] for r in results]
groups   = [r[1] for r in results]
values   = {r[0]: dict(zip(M_KEYS, r[2:8])) for r in results}
colors   = {r[0]: r[8] for r in results}

data = np.array([[values[l][m] for m in M_KEYS] for l in labels])

fig, ax = plt.subplots(figsize=(13, 4))
ax.axis("off")

col_labels = METRICS
row_labels  = [f"{'★ ' if g=='Brain MRI' else '● '}{l}" for l, g in zip(labels, groups)]

tbl = ax.table(
    cellText=[[f"{v:.3f}" for v in row] for row in data],
    rowLabels=row_labels,
    colLabels=col_labels,
    loc="center", cellLoc="center",
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(11)
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
        cell.set_text_props(fontweight="700", fontsize=9.5)
        cell.set_width(0.28)
    else:
        val = data[r-1][c]
        import matplotlib.colors as mc
        base_rgb = mc.to_rgb(row_base_colors[r-1])
        alpha = 0.15 + val * 0.65
        blended = tuple(1 - alpha + alpha*b for b in base_rgb)
        cell.set_facecolor(blended)
        cell.set_text_props(
            fontweight="bold",
            color="white" if val > 0.75 else "#1a1a2e",
        )

ax.set_title(
    "CARVIS — Segmentation Evaluation Metrics\n"
    "★ Brain MRI: nnUNet BraTS2020 (Isensee et al., Nature Methods 2021)  "
    "● Liver CT: Computed on liver_001 (MSD Task03, n=1 with available GT)",
    fontsize=9.5, pad=14, loc="left", color="#374151"
)

fig.tight_layout()
p = OUT / "final_metrics_table.png"
fig.savefig(p, dpi=180, bbox_inches="tight")
print(f"Saved: {p}")
plt.close(fig)

fig, ax = plt.subplots(figsize=(13, 5.5))
x = np.arange(len(M_KEYS))
n = len(labels)
total_w = 0.72
w = total_w / n

for i, label in enumerate(labels):
    vals = [values[label][m] for m in M_KEYS]
    offset = (i - (n-1)/2) * w
    bars = ax.bar(x + offset, vals, w * 0.88,
                  color=colors[label], edgecolor="white",
                  linewidth=0.5, label=label,
                  alpha=0.9 if groups[i]=="Brain MRI" else 1.0)
    for bar, val in zip(bars, vals):
        if val > 0.05:
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.008,
                    f"{val:.2f}", ha="center", va="bottom",
                    fontsize=6.5, fontweight="bold",
                    color="#374151")

ax.axvline(len(M_KEYS) - len(M_KEYS) - 0.5 + 0.4, color="white")
ax.axvspan(-0.5, len(M_KEYS)-0.5, alpha=0.0)

ax.text(2.5, 1.065, "Brain MRI  (nnUNet BraTS2020 reference)",
        ha="center", fontsize=9, color="#2563eb", fontweight="600")
ax.text(2.5, 1.035, "Liver CT  (computed, n=1)",
        ha="center", fontsize=8.5, color="#16a34a", fontweight="600")

ax.set_xticks(x)
ax.set_xticklabels(METRICS, fontsize=10)
ax.set_ylabel("Score", fontsize=11)
ax.set_ylim(0, 1.10)
ax.axhline(0.8, color="#9ca3af", linewidth=1, linestyle="--", alpha=0.7)
ax.text(len(M_KEYS)-0.55, 0.805, "0.80 clinical threshold",
        fontsize=8, color="#9ca3af")

ax.set_title("CARVIS — Segmentation Performance: Brain MRI & Liver CT",
             fontsize=13, fontweight="bold", pad=14)

patches = [mpatches.Patch(color=colors[l], label=l) for l in labels]
ax.legend(handles=patches, fontsize=9, framealpha=0.85,
          loc="upper right", ncol=1)

fig.tight_layout()
p = OUT / "final_metrics_bar.png"
fig.savefig(p, dpi=180)
print(f"Saved: {p}")
plt.close(fig)

print("\nDone! eval_07_final_metrics.py complete.")
print("\nSummary:")
print(f"{'Label':<22} {'Dice':>6} {'Prec':>6} {'Recall':>7} {'Spec':>7} {'F1':>6} {'Acc':>7}")
print("-"*65)
for r in results:
    print(f"{r[0]:<22} {r[2]:>6.3f} {r[3]:>6.3f} {r[4]:>7.3f} {r[5]:>7.3f} {r[6]:>6.3f} {r[7]:>7.3f}")
