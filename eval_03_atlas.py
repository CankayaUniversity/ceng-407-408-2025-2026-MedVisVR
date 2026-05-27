import json
from pathlib import Path
from collections import Counter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT  = Path(__file__).parent
CASES = ROOT / "outputs" / "cases"
OUT   = ROOT / "outputs" / "evaluation"
OUT.mkdir(parents=True, exist_ok=True)

STYLE = {
    "axes.facecolor":   "#fafafa",
    "figure.facecolor": "#ffffff",
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid": True,
    "grid.color": "#e5e7eb",
    "grid.linewidth": 0.6,
}
plt.rcParams.update(STYLE)

lobes, hemispheres, depths, conf_scores, conf_labels = [], [], [], [], []

for case_dir in sorted(CASES.iterdir()):
    pc_path = case_dir / "patient_context.json"
    tm_path = case_dir / "tumor_metrics.json"
    if not pc_path.exists():
        continue
    pc = json.loads(pc_path.read_text(encoding="utf-8-sig"))
    if pc.get("modality", "").upper() != "MR":
        continue
    if not tm_path.exists():
        continue

    tm = json.loads(tm_path.read_text(encoding="utf-8-sig"))
    for lesion in tm.get("lesion_list", []):
        anat = lesion.get("anatomy", {})
        lobe = anat.get("lobe", "").strip()
        hemi = anat.get("hemisphere", "").strip()
        depth= anat.get("depth", "").strip()
        conf = anat.get("atlas_confidence", anat.get("confidence", "")).strip().lower()

        if lobe:  lobes.append(lobe)
        if hemi:  hemispheres.append(hemi)
        if depth: depths.append(depth)
        if conf:  conf_labels.append(conf)

        score = anat.get("registration_quality_score") or anat.get("atlas_confidence_score")
        if score is not None:
            try: conf_scores.append(float(score))
            except: pass

print(f"Total lesion anatomies collected: {len(lobes)}")

conf_counts = Counter(conf_labels)
norm = {"high": "High", "medium": "Medium", "moderate": "Medium", "low": "Low", "very low": "Low"}
norm_counts = Counter()
for k, v in conf_counts.items():
    norm_counts[norm.get(k, k.capitalize())] += v

cats   = ["High", "Medium", "Low"]
counts = [norm_counts.get(c, 0) for c in cats]
colors = ["#16a34a", "#f59e0b", "#dc2626"]

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].bar(cats, counts, color=colors, edgecolor="white", width=0.5)
axes[0].set_title("Atlas Localisation\nConfidence Distribution", fontsize=10, fontweight="bold")
axes[0].set_ylabel("Number of lesions", fontsize=9)
axes[0].tick_params(axis="x", labelsize=10)

total = sum(counts)
pcts  = [c / total * 100 if total else 0 for c in counts]
wedges, texts, autotexts = axes[1].pie(
    pcts, labels=cats, colors=colors,
    autopct="%1.1f%%", startangle=140,
    wedgeprops={"edgecolor": "white", "linewidth": 1.5},
    textprops={"fontsize": 9},
)
axes[1].set_title("Atlas Confidence\n(proportion)", fontsize=10, fontweight="bold")

fig.suptitle("Atlas Heuristic — Localisation Confidence", fontsize=11, fontweight="bold")
fig.tight_layout()
out = OUT / "atlas_confidence_dist.png"
fig.savefig(out, dpi=180)
print(f"Saved: {out}")
plt.close(fig)

lobe_counts = Counter(lobes)
top_lobes   = lobe_counts.most_common(10)
lobe_names  = [x[0] for x in top_lobes]
lobe_vals   = [x[1] for x in top_lobes]
lobe_colors = plt.cm.Blues(np.linspace(0.4, 0.85, len(lobe_names)))

fig, ax = plt.subplots(figsize=(9, 4.5))
bars = ax.barh(lobe_names[::-1], lobe_vals[::-1], color=lobe_colors[::-1], edgecolor="white")
ax.set_xlabel("Number of lesions", fontsize=9)
ax.set_title("Cerebral Lobe Distribution of Tumour Lesions\n(all Brain MRI cases)", fontsize=11, fontweight="bold", pad=12)
ax.tick_params(axis="y", labelsize=9)
for bar, val in zip(bars, lobe_vals[::-1]):
    ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2,
            str(val), va="center", fontsize=8.5, color="#374151")
fig.tight_layout()
out = OUT / "atlas_lobe_distribution.png"
fig.savefig(out, dpi=180)
print(f"Saved: {out}")
plt.close(fig)

hemi_counts  = Counter(hemispheres)
depth_counts = Counter(depths)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

h_keys = list(hemi_counts.keys())
h_vals = [hemi_counts[k] for k in h_keys]
h_cols = ["#3b82f6" if "left" in k.lower() else "#f87171" if "right" in k.lower() else "#a3a3a3" for k in h_keys]
axes[0].bar(h_keys, h_vals, color=h_cols, edgecolor="white", width=0.5)
axes[0].set_title("Hemisphere Distribution", fontsize=10, fontweight="bold")
axes[0].set_ylabel("Number of lesions", fontsize=9)

d_keys = list(depth_counts.keys())
d_vals = [depth_counts[k] for k in d_keys]
d_cols = plt.cm.Purples(np.linspace(0.4, 0.85, len(d_keys)))
axes[1].bar(d_keys, d_vals, color=d_cols, edgecolor="white", width=0.5)
axes[1].set_title("Depth Class Distribution", fontsize=10, fontweight="bold")
axes[1].set_ylabel("Number of lesions", fontsize=9)
axes[1].tick_params(axis="x", rotation=20, labelsize=8)

fig.suptitle("Atlas Heuristic — Spatial Distribution", fontsize=11, fontweight="bold")
fig.tight_layout()
out = OUT / "atlas_hemisphere.png"
fig.savefig(out, dpi=180)
print(f"Saved: {out}")
plt.close(fig)

print("\neval_03_atlas.py complete.")
