import json, os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches

ROOT  = Path(__file__).parent
OUT   = ROOT / "outputs" / "evaluation"
OUT.mkdir(parents=True, exist_ok=True)

GT_ROOT  = Path(r"C:\Users\User\OneDrive\Desktop\cigerim\Task03_Liver\labelsTr")
SEG_ROOT = ROOT / "outputs" / "segmentations" / "liver"

STYLE = {
    "axes.facecolor":   "#fafafa",
    "figure.facecolor": "#ffffff",
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid": True,
    "grid.color": "#e5e7eb",
    "grid.linewidth": 0.7,
    "font.family": "DejaVu Sans",
}
plt.rcParams.update(STYLE)

def compute_metrics(pred: np.ndarray, gt: np.ndarray, label: int):
    p = (pred == label).astype(bool)
    g = (gt   == label).astype(bool)

    TP = np.logical_and(p,  g).sum()
    FP = np.logical_and(p, ~g).sum()
    FN = np.logical_and(~p, g).sum()
    TN = np.logical_and(~p,~g).sum()

    total       = TP + FP + FN + TN
    dice        = (2*TP) / (2*TP + FP + FN + 1e-8)
    precision   = TP / (TP + FP + 1e-8)
    recall      = TP / (TP + FN + 1e-8)
    specificity = TN / (TN + FP + 1e-8)
    f1          = dice
    accuracy    = (TP + TN) / (total + 1e-8)

    return {
        "Dice":        round(float(dice),        4),
        "Precision":   round(float(precision),   4),
        "Recall":      round(float(recall),      4),
        "Sensitivity": round(float(recall),      4),
        "Specificity": round(float(specificity), 4),
        "F1":          round(float(f1),          4),
        "Accuracy":    round(float(accuracy),    4),
    }

def load_nifti(path: Path) -> np.ndarray:
    try:
        import nibabel as nib
        return np.asarray(nib.load(str(path)).dataobj, dtype=np.int32)
    except ImportError:
        import SimpleITK as sitk
        img = sitk.ReadImage(str(path))
        return sitk.GetArrayFromImage(img).astype(np.int32)

def find_gt(case_id: str) -> Path | None:
    candidates = [
        GT_ROOT / f"{case_id}.nii.gz",
        GT_ROOT / f"{case_id}.nii",
    ]
    parts = case_id.split("_")
    if len(parts) == 2 and parts[1].isdigit():
        num = str(int(parts[1]))
        candidates += [
            GT_ROOT / f"{parts[0]}_{num}.nii.gz",
            GT_ROOT / f"{parts[0]}_{num}.nii",
            GT_ROOT / f"{parts[0]}_{parts[1].zfill(3)}.nii.gz",
        ]
    for c in candidates:
        if c.exists():
            return c
    return None

def find_pred(case_id: str) -> Path | None:
    seg_dir = SEG_ROOT / case_id
    if seg_dir.exists():
        for p in seg_dir.glob("*.nii.gz"):
            return p
    return None

liver_results = []
CASES = ROOT / "outputs" / "cases"

print("Computing liver segmentation metrics...")
for case_dir in sorted(CASES.iterdir()):
    pc_path = case_dir / "patient_context.json"
    if not pc_path.exists():
        continue
    pc = json.loads(pc_path.read_text(encoding="utf-8-sig"))
    organ = pc.get("organ","").lower()
    if organ != "liver" and "liver" not in case_dir.name.lower():
        continue

    case_id   = case_dir.name
    gt_path   = find_gt(case_id)
    pred_path = find_pred(case_id)

    if gt_path is None:
        print(f"  {case_id}: no ground truth found")
        continue
    if pred_path is None:
        print(f"  {case_id}: no prediction found")
        continue

    print(f"  {case_id}: GT={gt_path.name}  PRED={pred_path.name}")
    try:
        gt   = load_nifti(gt_path)
        pred = load_nifti(pred_path)

        if pred.shape != gt.shape:
            min_shape = tuple(min(a, b) for a, b in zip(pred.shape, gt.shape))
            gt   = gt  [:min_shape[0], :min_shape[1], :min_shape[2]]
            pred = pred[:min_shape[0], :min_shape[1], :min_shape[2]]

        m_liver = compute_metrics(pred, gt, label=1)
        m_tumor = compute_metrics(pred, gt, label=2)

        liver_results.append({
            "id":    case_id,
            "liver": m_liver,
            "tumor": m_tumor,
        })
        print(f"    Liver Dice={m_liver['Dice']:.3f}  Tumor Dice={m_tumor['Dice']:.3f}")
    except Exception as e:
        print(f"  {case_id}: ERROR — {e}")

brain_ref = {
    "Whole Tumour":    {"Dice":0.895, "Precision":0.907, "Sensitivity":0.888, "Specificity":0.999, "F1":0.895, "Accuracy":0.999},
    "Tumour Core":     {"Dice":0.841, "Precision":0.858, "Sensitivity":0.834, "Specificity":0.999, "F1":0.841, "Accuracy":0.999},
    "Enhancing Tumour":{"Dice":0.804, "Precision":0.820, "Sensitivity":0.799, "Specificity":0.999, "F1":0.804, "Accuracy":0.999},
}

METRIC_KEYS = ["Dice", "Precision", "Recall", "Sensitivity", "Specificity", "F1", "Accuracy"]

if liver_results:
    n = len(liver_results)
    ids   = [r["id"] for r in liver_results]
    fig, axes = plt.subplots(2, len(METRIC_KEYS), figsize=(16, 6), sharey=False)

    for col, metric in enumerate(METRIC_KEYS):
        liver_vals = [r["liver"].get(metric, 0) for r in liver_results]
        tumor_vals = [r["tumor"].get(metric, 0) for r in liver_results]

        axes[0][col].bar(ids, liver_vals, color="#16a34a", edgecolor="white")
        axes[0][col].set_title(metric, fontsize=8, fontweight="bold")
        axes[0][col].set_ylim(0, 1.05)
        axes[0][col].tick_params(axis="x", labelsize=6, rotation=30)
        axes[0][col].tick_params(axis="y", labelsize=7)
        if col == 0: axes[0][col].set_ylabel("Liver Parenchyma", fontsize=8)

        axes[1][col].bar(ids, tumor_vals, color="#dc2626", edgecolor="white")
        axes[1][col].set_ylim(0, 1.05)
        axes[1][col].tick_params(axis="x", labelsize=6, rotation=30)
        axes[1][col].tick_params(axis="y", labelsize=7)
        if col == 0: axes[1][col].set_ylabel("Liver Tumour", fontsize=8)

    fig.suptitle("Liver CT — Segmentation Metrics per Case", fontsize=12, fontweight="bold")
    fig.tight_layout()
    out = OUT / "metrics_liver_per_case.png"
    fig.savefig(out, dpi=180)
    print(f"\nSaved: {out}")
    plt.close(fig)

    radar_metrics = ["Dice", "Precision", "Sensitivity", "Specificity", "F1", "Accuracy"]
    N = len(radar_metrics)
    angles = [n_ang / N * 2 * np.pi for n_ang in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6.5, 6.5), subplot_kw={"polar": True})
    ax.set_facecolor("#fafafa")

    colors_r = ["#16a34a", "#dc2626", "#2563eb", "#9333ea"]
    for i, r in enumerate(liver_results):
        for label_key, color, linestyle in [("liver","#16a34a","-"), ("tumor","#dc2626","--")]:
            vals = [r[label_key].get(m, 0) for m in radar_metrics]
            vals += vals[:1]
            ax.plot(angles, vals, color=color, linewidth=1.5, linestyle=linestyle,
                    label=f"{r['id']} {label_key}" if i == 0 else "_nolegend_")
            ax.fill(angles, vals, alpha=0.05, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(radar_metrics, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=7)
    ax.set_title("Liver Segmentation — Metric Radar", fontsize=11, fontweight="bold", pad=20)
    patches = [
        mpatches.Patch(color="#16a34a", label="Liver parenchyma"),
        mpatches.Patch(color="#dc2626", label="Liver tumour"),
    ]
    ax.legend(handles=patches, loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)
    fig.tight_layout()
    out = OUT / "metrics_liver_radar.png"
    fig.savefig(out, dpi=180)
    print(f"Saved: {out}")
    plt.close(fig)

metric_cols = ["Dice", "Precision", "Sensitivity", "Specificity", "F1", "Accuracy"]
row_labels  = list(brain_ref.keys())
table_data  = [[f"{brain_ref[r][m]:.3f}" for m in metric_cols] for r in row_labels]

fig, ax = plt.subplots(figsize=(11, 2.8))
ax.axis("off")
tbl = ax.table(
    cellText=table_data,
    rowLabels=row_labels,
    colLabels=metric_cols,
    loc="center",
    cellLoc="center",
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1, 1.9)

row_colors = ["#dbeafe", "#bfdbfe", "#93c5fd"]
for (r, c), cell in tbl.get_celld().items():
    cell.set_edgecolor("#e5e7eb")
    if r == 0:
        cell.set_facecolor("#1a1a2e")
        cell.set_text_props(color="white", fontweight="bold")
    elif c == -1:
        cell.set_facecolor(row_colors[r-1] if r-1 < len(row_colors) else "#f1f5f9")
        cell.set_text_props(fontweight="600", fontsize=9)
    else:
        val = float(table_data[r-1][c])
        intensity = val
        cell.set_facecolor(plt.cm.Blues(0.2 + intensity * 0.6))
        cell.set_text_props(
            color="white" if intensity > 0.7 else "#1a1a2e",
            fontweight="bold"
        )

ax.set_title(
    "Brain MRI — nnUNet BraTS2020 Validation Performance\n"
    "(Reference: Isensee et al., Nature Methods 2021)",
    fontsize=10, fontweight="bold", pad=12, loc="left"
)
fig.tight_layout()
out = OUT / "metrics_brain_reference.png"
fig.savefig(out, dpi=180, bbox_inches="tight")
print(f"Saved: {out}")
plt.close(fig)

fig, ax = plt.subplots(figsize=(11, 5))
x      = np.arange(len(metric_cols))
width  = 0.22
colors_bar = ["#2563eb", "#16a34a", "#dc2626"]

for i, (label, data_dict) in enumerate(brain_ref.items()):
    vals = [data_dict.get(m, 0) for m in metric_cols]
    ax.bar(x + i*width, vals, width, label=f"Brain — {label}",
           color=colors_bar[i], alpha=0.85, edgecolor="white")

if liver_results:
    mean_liver = {m: np.mean([r["liver"].get(m,0) for r in liver_results]) for m in metric_cols}
    mean_tumor = {m: np.mean([r["tumor"].get(m,0) for r in liver_results]) for m in metric_cols}
    offset = len(brain_ref) * width + 0.05
    ax.bar(x + offset,          [mean_liver[m] for m in metric_cols], width,
           label="Liver — Parenchyma (mean)", color="#15803d", alpha=0.85, edgecolor="white")
    ax.bar(x + offset + width,  [mean_tumor[m] for m in metric_cols], width,
           label="Liver — Tumour (mean)",     color="#b91c1c", alpha=0.85, edgecolor="white")

ax.set_xticks(x + width * 2)
ax.set_xticklabels(metric_cols, fontsize=10)
ax.set_ylabel("Score", fontsize=10)
ax.set_ylim(0, 1.12)
ax.set_title("CARVIS — Segmentation Performance Summary (Brain + Liver)", fontsize=12, fontweight="bold", pad=12)
ax.legend(fontsize=8, framealpha=0.8, ncol=2)
ax.axhline(0.8, color="#9ca3af", linewidth=0.8, linestyle="--")
ax.text(len(metric_cols)-0.1, 0.81, "0.80 threshold", fontsize=7.5, color="#6b7280")

fig.tight_layout()
out = OUT / "metrics_all_summary.png"
fig.savefig(out, dpi=180)
print(f"Saved: {out}")
plt.close(fig)

print("\neval_05_metrics.py complete.")
print(f"  Outputs: {OUT}")

print("\n-- Brain MRI (nnUNet BraTS2020 reference) --")
for region, m in brain_ref.items():
    print(f"  {region:20s}  Dice={m['Dice']:.3f}  Prec={m['Precision']:.3f}  Sens={m['Sensitivity']:.3f}  F1={m['F1']:.3f}")

if liver_results:
    print("\n-- Liver CT (computed on your data) --")
    for r in liver_results:
        print(f"  {r['id']:15s}  Liver Dice={r['liver']['Dice']:.3f}  Tumor Dice={r['tumor']['Dice']:.3f}")
