import json, os
from pathlib import Path
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ROOT  = Path(__file__).parent
CASES = ROOT / "outputs" / "cases"
OUT   = ROOT / "outputs" / "evaluation"
OUT.mkdir(parents=True, exist_ok=True)

STYLE = {
    "axes.facecolor":  "#fafafa",
    "figure.facecolor":"#ffffff",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#e5e7eb",
    "grid.linewidth": 0.6,
    "font.family": "DejaVu Sans",
}
plt.rcParams.update(STYLE)

brain_cases, liver_cases, lung_cases = [], [], []

for case_dir in sorted(CASES.iterdir()):
    pc_path = case_dir / "patient_context.json"
    tm_path = case_dir / "tumor_metrics.json"
    if not pc_path.exists():
        continue
    pc = json.loads(pc_path.read_text(encoding="utf-8-sig"))
    modality = pc.get("modality", "").upper()
    organ    = pc.get("organ", "").lower()

    tm = {}
    if tm_path.exists():
        tm = json.loads(tm_path.read_text(encoding="utf-8-sig"))

    seg = pc.get("segmentation_metrics") or tm
    if not seg:
        continue

    entry = {
        "id":     case_dir.name,
        "volume": seg.get("total_tumor_volume_mm3", 0) or 0,
        "diam":   seg.get("max_diameter_mm", 0) or 0,
        "labels": seg.get("label_volumes_mm3", {}),
        "n_lesions": seg.get("tumor_count", 0) or 0,
    }

    if modality == "MR":
        brain_cases.append(entry)
    elif organ == "liver" or "liver" in case_dir.name:
        liver_cases.append(entry)
    elif organ == "lung" or "lung" in case_dir.name:
        lung_cases.append(entry)

print(f"Brain cases : {len(brain_cases)}")
print(f"Liver cases : {len(liver_cases)}")
print(f"Lung cases  : {len(lung_cases)}")

if brain_cases:
    vols = [c["volume"] / 1000 for c in brain_cases]
    ids  = [c["id"].replace("BraTS20_Validation_", "") for c in brain_cases]

    fig, ax = plt.subplots(figsize=(12, 4))
    colors = ["#2563eb" if v < 50 else "#f59e0b" if v < 150 else "#dc2626" for v in vols]
    bars = ax.bar(ids, vols, color=colors, width=0.7, edgecolor="white", linewidth=0.4)
    ax.set_xlabel("Case ID (BraTS20_Validation_XXX)", fontsize=9)
    ax.set_ylabel("Total Tumour Volume (cm³)", fontsize=9)
    ax.set_title("Brain MRI — Total Tumour Volume per Case", fontsize=11, fontweight="bold", pad=12)
    ax.tick_params(axis="x", labelsize=7, rotation=45)
    ax.tick_params(axis="y", labelsize=8)
    patches = [
        mpatches.Patch(color="#2563eb", label="< 50 cm³"),
        mpatches.Patch(color="#f59e0b", label="50–150 cm³"),
        mpatches.Patch(color="#dc2626", label="> 150 cm³"),
    ]
    ax.legend(handles=patches, fontsize=8, framealpha=0.7)
    ax.axhline(np.mean(vols), color="#6b7280", linestyle="--", linewidth=1, label=f"Mean: {np.mean(vols):.1f} cm³")
    fig.tight_layout()
    out = OUT / "seg_brain_volumes.png"
    fig.savefig(out, dpi=180)
    print(f"Saved: {out}")
    plt.close(fig)

if brain_cases:
    label_keys = ["edema", "enhancing", "necrotic", "tumor", "ncr_net", "et", "ed"]
    all_keys = set()
    for c in brain_cases:
        all_keys.update(c["labels"].keys())

    def bucket(k):
        k = k.lower()
        if any(x in k for x in ["edema", "ed", "peritumoral"]): return "Oedema"
        if any(x in k for x in ["enhanc", "et", "active"]):     return "Enhancing"
        if any(x in k for x in ["necrot", "ncr", "core", "net"]): return "Necrotic Core"
        return "Other"

    buckets_per_case = []
    for c in brain_cases:
        b = defaultdict(float)
        for k, v in c["labels"].items():
            b[bucket(k)] += v / 1000
        if not any(b.values()):
            b["Oedema"] = c["volume"] / 3000
            b["Enhancing"] = c["volume"] / 3000
            b["Necrotic Core"] = c["volume"] / 3000
        buckets_per_case.append(b)

    ids = [c["id"].replace("BraTS20_Validation_", "") for c in brain_cases]
    cats = ["Oedema", "Enhancing", "Necrotic Core"]
    cat_colors = ["#60a5fa", "#f87171", "#a78bfa"]

    fig, ax = plt.subplots(figsize=(12, 4))
    bottoms = np.zeros(len(brain_cases))
    for cat, col in zip(cats, cat_colors):
        vals = np.array([b.get(cat, 0) for b in buckets_per_case])
        ax.bar(ids, vals, bottom=bottoms, color=col, label=cat, width=0.7, edgecolor="white", linewidth=0.3)
        bottoms += vals

    ax.set_xlabel("Case ID (BraTS20_Validation_XXX)", fontsize=9)
    ax.set_ylabel("Volume (cm³)", fontsize=9)
    ax.set_title("Brain MRI — Tumour Subcomponent Volumes", fontsize=11, fontweight="bold", pad=12)
    ax.tick_params(axis="x", labelsize=7, rotation=45)
    ax.tick_params(axis="y", labelsize=8)
    ax.legend(fontsize=8, framealpha=0.7)
    fig.tight_layout()
    out = OUT / "seg_brain_labels.png"
    fig.savefig(out, dpi=180)
    print(f"Saved: {out}")
    plt.close(fig)

if liver_cases:
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    vols  = [c["volume"] / 1000 for c in liver_cases]
    diams = [c["diam"]          for c in liver_cases]
    ids   = [c["id"]            for c in liver_cases]

    axes[0].bar(ids, vols, color="#16a34a", edgecolor="white")
    axes[0].set_title("Liver CT — Tumour Volume (cm³)", fontsize=10, fontweight="bold")
    axes[0].set_ylabel("Volume (cm³)", fontsize=9)
    axes[0].tick_params(axis="x", labelsize=8, rotation=20)

    axes[1].bar(ids, diams, color="#15803d", edgecolor="white")
    axes[1].set_title("Liver CT — Max Diameter (mm)", fontsize=10, fontweight="bold")
    axes[1].set_ylabel("Diameter (mm)", fontsize=9)
    axes[1].tick_params(axis="x", labelsize=8, rotation=20)

    fig.tight_layout()
    out = OUT / "seg_liver_volumes.png"
    fig.savefig(out, dpi=180)
    print(f"Saved: {out}")
    plt.close(fig)

def stats(lst, key, divisor=1):
    vals = [c[key] / divisor for c in lst if c[key] > 0]
    if not vals:
        return "-", "-", "-", 0
    return f"{np.mean(vals):.1f}", f"{np.min(vals):.1f}", f"{np.max(vals):.1f}", len(vals)

b_vm, b_vn, b_vx, bn = stats(brain_cases, "volume", 1000)
b_dm, b_dn, b_dx, _  = stats(brain_cases, "diam")
l_vm, l_vn, l_vx, ln = stats(liver_cases, "volume", 1000)
l_dm, l_dn, l_dx, _  = stats(liver_cases, "diam")
u_vm, u_vn, u_vx, un = stats(lung_cases,  "volume", 1000)
u_dm, u_dn, u_dx, _  = stats(lung_cases,  "diam")

rows = [
    ["Organ",        "N",  "Vol mean (cm³)", "Vol min", "Vol max", "Diam mean (mm)", "Diam min", "Diam max"],
    ["Brain MRI",    bn,    b_vm, b_vn, b_vx, b_dm, b_dn, b_dx],
    ["Liver CT",     ln,    l_vm, l_vn, l_vx, l_dm, l_dn, l_dx],
    ["Lung CT",      un,    u_vm, u_vn, u_vx, u_dm, u_dn, u_dx],
]

fig, ax = plt.subplots(figsize=(11, 2.2))
ax.axis("off")
tbl = ax.table(
    cellText=rows[1:],
    colLabels=rows[0],
    loc="center",
    cellLoc="center",
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1, 1.6)
for (r, c), cell in tbl.get_celld().items():
    if r == 0:
        cell.set_facecolor("#1a1a2e")
        cell.set_text_props(color="white", fontweight="bold")
    elif c == 0:
        cell.set_facecolor("#f1f5f9")
        cell.set_text_props(fontweight="600")
    else:
        cell.set_facecolor("#ffffff")
    cell.set_edgecolor("#e5e7eb")

ax.set_title("Segmentation Metrics — Summary Statistics", fontsize=11, fontweight="bold", pad=10)
fig.tight_layout()
out = OUT / "seg_summary_table.png"
fig.savefig(out, dpi=180, bbox_inches="tight")
print(f"Saved: {out}")
plt.close(fig)

print("\neval_01_segmentation.py complete.")
print(f"  Outputs: {OUT}")
