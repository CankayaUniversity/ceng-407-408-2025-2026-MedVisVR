import json
from pathlib import Path
from collections import defaultdict
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

organs = defaultdict(list)
completeness = []

for case_dir in sorted(CASES.iterdir()):
    pc_path = case_dir / "patient_context.json"
    if not pc_path.exists():
        continue
    pc = json.loads(pc_path.read_text(encoding="utf-8-sig"))
    modality = pc.get("modality", "").upper()
    organ    = pc.get("organ", "").lower()

    if modality == "MR":
        organ = "brain"
    elif not organ:
        name = case_dir.name.lower()
        if "liver" in name: organ = "liver"
        elif "lung"  in name: organ = "lung"
        else: organ = "unknown"

    has_report  = (case_dir / "report_draft.md").exists()
    has_seg     = (case_dir / "tumor_metrics.json").exists()
    has_qa      = (case_dir / "qa_results.jsonl").exists()
    has_quality = (case_dir / "quality_report.json").exists()
    has_context = pc_path.exists()

    organs[organ].append(case_dir.name)
    completeness.append({
        "id":     case_dir.name,
        "organ":  organ,
        "context":  has_context,
        "seg":      has_seg,
        "report":   has_report,
        "qa":       has_qa,
        "quality":  has_quality,
    })

organ_counts = {k: len(v) for k, v in organs.items()}
total = sum(organ_counts.values())
print(f"Total cases: {total}")
for o, n in organ_counts.items():
    print(f"  {o}: {n}")

organ_order  = ["brain", "liver", "lung"]
org_labels   = ["Brain MRI", "Liver CT", "Lung CT"]
org_colors   = ["#2563eb", "#16a34a", "#d97706"]
org_vals     = [organ_counts.get(o, 0) for o in organ_order]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

wedges, texts, autotexts = axes[0].pie(
    org_vals,
    labels=[f"{l}\n({v})" for l, v in zip(org_labels, org_vals)],
    colors=org_colors,
    autopct="%1.0f%%",
    startangle=90,
    pctdistance=0.78,
    wedgeprops={"edgecolor": "white", "linewidth": 2, "width": 0.55},
    textprops={"fontsize": 9},
)
for at in autotexts:
    at.set_fontsize(9)
    at.set_fontweight("bold")
    at.set_color("white")
axes[0].set_title(f"Case Distribution by Organ\n(N = {total})", fontsize=10, fontweight="bold")

axes[1].barh(org_labels, org_vals, color=org_colors, edgecolor="white", height=0.5)
axes[1].set_xlabel("Number of cases", fontsize=9)
axes[1].set_title("Cases per Organ (absolute)", fontsize=10, fontweight="bold")
for i, v in enumerate(org_vals):
    axes[1].text(v + 0.1, i, str(v), va="center", fontsize=10, fontweight="600")
axes[1].set_xlim(0, max(org_vals) * 1.15)

fig.suptitle("CARVIS — Dataset Overview", fontsize=12, fontweight="bold")
fig.tight_layout()
out = OUT / "system_case_distribution.png"
fig.savefig(out, dpi=180)
print(f"Saved: {out}")
plt.close(fig)

components  = ["context", "seg", "report", "qa", "quality"]
comp_labels = ["Patient\nContext", "Segmentation\nMetrics", "Report\nDraft", "Q&A\nResults", "Quality\nReport"]

brain_cases = [c for c in completeness if c["organ"] == "brain"]
liver_cases = [c for c in completeness if c["organ"] == "liver"]
lung_cases  = [c for c in completeness if c["organ"] == "lung"]

def comp_rate(cases, comp):
    if not cases: return 0
    return sum(1 for c in cases if c[comp]) / len(cases) * 100

rows_data = []
row_labels = []
for grp, grp_cases in [("Brain MRI", brain_cases), ("Liver CT", liver_cases), ("Lung CT", lung_cases)]:
    if grp_cases:
        rows_data.append([comp_rate(grp_cases, c) for c in components])
        row_labels.append(f"{grp} (n={len(grp_cases)})")

if rows_data:
    data = np.array(rows_data)
    fig, ax = plt.subplots(figsize=(9, 3.2))
    im = ax.imshow(data, cmap="Blues", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(components)))
    ax.set_xticklabels(comp_labels, fontsize=9)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.set_title("Pipeline Component Availability per Organ Group (%)", fontsize=11, fontweight="bold", pad=10)
    for i in range(len(row_labels)):
        for j in range(len(components)):
            val = data[i, j]
            ax.text(j, i, f"{val:.0f}%",
                    ha="center", va="center",
                    fontsize=10, fontweight="bold",
                    color="white" if val > 55 else "#1a1a2e")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Availability (%)")
    fig.tight_layout()
    out = OUT / "system_report_completeness.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)

def pct(n, d): return f"{n/d*100:.0f}%" if d else "—"

b_n = len(brain_cases)
l_n = len(liver_cases)
u_n = len(lung_cases)

rows = [
    ["Organ",        "Cases", "Has Seg.", "Has Report", "Has Q&A", "Has Quality"],
    ["Brain MRI",    b_n,
     pct(sum(c["seg"]    for c in brain_cases), b_n),
     pct(sum(c["report"] for c in brain_cases), b_n),
     pct(sum(c["qa"]     for c in brain_cases), b_n),
     pct(sum(c["quality"]for c in brain_cases), b_n)],
    ["Liver CT",     l_n,
     pct(sum(c["seg"]    for c in liver_cases), l_n),
     pct(sum(c["report"] for c in liver_cases), l_n),
     pct(sum(c["qa"]     for c in liver_cases), l_n),
     pct(sum(c["quality"]for c in liver_cases), l_n)],
    ["Lung CT",      u_n,
     pct(sum(c["seg"]    for c in lung_cases), u_n),
     pct(sum(c["report"] for c in lung_cases), u_n),
     pct(sum(c["qa"]     for c in lung_cases), u_n),
     pct(sum(c["quality"]for c in lung_cases), u_n)],
    ["TOTAL",        total,
     pct(sum(c["seg"]    for c in completeness), total),
     pct(sum(c["report"] for c in completeness), total),
     pct(sum(c["qa"]     for c in completeness), total),
     pct(sum(c["quality"]for c in completeness), total)],
]

fig, ax = plt.subplots(figsize=(10, 2.5))
ax.axis("off")
tbl = ax.table(cellText=rows[1:], colLabels=rows[0], loc="center", cellLoc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(9.5)
tbl.scale(1, 1.7)
for (r, c), cell in tbl.get_celld().items():
    if r == 0:
        cell.set_facecolor("#1a1a2e")
        cell.set_text_props(color="white", fontweight="bold")
    elif r == len(rows) - 1:
        cell.set_facecolor("#f1f5f9")
        cell.set_text_props(fontweight="700")
    elif c == 0:
        cell.set_facecolor("#f8fafc")
        cell.set_text_props(fontweight="600")
    cell.set_edgecolor("#e5e7eb")

ax.set_title("System Component Coverage — All Cases", fontsize=11, fontweight="bold", pad=10)
fig.tight_layout()
out = OUT / "system_overview_table.png"
fig.savefig(out, dpi=180, bbox_inches="tight")
print(f"Saved: {out}")
plt.close(fig)

print("\neval_04_system.py complete.")
