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

records = []

for case_dir in sorted(CASES.iterdir()):
    pc_path = case_dir / "patient_context.json"
    qr_path = case_dir / "quality_report.json"
    qa_path = case_dir / "qa_results.jsonl"
    if not pc_path.exists():
        continue

    pc = json.loads(pc_path.read_text(encoding="utf-8-sig"))
    if pc.get("modality", "").upper() != "MR":
        continue

    entry = {
        "id": case_dir.name.replace("BraTS20_Validation_", ""),
        "total_q": 0, "insufficient": 0, "supported": 0,
        "avg_conf": 0.0,
        "conf_bands": {},
    }

    if qr_path.exists():
        qr = json.loads(qr_path.read_text(encoding="utf-8-sig"))
        entry["total_q"]     = qr.get("total_questions", 0)
        entry["insufficient"]= qr.get("insufficient_count", 0)
        entry["supported"]   = qr.get("evidence_coverage_count", 0)
        entry["avg_conf"]    = qr.get("avg_confidence", 0.0)
        entry["conf_bands"]  = qr.get("confidence_bands", {})
    elif qa_path.exists():
        rows = [json.loads(l) for l in qa_path.read_text(encoding="utf-8-sig").splitlines() if l.strip()]
        entry["total_q"] = len(rows)
        entry["insufficient"] = sum(1 for r in rows if str(r.get("answer","")).lower().startswith("insufficient"))
        entry["supported"]    = sum(1 for r in rows if r.get("evidence_ids"))
        confs = [float(r.get("confidence", 0)) for r in rows]
        entry["avg_conf"] = np.mean(confs) if confs else 0.0

    if entry["total_q"] > 0:
        records.append(entry)

print(f"Cases with Q&A data: {len(records)}")
if not records:
    print("No Q&A data found. Run Q&A for some cases first.")
    exit()

ids = [r["id"] for r in records]
total_qs    = [r["total_q"] for r in records]
insuff_rate = [r["insufficient"] / r["total_q"] * 100 if r["total_q"] else 0 for r in records]
supported_r = [r["supported"]   / r["total_q"] * 100 if r["total_q"] else 0 for r in records]
avg_confs   = [r["avg_conf"] for r in records]

fig, ax = plt.subplots(figsize=(13, 4.5))
x = np.arange(len(ids))
w = 0.38
b1 = ax.bar(x - w/2, supported_r, w, label="Evidence-supported (%)", color="#2563eb", edgecolor="white", linewidth=0.4)
b2 = ax.bar(x + w/2, insuff_rate, w, label="Insufficient evidence (%)", color="#f87171", edgecolor="white", linewidth=0.4)

ax.set_xticks(x)
ax.set_xticklabels(ids, fontsize=7, rotation=45)
ax.set_ylabel("Rate (%)", fontsize=9)
ax.set_ylim(0, 115)
ax.set_title("Clinical Q&A — Evidence Coverage vs. Insufficient Answer Rate", fontsize=11, fontweight="bold", pad=12)
ax.legend(fontsize=8, framealpha=0.7)
ax.axhline(np.mean(supported_r), color="#2563eb", linestyle="--", linewidth=0.8, alpha=0.6)
ax.axhline(np.mean(insuff_rate),  color="#f87171",  linestyle="--", linewidth=0.8, alpha=0.6)
fig.tight_layout()
out = OUT / "qa_evidence_coverage.png"
fig.savefig(out, dpi=180)
print(f"Saved: {out}")
plt.close(fig)

band_totals = defaultdict(int)
for r in records:
    for k, v in r["conf_bands"].items():
        band_totals[k] += v

if not band_totals:
    for conf in avg_confs:
        if   conf < 0.50: band_totals["lt_050"] += 1
        elif conf < 0.60: band_totals["050_059"] += 1
        elif conf < 0.70: band_totals["060_069"] += 1
        else:             band_totals["ge_070"]  += 1

band_labels = {"lt_050": "< 0.50", "050_059": "0.50–0.59", "060_069": "0.60–0.69", "ge_070": "≥ 0.70"}
band_colors = ["#e2e8f0", "#93c5fd", "#3b82f6", "#1d4ed8"]
keys   = ["lt_050", "050_059", "060_069", "ge_070"]
totals = [band_totals.get(k, 0) for k in keys]
labels = [band_labels[k] for k in keys]

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].bar(labels, totals, color=band_colors, edgecolor="white")
axes[0].set_title("Answer Confidence Distribution\n(all Q&A answers)", fontsize=10, fontweight="bold")
axes[0].set_ylabel("Number of answers", fontsize=9)
axes[0].tick_params(axis="x", labelsize=9)

axes[1].bar(ids, avg_confs, color="#3b82f6", edgecolor="white")
axes[1].set_title("Average Confidence per Case", fontsize=10, fontweight="bold")
axes[1].set_ylabel("Avg confidence score", fontsize=9)
axes[1].set_ylim(0, 1.0)
axes[1].tick_params(axis="x", labelsize=7, rotation=45)
axes[1].axhline(np.mean(avg_confs), color="#dc2626", linestyle="--", linewidth=1,
                label=f"Mean: {np.mean(avg_confs):.3f}")
axes[1].legend(fontsize=8)

fig.tight_layout()
out = OUT / "qa_confidence_dist.png"
fig.savefig(out, dpi=180)
print(f"Saved: {out}")
plt.close(fig)

n = len(records)
mean_total   = np.mean(total_qs)
mean_supp    = np.mean(supported_r)
mean_insuff  = np.mean(insuff_rate)
mean_conf    = np.mean(avg_confs)
total_q_all  = sum(total_qs)

rows = [
    ["Metric", "Value"],
    ["Cases evaluated", str(n)],
    ["Total Q&A answers", str(total_q_all)],
    [f"Avg questions / case", f"{mean_total:.1f}"],
    ["Evidence-supported rate (mean)", f"{mean_supp:.1f}%"],
    ["Insufficient evidence rate (mean)", f"{mean_insuff:.1f}%"],
    ["Average confidence score", f"{mean_conf:.3f}"],
]

fig, ax = plt.subplots(figsize=(7, 3))
ax.axis("off")
tbl = ax.table(cellText=rows[1:], colLabels=rows[0], loc="center", cellLoc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1, 1.7)
for (r, c), cell in tbl.get_celld().items():
    if r == 0:
        cell.set_facecolor("#1a1a2e")
        cell.set_text_props(color="white", fontweight="bold")
    elif c == 0:
        cell.set_facecolor("#f1f5f9")
        cell.set_text_props(fontweight="600")
    cell.set_edgecolor("#e5e7eb")

ax.set_title("Clinical Q&A — Performance Summary", fontsize=11, fontweight="bold", pad=10)
fig.tight_layout()
out = OUT / "qa_summary_table.png"
fig.savefig(out, dpi=180, bbox_inches="tight")
print(f"Saved: {out}")
plt.close(fig)

print("\neval_02_qa.py complete.")
