import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.image as mpimg
import matplotlib.colors as mc

ROOT  = Path(__file__).parent
CASES = ROOT / "outputs" / "cases"
OUT   = ROOT / "outputs" / "evaluation_v2"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "axes.facecolor":    "#ffffff",
    "figure.facecolor":  "#ffffff",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.color":        "#e5e7eb",
    "grid.linewidth":    0.8,
    "grid.alpha":        0.6,
    "font.family":       "DejaVu Sans",
    "axes.titlesize":    14,
    "axes.labelsize":    12,
    "xtick.labelsize":   11,
    "ytick.labelsize":   11,
})

BLUE   = "#2563eb"
LBLUE  = "#3b82f6"
LLBLUE = "#93c5fd"
GREEN  = "#16a34a"
RED    = "#dc2626"
GRAY   = "#6b7280"

print("Generating metrics_table_poster.png ...")

results = [
    ("Whole Tumour",      "Brain",  0.895, 0.907, 0.888, 0.999, 0.895, 0.944),
    ("Tumour Core",       "Brain",  0.841, 0.858, 0.834, 0.999, 0.841, 0.916),
    ("Enhancing Tumour",  "Brain",  0.804, 0.820, 0.799, 0.999, 0.804, 0.899),
    ("Liver Parenchyma",  "Liver",  0.758, 0.827, 0.700, 0.995, 0.758, 0.847),
    ("Liver Tumour",      "Liver",  0.057, 0.116, 0.038, 1.000, 0.057, 0.519),
]
COL_LABELS = ["Dice", "Precision", "Recall", "Specificity", "F1", "Bal. Accuracy"]
M_KEYS     = ["Dice", "Prec",      "Recall", "Spec",        "F1", "BalAcc"]
row_cols   = [BLUE, LBLUE, LLBLUE, GREEN, RED]

data = np.array([r[2:] for r in results])

fig, ax = plt.subplots(figsize=(12, 3.8))
ax.axis("off")

tbl = ax.table(
    cellText=[[f"{v:.3f}" for v in row] for row in data],
    rowLabels=[r[0] for r in results],
    colLabels=COL_LABELS,
    loc="center", cellLoc="center",
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(12)
tbl.scale(1, 2.4)

for (r, c), cell in tbl.get_celld().items():
    cell.set_edgecolor("#cbd5e1")
    if r == 0:
        cell.set_facecolor("#0f172a")
        cell.set_text_props(color="white", fontweight="bold", fontsize=11)
    elif c == -1:
        cell.set_facecolor(row_cols[r-1] + "18")
        cell.set_text_props(fontweight="700", fontsize=11, color="#1e293b")
        cell.set_width(0.22)
    else:
        val = data[r-1][c]
        base = mc.to_rgb(row_cols[r-1])
        a = 0.12 + val * 0.70
        blended = tuple(1 - a + a*b for b in base)
        cell.set_facecolor(blended)
        cell.set_text_props(
            fontweight="bold",
            color="white" if val > 0.72 else "#1e293b",
            fontsize=12,
        )

ax.set_title("Segmentation Evaluation Metrics", fontsize=15, fontweight="bold",
             pad=16, loc="center")

fig.text(0.02, 0.01,
    "Brain MRI: nnUNet BraTS2020 reference (Isensee et al., Nature Methods 2021)  |  "
    "Liver CT: liver_001, n=1 preliminary (MSD Task03)  |  "
    "Balanced Accuracy = (Sensitivity + Specificity) / 2  |  Raw Accuracy omitted",
    fontsize=8, color="#64748b", va="bottom")

fig.tight_layout(rect=[0, 0.06, 1, 1])
p = OUT / "metrics_table_poster.png"
fig.savefig(p, dpi=180, bbox_inches="tight", facecolor="white")
print(f"  Saved: {p}"); plt.close(fig)


print("Generating metrics_bar_poster.png ...")

labels_all = [r[0] for r in results]
colors_all = [BLUE, LBLUE, LLBLUE, GREEN, RED]
vals_all   = {r[0]: dict(zip(M_KEYS, r[2:])) for r in results}

fig, ax = plt.subplots(figsize=(14, 6))
x = np.arange(len(M_KEYS))
n = len(labels_all)
w = 0.68 / n

for i, label in enumerate(labels_all):
    vals   = [vals_all[label][m] for m in M_KEYS]
    offset = (i - (n-1)/2) * w
    is_liver = results[i][1] == "Liver"
    bars = ax.bar(x + offset, vals, w * 0.90,
                  color=colors_all[i],
                  edgecolor="#374151" if is_liver else "white",
                  linewidth=1.2 if is_liver else 0.5,
                  label=label,
                  alpha=1.0)
    for bar, val in zip(bars, vals):
        if 0.08 < val < 0.99:
            ax.text(bar.get_x() + bar.get_width()/2,
                    val + 0.011, f"{val:.2f}",
                    ha="center", va="bottom",
                    fontsize=8, fontweight="bold", color="#1e293b")

ax.set_xticks(x)
ax.set_xticklabels(["Dice", "Precision", "Recall", "Specificity", "F1", "Balanced\nAccuracy"],
                   fontsize=12)
ax.set_ylabel("Score", fontsize=12)
ax.set_ylim(0, 1.18)
ax.axhline(0.8, color="#94a3b8", linewidth=1.2, linestyle="--")
ax.text(len(M_KEYS) - 0.45, 0.815, "0.80 threshold", fontsize=9.5, color="#64748b")

ax.set_title("Segmentation Performance — Brain MRI & Liver CT", fontsize=15, fontweight="bold")

leg_patches = [mpatches.Patch(color=colors_all[i], label=labels_all[i])
               for i in range(len(labels_all))]
ax.legend(handles=leg_patches, fontsize=10.5, framealpha=0.9,
          loc="upper right", ncol=1, edgecolor="#e2e8f0")

fig.text(0.02, 0.01,
    "Brain: nnUNet BraTS2020 reference  |  Liver: n=1 preliminary (liver_001, MSD Task03)  |  "
    "Liver Tumour Dice low — voxel-alignment caveat  |  Raw Accuracy excluded",
    fontsize=8.5, color="#64748b")

fig.tight_layout(rect=[0, 0.05, 1, 1])
p = OUT / "metrics_bar_poster.png"
fig.savefig(p, dpi=180, bbox_inches="tight", facecolor="white")
print(f"  Saved: {p}"); plt.close(fig)


print("Generating qa_evidence_coverage_poster.png ...")

records = []
for case_dir in sorted(CASES.iterdir()):
    pc_path = case_dir / "patient_context.json"
    qr_path = case_dir / "quality_report.json"
    qa_path = case_dir / "qa_results.jsonl"
    if not pc_path.exists(): continue
    pc = json.loads(pc_path.read_text(encoding="utf-8-sig"))
    if pc.get("modality","").upper() != "MR": continue

    entry = {"id": case_dir.name.replace("BraTS20_Validation_", "Case "),
             "total_q": 0, "insufficient": 0, "supported": 0, "avg_conf": 0.0}

    if qr_path.exists():
        qr = json.loads(qr_path.read_text(encoding="utf-8-sig"))
        entry["total_q"]      = qr.get("total_questions", 0)
        entry["insufficient"] = qr.get("insufficient_count", 0)
        entry["supported"]    = qr.get("evidence_coverage_count", 0)
        entry["avg_conf"]     = qr.get("avg_confidence", 0.0)
    elif qa_path.exists():
        rows = [json.loads(l) for l in qa_path.read_text(encoding="utf-8-sig").splitlines() if l.strip()]
        entry["total_q"]      = len(rows)
        entry["insufficient"] = sum(1 for r in rows if str(r.get("answer","")).lower().startswith("insufficient"))
        entry["supported"]    = sum(1 for r in rows if r.get("evidence_ids"))
        confs = [float(r.get("confidence", 0)) for r in rows if r.get("confidence")]
        entry["avg_conf"]     = np.mean(confs) if confs else 0.0

    if entry["total_q"] > 0:
        records.append(entry)

if records:
    ids         = [r["id"] for r in records]
    supp_rate   = [r["supported"]    / r["total_q"] * 100 for r in records]
    insuff_rate = [r["insufficient"] / r["total_q"] * 100 for r in records]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    x = np.arange(len(ids))
    axes[0].bar(x, supp_rate,   0.55, label="Evidence-supported", color=BLUE,  edgecolor="white")
    axes[0].bar(x, insuff_rate, 0.55, label="Insufficient evidence", color="#f87171", edgecolor="white",
                bottom=supp_rate, alpha=0.85)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(ids, rotation=35, ha="right", fontsize=10)
    axes[0].set_ylabel("Percentage of answers (%)", fontsize=12)
    axes[0].set_ylim(0, 120)
    axes[0].axhline(np.mean(supp_rate), color=BLUE, linestyle="--", linewidth=1.2, alpha=0.7)
    axes[0].set_title("Evidence Coverage per Case", fontsize=13, fontweight="bold")
    axes[0].legend(fontsize=10.5, framealpha=0.9)
    for xi, (s, ins) in enumerate(zip(supp_rate, insuff_rate)):
        axes[0].text(xi, s + ins + 2, f"{s:.0f}%", ha="center", fontsize=9, color=BLUE, fontweight="bold")

    avg_confs = [r["avg_conf"] for r in records]
    bars = axes[1].bar(ids, avg_confs, color=[BLUE]*len(records), edgecolor="white", width=0.5)
    for bar, val in zip(bars, avg_confs):
        axes[1].text(bar.get_x()+bar.get_width()/2, val+0.01, f"{val:.2f}",
                     ha="center", va="bottom", fontsize=10, fontweight="bold")
    axes[1].set_xticks(np.arange(len(ids)))
    axes[1].set_xticklabels(ids, rotation=35, ha="right")
    axes[1].set_ylabel("Average confidence score", fontsize=12)
    axes[1].set_ylim(0, 1.1)
    axes[1].axhline(np.mean(avg_confs), color=RED, linestyle="--", linewidth=1.2,
                    label=f"Mean: {np.mean(avg_confs):.2f}")
    axes[1].legend(fontsize=10.5)
    axes[1].set_title("Q&A Confidence Score per Case", fontsize=13, fontweight="bold")

    fig.suptitle("Clinical Q&A System Performance — Brain MRI Cases", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p = OUT / "qa_evidence_coverage_poster.png"
    fig.savefig(p, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"  Saved: {p}"); plt.close(fig)


print("Generating report_quality_bars_poster.png ...")

SECTION_ALIASES = {
    "Clinical Summary":        ["clinical summary","summary","impression"],
    "Imaging Findings":        ["imaging findings","findings","mri findings","ct findings"],
    "Tumour Characteristics":  ["tumour characteristics","tumor characteristics","characteristics","tumour metrics"],
    "Anatomical Localisation": ["anatomical localisation","atlas localisation","localisation","anatomy","lobe"],
    "Differential Diagnosis":  ["differential","ddx","differential diagnosis"],
    "Recommendations":         ["recommend","next step","management","follow"],
    "Clinical Note":           ["clinical note","clinical history","history","note"],
}
REPORT_SECTIONS = list(SECTION_ALIASES.keys())
CLINICAL_TERMS  = [
    "tumour","tumor","lesion","neoplasm","glioma","glioblastoma",
    "metastasis","enhancement","oedema","edema","necrosis","mass",
    "liver","hepatic","lung","pulmonary","MRI","CT","hemisphere","lobe",
    "parenchyma","ventricle","cortex","Dice","segmentation","volume",
    "radiotherapy","chemotherapy","prognosis","treatment","resection",
]

def sec_score(text, sec):
    return 1 if any(kw in text.lower() for kw in SECTION_ALIASES[sec]) else 0

def clin_density(text):
    words = text.lower().split()
    return sum(1 for w in words if any(t in w for t in CLINICAL_TERMS)) / max(len(words),1) * 100

report_rows = []
for case_dir in sorted(CASES.iterdir()):
    pc_path = case_dir / "patient_context.json"
    rp_path = case_dir / "report_draft.md"
    if not pc_path.exists() or not rp_path.exists(): continue
    pc    = json.loads(pc_path.read_text(encoding="utf-8-sig"))
    organ = pc.get("organ","").lower() or ("brain" if pc.get("modality","").upper()=="MR" else "unknown")
    if organ not in ("brain","liver","lung"):
        organ = ("liver" if "liver" in case_dir.name else
                 "lung"  if "lung"  in case_dir.name else "brain")
    text   = rp_path.read_text(encoding="utf-8-sig", errors="replace")
    scores = {s: sec_score(text, s) for s in REPORT_SECTIONS}
    report_rows.append({
        "id": case_dir.name.replace("BraTS20_Validation_","Case "),
        "organ": organ,
        "word_count":   len(text.split()),
        "completeness": sum(scores.values()) / len(REPORT_SECTIONS) * 100,
        "density":      clin_density(text),
        **scores,
    })

report_rows.sort(key=lambda r: (r["organ"], r["id"]))
organ_color = {"brain": BLUE, "liver": GREEN, "lung": "#d97706"}

if report_rows:
    ids_r  = [r["id"] for r in report_rows]
    cols_r = [organ_color.get(r["organ"], GRAY) for r in report_rows]
    wc     = [r["word_count"]   for r in report_rows]
    comp   = [r["completeness"] for r in report_rows]
    dens   = [r["density"]      for r in report_rows]

    fig, axes = plt.subplots(1, 3, figsize=(16, 6), sharey=False)
    fig.patch.set_facecolor("white")

    panel_data = [
        (axes[0], wc,   "Word Count",               "Words per Report"),
        (axes[1], comp, "Section Completeness (%)",  "% of Sections Present"),
        (axes[2], dens, "Clinical Term Density (%)", "% Clinical Terms"),
    ]
    for ax_, vals, title, ylabel in panel_data:
        bars = ax_.bar(ids_r, vals, color=cols_r, edgecolor="white", width=0.6)
        ax_.set_title(title, fontsize=13, fontweight="bold", pad=10)
        ax_.set_ylabel(ylabel, fontsize=11)
        ax_.set_xticks(range(len(ids_r)))
        ax_.set_xticklabels(ids_r, rotation=40, ha="right", fontsize=8.5)
        mean_val = np.mean(vals)
        ax_.axhline(mean_val, color=RED, linestyle="--", linewidth=1.3,
                    label=f"Mean: {mean_val:.0f}" if ylabel != "% Clinical Terms"
                           else f"Mean: {mean_val:.1f}%")
        ax_.legend(fontsize=10)
        for bar, val in zip(bars, vals):
            ax_.text(bar.get_x()+bar.get_width()/2, val + max(vals)*0.01,
                     f"{val:.0f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    patches = [
        mpatches.Patch(color=BLUE,      label="Brain MRI"),
        mpatches.Patch(color=GREEN,     label="Liver CT  (n=1 GT)"),
        mpatches.Patch(color="#d97706", label="Lung CT  (inference only)"),
    ]
    fig.legend(handles=patches, loc="lower center", ncol=3, fontsize=11,
               framealpha=0.9, bbox_to_anchor=(0.5, -0.02), edgecolor="#e2e8f0")

    fig.suptitle("Automated Clinical Report Quality — All Cases", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0.07, 1, 0.97])
    p = OUT / "report_quality_bars_poster.png"
    fig.savefig(p, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"  Saved: {p}"); plt.close(fig)


print("Generating report_quality_heatmap_poster.png ...")

SHORT_SECTIONS = [
    "Clinical\nSummary", "Imaging\nFindings", "Tumour\nCharact.",
    "Anatomical\nLocalisation", "Differential\nDiagnosis",
    "Recommen-\ndations", "Clinical\nNote",
]

if report_rows:
    n_cases = len(report_rows)
    mat = np.array([[r[s] for s in REPORT_SECTIONS] for r in report_rows], dtype=float)
    ids_short = [r["id"] for r in report_rows]

    fig_h = max(6, n_cases * 0.52 + 2.5)
    fig, ax = plt.subplots(figsize=(13, fig_h))
    fig.patch.set_facecolor("white")

    cmap_custom = matplotlib.colors.LinearSegmentedColormap.from_list(
        "presence", ["#f8fafc", "#16a34a"])
    im = ax.imshow(mat, cmap=cmap_custom, vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(len(SHORT_SECTIONS)))
    ax.set_xticklabels(SHORT_SECTIONS, fontsize=11, ha="center")
    ax.set_yticks(range(n_cases))
    ax.set_yticklabels(ids_short, fontsize=10)

    for i in range(n_cases):
        for j in range(len(REPORT_SECTIONS)):
            v = mat[i, j]
            ax.text(j, i, "Yes" if v else "No",
                    ha="center", va="center", fontsize=10.5, fontweight="bold",
                    color="white" if v else "#94a3b8")

    prev = None
    organ_start = {}
    for i, r in enumerate(report_rows):
        if r["organ"] != prev:
            if prev is not None:
                ax.axhline(i-0.5, color="#475569", linewidth=2)
            organ_start[r["organ"]] = i
            prev = r["organ"]

    from matplotlib.transforms import blended_transform_factory
    trans = blended_transform_factory(ax.transAxes, ax.transData)
    organ_groups = {}
    for i, r in enumerate(report_rows):
        organ_groups.setdefault(r["organ"], []).append(i)
    org_display = {"brain": ("Brain MRI", BLUE),
                   "liver": ("Liver CT\n(n=1 GT)", GREEN),
                   "lung":  ("Lung CT\n(inf. only)", "#d97706")}
    for org, idxs in organ_groups.items():
        name, col = org_display.get(org, (org, GRAY))
        ax.text(1.02, np.mean(idxs), name, transform=trans,
                va="center", ha="left", fontsize=10, fontweight="bold", color=col)

    ax.set_title("Report Section Coverage — All Cases", fontsize=15, fontweight="bold", pad=14)
    ax.set_xlabel("Report Section", fontsize=12, labelpad=10)

    fig.text(0.02, 0.005,
        "Green = section present in report  |  "
        "Liver CT: GT available for liver_001 only  |  Lung CT: inference only, no GT",
        fontsize=9, color="#64748b")

    fig.tight_layout(rect=[0, 0.04, 0.93, 1])
    p = OUT / "report_quality_heatmap_poster.png"
    fig.savefig(p, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"  Saved: {p}"); plt.close(fig)


print("Generating training_combined_poster.png ...")

BRAIN_DIR = ROOT / "nnunet_brain"
LIVER_DIR = ROOT / "nnunet_liver" / "nnUNetTrainerV2__nnUNetPlansv2.1"

def get_fold0(base_dir):
    p = base_dir / "fold_0" / "progress.png"
    return p if p.exists() else None

brain_p = get_fold0(BRAIN_DIR)
liver_p = get_fold0(LIVER_DIR)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.patch.set_facecolor("white")

for ax_, img_path, title, subtitle, col in [
    (axes[0], brain_p,
     "Brain MRI Segmentation",
     "nnUNet v2  |  BraTS-style multi-sequence MRI  |  1000 epochs  |  5-fold CV",
     "#1d4ed8"),
    (axes[1], liver_p,
     "Liver CT Segmentation",
     "nnUNet v1  |  MSD Task03_Liver  |  1000 epochs  |  5-fold CV",
     "#15803d"),
]:
    if img_path and img_path.exists():
        img = mpimg.imread(str(img_path))
        if "brain" in str(img_path):
            h = img.shape[0]
            img = img[:h//3, :, :]
        ax_.imshow(img)
    else:
        ax_.text(0.5, 0.5, "Training log\nnot available",
                 ha="center", va="center", fontsize=14, color=GRAY,
                 transform=ax_.transAxes)
        ax_.set_facecolor("#f8fafc")
    ax_.axis("off")
    ax_.set_title(title, fontsize=14, fontweight="bold", color=col, pad=10)
    ax_.text(0.5, -0.04, subtitle, transform=ax_.transAxes,
             ha="center", fontsize=10, color="#475569")

legend_items = [
    mpatches.Patch(color="#4472c4", label="Train Loss"),
    mpatches.Patch(color="#ed7d31", label="Validation Loss"),
    mpatches.Patch(color="#70ad47", label="Pseudo Dice (Val.)"),
]
fig.legend(handles=legend_items, loc="lower center", ncol=3,
           fontsize=11, framealpha=0.9, bbox_to_anchor=(0.5, -0.02),
           edgecolor="#e2e8f0")

fig.suptitle("Segmentation Model Training Curves (Fold 0 of 5)", fontsize=16, fontweight="bold")
fig.tight_layout(rect=[0, 0.06, 1, 0.97])
p = OUT / "training_combined_poster.png"
fig.savefig(p, dpi=160, bbox_inches="tight", facecolor="white")
print(f"  Saved: {p}"); plt.close(fig)


print("\n" + "="*55)
print("Done. eval_v2_poster_figures.py complete.")
print(f"\nAll poster figures saved to:")
print(f"  {OUT}")
print("\nFiles:")
for name in [
    "metrics_table_poster.png",
    "metrics_bar_poster.png",
    "qa_evidence_coverage_poster.png",
    "report_quality_bars_poster.png",
    "report_quality_heatmap_poster.png",
    "training_combined_poster.png",
]:
    exists = (OUT / name).exists()
    print(f"  {'OK' if exists else 'MISSING':6}  {name}")