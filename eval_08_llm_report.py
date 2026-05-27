import json, re
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT  = Path(__file__).parent
CASES = ROOT / "outputs" / "cases"
OUT   = ROOT / "outputs" / "evaluation"
OUT.mkdir(parents=True, exist_ok=True)

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

benchmarks = [
    ("MMLU (5-shot)",            74.2, 73.0, 64.2, 82.0),
    ("HellaSwag (0-shot)",       80.7, 81.1, 81.3, 88.0),
    ("HumanEval (pass@1)",       84.8, 72.6, 36.6, 87.2),
    ("GSM8K (8-shot)",           91.6, 84.5, 52.2, 93.0),
    ("MATH (4-shot)",            75.4, 51.9, 13.1, 70.2),
    ("ARC-Challenge (0-shot)",   65.0, 59.5, 60.0, 75.7),
    ("TruthfulQA (0-shot)",      54.2, 44.0, 42.3, 65.0),
]

bench_names  = [b[0] for b in benchmarks]
q25_vals     = [b[1] for b in benchmarks]
llama_vals   = [b[2] for b in benchmarks]
mistral_vals = [b[3] for b in benchmarks]
gpt_vals     = [b[4] for b in benchmarks]

x   = np.arange(len(bench_names))
w   = 0.20
fig, ax = plt.subplots(figsize=(14, 5.5))

ax.bar(x - 1.5*w, q25_vals,     w, label="Qwen2.5-7B (CARVIS)",   color="#2563eb", edgecolor="white")
ax.bar(x - 0.5*w, llama_vals,   w, label="LLaMA-3.1-8B",           color="#9ca3af", edgecolor="white", alpha=0.8)
ax.bar(x + 0.5*w, mistral_vals, w, label="Mistral-7B-v0.3",         color="#d1d5db", edgecolor="white", alpha=0.8)
ax.bar(x + 1.5*w, gpt_vals,     w, label="GPT-4o-mini (reference)", color="#f59e0b", edgecolor="white", alpha=0.7)

for i, v in enumerate(q25_vals):
    ax.text(x[i] - 1.5*w, v + 0.6, f"{v}", ha="center", va="bottom",
            fontsize=7.5, fontweight="bold", color="#1d4ed8")

ax.set_xticks(x)
ax.set_xticklabels(bench_names, fontsize=9, rotation=15, ha="right")
ax.set_ylabel("Score (%)", fontsize=10)
ax.set_ylim(0, 105)
ax.axhline(70, color="#9ca3af", linewidth=0.8, linestyle="--", alpha=0.6)
ax.text(len(bench_names)-0.5, 71, "70% baseline", fontsize=8, color="#9ca3af")
ax.set_title(
    "LLM Model Quality — Qwen2.5-7B-Instruct vs Comparable Open-Source Models\n"
    "Source: Qwen2.5 Technical Report (2024). CARVIS uses Q4_K_M quantised local inference.",
    fontsize=11, fontweight="bold", pad=14)
ax.legend(fontsize=9, framealpha=0.85, ncol=4, loc="upper left")

fig.tight_layout()
p = OUT / "llm_benchmark_table.png"
fig.savefig(p, dpi=180, bbox_inches="tight")
print(f"Saved: {p}")
plt.close(fig)

REPORT_SECTIONS = [
    "Clinical Summary",
    "Imaging Findings",
    "Tumour Characteristics",
    "Atlas Localisation",
    "Differential Diagnosis",
    "Recommendations",
    "Clinical Note",
]

CLINICAL_TERMS = [
    "tumour","tumor","lesion","neoplasm","glioma","glioblastoma",
    "metastasis","enhancement","oedema","edema","necrosis",
    "mass","infiltration","resection","biopsy","MRI","CT",
    "hemisphere","lobe","parenchyma","ventricle","midline",
    "prognosis","treatment","radiotherapy","chemotherapy",
    "liver","hepatic","cirrhosis","hepatocellular",
    "lung","pulmonary","nodule","consolidation",
    "frontal","parietal","temporal","occipital","cerebellum",
    "white matter","grey matter","cortex","subcortical",
    "Dice","segmentation","atlas","confidence","volume","diameter",
]

def section_score(text, section_name):
    aliases = {
        "Clinical Summary":       ["clinical summary","summary","impression"],
        "Imaging Findings":       ["imaging findings","findings","mri findings","ct findings"],
        "Tumour Characteristics": ["tumour characteristics","tumor characteristics","characteristics","tumour metrics","tumor metrics"],
        "Atlas Localisation":     ["atlas localisation","atlas localization","anatomical localisation","localisation","anatomy"],
        "Differential Diagnosis": ["differential","ddx","differential diagnosis"],
        "Recommendations":        ["recommend","next step","management","follow"],
        "Clinical Note":          ["clinical note","clinical history","history","note"],
    }
    kws = aliases.get(section_name, [section_name.lower()])
    return 1 if any(kw in text.lower() for kw in kws) else 0

def clinical_density(text):
    words = text.lower().split()
    if not words: return 0.0
    hits = sum(1 for w in words if any(t in w for t in CLINICAL_TERMS))
    return hits / len(words) * 100

report_rows = []

for case_dir in sorted(CASES.iterdir()):
    pc_path = case_dir / "patient_context.json"
    rp_path = case_dir / "report_draft.md"
    if not pc_path.exists() or not rp_path.exists():
        continue
    pc = json.loads(pc_path.read_text(encoding="utf-8-sig"))
    organ = pc.get("organ", "").lower() or ("brain" if pc.get("modality","").upper()=="MR" else "unknown")
    if organ not in ("brain","liver","lung"):
        if "liver" in case_dir.name: organ = "liver"
        elif "lung" in case_dir.name: organ = "lung"
        else: organ = "brain"

    text = rp_path.read_text(encoding="utf-8-sig", errors="replace")
    words = len(text.split())
    scores = {s: section_score(text, s) for s in REPORT_SECTIONS}
    completeness = sum(scores.values()) / len(REPORT_SECTIONS) * 100
    density = clinical_density(text)

    report_rows.append({
        "id": case_dir.name,
        "organ": organ,
        "word_count": words,
        "completeness": completeness,
        "density": density,
        **scores,
    })

print(f"\nCases with reports: {len(report_rows)}")
for r in report_rows:
    print(f"  {r['id']:35s}  words={r['word_count']:4d}  complete={r['completeness']:.0f}%  density={r['density']:.1f}%")

if report_rows:
    report_rows.sort(key=lambda r: (r["organ"], r["id"]))
    case_ids = [r["id"].replace("BraTS20_Validation_","BV_") for r in report_rows]
    mat = np.array([[r[s] for s in REPORT_SECTIONS] for r in report_rows], dtype=float)
    organ_colors = {"brain":"#dbeafe","liver":"#dcfce7","lung":"#fef9c3"}

    fig, ax = plt.subplots(figsize=(14, max(5, len(report_rows)*0.45 + 2)))
    im = ax.imshow(mat, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(len(REPORT_SECTIONS)))
    ax.set_xticklabels(REPORT_SECTIONS, fontsize=9, rotation=20, ha="right")
    ax.set_yticks(range(len(case_ids)))
    ax.set_yticklabels(case_ids, fontsize=8)

    for i in range(len(report_rows)):
        for j in range(len(REPORT_SECTIONS)):
            v = mat[i, j]
            ax.text(j, i, "yes" if v else "—", ha="center", va="center",
                    fontsize=8.5, fontweight="bold",
                    color="white" if v else "#9ca3af")

    prev = None
    for i, r in enumerate(report_rows):
        if r["organ"] != prev and prev is not None:
            ax.axhline(i - 0.5, color="white", linewidth=2)
        prev = r["organ"]

    fig.colorbar(im, ax=ax, shrink=0.6, label="Section present (1=yes, 0=no)")
    ax.set_title("Report Structural Completeness — Section Coverage per Case",
                 fontsize=12, fontweight="bold", pad=14)

    patches = [
        mpatches.Patch(color="#3b82f6", label="Brain MRI"),
        mpatches.Patch(color="#16a34a", label="Liver CT"),
        mpatches.Patch(color="#ca8a04", label="Lung CT"),
    ]
    ax.legend(handles=patches, loc="lower right", fontsize=8.5,
              bbox_to_anchor=(1.13, -0.02), framealpha=0.85)

    fig.tight_layout()
    p = OUT / "report_quality_heatmap.png"
    fig.savefig(p, dpi=180, bbox_inches="tight")
    print(f"Saved: {p}")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    short_ids = [r["id"].replace("BraTS20_Validation_","BV_") for r in report_rows]
    colors_organ = [("#3b82f6" if r["organ"]=="brain"
                     else "#16a34a" if r["organ"]=="liver" else "#ca8a04")
                    for r in report_rows]

    wc = [r["word_count"] for r in report_rows]
    axes[0].barh(short_ids, wc, color=colors_organ, edgecolor="white")
    axes[0].set_xlabel("Word count", fontsize=9)
    axes[0].set_title("Report Word Count\nper Case", fontsize=10, fontweight="bold")
    axes[0].axvline(np.mean(wc), color="#dc2626", linestyle="--", linewidth=1,
                    label=f"Mean: {np.mean(wc):.0f}")
    axes[0].legend(fontsize=8)

    comp = [r["completeness"] for r in report_rows]
    axes[1].barh(short_ids, comp, color=colors_organ, edgecolor="white")
    axes[1].set_xlabel("Completeness (%)", fontsize=9)
    axes[1].set_xlim(0, 110)
    axes[1].set_title("Report Section\nCompleteness (%)", fontsize=10, fontweight="bold")
    axes[1].axvline(np.mean(comp), color="#dc2626", linestyle="--", linewidth=1,
                    label=f"Mean: {np.mean(comp):.0f}%")
    axes[1].legend(fontsize=8)
    for i, v in enumerate(comp):
        axes[1].text(v + 0.5, i, f"{v:.0f}%", va="center", fontsize=7.5)

    dens = [r["density"] for r in report_rows]
    axes[2].barh(short_ids, dens, color=colors_organ, edgecolor="white")
    axes[2].set_xlabel("Clinical terms (%)", fontsize=9)
    axes[2].set_title("Clinical Terminology\nDensity (%)", fontsize=10, fontweight="bold")
    axes[2].axvline(np.mean(dens), color="#dc2626", linestyle="--", linewidth=1,
                    label=f"Mean: {np.mean(dens):.1f}%")
    axes[2].legend(fontsize=8)

    patches = [
        mpatches.Patch(color="#3b82f6", label="Brain MRI"),
        mpatches.Patch(color="#16a34a", label="Liver CT"),
        mpatches.Patch(color="#ca8a04", label="Lung CT"),
    ]
    fig.legend(handles=patches, loc="lower center", ncol=3, fontsize=9,
               framealpha=0.85, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("CARVIS — Report Quality Metrics", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.15)
    p = OUT / "report_quality_bars.png"
    fig.savefig(p, dpi=180, bbox_inches="tight")
    print(f"Saved: {p}")
    plt.close(fig)

COMPONENTS = [
    ("Patient Context",     "patient_context.json"),
    ("Segmentation Model",  "tumor_metrics.json"),
    ("Report (LLM)",        "report_draft.md"),
    ("Q&A System",          "qa_results.jsonl"),
    ("Quality Report",      "quality_report.json"),
    ("Atlas Heuristic",     None),
    ("RAG Evidence",        None),
]

comp_labels = [c[0] for c in COMPONENTS]
comp_matrix = []
case_ids_sys = []
case_organs  = []

for case_dir in sorted(CASES.iterdir()):
    pc_path = case_dir / "patient_context.json"
    if not pc_path.exists():
        continue
    pc = json.loads(pc_path.read_text(encoding="utf-8-sig"))
    organ = pc.get("organ","").lower() or ("brain" if pc.get("modality","").upper()=="MR" else "unknown")
    if organ not in ("brain","liver","lung"):
        if "liver" in case_dir.name: organ = "liver"
        elif "lung" in case_dir.name: organ = "lung"
        else: organ = "brain"

    row = []
    for comp_name, fname in COMPONENTS:
        if fname is not None:
            row.append(1.0 if (case_dir / fname).exists() else 0.0)
        elif comp_name == "Atlas Heuristic":
            tm = case_dir / "tumor_metrics.json"
            if tm.exists():
                try:
                    d = json.loads(tm.read_text(encoding="utf-8-sig"))
                    has_atlas = any("anatomy" in l for l in d.get("lesion_list",[]))
                    row.append(1.0 if has_atlas else 0.5)
                except:
                    row.append(0.0)
            else:
                row.append(0.0)
        elif comp_name == "RAG Evidence":
            qa = case_dir / "qa_results.jsonl"
            if qa.exists():
                try:
                    lines = [l for l in qa.read_text(encoding="utf-8-sig").splitlines() if l.strip()]
                    rows_qa = [json.loads(l) for l in lines]
                    has_rag = any(r.get("evidence_ids") or r.get("evidence") for r in rows_qa)
                    row.append(1.0 if has_rag else 0.5)
                except:
                    row.append(0.5)
            else:
                row.append(0.0)

    comp_matrix.append(row)
    cid = case_dir.name.replace("BraTS20_Validation_","BV_")
    case_ids_sys.append(cid)
    case_organs.append(organ)

mat2 = np.array(comp_matrix)

fig, ax = plt.subplots(figsize=(13, max(5, len(case_ids_sys)*0.38 + 2)))
cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
    "traffic", ["#fee2e2","#fef9c3","#dcfce7"])
im = ax.imshow(mat2, cmap=cmap, vmin=0, vmax=1, aspect="auto")

ax.set_xticks(range(len(comp_labels)))
ax.set_xticklabels(comp_labels, fontsize=9, rotation=18, ha="right")
ax.set_yticks(range(len(case_ids_sys)))
ax.set_yticklabels(case_ids_sys, fontsize=7.5)

labels_map = {1.0: "Active", 0.5: "Partial", 0.0: "—"}
for i in range(len(case_ids_sys)):
    for j in range(len(comp_labels)):
        v = mat2[i, j]
        txt = labels_map.get(v, f"{v:.0f}")
        ax.text(j, i, txt, ha="center", va="center",
                fontsize=8, fontweight="bold",
                color="#1a1a2e" if v < 0.8 else "#166534")

prev = None
for i, org in enumerate(case_organs):
    if org != prev and prev is not None:
        ax.axhline(i - 0.5, color="white", linewidth=2.5)
    prev = org

from matplotlib.transforms import blended_transform_factory
trans = blended_transform_factory(ax.transAxes, ax.transData)
organ_groups = defaultdict(list)
for i, org in enumerate(case_organs):
    organ_groups[org].append(i)
org_color_map = {"brain":"#2563eb","liver":"#16a34a","lung":"#ca8a04"}
for org, idxs in organ_groups.items():
    mid = np.mean(idxs)
    ax.text(1.01, mid, org.title(), transform=trans, va="center", ha="left",
            fontsize=8.5, fontweight="bold", color=org_color_map.get(org,"#374151"))

fig.colorbar(im, ax=ax, shrink=0.5, ticks=[0,0.5,1],
             label="0=Inactive  0.5=Partial  1=Active")
ax.set_title("CARVIS — AI System Component Usage per Case",
             fontsize=12, fontweight="bold", pad=14)
fig.tight_layout()
p = OUT / "system_components_heatmap.png"
fig.savefig(p, dpi=180, bbox_inches="tight")
print(f"Saved: {p}")
plt.close(fig)

print("\nDone! eval_08_llm_report.py complete.")
print("\nSummary of outputs:")
for f in ["llm_benchmark_table.png","report_quality_heatmap.png",
          "report_quality_bars.png","system_components_heatmap.png"]:
    print(f"  {OUT/f}")
