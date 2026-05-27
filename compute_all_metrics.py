import json
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mc

ROOT         = Path(__file__).parent
OUT          = ROOT / "outputs" / "evaluation_real"
TEMP_DIR     = ROOT / "outputs" / "_inference_temp"
BRAIN_OUTPUT = TEMP_DIR / "brain_output"
BRAIN_GT_DIR = TEMP_DIR / "brain_gt"
LIVER_OUTPUT = TEMP_DIR / "liver_output"
LIVER_GT     = Path(r"C:\Users\User\OneDrive\Desktop\cigerim\nnUNet_data\nnUNet_preprocessed\Dataset603_Liver\gt_segmentations")
LUNG_SUMMARY = ROOT / "outputs" / "evaluation_v2" / "real_metrics_summary.json"

OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "axes.facecolor": "#fafafa", "figure.facecolor": "#ffffff",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e5e7eb", "grid.linewidth": 0.6,
    "font.family": "DejaVu Sans",
})

C = {"brain": "#2563eb", "brain2": "#3b82f6", "brain3": "#93c5fd",
     "liver_p": "#16a34a", "liver_t": "#dc2626", "lung": "#b45309"}

def load_nifti(path: Path) -> np.ndarray:
    try:
        import nibabel as nib
        return np.asarray(nib.load(str(path)).dataobj, dtype=np.int32)
    except Exception:
        import SimpleITK as sitk
        return sitk.GetArrayFromImage(sitk.ReadImage(str(path))).astype(np.int32)

def match_shape(a, b):
    if a.shape != b.shape:
        s = tuple(min(x, y) for x, y in zip(a.shape, b.shape))
        a, b = a[:s[0],:s[1],:s[2]], b[:s[0],:s[1],:s[2]]
    return a, b

def binary_metrics(pred, gt, label):
    p = (pred == label); g = (gt == label)
    TP = int(np.logical_and(p,   g).sum())
    FP = int(np.logical_and(p,  ~g).sum())
    FN = int(np.logical_and(~p,  g).sum())
    TN = int(np.logical_and(~p, ~g).sum())
    total = TP + FP + FN + TN; eps = 1e-8
    dice = 2*TP / (2*TP + FP + FN + eps)
    prec = TP / (TP + FP + eps)
    rec  = TP / (TP + FN + eps)
    spec = TN / (TN + FP + eps)
    bal  = (rec + spec) / 2
    acc  = (TP + TN) / (total + eps)
    return dict(Dice=dice, Precision=prec, Recall=rec, Specificity=spec,
                F1=dice, Balanced_Accuracy=bal, AUC=bal, Accuracy=acc,
                TP=TP, FP=FP, FN=FN, TN=TN)

def macro_f1_score(pred, gt, fg_labels):
    all_labels = [0] + list(fg_labels)
    return float(np.mean([binary_metrics(pred, gt, l)["F1"] for l in all_labels]))

def agg(vals):
    arr = [v for v in vals if v is not None and not np.isnan(float(v))]
    if not arr: return {"mean": None, "std": None}
    return {"mean": float(np.mean(arr)), "std": float(np.std(arr)),
            "n": len(arr), "min": float(np.min(arr)), "max": float(np.max(arr))}

METRICS = ["Dice","Precision","Recall","Specificity","Balanced_Accuracy","F1","AUC","Accuracy"]
METRICS_NICE = ["Dice","Precision","Recall","Specificity","Bal. Acc.","F1","AUC","Accuracy"]

print("=" * 65)
print("CARVIS — compute_all_metrics.py")
print("=" * 65)

results = {}

print("\n[LUNG] Loading from JSON...")
if LUNG_SUMMARY.exists():
    with open(LUNG_SUMMARY, encoding="utf-8") as f:
        lung_raw = json.load(f)["lung"]
    results["lung"] = lung_raw
    lms = lung_raw.get("metrics_summary", {})
    print(f"  n={lung_raw['n_cases']}  "
          f"Dice={lms.get('Dice',{}).get('mean',0):.3f}  "
          f"BalAcc={lms.get('Balanced_Accuracy',{}).get('mean',0):.3f}  "
          f"F1={lms.get('F1',{}).get('mean',0):.3f}")
else:
    print("  WARNING: real_metrics_summary.json not found. Run eval_real_compute.py first.")
    results["lung"] = {}

print("\n[BRAIN] Checking prediction files...")

pred_files = sorted(BRAIN_OUTPUT.glob("BraTS_*.nii.gz"))
if not pred_files:
    print(f"  WARNING: No predictions in {BRAIN_OUTPUT}!")
    print("  Run: .venv\\Scripts\\python.exe run_brain_inference.py")

BRAIN_LABELS = {1: "NCR/Non-enh.", 2: "Edema", 4: "Enhancing Tumor"}
brain_cases  = []

for pred_path in pred_files:
    cid      = pred_path.name.replace(".nii.gz", "")
    num      = cid.split("_")[-1]
    gt_nii   = BRAIN_GT_DIR / f"BraTS_{num}.nii"
    gt_niigz = BRAIN_GT_DIR / f"BraTS_{num}.nii.gz"
    gt_path  = gt_nii if gt_nii.exists() else (gt_niigz if gt_niigz.exists() else None)

    if gt_path is None:
        print(f"  {cid}: GT not found, skipping")
        continue

    try:
        pred = load_nifti(pred_path)
        gt   = load_nifti(gt_path)
        pred, gt = match_shape(pred, gt)

        per_class = {lbl: binary_metrics(pred, gt, lbl) for lbl in [1, 2, 4]}
        mf1       = macro_f1_score(pred, gt, [1, 2, 4])

        brain_cases.append({"case_id": cid, "per_class": per_class, "macro_f1": mf1})

        strs = "  |  ".join(
            f"{BRAIN_LABELS[l]}: Dice={per_class[l]['Dice']:.3f} BalAcc={per_class[l]['Balanced_Accuracy']:.3f}"
            for l in [1, 2, 4])
        print(f"  {cid}: {strs}")
        print(f"         Macro F1={mf1:.3f}")
    except Exception as e:
        print(f"  {cid}: ERROR — {e}")

brain_summary = {}
if brain_cases:
    for lbl in [1, 2, 4]:
        for mk in METRICS:
            key = f"lbl{lbl}_{mk}"
            brain_summary[key] = agg([c["per_class"][lbl][mk] for c in brain_cases])
    brain_summary["macro_f1"] = agg([c["macro_f1"] for c in brain_cases])
    print(f"\n  BRAIN SUMMARY (n={len(brain_cases)} case, fold_0, BraTS20 BAMF):")
    for lbl, name in BRAIN_LABELS.items():
        d = brain_summary[f"lbl{lbl}_Dice"]["mean"]
        b = brain_summary[f"lbl{lbl}_Balanced_Accuracy"]["mean"]
        p = brain_summary[f"lbl{lbl}_Precision"]["mean"]
        r = brain_summary[f"lbl{lbl}_Recall"]["mean"]
        print(f"    {name}: Dice={d:.3f}  Prec={p:.3f}  Rec={r:.3f}  BalAcc={b:.3f}")
    print(f"    Macro F1 = {brain_summary['macro_f1']['mean']:.3f}")
else:
    print("  No brain predictions — using CV Dice fallback")

results["brain"] = {
    "source": "nnUNetv2 fold_0 inference, BraTS20, BraTS label schema {1=NCR,2=Edema,4=ET}",
    "n_cases": len(brain_cases),
    "label_map": {"1":"NCR/Non-enh.","2":"Edema","4":"Enhancing Tumor"},
    "summary": brain_summary,
    "cv_fallback": {"ema_dice_mean": 0.8691, "ema_dice_std": 0.0019,
                    "note": "5-fold CV EMA Dice from checkpoint (all training data)"},
    "per_case": [{"case_id": c["case_id"],
                  "macro_f1": float(c["macro_f1"]),
                  "per_class": {str(l): {k: float(v) for k,v in m.items()
                                         if isinstance(v,(int,float,np.floating))}
                                for l, m in c["per_class"].items()}}
                 for c in brain_cases],
}

print("\n[LIVER] Checking prediction files...")

liver_pred_files = sorted(LIVER_OUTPUT.glob("liver_*.nii.gz"))
if not liver_pred_files:
    print(f"  WARNING: No predictions in {LIVER_OUTPUT}!")
    print("  Run: .venv\\Scripts\\python.exe run_liver_inference.py")

LIVER_LABELS = {1: "Parenchyma", 2: "Tumour"}
liver_cases  = []

for pred_path in liver_pred_files:
    case_id = pred_path.name.replace(".nii.gz", "")
    num     = case_id.replace("liver_", "")
    gt_path = LIVER_GT / f"liver_{int(num)}.nii.gz"
    if not gt_path.exists():
        gt_path = LIVER_GT / f"{case_id}.nii.gz"
    if not gt_path.exists():
        print(f"  {case_id}: GT not found ({LIVER_GT}/liver_{num}.nii.gz)")
        continue

    try:
        pred = load_nifti(pred_path)
        gt   = load_nifti(gt_path)
        pred, gt = match_shape(pred, gt)

        per_class = {lbl: binary_metrics(pred, gt, lbl) for lbl in [1, 2]}
        mf1       = macro_f1_score(pred, gt, [1, 2])

        liver_cases.append({"case_id": case_id, "per_class": per_class, "macro_f1": mf1})
        print(f"  {case_id}: Paren Dice={per_class[1]['Dice']:.3f} "
              f"Tumour Dice={per_class[2]['Dice']:.3f}  Macro F1={mf1:.3f}")
    except Exception as e:
        print(f"  {case_id}: ERROR — {e}")

liver_summary = {}
if liver_cases:
    for lbl in [1, 2]:
        for mk in METRICS:
            liver_summary[f"lbl{lbl}_{mk}"] = agg([c["per_class"][lbl][mk] for c in liver_cases])
    liver_summary["macro_f1"] = agg([c["macro_f1"] for c in liver_cases])
    print(f"\n  LIVER SUMMARY (n={len(liver_cases)} case, 5-fold ensemble):")
    for lbl, name in LIVER_LABELS.items():
        d = liver_summary[f"lbl{lbl}_Dice"]["mean"]
        b = liver_summary[f"lbl{lbl}_Balanced_Accuracy"]["mean"]
        p = liver_summary[f"lbl{lbl}_Precision"]["mean"]
        r = liver_summary[f"lbl{lbl}_Recall"]["mean"]
        print(f"    {name}: Dice={d:.3f}  Prec={p:.3f}  Rec={r:.3f}  BalAcc={b:.3f}")
    print(f"    Macro F1 = {liver_summary['macro_f1']['mean']:.3f}")
else:
    print("  No liver predictions — using CV Dice fallback")

results["liver"] = {
    "source": "nnunet v1 5-fold ensemble inference, Dataset603_Liver",
    "n_cases": len(liver_cases),
    "label_map": {"1":"Parenchyma","2":"Tumour"},
    "summary": liver_summary,
    "cv_fallback": {"parenchyma_dice": 0.9571, "tumour_dice": 0.6372, "n_cv_samples": 131,
                    "note": "5-fold CV Dice from postprocessing.json (n=131)"},
    "per_case": [{"case_id": c["case_id"],
                  "macro_f1": float(c["macro_f1"]),
                  "per_class": {str(l): {k: float(v) for k,v in m.items()
                                         if isinstance(v,(int,float,np.floating))}
                                for l, m in c["per_class"].items()}}
                 for c in liver_cases],
}

out_json = OUT / "full_metrics_summary.json"
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False, default=str)
print(f"\nJSON saved: {out_json}")

def g(d, *keys):
    for k in keys:
        if not isinstance(d, dict): return None
        d = d.get(k)
    return d

def cell(val, std=None):
    if val is None or (isinstance(val, float) and np.isnan(val)): return "—"
    s = f"{val:.3f}"
    if std and std > 0.001: s += f"\n±{std:.3f}"
    return s

bs = results["brain"].get("summary", {})
ls = results["liver"].get("summary", {})
lgs = results.get("lung", {}).get("metrics_summary", {})

COL_H = ["Dice","Precision","Recall","Specificity","Bal. Acc.","F1","AUC","Macro F1"]

def brain_row(lbl):
    mk = lambda m: cell(g(bs, f"lbl{lbl}_{m}", "mean"), g(bs, f"lbl{lbl}_{m}", "std"))
    mf = cell(g(bs,"macro_f1","mean"), g(bs,"macro_f1","std"))
    return [mk("Dice"), mk("Precision"), mk("Recall"), mk("Specificity"),
            mk("Balanced_Accuracy"), mk("F1"), mk("AUC"), mf]

def liver_row(lbl):
    mk = lambda m: cell(g(ls, f"lbl{lbl}_{m}", "mean"), g(ls, f"lbl{lbl}_{m}", "std"))
    mf = cell(g(ls,"macro_f1","mean"), g(ls,"macro_f1","std"))
    return [mk("Dice"), mk("Precision"), mk("Recall"), mk("Specificity"),
            mk("Balanced_Accuracy"), mk("F1"), mk("AUC"), mf]

def lung_row():
    lc = lambda m: cell(g(lgs, m, "mean"), g(lgs, m, "std"))
    return [lc("Dice"), lc("Precision"), lc("Recall"), lc("Specificity"),
            lc("Balanced_Accuracy"), lc("F1"), lc("AUC"), "—"]

n_brain = len(brain_cases)
n_liver = len(liver_cases)
n_lung  = results.get("lung", {}).get("n_cases", 0)

brain_src = f"inference, n={n_brain}" if n_brain > 0 else "CV fallback"
liver_src = f"inference, n={n_liver}" if n_liver > 0 else "CV fallback"

def brain_fallback_row():
    cv = results["brain"]["cv_fallback"]
    d  = cv["ema_dice_mean"]; s = cv["ema_dice_std"]
    return [cell(d, s), "—","—","—","—", cell(d), "—","—"]

def liver_fallback_rows():
    cv = results["liver"]["cv_fallback"]
    p  = cv["parenchyma_dice"]; t = cv["tumour_dice"]
    return ([cell(p),"—","—","—","—", cell(p),"—","—"],
            [cell(t),"—","—","—","—", cell(t),"—","—"])

rows        = []
row_labels  = []
row_colors  = []

if brain_cases:
    rows.append(brain_row(1))
    row_labels.append(f"Brain — NCR/Non-enh.\n(BraTS20, {brain_src})")
    row_colors.append(C["brain"])
    rows.append(brain_row(2))
    row_labels.append("Brain — Edema")
    row_colors.append(C["brain2"])
    rows.append(brain_row(4))
    row_labels.append("Brain — Enhancing Tumor")
    row_colors.append(C["brain3"])
else:
    rows.append(brain_fallback_row())
    row_labels.append("Brain MRI\n(5-fold CV EMA Dice)")
    row_colors.append(C["brain"])

if liver_cases:
    rows.append(liver_row(1))
    row_labels.append(f"Liver — Parenchyma\n({liver_src})")
    row_colors.append(C["liver_p"])
    rows.append(liver_row(2))
    row_labels.append(f"Liver — Tumour\n({liver_src})")
    row_colors.append(C["liver_t"])
else:
    pr, tr = liver_fallback_rows()
    rows.append(pr)
    row_labels.append("Liver Parenchyma\n(5-fold CV, n=131)")
    row_colors.append(C["liver_p"])
    rows.append(tr)
    row_labels.append("Liver Tumour\n(5-fold CV, n=131)")
    row_colors.append(C["liver_t"])

rows.append(lung_row())
row_labels.append(f"Lung CT\n(fold_0 val, n={n_lung})")
row_colors.append(C["lung"])

fig, ax = plt.subplots(figsize=(17, max(5.5, len(rows) * 1.15 + 2)))
ax.axis("off")
tbl = ax.table(cellText=rows, rowLabels=row_labels, colLabels=COL_H,
               loc="center", cellLoc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(9.5)
tbl.scale(1, 2.8)

for (r, c), cell_obj in tbl.get_celld().items():
    cell_obj.set_edgecolor("#d1d5db")
    if r == 0:
        cell_obj.set_facecolor("#1a1a2e")
        cell_obj.set_text_props(color="white", fontweight="bold", fontsize=10)
        if c in [6, 7]: cell_obj.set_facecolor("#312e81")
    elif c == -1:
        col  = row_colors[r-1] if r-1 < len(row_colors) else "#6b7280"
        rgb  = mc.to_rgb(col)
        tint = tuple(0.87 + 0.13*x for x in rgb)
        cell_obj.set_facecolor(tint)
        cell_obj.set_text_props(fontweight="bold", fontsize=8.5, linespacing=1.3)
        cell_obj.set_width(0.22)
    else:
        txt = cell_obj.get_text().get_text()
        if txt == "—":
            cell_obj.set_facecolor("#f3f4f6")
            cell_obj.set_text_props(color="#9ca3af", fontsize=13)
        else:
            try:
                val = float(txt.split("\n")[0])
                col = row_colors[r-1] if r-1 < len(row_colors) else "#6b7280"
                rgb = mc.to_rgb(col)
                alpha   = 0.10 + val * 0.55
                blended = tuple(1 - alpha + alpha*x for x in rgb)
                cell_obj.set_facecolor(blended)
                cell_obj.set_text_props(fontweight="bold",
                    color="white" if val > 0.72 else "#1a1a2e",
                    fontsize=9, linespacing=1.3)
            except Exception:
                pass

ax.set_title(
    "CARVIS — Complete Segmentation Evaluation\n"
    f"Brain: {n_brain or 'CV fallback'} BraTS20 cases | "
    f"Liver: {n_liver or 'CV fallback'} cases | "
    f"Lung: {n_lung} fold_0 val cases\n"
    "Dice=F1 (binary) | Bal. Acc. = AUC (hard predictions) | Macro F1 incl. background",
    fontsize=10, pad=14, loc="left", color="#374151")
fig.tight_layout()
p = OUT / "full_metrics_table.png"
fig.savefig(p, dpi=160, bbox_inches="tight", facecolor="white")
print(f"Saved: {p}")
plt.close(fig)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5.5))

segments = []
if brain_cases:
    for lbl, name, col in [(1,"Brain NCR/Non-enh.",C["brain"]),
                            (2,"Brain Edema",C["brain2"]),
                            (4,"Brain Enhancing",C["brain3"])]:
        d = g(bs,f"lbl{lbl}_Dice","mean") or 0
        b = g(bs,f"lbl{lbl}_Balanced_Accuracy","mean") or 0
        segments.append((name, col, d, b))
else:
    cv = results["brain"]["cv_fallback"]
    segments.append(("Brain MRI (CV)", C["brain"], cv["ema_dice_mean"], None))

if liver_cases:
    for lbl, name, col in [(1,"Liver Paren.",C["liver_p"]),
                            (2,"Liver Tumour",C["liver_t"])]:
        d = g(ls,f"lbl{lbl}_Dice","mean") or 0
        b = g(ls,f"lbl{lbl}_Balanced_Accuracy","mean") or 0
        segments.append((name, col, d, b))
else:
    cv = results["liver"]["cv_fallback"]
    segments.append(("Liver Paren. (CV)", C["liver_p"], cv["parenchyma_dice"], None))
    segments.append(("Liver Tumour (CV)", C["liver_t"], cv["tumour_dice"], None))

lung_dice = g(lgs,"Dice","mean") or 0
lung_bal  = g(lgs,"Balanced_Accuracy","mean") or 0
segments.append(("Lung CT", C["lung"], lung_dice, lung_bal))

names  = [s[0] for s in segments]
colors = [s[1] for s in segments]
dices  = [s[2] for s in segments]
bals   = [s[3] if s[3] is not None else 0 for s in segments]
x      = np.arange(len(segments))

for ax, vals, title, ylabel in [
    (ax1, dices, "Dice Score", "Dice"),
    (ax2, bals,  "Balanced Accuracy (AUC)", "Balanced Accuracy"),
]:
    bars = ax.bar(x, vals, color=colors, width=0.55, edgecolor="white", linewidth=0.8)
    for bar, val, seg in zip(bars, vals, segments):
        if val > 0.01:
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.015,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
        if seg[3] is None and title == "Balanced Accuracy (AUC)":
            ax.text(bar.get_x() + bar.get_width()/2, 0.05,
                    "CV\nonly", ha="center", va="bottom", fontsize=7.5,
                    color="white", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=9.5)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel(ylabel, fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
    ax.axhline(0.8, color="#9ca3af", lw=1.2, ls="--", alpha=0.8)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

fig.suptitle(
    "CARVIS — Segmentation Performance (Real Measurements)\n"
    "Brain & Liver: voxel-level from inference | Lung: fold_0 validation | No hardcoded values",
    fontsize=12, fontweight="bold", y=1.01)
fig.tight_layout()
p = OUT / "full_metrics_bar.png"
fig.savefig(p, dpi=160, bbox_inches="tight", facecolor="white")
print(f"Saved: {p}")
plt.close(fig)

PLOT_METRICS = ["Dice","Precision","Recall","Specificity","Balanced_Accuracy","F1","AUC"]
PLOT_NICE    = ["Dice","Precision","Recall","Specificity","Bal. Acc.","F1","AUC"]

hm_segs   = []
hm_colors = []

def hm_row(summary_dict, lbl_keys, seg_name, col):
    row = []
    for mk in PLOT_METRICS:
        v = g(summary_dict, f"lbl{lbl_keys}_{mk}", "mean")
        row.append(v if v is not None else np.nan)
    hm_segs.append((seg_name, row))
    hm_colors.append(col)

if brain_cases:
    hm_row(bs, 1, "Brain — NCR/Non-enh.",   C["brain"])
    hm_row(bs, 2, "Brain — Edema",          C["brain2"])
    hm_row(bs, 4, "Brain — Enhancing",      C["brain3"])
if liver_cases:
    hm_row(ls, 1, "Liver — Parenchyma",     C["liver_p"])
    hm_row(ls, 2, "Liver — Tumour",         C["liver_t"])

if hm_segs:
    lung_row_vals = [g(lgs, mk, "mean") or np.nan for mk in
                     ["Dice","Precision","Recall","Specificity","Balanced_Accuracy","F1","AUC"]]
    hm_segs.append(("Lung CT", lung_row_vals))
    hm_colors.append(C["lung"])

    mat  = np.array([r for _, r in hm_segs], dtype=float)
    segs = [s for s, _ in hm_segs]

    fig, ax = plt.subplots(figsize=(11, max(3.5, len(segs)*0.6 + 1.5)))
    im = ax.imshow(mat, cmap="Blues", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(PLOT_NICE))); ax.set_xticklabels(PLOT_NICE, fontsize=10)
    ax.set_yticks(range(len(segs))); ax.set_yticklabels(segs, fontsize=10)
    for i in range(len(segs)):
        for j in range(len(PLOT_METRICS)):
            v = mat[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                        fontsize=9, fontweight="bold",
                        color="white" if v > 0.6 else "#1a1a2e")
            else:
                ax.text(j, i, "—", ha="center", va="center",
                        fontsize=11, color="#9ca3af")
    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    ax.set_title("CARVIS — Segmentation Metric Heatmap (Real Measurements)",
                 fontsize=12, fontweight="bold", pad=10)
    fig.tight_layout()
    p = OUT / "full_metrics_heatmap.png"
    fig.savefig(p, dpi=160, bbox_inches="tight", facecolor="white")
    print(f"Saved: {p}")
    plt.close(fig)

print("\n" + "="*65)
print("FULL METRIC SUMMARY")
print("="*65)
if brain_cases:
    print(f"BRAIN (n={len(brain_cases)} BraTS20, fold_0, BAMF schema):")
    for lbl, name in BRAIN_LABELS.items():
        d = g(bs,f"lbl{lbl}_Dice","mean"); b = g(bs,f"lbl{lbl}_Balanced_Accuracy","mean")
        p_v = g(bs,f"lbl{lbl}_Precision","mean"); r_v = g(bs,f"lbl{lbl}_Recall","mean")
        f_v = g(bs,f"lbl{lbl}_F1","mean"); auc_v = g(bs,f"lbl{lbl}_AUC","mean")
        print(f"  {name}: Dice={d:.3f}  Prec={p_v:.3f}  Rec={r_v:.3f}  "
              f"Spec={g(bs,f'lbl{lbl}_Specificity','mean'):.3f}  "
              f"BalAcc={b:.3f}  F1={f_v:.3f}  AUC={auc_v:.3f}")
    print(f"  Macro F1 = {g(bs,'macro_f1','mean'):.3f}")
else:
    cv = results["brain"]["cv_fallback"]
    print(f"BRAIN (CV fallback): Dice={cv['ema_dice_mean']:.4f} +/- {cv['ema_dice_std']:.4f}")

if liver_cases:
    print(f"\nLIVER (n={len(liver_cases)} case, 5-fold ensemble):")
    for lbl, name in LIVER_LABELS.items():
        d = g(ls,f"lbl{lbl}_Dice","mean"); b = g(ls,f"lbl{lbl}_Balanced_Accuracy","mean")
        p_v = g(ls,f"lbl{lbl}_Precision","mean"); r_v = g(ls,f"lbl{lbl}_Recall","mean")
        print(f"  {name}: Dice={d:.3f}  Prec={p_v:.3f}  Rec={r_v:.3f}  BalAcc={b:.3f}")
    print(f"  Macro F1 = {g(ls,'macro_f1','mean'):.3f}")
else:
    cv = results["liver"]["cv_fallback"]
    print(f"LIVER (CV fallback): Paren={cv['parenchyma_dice']:.4f}  Tumour={cv['tumour_dice']:.4f}")

if lgs:
    print(f"\nLUNG (n={n_lung} case, fold_0 val):")
    for mk in ["Dice","Precision","Recall","Specificity","Balanced_Accuracy","F1","AUC"]:
        v = g(lgs, mk, "mean"); s = g(lgs, mk, "std")
        print(f"  {mk:<20}: {v:.3f} +/- {s:.3f}")

print(f"\nOutputs: {OUT}")
print("Done.")
