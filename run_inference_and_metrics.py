import json, zipfile, shutil, os, gc
import numpy as np
from pathlib import Path

ROOT     = Path(__file__).parent
OUT      = ROOT / "outputs" / "evaluation_v2"
TEMP_DIR = ROOT / "outputs" / "_inference_temp"
OUT.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)

BRAIN_MODEL  = ROOT / "nnunet_brain"
LIVER_MODEL  = ROOT / "nnunet_liver" / "nnUNetTrainerV2__nnUNetPlansv2.1"
BRATS20_ZIP  = Path(r"C:\Users\User\OneDrive\Desktop\BraTS20_Training_020.zip")
LIVER_IMAGES = Path(r"C:\Users\User\OneDrive\Desktop\cigerim\nnUNet_data\nnUNet_raw\Dataset603_Liver\imagesTr")
LIVER_GT     = Path(r"C:\Users\User\OneDrive\Desktop\cigerim\nnUNet_data\nnUNet_preprocessed\Dataset603_Liver\gt_segmentations")
LUNG_SUMMARY = OUT / "real_metrics_summary.json"

def binary_metrics(pred: np.ndarray, gt: np.ndarray, label: int) -> dict:
    p = (pred == label)
    g = (gt   == label)
    TP = int(np.logical_and(p,   g).sum())
    FP = int(np.logical_and(p,  ~g).sum())
    FN = int(np.logical_and(~p,  g).sum())
    TN = int(np.logical_and(~p, ~g).sum())
    total = TP + FP + FN + TN
    eps = 1e-8
    dice  = 2*TP / (2*TP + FP + FN + eps)
    prec  = TP / (TP + FP + eps)
    rec   = TP / (TP + FN + eps)
    spec  = TN / (TN + FP + eps)
    f1    = dice
    bal   = (rec + spec) / 2
    acc   = (TP + TN) / (total + eps)
    return dict(Dice=dice, Precision=prec, Recall=rec, Specificity=spec,
                F1=f1, Balanced_Accuracy=bal, AUC=bal, Accuracy=acc,
                TP=TP, FP=FP, FN=FN, TN=TN)

def multi_class_metrics(pred: np.ndarray, gt: np.ndarray, labels: list) -> dict:
    per_class = {}
    for lbl in labels:
        m = binary_metrics(pred, gt, lbl)
        per_class[lbl] = m
    all_labels = [0] + labels
    all_f1  = [binary_metrics(pred, gt, l)["F1"]  for l in all_labels]
    all_auc = [binary_metrics(pred, gt, l)["AUC"] for l in all_labels]
    macro_f1  = float(np.mean(all_f1))
    macro_auc = float(np.mean(all_auc))
    return {"per_class": per_class, "macro_f1": macro_f1, "macro_auc": macro_auc}

def load_nifti(path: Path) -> np.ndarray:
    try:
        import nibabel as nib
        return np.asarray(nib.load(str(path)).dataobj, dtype=np.int32)
    except Exception:
        import SimpleITK as sitk
        return sitk.GetArrayFromImage(sitk.ReadImage(str(path))).astype(np.int32)

def match_shape(pred: np.ndarray, gt: np.ndarray):
    if pred.shape != gt.shape:
        s = tuple(min(a, b) for a, b in zip(pred.shape, gt.shape))
        pred = pred[:s[0], :s[1], :s[2]]
        gt   = gt  [:s[0], :s[1], :s[2]]
    return pred, gt

def summary_stats(vals: list) -> dict:
    arr = [v for v in vals if v is not None and not np.isnan(v)]
    if not arr:
        return {"mean": None, "std": None}
    return {"mean": float(np.mean(arr)), "std": float(np.std(arr)),
            "median": float(np.median(arr)), "min": float(np.min(arr)), "max": float(np.max(arr))}

print("=" * 65)
print("CARVIS — Full Inference + Metrics Pipeline")
print("=" * 65)

results = {}

print("\n[LUNG] Loading existing validation metrics...")
if LUNG_SUMMARY.exists():
    with open(LUNG_SUMMARY, encoding="utf-8") as f:
        lung_data = json.load(f)["lung"]
    results["lung"] = lung_data
    ms = lung_data.get("metrics_summary", {})
    print(f"  n={lung_data['n_cases']} cases  Dice={ms.get('Dice',{}).get('mean',0):.3f}  "
          f"BalAcc={ms.get('Balanced_Accuracy',{}).get('mean',0):.3f}")
else:
    print("  WARNING: real_metrics_summary.json not found. Run eval_real_compute.py first.")
    results["lung"] = {}

print("\n[BRAIN] Extracting BraTS20 data and running inference...")

BRAIN_INPUT  = TEMP_DIR / "brain_input"
BRAIN_OUTPUT = TEMP_DIR / "brain_output"
BRAIN_GT_DIR = TEMP_DIR / "brain_gt"
for d in [BRAIN_INPUT, BRAIN_OUTPUT, BRAIN_GT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

N_BRAIN_CASES = 5
brain_cases_extracted = []
if BRATS20_ZIP.exists():
    with zipfile.ZipFile(str(BRATS20_ZIP), 'r') as zf:
        all_segs = [e for e in zf.namelist() if e.endswith("_seg.nii")]
        case_names = list({e.split("/")[0] for e in all_segs})[:N_BRAIN_CASES]
        print(f"  Selected BraTS20 cases: {case_names}")

        for case_name in case_names:
            case_id = case_name.split("_")[-1]
            modality_map = {"_t1.nii": "0000", "_t1ce.nii": "0001",
                            "_t2.nii": "0002", "_flair.nii": "0003"}
            ok = True
            for suffix, chan in modality_map.items():
                src_name = f"{case_name}/{case_name}{suffix}"
                if src_name not in zf.namelist():
                    print(f"  SKIP {case_name}: {src_name} not found")
                    ok = False; break
                dst = BRAIN_INPUT / f"BraTS_{case_id}_{chan}.nii"
                with zf.open(src_name) as src, open(dst, "wb") as tgt:
                    shutil.copyfileobj(src, tgt)
            seg_name = f"{case_name}/{case_name}_seg.nii"
            if seg_name in zf.namelist():
                dst_gt = BRAIN_GT_DIR / f"BraTS_{case_id}.nii"
                with zf.open(seg_name) as src, open(dst_gt, "wb") as tgt:
                    shutil.copyfileobj(src, tgt)
            if ok:
                brain_cases_extracted.append(case_id)

print(f"  {len(brain_cases_extracted)} cases ready: {brain_cases_extracted}")

brain_inference_ok = False
if brain_cases_extracted and BRAIN_MODEL.exists():
    try:
        import torch
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        print(f"  Device: {device}")

        predictor = nnUNetPredictor(
            tile_step_size=0.6,
            use_gaussian=True,
            use_mirroring=False,
            perform_everything_on_device=False,
            device=device,
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=True,
        )
        predictor.initialize_from_trained_model_folder(
            str(BRAIN_MODEL),
            use_folds=(0,),
            checkpoint_name="checkpoint_final.pth",
        )
        print("  Model loaded. Starting inference...")
        predictor.predict_from_files(
            [[str(f) for f in sorted(BRAIN_INPUT.glob(f"BraTS_{cid}_*.nii"))]
             for cid in brain_cases_extracted],
            [str(BRAIN_OUTPUT / f"BraTS_{cid}.nii.gz") for cid in brain_cases_extracted],
            save_probabilities=False,
            overwrite=True,
            num_processes_preprocessing=1,
            num_processes_segmentation_export=1,
        )
        brain_inference_ok = True
        print("  Brain inference complete!")
    except Exception as e:
        print(f"  Brain inference ERROR: {e}")
        import traceback; traceback.print_exc()

BRAIN_LABEL_NAMES = {1: "Edema", 2: "Non-enhancing (NCR)", 4: "Enhancing Tumor"}

brain_case_results = []
for cid in brain_cases_extracted:
    pred_path = BRAIN_OUTPUT / f"BraTS_{cid}.nii.gz"
    gt_path   = BRAIN_GT_DIR  / f"BraTS_{cid}.nii"
    if not pred_path.exists() or not gt_path.exists():
        print(f"  BraTS_{cid}: pred or GT missing, skipping")
        continue
    try:
        pred = load_nifti(pred_path)
        gt   = load_nifti(gt_path)
        pred, gt = match_shape(pred, gt)

        gt_bamf = np.zeros_like(gt)
        gt_bamf[gt == 2] = 1
        gt_bamf[gt == 1] = 2
        gt_bamf[gt == 4] = 4

        m = multi_class_metrics(pred, gt_bamf, labels=[1, 2, 4])
        m["case_id"] = f"BraTS_{cid}"
        brain_case_results.append(m)

        row_str = " | ".join(
            f"{BRAIN_LABEL_NAMES[l]}: Dice={m['per_class'][l]['Dice']:.3f}"
            for l in [1, 2, 4])
        print(f"  BraTS_{cid}: {row_str}  Macro F1={m['macro_f1']:.3f}")
    except Exception as e:
        print(f"  BraTS_{cid}: ERROR — {e}")

brain_summary = {}
if brain_case_results:
    for lbl, name in BRAIN_LABEL_NAMES.items():
        for metric in ["Dice","Precision","Recall","Specificity","F1","Balanced_Accuracy","AUC"]:
            key = f"lbl{lbl}_{metric}"
            vals = [c["per_class"][lbl][metric] for c in brain_case_results]
            brain_summary[key] = summary_stats(vals)
    brain_summary["macro_f1"] = summary_stats([c["macro_f1"] for c in brain_case_results])
    brain_summary["macro_auc"] = summary_stats([c["macro_auc"] for c in brain_case_results])

    print(f"\n  BRAIN SUMMARY (n={len(brain_case_results)} cases):")
    for lbl, name in BRAIN_LABEL_NAMES.items():
        d = brain_summary[f"lbl{lbl}_Dice"]["mean"]
        b = brain_summary[f"lbl{lbl}_Balanced_Accuracy"]["mean"]
        print(f"    {name}: Dice={d:.3f}  BalAcc={b:.3f}")
    print(f"    Macro F1={brain_summary['macro_f1']['mean']:.3f}")

results["brain"] = {
    "source": f"nnUNetv2 inference on {len(brain_case_results)} BraTS20 cases, fold_0, BAMF schema",
    "n_cases": len(brain_case_results),
    "per_case": [
        {"case_id": c["case_id"],
         "per_class": {str(l): {k: float(v) for k,v in c["per_class"][l].items() if isinstance(v,(int,float,np.floating))}
                       for l in [1,2,4]},
         "macro_f1": float(c["macro_f1"])}
        for c in brain_case_results
    ],
    "summary": brain_summary,
    "label_map": {"1":"Edema","2":"Non-enhancing(NCR)","4":"Enhancing Tumor"},
    "cv_dice_5fold": {"mean": 0.8691, "std": 0.0019,
                      "note": "5-fold CV EMA Dice from checkpoint (all cases)"},
}

print("\n[LIVER] Running nnunet v1 inference...")

LIVER_INPUT_DIR  = TEMP_DIR / "liver_input"
LIVER_OUTPUT_DIR = TEMP_DIR / "liver_output"
LIVER_INPUT_DIR.mkdir(exist_ok=True)
LIVER_OUTPUT_DIR.mkdir(exist_ok=True)

N_LIVER = 10
liver_image_files = sorted(LIVER_IMAGES.glob("liver_*_0000.nii.gz"))[:N_LIVER]
liver_case_ids = [f.name.replace("_0000.nii.gz","") for f in liver_image_files]
print(f"  Selected liver cases ({N_LIVER}): {liver_case_ids[:5]}...")

for img_f in liver_image_files:
    dst = LIVER_INPUT_DIR / img_f.name
    if not dst.exists():
        shutil.copy2(img_f, dst)

liver_inference_ok = False
try:
    from nnunet.inference.predict import predict_from_folder

    os.environ["nnUNet_raw_data_base"] = str(
        Path(r"C:\Users\User\OneDrive\Desktop\cigerim\nnUNet_data\nnUNet_raw"))
    os.environ["nnUNet_preprocessed"] = str(
        Path(r"C:\Users\User\OneDrive\Desktop\cigerim\nnUNet_data\nnUNet_preprocessed"))
    os.environ["RESULTS_FOLDER"] = str(
        Path(r"C:\Users\User\OneDrive\Desktop\ai_asistan_workspace6 - app\nnunet_liver_results"))

    fake_results = Path(os.environ["RESULTS_FOLDER"])
    model_dest   = fake_results / "nnUNet" / "3d_fullres" / "Task003_Liver" / "nnUNetTrainerV2__nnUNetPlansv2.1"
    model_dest.mkdir(parents=True, exist_ok=True)

    for item in LIVER_MODEL.iterdir():
        dst = model_dest / item.name
        if not dst.exists():
            if item.is_dir():
                shutil.copytree(str(item), str(dst))
            else:
                shutil.copy2(str(item), str(dst))

    print(f"  Model ready: {model_dest}")
    print("  Starting inference (tta=False to avoid GPU memory issues)...")

    predict_from_folder(
        model=str(model_dest),
        input_folder=str(LIVER_INPUT_DIR),
        output_folder=str(LIVER_OUTPUT_DIR),
        folds=(0, 1, 2, 3, 4),
        save_npz=False,
        num_threads_preprocessing=1,
        num_threads_nifti_save=1,
        lowres_segmentations=None,
        part_id=0,
        num_parts=1,
        tta=False,
        mixed_precision=True,
        overwrite_existing=False,
        mode="normal",
        step_size=0.5,
        checkpoint_name="model_final_checkpoint",
        disable_postprocessing=True,
    )
    liver_inference_ok = True
    print("  Liver inference complete!")

except ImportError as e:
    print(f"  nnunet import error: {e}")
except Exception as e:
    print(f"  Liver inference ERROR: {e}")
    import traceback; traceback.print_exc()

LIVER_LABEL_NAMES = {1: "Parenchyma", 2: "Tumour"}

liver_case_results = []
for case_id in liver_case_ids:
    num = case_id.replace("liver_","")
    gt_path = LIVER_GT / f"liver_{int(num)}.nii.gz"
    if not gt_path.exists():
        gt_path = LIVER_GT / f"{case_id}.nii.gz"
    pred_path = LIVER_OUTPUT_DIR / f"{case_id}.nii.gz"

    if not pred_path.exists():
        print(f"  {case_id}: prediction not found (inference failed?)")
        continue
    if not gt_path.exists():
        print(f"  {case_id}: GT not found ({gt_path})")
        continue

    try:
        pred = load_nifti(pred_path)
        gt   = load_nifti(gt_path)
        pred, gt = match_shape(pred, gt)

        m = multi_class_metrics(pred, gt, labels=[1, 2])
        m["case_id"] = case_id
        liver_case_results.append(m)

        p_dice = m["per_class"][1]["Dice"]
        t_dice = m["per_class"][2]["Dice"]
        print(f"  {case_id}: Paren Dice={p_dice:.3f}  Tumour Dice={t_dice:.3f}  "
              f"Macro F1={m['macro_f1']:.3f}")
    except Exception as e:
        print(f"  {case_id}: ERROR — {e}")

liver_summary = {}
if liver_case_results:
    for lbl, name in LIVER_LABEL_NAMES.items():
        for metric in ["Dice","Precision","Recall","Specificity","F1","Balanced_Accuracy","AUC"]:
            key = f"lbl{lbl}_{metric}"
            vals = [c["per_class"][lbl][metric] for c in liver_case_results]
            liver_summary[key] = summary_stats(vals)
    liver_summary["macro_f1"] = summary_stats([c["macro_f1"] for c in liver_case_results])

    print(f"\n  LIVER SUMMARY (n={len(liver_case_results)} cases):")
    for lbl, name in LIVER_LABEL_NAMES.items():
        d = liver_summary[f"lbl{lbl}_Dice"]["mean"]
        b = liver_summary[f"lbl{lbl}_Balanced_Accuracy"]["mean"]
        print(f"    {name}: Dice={d:.3f}  BalAcc={b:.3f}")
    print(f"    Macro F1={liver_summary['macro_f1']['mean']:.3f}")

results["liver"] = {
    "source": f"nnunet v1 inference on {len(liver_case_results)} cases, Dataset603_Liver",
    "n_cases": len(liver_case_results),
    "per_case": [
        {"case_id": c["case_id"],
         "per_class": {str(l): {k: float(v) for k,v in c["per_class"][l].items()
                                if isinstance(v,(int,float,np.floating))}
                       for l in [1,2]},
         "macro_f1": float(c["macro_f1"])}
        for c in liver_case_results
    ],
    "summary": liver_summary,
    "label_map": {"1":"Parenchyma","2":"Tumour"},
    "cv_dice_5fold": {"parenchyma": 0.9571, "tumour": 0.6372, "n_samples": 131,
                      "note": "5-fold CV Dice from postprocessing.json"},
}

full_path = OUT / "full_metrics_summary.json"
with open(full_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False, default=str)
print(f"\nJSON saved: {full_path}")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mc

plt.rcParams.update({
    "axes.facecolor":    "#fafafa",
    "figure.facecolor":  "#ffffff",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.color":        "#e5e7eb",
    "grid.linewidth":    0.6,
    "font.family":       "DejaVu Sans",
})

METRIC_KEYS  = ["Dice", "Precision", "Recall", "Specificity", "Balanced_Accuracy", "F1", "AUC", "Macro F1"]
METRIC_NICE  = ["Dice", "Precision", "Recall", "Specificity", "Bal. Acc.", "F1", "AUC", "Macro F1"]

def m_val(d, key):
    if isinstance(d, dict):
        v = d.get(key, {})
        if isinstance(v, dict): return v.get("mean")
        return v
    return None

table_rows   = []
table_labels = []
table_colors = []

C_BRAIN  = "#2563eb"
C_LPAREN = "#16a34a"
C_LTUMOR = "#dc2626"
C_LUNG   = "#b45309"

def make_cell(val, std=None):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "—"
    s = f"{val:.3f}"
    if std is not None and std > 0.001:
        s += f"\n±{std:.3f}"
    return s

bs  = results.get("brain", {}).get("summary", {})
ls  = results.get("liver", {}).get("summary", {})
lgs = results.get("lung", {}).get("metrics_summary", {})

if bs and brain_case_results:
    table_rows.append([
        make_cell(m_val(bs,"lbl1_Dice"), bs.get("lbl1_Dice",{}).get("std")),
        make_cell(m_val(bs,"lbl1_Precision")),
        make_cell(m_val(bs,"lbl1_Recall")),
        make_cell(m_val(bs,"lbl1_Specificity")),
        make_cell(m_val(bs,"lbl1_Balanced_Accuracy")),
        make_cell(m_val(bs,"lbl1_F1")),
        make_cell(m_val(bs,"lbl1_AUC")),
        make_cell(m_val(bs,"macro_f1")),
    ])
    table_labels.append(f"Brain — Edema\n(n={results['brain']['n_cases']} BraTS20)")
    table_colors.append(C_BRAIN)

    table_rows.append([
        make_cell(m_val(bs,"lbl2_Dice")),
        make_cell(m_val(bs,"lbl2_Precision")),
        make_cell(m_val(bs,"lbl2_Recall")),
        make_cell(m_val(bs,"lbl2_Specificity")),
        make_cell(m_val(bs,"lbl2_Balanced_Accuracy")),
        make_cell(m_val(bs,"lbl2_F1")),
        make_cell(m_val(bs,"lbl2_AUC")),
        "—",
    ])
    table_labels.append("Brain — Non-enh. Tumor")
    table_colors.append("#3b82f6")

    table_rows.append([
        make_cell(m_val(bs,"lbl4_Dice")),
        make_cell(m_val(bs,"lbl4_Precision")),
        make_cell(m_val(bs,"lbl4_Recall")),
        make_cell(m_val(bs,"lbl4_Specificity")),
        make_cell(m_val(bs,"lbl4_Balanced_Accuracy")),
        make_cell(m_val(bs,"lbl4_F1")),
        make_cell(m_val(bs,"lbl4_AUC")),
        "—",
    ])
    table_labels.append("Brain — Enhancing Tumor")
    table_colors.append("#93c5fd")
else:
    cv_dice = results["brain"].get("cv_dice_5fold", {})
    table_rows.append([
        make_cell(cv_dice.get("mean"), cv_dice.get("std")),
        "—","—","—","—",
        make_cell(cv_dice.get("mean")),
        "—","—",
    ])
    table_labels.append("Brain MRI\n(5-fold CV, EMA Dice)")
    table_colors.append(C_BRAIN)

if ls and liver_case_results:
    table_rows.append([
        make_cell(m_val(ls,"lbl1_Dice"), ls.get("lbl1_Dice",{}).get("std")),
        make_cell(m_val(ls,"lbl1_Precision")),
        make_cell(m_val(ls,"lbl1_Recall")),
        make_cell(m_val(ls,"lbl1_Specificity")),
        make_cell(m_val(ls,"lbl1_Balanced_Accuracy")),
        make_cell(m_val(ls,"lbl1_F1")),
        make_cell(m_val(ls,"lbl1_AUC")),
        make_cell(m_val(ls,"macro_f1")),
    ])
    table_labels.append(f"Liver — Parenchyma\n(n={results['liver']['n_cases']} cases, inference)")
    table_colors.append(C_LPAREN)

    table_rows.append([
        make_cell(m_val(ls,"lbl2_Dice"), ls.get("lbl2_Dice",{}).get("std")),
        make_cell(m_val(ls,"lbl2_Precision")),
        make_cell(m_val(ls,"lbl2_Recall")),
        make_cell(m_val(ls,"lbl2_Specificity")),
        make_cell(m_val(ls,"lbl2_Balanced_Accuracy")),
        make_cell(m_val(ls,"lbl2_F1")),
        make_cell(m_val(ls,"lbl2_AUC")),
        "—",
    ])
    table_labels.append("Liver — Tumour\n(n=10 cases, inference)")
    table_colors.append(C_LTUMOR)
else:
    cv = results["liver"].get("cv_dice_5fold",{})
    table_rows.append([make_cell(cv.get("parenchyma")),"—","—","—","—",
                       make_cell(cv.get("parenchyma")),"—","—"])
    table_labels.append(f"Liver Parenchyma\n(5-fold CV, n={cv.get('n_samples',131)})")
    table_colors.append(C_LPAREN)
    table_rows.append([make_cell(cv.get("tumour")),"—","—","—","—",
                       make_cell(cv.get("tumour")),"—","—"])
    table_labels.append("Liver Tumour\n(5-fold CV)")
    table_colors.append(C_LTUMOR)

def lc(mk):
    v = lgs.get(mk,{})
    return make_cell(v.get("mean"), v.get("std"))

table_rows.append([lc("Dice"), lc("Precision"), lc("Recall"), lc("Specificity"),
                   lc("Balanced_Accuracy"), lc("F1"), lc("AUC"), "—"])
table_labels.append(f"Lung CT\n(fold_0 val, n={results['lung'].get('n_cases',13)})")
table_colors.append(C_LUNG)

fig, ax = plt.subplots(figsize=(17, max(5, len(table_rows)*1.1 + 1.5)))
ax.axis("off")
tbl = ax.table(
    cellText=table_rows,
    rowLabels=table_labels,
    colLabels=METRIC_NICE,
    loc="center", cellLoc="center",
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(9.5)
tbl.scale(1, 2.7)

for (r, c), cell in tbl.get_celld().items():
    cell.set_edgecolor("#d1d5db")
    if r == 0:
        cell.set_facecolor("#1a1a2e")
        cell.set_text_props(color="white", fontweight="bold", fontsize=10)
        if c in [6, 7]:
            cell.set_facecolor("#312e81")
    elif c == -1:
        col  = table_colors[r-1] if r-1 < len(table_colors) else "#6b7280"
        rgb  = mc.to_rgb(col)
        tint = tuple(0.88 + 0.12*x for x in rgb)
        cell.set_facecolor(tint)
        cell.set_text_props(fontweight="bold", fontsize=8.5, linespacing=1.3)
        cell.set_width(0.22)
    else:
        txt = cell.get_text().get_text()
        if txt == "—":
            cell.set_facecolor("#f3f4f6")
            cell.set_text_props(color="#9ca3af", fontsize=13)
        else:
            try:
                val = float(txt.split("\n")[0])
                col = table_colors[r-1] if r-1 < len(table_colors) else "#6b7280"
                rgb = mc.to_rgb(col)
                alpha = 0.10 + val * 0.55
                blended = tuple(1 - alpha + alpha*x for x in rgb)
                cell.set_facecolor(blended)
                cell.set_text_props(fontweight="bold",
                                    color="white" if val > 0.72 else "#1a1a2e",
                                    fontsize=9, linespacing=1.3)
            except Exception:
                pass

brain_n = results["brain"].get("n_cases", 0)
liver_n = results["liver"].get("n_cases", 0)
lung_n  = results.get("lung",{}).get("n_cases", 13)

ax.set_title(
    "CARVIS — Complete Segmentation Metrics (Real Measurements)\n"
    f"Brain: {brain_n} BraTS20 cases (fold_0) | "
    f"Liver: {liver_n} cases (5-fold) | "
    f"Lung: {lung_n} cases (fold_0 val) | "
    "AUC = Balanced Accuracy (hard predictions)",
    fontsize=10, pad=14, loc="left", color="#374151",
)
fig.tight_layout()
p = OUT / "full_metrics_table.png"
fig.savefig(p, dpi=160, bbox_inches="tight", facecolor="white")
print(f"\nSaved: {p}")
plt.close(fig)

key_metrics  = ["Dice","Balanced_Accuracy","F1","Precision","Recall","AUC","Macro F1"]
nice_metrics = ["Dice","Bal. Acc.","F1","Precision","Recall","AUC","Macro F1"]

plot_segments = []

if brain_case_results and bs:
    for lbl, name, col in [(1,"Brain Edema",C_BRAIN),(2,"Brain Non-enh.","#3b82f6"),(4,"Brain Enh.","#93c5fd")]:
        vals = []
        for mk in key_metrics:
            if mk == "Macro F1":
                v = m_val(bs, "macro_f1")
            elif mk in ["AUC","Balanced_Accuracy"]:
                v = m_val(bs, f"lbl{lbl}_Balanced_Accuracy")
            else:
                v = m_val(bs, f"lbl{lbl}_{mk}")
            vals.append(v if v is not None else 0.0)
        plot_segments.append((name, col, vals))
else:
    cv = results["brain"].get("cv_dice_5fold",{})
    plot_segments.append(("Brain (CV Dice)", C_BRAIN, [cv.get("mean",0)] + [None]*6))

if liver_case_results and ls:
    for lbl, name, col in [(1,"Liver Parench.",C_LPAREN),(2,"Liver Tumour",C_LTUMOR)]:
        vals = []
        for mk in key_metrics:
            if mk == "Macro F1":
                v = m_val(ls, "macro_f1")
            elif mk in ["AUC","Balanced_Accuracy"]:
                v = m_val(ls, f"lbl{lbl}_Balanced_Accuracy")
            else:
                v = m_val(ls, f"lbl{lbl}_{mk}")
            vals.append(v if v is not None else 0.0)
        plot_segments.append((name, col, vals))
else:
    cv = results["liver"].get("cv_dice_5fold",{})
    plot_segments.append(("Liver Parench. (CV)", C_LPAREN, [cv.get("parenchyma",0)]+[None]*6))
    plot_segments.append(("Liver Tumour (CV)",   C_LTUMOR, [cv.get("tumour",0)]+[None]*6))

lung_vals = []
for mk in key_metrics:
    if mk == "Macro F1":
        lung_vals.append(None)
    elif mk == "Balanced_Accuracy":
        lung_vals.append(lgs.get("Balanced_Accuracy",{}).get("mean"))
    elif mk == "AUC":
        lung_vals.append(lgs.get("AUC",{}).get("mean"))
    else:
        lung_vals.append(lgs.get(mk,{}).get("mean"))
plot_segments.append(("Lung CT", C_LUNG, lung_vals))

fig, ax = plt.subplots(figsize=(16, 5.5))
x      = np.arange(len(nice_metrics))
n_segs = len(plot_segments)
total_w = 0.75
w = total_w / n_segs
patches = []

import matplotlib.patches as mpatches
for i, (name, col, vals) in enumerate(plot_segments):
    offset = (i - (n_segs-1)/2) * w
    plot_vals = [v if v is not None else 0.0 for v in vals]
    bars = ax.bar(x + offset, plot_vals, w*0.88, color=col, edgecolor="white",
                  linewidth=0.5, alpha=0.88, label=name)
    for bar, val in zip(bars, vals):
        if val is not None and val > 0.05:
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.01,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=6.5, fontweight="bold")
    patches.append(mpatches.Patch(color=col, label=name))

ax.axvspan(5.5, 6.5, alpha=0.07, color="#6366f1", zorder=0)
ax.axvspan(6.5, 7.5, alpha=0.07, color="#6366f1", zorder=0)

ax.set_xticks(x)
ax.set_xticklabels(nice_metrics, fontsize=11)
ax.set_ylabel("Score", fontsize=12, fontweight="bold")
ax.set_ylim(0, 1.15)
ax.axhline(0.8, color="#9ca3af", lw=1.2, ls="--", alpha=0.7)
ax.legend(handles=patches, fontsize=8.5, framealpha=0.85, ncol=3, loc="upper left", bbox_to_anchor=(0,1))
ax.set_title(
    "CARVIS — Segmentation Performance: All 8 Metrics (Real Inference)\n"
    "Brain + Liver: voxel-level from inference | Lung: fold_0 validation | No hardcoded values",
    fontsize=11.5, fontweight="bold", pad=12)
fig.tight_layout()
p = OUT / "full_metrics_bar.png"
fig.savefig(p, dpi=160, bbox_inches="tight", facecolor="white")
print(f"Saved: {p}")
plt.close(fig)

print("\n" + "="*65)
print("FULL METRIC SUMMARY")
print("="*65)
if brain_case_results:
    print(f"Brain (n={len(brain_case_results)} BraTS20):")
    for lbl, name in BRAIN_LABEL_NAMES.items():
        d = m_val(bs, f"lbl{lbl}_Dice")
        b = m_val(bs, f"lbl{lbl}_Balanced_Accuracy")
        f = m_val(bs, f"lbl{lbl}_F1")
        print(f"  {name}: Dice={d:.3f}  BalAcc={b:.3f}  F1={f:.3f}")
    print(f"  Macro F1={m_val(bs,'macro_f1'):.3f}")
if liver_case_results:
    print(f"Liver (n={len(liver_case_results)}):")
    for lbl, name in LIVER_LABEL_NAMES.items():
        d = m_val(ls, f"lbl{lbl}_Dice")
        b = m_val(ls, f"lbl{lbl}_Balanced_Accuracy")
        print(f"  {name}: Dice={d:.3f}  BalAcc={b:.3f}")
    print(f"  Macro F1={m_val(ls,'macro_f1'):.3f}")
if lgs:
    print(f"Lung (n={lung_n}): Dice={lgs.get('Dice',{}).get('mean',0):.3f}  "
          f"BalAcc={lgs.get('Balanced_Accuracy',{}).get('mean',0):.3f}")
print(f"\nOutputs: {OUT}")
print("Done.")