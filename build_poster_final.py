import shutil
from pathlib import Path

ROOT    = Path(__file__).parent
SRC_V2  = ROOT / "outputs" / "evaluation_v2"
DEST    = ROOT / "outputs" / "poster_final"
DEST.mkdir(parents=True, exist_ok=True)

FIGURES = [
    (
        "metrics_table_poster.png",
        "01_segmentation_metrics_table.png",
        "Segmentation Metrics",
        "Heatmap table: Dice, Precision, Recall, Specificity, F1, Balanced Accuracy "
        "for Brain MRI (3 sub-regions) and Liver CT (n=1 preliminary)"
    ),
    (
        "metrics_bar_poster.png",
        "02_segmentation_metrics_bar.png",
        "Segmentation Metrics",
        "Grouped bar chart: all 6 metrics side-by-side for Brain MRI and Liver CT"
    ),
    (
        "auc_macro_table_v2.png",
        "03_segmentation_auc_macro_table.png",
        "Segmentation Metrics",
        "Full 7-metric table including AUC (Balanced Acc.) and Macro F1 per segment"
    ),
    (
        "auc_macro_bar_v2.png",
        "04_segmentation_auc_macro_bar.png",
        "Segmentation Metrics",
        "Bar chart: 7 metrics including AUC and Macro F1"
    ),
    (
        "training_combined_poster.png",
        "05_training_curves_combined.png",
        "Training Curves",
        "Brain MRI (fold 0) and Liver CT (fold 0) training curves side-by-side: "
        "Train Loss, Val Loss, Pseudo Dice over 1000 epochs"
    ),
    (
        "training_brain_all_folds_v2.png",
        "06_training_brain_5folds.png",
        "Training Curves",
        "Brain MRI nnUNet v2 — all 5 folds training curves (1000 epochs each)"
    ),
    (
        "training_liver_all_folds_v2.png",
        "07_training_liver_5folds.png",
        "Training Curves",
        "Liver CT nnUNet v1 TrainerV2 — all 5 folds training curves (1000 epochs each)"
    ),
    (
        "llm_benchmark_v2.png",
        "08_llm_model_benchmarks.png",
        "LLM Model Quality",
        "Qwen2.5-7B-Instruct vs LLaMA-3.1-8B, Mistral-7B, GPT-4o-mini "
        "on MMLU, HellaSwag, HumanEval, GSM8K, MATH, ARC, TruthfulQA"
    ),
    (
        "qa_evidence_coverage_poster.png",
        "09_qa_evidence_coverage.png",
        "Q&A System",
        "Per-case evidence-supported vs insufficient answer rate; "
        "average confidence score per Brain MRI case"
    ),
    (
        "qa_auc_roc_v2.png",
        "10_qa_roc_curve.png",
        "Q&A System",
        "ROC curve: Q&A confidence score as predictor of evidence-supported answers "
        "(n=140 answers, AUC=0.522)"
    ),
    (
        "qa_summary_table.png",
        "11_qa_summary_table.png",
        "Q&A System",
        "Summary table: total Q&A answers, evidence-supported rate, "
        "insufficient rate, average confidence"
    ),
    (
        "atlas_confidence_dist.png",
        "12_atlas_confidence_distribution.png",
        "Atlas Heuristic",
        "Anatomical localisation confidence: High / Medium / Low "
        "bar chart and pie chart across all Brain MRI cases"
    ),
    (
        "atlas_lobe_distribution.png",
        "13_atlas_lobe_distribution.png",
        "Atlas Heuristic",
        "Tumour lesion count per cerebral lobe (frontal, parietal, temporal, etc.)"
    ),
    (
        "atlas_hemisphere.png",
        "14_atlas_hemisphere_depth.png",
        "Atlas Heuristic",
        "Lesion hemisphere (left/right) and depth class distributions"
    ),
    (
        "report_quality_heatmap_poster.png",
        "15_report_section_coverage.png",
        "Report Quality",
        "Section coverage heatmap: which of the 7 report sections are present "
        "for each case (Brain / Liver / Lung)"
    ),
    (
        "report_quality_bars_poster.png",
        "16_report_quality_bars.png",
        "Report Quality",
        "Three-panel chart: word count, section completeness (%), "
        "clinical term density (%) per case"
    ),
    (
        "system_case_distribution.png",
        "17_system_case_distribution.png",
        "System Overview",
        "Case distribution by organ: Brain MRI (n=22), Liver CT (n=4), Lung CT (n=2) "
        "donut chart and bar chart"
    ),
    (
        "system_components_heatmap_v2.png",
        "18_system_components_heatmap.png",
        "System Overview",
        "AI component usage per case: Patient Context, Segmentation, Report (LLM), "
        "Q&A, Quality Report, Anatomical Labelling, RAG Evidence"
    ),
    (
        "system_report_completeness.png",
        "19_system_pipeline_completeness.png",
        "System Overview",
        "Pipeline component availability (%) per organ group — heatmap"
    ),
    (
        "system_overview_table.png",
        "20_system_overview_table.png",
        "System Overview",
        "Summary table: cases per organ, coverage rates for each pipeline component"
    ),
    (
        "poster_real_metrics_table.png",
        "25_REAL_metrics_table.png",
        "Real Measurements",
        "Segmentation metrics from real CV data: Brain 5-fold EMA Dice, "
        "Liver 5-fold CV Dice (n=131), Lung 13-case voxel-level computation"
    ),
    (
        "poster_real_metrics_bar.png",
        "26_REAL_metrics_bar.png",
        "Real Measurements",
        "Bar chart: Dice comparison (Brain/Liver/Lung) + Lung full metric profile"
    ),
    (
        "poster_real_brain_classes.png",
        "27_REAL_brain_per_class_dice.png",
        "Real Measurements",
        "Brain MRI per-class Dice across 5 folds: edema, non-enhancing, enhancing"
    ),
    (
        "poster_real_lung_detail.png",
        "28_REAL_lung_per_case_detail.png",
        "Real Measurements",
        "Lung CT per-case validation metrics: Dice, Balanced Acc, Precision, Recall (n=13)"
    ),
    (
        "seg_brain_volumes.png",
        "21_seg_brain_tumour_volumes.png",
        "Segmentation Analysis",
        "Brain MRI: tumour sub-region volumes per case (bar chart)"
    ),
    (
        "seg_brain_labels.png",
        "22_seg_brain_label_breakdown.png",
        "Segmentation Analysis",
        "Brain MRI: stacked label breakdown per case"
    ),
    (
        "seg_liver_volumes.png",
        "23_seg_liver_volumes.png",
        "Segmentation Analysis",
        "Liver CT: parenchyma and tumour volumes per case"
    ),
    (
        "seg_summary_table.png",
        "24_seg_summary_table.png",
        "Segmentation Analysis",
        "Summary table: mean volumes and diameters per organ group"
    ),
]

print("Building poster_final/ folder...")
print(f"Destination: {DEST}\n")

manifest_lines = [
    "CARVIS — Poster Final Figure Manifest",
    "=" * 60,
    "",
]

copied, missing = 0, 0
current_cat = None

for src_name, dst_name, category, description in FIGURES:
    src = SRC_V2 / src_name
    dst = DEST  / dst_name

    if category != current_cat:
        manifest_lines.append(f"\n{'─'*60}")
        manifest_lines.append(f"  {category.upper()}")
        manifest_lines.append(f"{'─'*60}")
        current_cat = category
        print(f"\n  [{category}]")

    if src.exists():
        shutil.copy2(src, dst)
        status = "OK"
        copied += 1
    else:
        status = "MISSING"
        missing += 1

    manifest_lines.append(f"  {dst_name}")
    manifest_lines.append(f"    {description}")
    manifest_lines.append(f"    Source: {src_name}  [{status}]")
    manifest_lines.append("")
    print(f"  {'  OK' if status=='OK' else 'MISS'}  {dst_name}")

manifest_lines += [
    "=" * 60,
    f"Total: {copied} figures copied, {missing} missing",
    "",
    "Folder: outputs/poster_final/",
    "Generated by: build_poster_final.py",
]
manifest_path = DEST / "poster_final_manifest.txt"
manifest_path.write_text("\n".join(manifest_lines), encoding="utf-8")

print(f"\n{'='*55}")
print(f"Done.  {copied} figures copied  |  {missing} missing")
print(f"Manifest: {manifest_path}")
print(f"\nFinal listing ({DEST.name}/):")
for f in sorted(DEST.iterdir()):
    size_kb = f.stat().st_size // 1024
    print(f"  {f.name:<50}  {size_kb:>5} KB")
