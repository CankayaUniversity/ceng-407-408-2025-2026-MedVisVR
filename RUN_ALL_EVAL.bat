@echo off
cd /d "%~dp0"
echo.
echo  ================================================
echo   CARVIS -- Evaluation Scripts
echo  ================================================
echo.

echo [1/9] Segmentation volume charts...
.venv\Scripts\python.exe eval_01_segmentation.py
if errorlevel 1 echo [WARN] eval_01 finished with errors.
echo.

echo [2/9] Q^&A performance (evidence coverage, confidence)...
.venv\Scripts\python.exe eval_02_qa.py
if errorlevel 1 echo [WARN] eval_02 finished with errors.
echo.

echo [3/9] Atlas heuristic (lobe, hemisphere, confidence)...
.venv\Scripts\python.exe eval_03_atlas.py
if errorlevel 1 echo [WARN] eval_03 finished with errors.
echo.

echo [4/9] System overview (case distribution, completeness)...
.venv\Scripts\python.exe eval_04_system.py
if errorlevel 1 echo [WARN] eval_04 finished with errors.
echo.

echo [5/9] Segmentation metrics - brain reference + liver (Dice/F1/Precision/Recall)...
.venv\Scripts\python.exe eval_05_metrics.py
if errorlevel 1 echo [WARN] eval_05 finished with errors.
echo.

echo [6/9] Final poster metrics table + bar chart (eval_07)...
.venv\Scripts\python.exe eval_07_final_metrics.py
if errorlevel 1 echo [WARN] eval_07 finished with errors.
echo.

echo [7/9] LLM benchmark + report quality + system components (eval_08)...
.venv\Scripts\python.exe eval_08_llm_report.py
if errorlevel 1 echo [WARN] eval_08 finished with errors.
echo.

echo [8/9] AUC Score + Macro F1 (eval_09)...
.venv\Scripts\python.exe eval_09_auc_macro.py
if errorlevel 1 echo [WARN] eval_09 finished with errors.
echo.

echo [9/9] Training curves - Brain + Liver 5-fold (eval_10)...
.venv\Scripts\python.exe eval_10_training_curves.py
if errorlevel 1 echo [WARN] eval_10 finished with errors.
echo.

echo  ================================================
echo   Done! All figures saved to: outputs\evaluation\
echo  ================================================
echo.
echo  Outputs summary:
echo    eval_01  -- seg_brain_volumes.png, seg_liver_volumes.png, seg_summary_table.png
echo    eval_02  -- qa_evidence_coverage.png, qa_confidence_dist.png, qa_summary_table.png
echo    eval_03  -- atlas_confidence_dist.png, atlas_lobe_distribution.png, atlas_hemisphere.png
echo    eval_04  -- system_case_distribution.png, system_report_completeness.png, system_overview_table.png
echo    eval_05  -- metrics_brain_reference.png, metrics_liver_per_case.png, metrics_all_summary.png
echo    eval_07  -- final_metrics_table.png, final_metrics_bar.png   (POSTER MAIN)
echo    eval_08  -- llm_benchmark_table.png, report_quality_heatmap.png, system_components_heatmap.png
echo    eval_09  -- auc_macro_table.png, auc_macro_bar.png, qa_auc_roc.png
echo    eval_10  -- training_brain_all_folds.png, training_liver_all_folds.png, training_combined.png
echo.
pause
