@echo off
cd /d "%~dp0"
echo.
echo  =====================================================
echo   CARVIS -- Evaluation v2  (outputs\evaluation_v2\)
echo  =====================================================
echo   Fixes applied:
echo     - Raw Accuracy removed (replaced by Balanced Accuracy)
echo     - Liver metrics labelled as n=1 preliminary
echo     - MSD Lung excluded from metric tables (no GT)
echo     - BraTS-protocol -> BraTS-style multi-sequence MRI
echo     - pulmonary nodule -> lung tumour
echo     - Couinaud removed -> anatomical region labelling
echo  =====================================================
echo.

echo [1/5] Base scripts (eval_01-05) + copy to evaluation_v2...
.venv\Scripts\python.exe eval_v2_copy_base_outputs.py
if errorlevel 1 echo [WARN] Step 1 finished with errors.
echo.

echo [2/5] Final metrics table + bar (Balanced Accuracy, n=1 note)...
.venv\Scripts\python.exe eval_v2_07_metrics.py
if errorlevel 1 echo [WARN] eval_v2_07 finished with errors.
echo.

echo [3/5] LLM benchmark + report quality + system components...
.venv\Scripts\python.exe eval_v2_08_llm_report.py
if errorlevel 1 echo [WARN] eval_v2_08 finished with errors.
echo.

echo [4/5] AUC + Macro F1 (Balanced Accuracy, honest labels)...
.venv\Scripts\python.exe eval_v2_09_auc_macro.py
if errorlevel 1 echo [WARN] eval_v2_09 finished with errors.
echo.

echo [5/5] Training curves - Brain + Liver 5-fold...
.venv\Scripts\python.exe eval_v2_10_training.py
if errorlevel 1 echo [WARN] eval_v2_10 finished with errors.
echo.

echo  =====================================================
echo   Done!  All figures -> outputs\evaluation_v2\
echo  =====================================================
echo.
echo  Key poster figures (v2):
echo    final_metrics_table_v2.png      Heatmap table (7 metrics, no Accuracy)
echo    final_metrics_bar_v2.png        Grouped bar chart
echo    auc_macro_table_v2.png          Full 7-metric table incl AUC + Macro F1
echo    auc_macro_bar_v2.png            Bar chart
echo    llm_benchmark_v2.png            Qwen2.5-7B vs others
echo    report_quality_heatmap_v2.png   Section coverage per case
echo    system_components_heatmap_v2.png
echo    training_combined_v2.png        Brain + Liver training curves
echo    training_brain_all_folds_v2.png
echo    training_liver_all_folds_v2.png
echo.
pause
