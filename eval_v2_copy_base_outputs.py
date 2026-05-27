import subprocess, shutil, sys
from pathlib import Path

ROOT   = Path(__file__).parent
SRC    = ROOT / "outputs" / "evaluation"
DST    = ROOT / "outputs" / "evaluation_v2"
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
DST.mkdir(parents=True, exist_ok=True)

BASE_SCRIPTS = [
    "eval_01_segmentation.py",
    "eval_02_qa.py",
    "eval_03_atlas.py",
    "eval_04_system.py",
    "eval_05_metrics.py",
]

FILES_TO_COPY = [
    "seg_brain_volumes.png",
    "seg_brain_labels.png",
    "seg_liver_volumes.png",
    "seg_summary_table.png",
    "qa_evidence_coverage.png",
    "qa_confidence_dist.png",
    "qa_summary_table.png",
    "atlas_confidence_dist.png",
    "atlas_lobe_distribution.png",
    "atlas_hemisphere.png",
    "system_case_distribution.png",
    "system_report_completeness.png",
    "system_overview_table.png",
    "metrics_brain_reference.png",
]

print("="*60)
print("Step 1 — Running base eval scripts (01-05)...")
print("="*60)
for script in BASE_SCRIPTS:
    print(f"\n  Running {script} ...")
    result = subprocess.run(
        [str(PYTHON), script],
        cwd=str(ROOT),
        capture_output=False,
    )
    if result.returncode != 0:
        print(f"  [WARN] {script} exited with code {result.returncode}")
    else:
        print(f"  [OK] {script} complete.")

print("\n" + "="*60)
print("Step 2 — Copying outputs to evaluation_v2/ ...")
print("="*60)
copied  = []
skipped = []
for fname in FILES_TO_COPY:
    src_file = SRC / fname
    dst_file = DST / fname
    if src_file.exists():
        shutil.copy2(src_file, dst_file)
        copied.append(fname)
        print(f"  Copied: {fname}")
    else:
        skipped.append(fname)
        print(f"  [SKIP] Not found: {fname}")

print(f"\nCopied {len(copied)} files, skipped {len(skipped)}.")
print(f"\nFinal evaluation_v2/ contents:")
for f in sorted(DST.iterdir()):
    print(f"  {f.name}")

print("\nDone. eval_v2_copy_base_outputs.py complete.")