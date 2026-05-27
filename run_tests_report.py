import subprocess, sys, json, re, time
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ROOT     = Path(__file__).parent
TESTS    = ROOT / "tests"
OUT_DIR  = ROOT / "outputs" / "evaluation_real" / "test_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PYTHON   = sys.executable

TEST_FILES = [
    ("test_schema_validation.py",    "Schema Validation",     "unit"),
    ("test_report_generator.py",     "Report Generator",      "unit"),
    ("test_qa_engine.py",            "QA Engine",             "unit"),
    ("test_metrics.py",              "Metrics",               "unit"),
    ("test_guideline_retrieval.py",  "Guideline Retrieval",   "unit"),
    ("test_case_repository.py",      "Case Repository",       "unit"),
    ("test_cases_api.py",            "Cases API",             "integration"),
    ("test_reporting_extensions.py", "Reporting Extensions",  "integration"),
    ("test_performance.py",          "Performance",           "integration"),
    ("test_integration.py",          "Integration E2E",       "integration"),
    ("test_golden_case.py",          "Golden Case",           "integration"),
]

print("=" * 60)
print("CARVIS — Test Suite Runner")
print("=" * 60)
print(f"Tests dir : {TESTS}")
print(f"Output    : {OUT_DIR}")
print()

results = []

for fname, label, category in TEST_FILES:
    fpath = TESTS / fname
    if not fpath.exists():
        print(f"  SKIP  {fname}  (file not found)")
        results.append({
            "file": fname, "label": label, "category": category,
            "status": "SKIP", "passed": 0, "failed": 0,
            "errors": 0, "duration": 0, "output": "File not found"
        })
        continue

    print(f"  RUN   {fname} ...", end=" ", flush=True)
    t0 = time.time()

    proc = subprocess.run(
        [PYTHON, "-m", "pytest", str(fpath),
         "-v", "--tb=line", "--no-header",
         "--ignore-glob=*benchmark*"],
        capture_output=True, text=True, cwd=str(ROOT), timeout=300
    )
    dt = time.time() - t0
    out = proc.stdout + proc.stderr

    passed = failed = errors = 0
    status = "UNKNOWN"

    m_pass  = re.search(r'(\d+) passed',  out)
    m_fail  = re.search(r'(\d+) failed',  out)
    m_err   = re.search(r'(\d+) error',   out)
    m_skip  = re.search(r'(\d+) skipped', out)

    if m_pass:  passed  = int(m_pass.group(1))
    if m_fail:  failed  = int(m_fail.group(1))
    if m_err:   errors  = int(m_err.group(1))
    skipped = int(m_skip.group(1)) if m_skip else 0

    if proc.returncode == 0 and (passed > 0 or skipped > 0):
        status = "PASS"
    elif passed == 0 and failed == 0 and errors == 0:
        status = "ERROR"
    elif failed > 0 or errors > 0:
        status = "FAIL"
    elif skipped > 0 and passed == 0:
        status = "SKIP"
    else:
        status = "FAIL"

    icon = {"PASS": "✓", "FAIL": "✗", "SKIP": "—", "UNKNOWN": "?"}.get(status, "?")
    print(f"{icon}  {passed}P / {failed}F / {errors}E  ({dt:.1f}s)")

    results.append({
        "file": fname, "label": label, "category": category,
        "status": status, "passed": passed, "failed": failed,
        "errors": errors, "skipped": skipped,
        "duration": round(dt, 2),
        "output": out[-3000:] if len(out) > 3000 else out
    })

total_p = sum(r["passed"]  for r in results)
total_f = sum(r["failed"]  for r in results)
total_e = sum(r["errors"]  for r in results)
total_s = sum(r["skipped"] for r in results)
n_pass  = sum(1 for r in results if r["status"] == "PASS")
n_fail  = sum(1 for r in results if r["status"] == "FAIL")
n_skip  = sum(1 for r in results if r["status"] == "SKIP")

summary = {
    "total_files": len(results),
    "files_pass": n_pass, "files_fail": n_fail, "files_skip": n_skip,
    "total_tests_passed": total_p, "total_tests_failed": total_f,
    "total_tests_errors": total_e, "total_tests_skipped": total_s,
    "results": results
}

json_path = OUT_DIR / "test_results.json"
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print(f"\nJSON saved: {json_path}")

STATUS_COLOR   = {"PASS": "#16a34a", "FAIL": "#dc2626", "SKIP": "#6b7280", "ERROR": "#f59e0b", "UNKNOWN": "#9ca3af"}
CATEGORY_COLOR = {"unit": "#2563eb", "integration": "#7c3aed"}

fig, ax = plt.subplots(figsize=(14, max(5, len(results) * 0.75 + 3)))
ax.axis("off")

col_headers = ["Test Module", "Category", "Status", "Passed", "Failed", "Errors", "Time (s)"]
table_data = []

for r in results:
    table_data.append([
        r["label"],
        r["category"].upper(),
        r["status"],
        str(r["passed"]),
        str(r["failed"]) if r["failed"] > 0 else "—",
        str(r["errors"]) if r["errors"] > 0 else "—",
        f"{r['duration']:.1f}s",
    ])

tbl = ax.table(
    cellText=table_data,
    colLabels=col_headers,
    loc="center", cellLoc="center"
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1, 2.2)

for (row, col), cell in tbl.get_celld().items():
    cell.set_edgecolor("#e5e7eb")
    if row == 0:
        cell.set_facecolor("#1e1b4b")
        cell.set_text_props(color="white", fontweight="bold", fontsize=10.5)
    else:
        r_data = results[row - 1]
        sc = STATUS_COLOR.get(r_data["status"], "#9ca3af")
        if col == 2:
            cell.set_facecolor(sc)
            cell.set_text_props(color="white", fontweight="bold")
        elif col == 0:
            cat_c = CATEGORY_COLOR.get(r_data["category"], "#6b7280")
            import matplotlib.colors as mc
            rgb = mc.to_rgb(cat_c)
            tint = tuple(0.88 + 0.12*x for x in rgb)
            cell.set_facecolor(tint)
            cell.set_text_props(fontweight="bold")
        elif col in [3, 4, 5]:
            txt = cell.get_text().get_text()
            if txt not in ["—", "0"] and txt.isdigit() and int(txt) > 0:
                bad = col in [4, 5]
                cell.set_facecolor("#fef2f2" if bad else "#f0fdf4")
                cell.set_text_props(
                    color="#dc2626" if bad else "#16a34a",
                    fontweight="bold"
                )
            else:
                cell.set_facecolor("#f9fafb")
                cell.set_text_props(color="#9ca3af")
        else:
            cell.set_facecolor("#f9fafb")

pass_rate = f"{n_pass}/{len(results)} files passed"
test_rate = f"{total_p} tests passed, {total_f + total_e} failed"
ax.set_title(
    f"CARVIS — Test Suite Results\n{pass_rate}  |  {test_rate}",
    fontsize=13, fontweight="bold", pad=16, color="#1e1b4b"
)

patches = [
    mpatches.Patch(color="#16a34a", label="PASS"),
    mpatches.Patch(color="#dc2626", label="FAIL"),
    mpatches.Patch(color="#6b7280", label="SKIP"),
    mpatches.Patch(color=CATEGORY_COLOR["unit"], label="Unit test"),
    mpatches.Patch(color=CATEGORY_COLOR["integration"], label="Integration test"),
]
ax.legend(handles=patches, loc="lower right", fontsize=9,
          bbox_to_anchor=(1.0, -0.02), framealpha=0.9)

fig.tight_layout()
png_path = OUT_DIR / "test_results_table.png"
fig.savefig(png_path, dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"PNG saved: {png_path}")

fig2, ax2 = plt.subplots(figsize=(10, 4))
cats = ["PASS", "FAIL", "SKIP"]
vals = [n_pass, n_fail, n_skip]
cols = [STATUS_COLOR[c] for c in cats]
bars = ax2.bar(cats, vals, color=cols, width=0.4, edgecolor="white")
for bar, val in zip(bars, vals):
    if val > 0:
        ax2.text(bar.get_x() + bar.get_width()/2, val + 0.08,
                 str(val), ha="center", va="bottom", fontsize=14, fontweight="bold")
ax2.set_ylabel("Test Files", fontsize=12)
ax2.set_title(f"Test Suite Summary — {len(results)} test files\n"
              f"{total_p} individual tests passed, {total_f + total_e} failed",
              fontsize=12, fontweight="bold")
ax2.set_ylim(0, max(vals) + 2)
ax2.spines[["top","right"]].set_visible(False)
fig2.tight_layout()
bar_path = OUT_DIR / "test_results_bar.png"
fig2.savefig(bar_path, dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig2)
print(f"PNG saved: {bar_path}")

print(f"\n{'='*60}")
print(f"TEST SUMMARY")
print(f"{'='*60}")
print(f"  Files: {n_pass} PASS / {n_fail} FAIL / {n_skip} SKIP  ({len(results)} total)")
print(f"  Tests: {total_p} passed / {total_f} failed / {total_e} errors / {total_s} skipped")
print(f"\nOutputs: {OUT_DIR}")