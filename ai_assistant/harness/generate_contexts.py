import argparse
import json
from pathlib import Path

from ai_assistant.core.context_builder import build_patient_context_from_case, save_patient_context


def discover_cases(nifti_root: Path) -> list[str]:
    return sorted([p.name for p in nifti_root.iterdir() if p.is_dir()])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate patient_context.json files from NIfTI (+ optional DICOM)"
    )
    parser.add_argument("--manifest", default="", help="Optional path to cases_manifest.json")
    parser.add_argument(
        "--dicom-root",
        default="",
        help="Optional root folder containing case DICOM folders",
    )
    parser.add_argument("--nifti-root", required=True, help="Root folder containing case NIfTI folders")
    parser.add_argument("--out-root", required=True, help="Output root for per-case patient_context.json")
    parser.add_argument("--modality", default="MR", help="Default modality when DICOM metadata is missing")
    args = parser.parse_args()

    nifti_root = Path(args.nifti_root)
    if args.manifest:
        manifest_path = Path(args.manifest)
        with manifest_path.open("r", encoding="utf-8-sig") as f:
            manifest = json.load(f)
        cases = manifest.get("cases", [])
    else:
        cases = discover_cases(nifti_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    generated = 0
    for case in cases:
        nifti_case = nifti_root / case
        if not nifti_case.exists():
            print(f"SKIP {case}: missing NIfTI folder")
            continue

        if args.dicom_root:
            dicom_case = Path(args.dicom_root) / case
        else:
            dicom_case = nifti_case

        context = build_patient_context_from_case(
            patient_id=case,
            dicom_case_dir=str(dicom_case),
            nifti_case_dir=str(nifti_case),
            modality=args.modality,
        )

        out_path = out_root / case / "patient_context.json"
        save_patient_context(context, str(out_path))
        print(f"OK {case}: {out_path}")
        generated += 1

    print(f"DONE generated={generated} expected={len(cases)}")


if __name__ == "__main__":
    main()
