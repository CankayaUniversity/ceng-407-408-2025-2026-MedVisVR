import argparse
import shutil
from pathlib import Path


CHANNEL_MAP = {
    "t1": "0000",
    "t1ce": "0001",
    "t2": "0002",
    "flair": "0003",
}


def _find_sequence_file(case_dir: Path, sequence: str) -> Path:
    matches = sorted(case_dir.glob(f"*_{sequence}.nii")) + sorted(case_dir.glob(f"*_{sequence}.nii.gz"))
    if not matches:
        raise FileNotFoundError(f"missing {sequence} file in {case_dir}")
    return matches[0]


def _copy_or_convert(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    is_nii_gz = len(src.suffixes) >= 2 and src.suffixes[-2:] == [".nii", ".gz"]
    if is_nii_gz:
        shutil.copy2(src, dst)
        return

    import SimpleITK as sitk  

    img = sitk.ReadImage(str(src))
    sitk.WriteImage(img, str(dst), useCompression=True)


def discover_cases(brain_root: Path) -> list[str]:
    return sorted([p.name for p in brain_root.iterdir() if p.is_dir()])


def prepare_case(case_id: str, brain_root: Path, out_root: Path) -> None:
    case_dir = brain_root / case_id
    for sequence, channel in CHANNEL_MAP.items():
        src = _find_sequence_file(case_dir, sequence)
        dst = out_root / f"{case_id}_{channel}.nii.gz"
        _copy_or_convert(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare nnUNet inference inputs from brain_data cases")
    parser.add_argument("--brain-root", default="brain_data", help="Root folder containing per-case MRI sequences")
    parser.add_argument("--out-root", default="nnunet_inference_input", help="nnUNet input folder")
    args = parser.parse_args()

    brain_root = Path(args.brain_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    cases = discover_cases(brain_root)
    for case_id in cases:
        prepare_case(case_id, brain_root, out_root)
        print(f"OK prepared={case_id}")

    print(f"DONE prepared_cases={len(cases)} out={out_root}")


if __name__ == "__main__":
    main()
