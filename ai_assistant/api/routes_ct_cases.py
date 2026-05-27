import os
from pathlib import Path
from fastapi import APIRouter

router = APIRouter()


def _ws() -> Path:
    return Path(os.environ.get("AI_WORKSPACE_ROOT", os.getcwd()))


def _list_cases(data_dir: Path, seg_dir: Path) -> list[dict]:
    if not data_dir.exists():
        return []
    cases = []
    for folder in sorted(data_dir.iterdir()):
        if not folder.is_dir():
            continue
        nifti_files = list(folder.glob("*.nii.gz")) + list(folder.glob("*.nii"))
        if not nifti_files:
            continue
        mask_dir = seg_dir / folder.name
        has_mask = mask_dir.exists() and bool(list(mask_dir.glob("*.nii.gz")))
        cases.append({
            "case_id":  folder.name,
            "has_mask": has_mask,
            "nifti":    nifti_files[0].name,
        })
    return cases


@router.get("/lung")
def list_lung_cases():
    ws = _ws()
    return _list_cases(
        data_dir=ws / "lung_data",
        seg_dir=ws / "outputs" / "segmentations" / "lung",
    )


@router.get("/liver")
def list_liver_cases():
    ws = _ws()
    return _list_cases(
        data_dir=ws / "liver_data",
        seg_dir=ws / "outputs" / "segmentations" / "liver",
    )
