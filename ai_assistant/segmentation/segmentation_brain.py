import os
import shutil
import tempfile
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

WORKSPACE_ROOT = Path(os.environ.get("AI_WORKSPACE_ROOT", os.getcwd()))

BRAIN_MODEL_DIR = WORKSPACE_ROOT / "nnunet_brain"


def run_brain_segmentation(input_nifti_path: str, output_dir: str) -> dict:
    try:
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
        import torch
    except ImportError as e:
        return {"success": False, "output_file": None,
                "error": f"nnunetv2 import error: {e}"}

    input_path = Path(input_nifti_path)
    if not input_path.exists():
        return {"success": False, "output_file": None,
                "error": f"Input file not found: {input_nifti_path}"}

    if not BRAIN_MODEL_DIR.exists():
        return {"success": False, "output_file": None,
                "error": f"Brain model directory not found: {BRAIN_MODEL_DIR}"}

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp_in:
        tmp_input = Path(tmp_in)
        case_dir = input_path.parent

        fname = input_path.name
        prefix = fname.replace(".nii.gz", "").replace(".nii", "")
        for brats_suf in ("_flair", "_t1ce", "_t1", "_t2"):
            if prefix.endswith(brats_suf):
                prefix = prefix[: -len(brats_suf)]
                break

        CHANNEL_SUFFIXES = [
            ("_flair", "_0000"),
            ("_t1",    "_0001"),
            ("_t1ce",  "_0002"),
            ("_t2",    "_0003"),
        ]

        def _copy_to_tmp(src: Path, dst: Path):
            if str(src).endswith(".nii.gz"):
                shutil.copy(src, dst)
            else:
                import gzip
                with open(src, "rb") as f_in, gzip.open(dst, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)

        dest_files = []
        for seq_suf, ch_suf in CHANNEL_SUFFIXES:
            candidate_gz  = case_dir / f"{prefix}{seq_suf}.nii.gz"
            candidate_nii = case_dir / f"{prefix}{seq_suf}.nii"
            dst = tmp_input / f"{prefix}{ch_suf}.nii.gz"

            if candidate_gz.exists():
                _copy_to_tmp(candidate_gz, dst)
                dest_files.append(str(dst))
            elif candidate_nii.exists():
                _copy_to_tmp(candidate_nii, dst)
                dest_files.append(str(dst))
            elif seq_suf == "_flair":
                _copy_to_tmp(input_path, dst)
                dest_files.append(str(dst))

        n_channels = len(dest_files)
        logger.info(f"Brain segmentation: {n_channels} channel(s) found for {prefix}")
        if n_channels < 4:
            logger.warning(
                f"Only {n_channels}/4 BraTS sequences found in {case_dir}. "
                "Segmentation may fail — ensure _flair, _t1, _t1ce, _t2 files are present."
            )

        try:
            predictor = nnUNetPredictor(
                tile_step_size=0.5,
                use_gaussian=True,
                use_mirroring=True,
                perform_everything_on_device=True,
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                verbose=False,
                verbose_preprocessing=False,
                allow_tqdm=True,
            )
            predictor.initialize_from_trained_model_folder(
                model_training_output_dir=str(BRAIN_MODEL_DIR),
                use_folds=(0, 1, 2, 3, 4),
                checkpoint_name="checkpoint_final.pth",
            )
            predictor.predict_from_files(
                list_of_lists_or_source_folder=[dest_files],
                output_folder_or_list_of_truncated_output_files=str(output_path),
                save_probabilities=False,
                overwrite=True,
                num_processes_preprocessing=1,
                num_processes_segmentation_export=1,
            )
        except Exception as e:
            logger.error(f"Brain segmentation error: {e}")
            return {"success": False, "output_file": None, "error": str(e)}

    output_files = list(output_path.glob("*.nii.gz"))
    output_file = str(output_files[0]) if output_files else None

    return {
        "success": True,
        "output_file": output_file,
        "labels": {"0": "Background", "1": "Brain Tumor"},
        "error": None,
    }
