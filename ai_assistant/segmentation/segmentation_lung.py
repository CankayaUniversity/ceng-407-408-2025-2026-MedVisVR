import os
import shutil
import tempfile
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

WORKSPACE_ROOT = Path(os.environ.get("AI_WORKSPACE_ROOT", os.getcwd()))

LUNG_MODEL_DIR = WORKSPACE_ROOT / "nnunet_lung"


def run_lung_segmentation(input_nifti_path: str, output_dir: str) -> dict:
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

    if not LUNG_MODEL_DIR.exists():
        return {"success": False, "output_file": None,
                "error": f"Lung model directory not found: {LUNG_MODEL_DIR}"}

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp_in:
        tmp_input = Path(tmp_in)
        stem = input_path.name.replace(".nii.gz", "").replace(".nii", "")
        dest = tmp_input / f"{stem}_0000.nii.gz"

        if str(input_path).endswith(".nii.gz"):
            shutil.copy(input_path, dest)
        else:
            import gzip
            with open(input_path, "rb") as f_in:
                with gzip.open(dest, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)

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
                model_training_output_dir=str(LUNG_MODEL_DIR),
                use_folds=(0,),
                checkpoint_name="checkpoint_final.pth",
            )
            predictor.predict_from_files(
                list_of_lists_or_source_folder=[[str(dest)]],
                output_folder_or_list_of_truncated_output_files=str(output_path),
                save_probabilities=False,
                overwrite=True,
                num_processes_preprocessing=1,
                num_processes_segmentation_export=1,
            )
        except Exception as e:
            logger.error(f"Lung segmentation error: {e}")
            return {"success": False, "output_file": None, "error": str(e)}

    output_files = list(output_path.glob("*.nii.gz"))
    output_file = str(output_files[0]) if output_files else None

    return {
        "success": True,
        "output_file": output_file,
        "labels": {"0": "Background", "1": "Lung Tumor"},
        "error": None,
    }
