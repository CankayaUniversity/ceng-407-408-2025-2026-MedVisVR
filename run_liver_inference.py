if __name__ == '__main__':
    import os, shutil
    from pathlib import Path

    ROOT         = Path(__file__).parent
    LIVER_MODEL  = ROOT / "nnunet_liver" / "nnUNetTrainerV2__nnUNetPlansv2.1"
    LIVER_IMAGES = Path(r"C:\Users\User\OneDrive\Desktop\cigerim\nnUNet_data\nnUNet_raw\Dataset603_Liver\imagesTr")
    TEMP_DIR     = ROOT / "outputs" / "_inference_temp"
    LIVER_INPUT  = TEMP_DIR / "liver_input"
    LIVER_OUTPUT = TEMP_DIR / "liver_output"
    RESULTS_BASE = ROOT / "nnunet_liver_results"

    LIVER_INPUT.mkdir(parents=True, exist_ok=True)
    LIVER_OUTPUT.mkdir(parents=True, exist_ok=True)

    N_LIVER = 10
    img_files      = sorted(LIVER_IMAGES.glob("liver_*_0000.nii.gz"))[:N_LIVER]
    liver_case_ids = [f.name.replace("_0000.nii.gz", "") for f in img_files]
    print(f"Selected {N_LIVER} liver cases: {liver_case_ids}")

    for img_f in img_files:
        dst = LIVER_INPUT / img_f.name
        if not dst.exists():
            shutil.copy2(img_f, dst)
    print(f"Input ready: {LIVER_INPUT}")

    os.environ["nnUNet_raw_data_base"] = str(
        Path(r"C:\Users\User\OneDrive\Desktop\cigerim\nnUNet_data\nnUNet_raw"))
    os.environ["nnUNet_preprocessed"]  = str(
        Path(r"C:\Users\User\OneDrive\Desktop\cigerim\nnUNet_data\nnUNet_preprocessed"))
    os.environ["RESULTS_FOLDER"]       = str(RESULTS_BASE)

    model_dest = RESULTS_BASE / "nnUNet" / "3d_fullres" / "Task003_Liver" / "nnUNetTrainerV2__nnUNetPlansv2.1"
    model_dest.mkdir(parents=True, exist_ok=True)

    for item in LIVER_MODEL.iterdir():
        dst = model_dest / item.name
        if dst.exists():
            continue
        if item.is_dir():
            shutil.copytree(str(item), str(dst))
        else:
            shutil.copy2(str(item), str(dst))
    print(f"Model ready: {model_dest}")

    import torch, numpy as np

    try:
        torch.serialization.add_safe_globals([
            np.core.multiarray.scalar,
            np.dtype,
            np.ndarray,
        ])
    except Exception:
        pass

    _orig_load = torch.load
    def _patched_load(f, map_location=None, **kwargs):
        kwargs.setdefault('weights_only', False)
        return _orig_load(f, map_location=map_location, **kwargs)
    torch.load = _patched_load
    print("torch.load patched for PyTorch 2.6 compatibility (weights_only=False)")

    from nnunet.inference.predict import predict_from_folder

    print("\nStarting inference (tta=False, 5-fold ensemble)...")
    predict_from_folder(
        model=str(model_dest),
        input_folder=str(LIVER_INPUT),
        output_folder=str(LIVER_OUTPUT),
        folds=(0, 1, 2, 3, 4),
        save_npz=False,
        num_threads_preprocessing=1,
        num_threads_nifti_save=1,
        lowres_segmentations=None,
        part_id=0,
        num_parts=1,
        tta=False,
        mixed_precision=True,
        overwrite_existing=False,
        mode="normal",
        step_size=0.5,
        checkpoint_name="model_final_checkpoint",
        disable_postprocessing=True,
    )

    print("\nInference complete!")
    print(f"Predictions: {LIVER_OUTPUT}")
    preds = list(LIVER_OUTPUT.glob("*.nii.gz"))
    print(f"Files generated: {len(preds)}")
    for p in preds:
        print(f"  {p.name}  ({p.stat().st_size // 1024} KB)")

    print("\nNext step: .venv\\Scripts\\python.exe compute_all_metrics.py")