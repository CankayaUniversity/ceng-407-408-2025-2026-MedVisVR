if __name__ == '__main__':
    import zipfile, shutil, os
    import torch
    from pathlib import Path

    ROOT         = Path(__file__).parent
    BRAIN_MODEL  = ROOT / "nnunet_brain"
    BRATS20_ZIP  = Path(r"C:\Users\User\OneDrive\Desktop\BraTS20_Training_020.zip")
    TEMP_DIR     = ROOT / "outputs" / "_inference_temp"
    BRAIN_INPUT  = TEMP_DIR / "brain_input"
    BRAIN_OUTPUT = TEMP_DIR / "brain_output"
    BRAIN_GT_DIR = TEMP_DIR / "brain_gt"

    for d in [BRAIN_INPUT, BRAIN_OUTPUT, BRAIN_GT_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    N_CASES = 5
    print(f"Opening BraTS20 zip: {BRATS20_ZIP}")
    brain_cases = []

    with zipfile.ZipFile(str(BRATS20_ZIP), 'r') as zf:
        all_segs   = [e for e in zf.namelist() if e.endswith("_seg.nii")]
        case_names = sorted({e.split("/")[0] for e in all_segs})[:N_CASES]
        print(f"Selected cases: {case_names}")

        modality_map = {
            "_t1.nii":    "0000",
            "_t1ce.nii":  "0001",
            "_t2.nii":    "0002",
            "_flair.nii": "0003",
        }

        for case_name in case_names:
            cid = case_name.split("_")[-1]
            ok  = True

            for suffix, chan in modality_map.items():
                src_name = f"{case_name}/{case_name}{suffix}"
                if src_name not in zf.namelist():
                    print(f"  SKIP {case_name}: {src_name} not found")
                    ok = False; break
                dst = BRAIN_INPUT / f"BraTS_{cid}_{chan}.nii"
                if not dst.exists():
                    with zf.open(src_name) as src, open(dst, "wb") as tgt:
                        shutil.copyfileobj(src, tgt)

            seg_name = f"{case_name}/{case_name}_seg.nii"
            if seg_name in zf.namelist():
                dst_gt = BRAIN_GT_DIR / f"BraTS_{cid}.nii"
                if not dst_gt.exists():
                    with zf.open(seg_name) as src, open(dst_gt, "wb") as tgt:
                        shutil.copyfileobj(src, tgt)
            if ok:
                brain_cases.append(cid)

    print(f"\n{len(brain_cases)} cases ready: {brain_cases}")
    print(f"Input folder: {BRAIN_INPUT}")

    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"\nDevice: {device}")
    if device.type == "cpu":
        print("  WARNING: GPU not found, running on CPU (may be very slow)")

    predictor = nnUNetPredictor(
        tile_step_size=0.6,
        use_gaussian=True,
        use_mirroring=False,
        perform_everything_on_device=False,
        device=device,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True,
    )

    print("Loading model...")
    predictor.initialize_from_trained_model_folder(
        str(BRAIN_MODEL),
        use_folds=(0,),
        checkpoint_name="checkpoint_final.pth",
    )
    print("Model loaded.")

    list_of_lists = []
    output_files  = []
    for cid in brain_cases:
        modalities = sorted(BRAIN_INPUT.glob(f"BraTS_{cid}_*.nii"))
        if len(modalities) == 4:
            list_of_lists.append([str(m) for m in modalities])
            output_files.append(str(BRAIN_OUTPUT / f"BraTS_{cid}.nii.gz"))
        else:
            print(f"  SKIP BraTS_{cid}: {len(modalities)} modalities found (expected 4)")

    print(f"\nStarting inference ({len(list_of_lists)} cases)...")
    predictor.predict_from_files(
        list_of_lists_or_source_folder=list_of_lists,
        output_folder_or_list_of_truncated_output_files=output_files,
        save_probabilities=False,
        overwrite=True,
        num_processes_preprocessing=1,
        num_processes_segmentation_export=1,
    )

    print("\nInference complete!")
    print(f"Predictions: {BRAIN_OUTPUT}")
    preds = list(BRAIN_OUTPUT.glob("*.nii.gz"))
    print(f"Files generated: {len(preds)}")
    for p in preds:
        print(f"  {p.name}  ({p.stat().st_size // 1024} KB)")

    print("\nNext step: .venv\\Scripts\\python.exe compute_all_metrics.py")