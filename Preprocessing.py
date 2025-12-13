import os
import glob
import torch
import numpy as np
import nibabel as nib
from monai.utils import first
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    NormalizeIntensityd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    RandShiftIntensityd,
    ConvertToMultiChannelBasedOnBratsClassesd,
    EnsureTyped
)
from monai.data import Dataset, DataLoader
from monai.utils import set_determinism

def main():
    set_determinism(seed=0)

    # Define data and output directories
    data_dir = r"user\Desktop\project\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData"
    output_dir = r"user\Desktop\project\Preprocessed_Output"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Data Directory: {data_dir}")
    print(f"Output Directory: {output_dir}")

    train_images = sorted(glob.glob(os.path.join(data_dir, "BraTS20_Training_*")))

    if len(train_images) == 0:
        print("ERROR: No files found! Please check the data directory path.")
        return

    print(f"Total number of patients to process: {len(train_images)}")

    data_dicts = []
    for patient_dir in train_images:
        patient_name = os.path.basename(patient_dir)
        data_dicts.append({
            "image": [
                os.path.join(patient_dir, f"{patient_name}_t1.nii"),
                os.path.join(patient_dir, f"{patient_name}_t1ce.nii"),
                os.path.join(patient_dir, f"{patient_name}_t2.nii"),
                os.path.join(patient_dir, f"{patient_name}_flair.nii"),
            ],
            "label": os.path.join(patient_dir, f"{patient_name}_seg.nii"),
            "name": patient_name
        })

    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            ConvertToMultiChannelBasedOnBratsClassesd(keys=["label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest"),
            ),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=[128, 128, 128],
                pos=2,
                neg=1,
                num_samples=1,
                image_key="image",
                image_threshold=0,
            ),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
            RandShiftIntensityd(keys=["image"], offsets=0.10, prob=0.5),
            EnsureTyped(keys=["image", "label"]),
        ]
    )

    ds = Dataset(data=data_dicts, transform=train_transforms)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    print("\nStarting processing...")

    for i, batch in enumerate(loader):
        img_tensor = batch["image"][0]
        lbl_tensor = batch["label"][0]
        name = batch["name"][0]
        
        print(f"[{i+1}/{len(train_images)}] Saving: {name}")

        img_data = img_tensor.detach().cpu().numpy()
        img_data = np.moveaxis(img_data, 0, -1)
        new_img = nib.Nifti1Image(img_data, np.eye(4))
        nib.save(new_img, os.path.join(output_dir, f"{name}_aug_image.nii"))

        lbl_data = lbl_tensor.detach().cpu().numpy()
        lbl_data = np.moveaxis(lbl_data, 0, -1)
        new_lbl = nib.Nifti1Image(lbl_data, np.eye(4))
        nib.save(new_lbl, os.path.join(output_dir, f"{name}_aug_label.nii"))

    print("\nALL OPERATIONS COMPLETED!")
    print(f"You can check the files here: {output_dir}")

if __name__ == "__main__":
    main()
