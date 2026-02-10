# MedVisVR

<p align="center">
  <img src="https://github.com/user-attachments/assets/110dd15e-0a21-4cbe-a9b9-46f5ccebe5b9" alt="MedVisVR Logo" width="600"/>
</p>

<p align="center">
  <strong>Medical Visualization in Virtual Reality for Brain Tumor Segmentation</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue" alt="Python 3.8+"/>
  <img src="https://img.shields.io/badge/MONAI-Latest-green" alt="MONAI"/>
  <img src="https://img.shields.io/badge/PyTorch-Latest-red" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/License-MIT-yellow" alt="License"/>
</p>

---

## Overview

**MedVisVR** is an advanced medical visualization platform that leverages Virtual Reality technology to provide immersive 3D visualization and analysis of brain tumor segmentation data. Built for the CENG 407/408 Senior Design Project (2025-2026), this project integrates state-of-the-art deep learning techniques with interactive VR environments to enhance medical image analysis and diagnostic capabilities.

The system processes multi-modal MRI scans from the BraTS (Brain Tumor Segmentation) dataset and presents them in an intuitive, three-dimensional virtual reality interface, enabling medical professionals and researchers to explore and analyze brain tumors in unprecedented detail.

---

## Key Features

- **3D Medical Image Visualization**: Immersive VR-based visualization of MRI brain scans
- **Multi-Modal MRI Support**: Processing of T1, T1ce, T2, and FLAIR MRI sequences
- **Advanced Preprocessing Pipeline**: Automated data preprocessing using MONAI framework
- **Brain Tumor Segmentation**: Integration with BraTS2020 dataset for tumor analysis
- **Interactive VR Interface**: Intuitive controls for medical data exploration
- **Real-time Rendering**: Efficient 3D rendering of volumetric medical data

---

## Technology Stack

### Core Technologies
- **Python 3.8+**: Primary programming language
- **PyTorch**: Deep learning framework
- **MONAI**: Medical Open Network for AI
- **Unity/Unreal Engine**: VR environment development
- **Nibabel**: Neuroimaging data I/O

### VR Platform
- **Meta Quest 2/3** (Recommended)
- **HTC Vive**
- **Valve Index**

### Medical Imaging Standards
- **NIfTI Format**: Neuroimaging Informatics Technology Initiative
- **DICOM Support**: Digital Imaging and Communications in Medicine

---

## Installation

### Prerequisites

```bash
# Python 3.8 or higher
python --version

# CUDA-enabled GPU (recommended for preprocessing)
nvidia-smi
```

### Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-org/ceng-407-408-2025-2026-MedVisVR.git
   cd ceng-407-408-2025-2026-MedVisVR
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install torch torchvision torchaudio
   pip install monai nibabel numpy
   pip install -r requirements.txt
   ```

4. **VR Setup**
   - Install VR runtime (Oculus, SteamVR, etc.)
   - Connect your VR headset
   - Calibrate play area

---

## Dataset

This project uses the **BraTS2020 (Brain Tumor Segmentation Challenge 2020)** dataset, which includes:

- **369 Training Cases**: Multi-institutional pre-operative MRI scans
- **4 MRI Modalities**: T1, T1ce (contrast-enhanced), T2, FLAIR
- **Manual Segmentations**: Expert-annotated tumor regions
- **3 Tumor Sub-regions**: Enhancing tumor, peritumoral edema, necrotic core

### Data Structure
```
BraTS2020_TrainingData/
└── MICCAI_BraTS2020_TrainingData/
    ├── BraTS20_Training_001/
    │   ├── BraTS20_Training_001_t1.nii
    │   ├── BraTS20_Training_001_t1ce.nii
    │   ├── BraTS20_Training_001_t2.nii
    │   ├── BraTS20_Training_001_flair.nii
    │   └── BraTS20_Training_001_seg.nii
    └── ...
```

---

## Usage

### Data Preprocessing

```bash
# Run preprocessing pipeline
python preprocessing.py
```

The preprocessing pipeline includes:
- **Loading**: Multi-modal MRI loading
- **Orientation**: Standardization to RAS orientation
- **Resampling**: Uniform 1mm³ spacing
- **Normalization**: Intensity normalization
- **Augmentation**: Random flips, rotations, intensity shifts
- **Cropping**: 128×128×128 patches

### VR Application

1. **Launch the VR Environment**
   ```bash
   python main.py --mode vr
   ```

2. **Load Preprocessed Data**
   - Use VR controllers to navigate the file menu
   - Select patient data from the preprocessed output

3. **Interactive Controls**
   - **Grip**: Rotate volume
   - **Trigger**: Select/Interact
   - **Thumbstick**: Navigate slices
   - **Menu Button**: Settings

---

## Project Structure

```
MedVisVR/
├── src/
│   ├── preprocessing/     # Data preprocessing modules
│   ├── visualization/     # VR rendering engine
│   ├── segmentation/      # Tumor segmentation models
│   └── utils/             # Utility functions
├── data/
│   ├── raw/               # Raw BraTS data
│   └── preprocessed/      # Processed outputs
├── models/                # Trained model weights
├── config/                # Configuration files
├── tests/                 # Unit and integration tests
└── docs/                  # Additional documentation
```

---

## Documentation

For detailed documentation, please visit our [Documents Repository](https://github.com/your-org/ceng-407-408-2025-2026-MedVisVR-Documents):

- **Literature Review**: State-of-the-art analysis
- **Methodology Report**: Technical approach and architecture
- **Dataset Description**: BraTS2020 preprocessing details
- **Final Report**: Complete project documentation

---

## Development Team

**CENG 407/408 Senior Design Project (2025-2026)**

### Team Members
| Student Number | Name |
|----------------|------|
| 202111012 | Alperen Berke Çetinkaya |
| 202211052 | Muhammed Yusuf Özcan |
| 202211011 | Sezer Ataş |
| 202211061 | Mete Serpil |

- **Advisor**: Assoc. Prof. Dr. Gül Tokdemir
- **Institution**: Çankaya University

---

## Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **BraTS Challenge Organizers**: For providing the comprehensive dataset
- **MONAI Consortium**: For the medical imaging framework
- **PyTorch Team**: For the deep learning platform
- **VR Community**: For development tools and resources

---

## Citation

If you use this project in your research, please cite:

```bibtex
@misc{medvisvr2025,
  title={MedVisVR: Medical Visualization in Virtual Reality for Brain Tumor Segmentation},
  author={[Your Team Names]},
  year={2025},
  publisher={GitHub},
  url={https://github.com/your-org/ceng-407-408-2025-2026-MedVisVR}
}
```

---

## Contact

For questions, issues, or collaboration:

- **Project Repository**: [GitHub Issues](https://github.com/your-org/ceng-407-408-2025-2026-MedVisVR/issues)
- **Email**: [your-email@example.com]
- **Documentation**: [Project Wiki](https://github.com/your-org/ceng-407-408-2025-2026-MedVisVR/wiki)

---

<p align="center">
  <img src="https://github.com/user-attachments/assets/fa5e2691-f3a3-4b0c-837a-81f2d7f2283c" alt="MedVisVR Interface"/>
</p>

<p align="center">
  Made with ❤️ for advancing medical visualization technology
</p>
