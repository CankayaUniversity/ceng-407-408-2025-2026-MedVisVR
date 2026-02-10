# MedVisVR - Project Documentation

<p align="center">
  <strong>Comprehensive Documentation for Medical Visualization in Virtual Reality</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Status-Active-success" alt="Status"/>
  <img src="https://img.shields.io/badge/Academic-CENG%20407%2F408-blue" alt="Course"/>
  <img src="https://img.shields.io/badge/Year-2025--2026-orange" alt="Year"/>
</p>

---

## Repository Overview

This repository contains all academic and technical documentation for the **MedVisVR** project, a senior design project focused on developing a Virtual Reality platform for medical image visualization and brain tumor segmentation analysis.

---

## Table of Contents

1. [Project Reports](#project-reports)
2. [Dataset Information](#dataset-information)
3. [Methodology Documentation](#methodology-documentation)
4. [Source Code](#source-code)
5. [Visual Assets](#visual-assets)
6. [How to Use](#how-to-use)

---

## Project Reports

### 📄 Literature Review
**File**: `Ceng407_Literature_Review.pdf`

Comprehensive analysis of existing research in:
- Medical image visualization techniques
- Virtual Reality applications in healthcare
- Brain tumor segmentation methods
- Deep learning approaches for medical imaging
- State-of-the-art VR platforms and frameworks

**Key Topics Covered**:
- Current challenges in medical visualization
- VR technology in clinical settings
- Machine learning for tumor segmentation
- Comparative analysis of existing solutions

---

### 📄 Methodology Report
**File**: `Ceng407_Methodology_Report.pdf`

Detailed technical approach and system architecture:
- Project objectives and scope
- System design and architecture
- Technology stack selection
- Development methodology
- Implementation timeline
- Risk analysis and mitigation strategies

**Includes**:
- System architecture diagrams
- Data flow illustrations
- Component interaction models
- Development milestones

**Visual References**:
- `methodology_photo1.png` - `methodology_photo9.png`: Detailed methodology diagrams

---

### 📄 Dataset Description & Preprocessing
**File**: `Dataset_Description_&_Preprocessing_MedVisVR.pdf`

Comprehensive guide to data handling:
- BraTS2020 dataset overview
- Data acquisition protocols
- Preprocessing pipeline details
- Augmentation techniques
- Quality assurance procedures

**Preprocessing Steps**:
1. Data loading and validation
2. Orientation standardization (RAS)
3. Spatial resampling (1mm³ isotropic)
4. Intensity normalization
5. Multi-channel conversion
6. Data augmentation
7. Patch extraction (128³)

**Dataset Visualizations**:
- `dataset_1.jpg`: Multi-modal MRI sequences
- `dataset_2.jpg`: Segmentation labels
- `dataset_3.jpg`: 3D reconstruction examples

---

### 📄 Final Report
**File**: `Final_report.pdf`

Complete project documentation including:
- Executive summary
- Literature review synthesis
- Detailed methodology
- Implementation details
- Results and evaluation
- User testing outcomes
- Future work and recommendations
- Conclusions

---

## Dataset Information

### BraTS2020 Dataset

The project utilizes the **Brain Tumor Segmentation Challenge 2020** dataset:

**Dataset Characteristics**:
- **Size**: 369 training cases
- **Modalities**: 4 (T1, T1ce, T2, FLAIR)
- **Resolution**: 240×240×155 voxels
- **Annotations**: Expert manual segmentations
- **Format**: NIfTI (.nii files)

**Tumor Regions**:
1. **Necrotic and Non-Enhancing Tumor Core** (NCR/NET) - Label 1
2. **Peritumoral Edema** (ED) - Label 2
3. **GD-enhancing Tumor** (ET) - Label 4

**Data Organization**:
```
BraTS20_Training_XXX/
├── BraTS20_Training_XXX_t1.nii      # T1-weighted
├── BraTS20_Training_XXX_t1ce.nii    # T1-contrast enhanced
├── BraTS20_Training_XXX_t2.nii      # T2-weighted
├── BraTS20_Training_XXX_flair.nii   # FLAIR sequence
└── BraTS20_Training_XXX_seg.nii     # Segmentation mask
```

---

## Methodology Documentation

### System Architecture

The MedVisVR system consists of three main components:

1. **Data Processing Module**
   - Automated preprocessing pipeline
   - Data validation and quality control
   - Format conversion utilities

2. **Visualization Engine**
   - VR rendering system
   - Interactive 3D volume rendering
   - Multi-modal data fusion

3. **User Interface**
   - VR interaction controls
   - Menu navigation system
   - Settings and preferences

### Visual Documentation

**Methodology Diagrams** (9 detailed illustrations):
- `methodology_photo1.png`: Overall system architecture
- `methodology_photo2.png`: Data preprocessing workflow
- `methodology_photo3.png`: VR rendering pipeline
- `methodology_photo4.png`: User interaction flow
- `methodology_photo5.png`: Segmentation integration
- `methodology_photo6.png`: Performance optimization
- `methodology_photo7.png`: Quality assurance process
- `methodology_photo8.png`: Testing framework
- `methodology_photo9.png`: Deployment architecture

---

## Visual Assets

### Dataset Visualization Examples

1. **dataset_1.jpg**: Multi-modal MRI display
   - Shows all 4 MRI sequences (T1, T1ce, T2, FLAIR)
   - Demonstrates different tissue contrasts
   - Highlights tumor visibility per modality

2. **dataset_2.jpg**: Segmentation masks
   - Color-coded tumor regions
   - Label overlay on anatomical scans
   - Ground truth annotations

3. **dataset_3.jpg**: 3D reconstructions
   - Volumetric rendering examples
   - Multi-planar views
   - Tumor region visualization

### Methodology Visualizations

Nine comprehensive diagrams illustrating:
- System component interactions
- Data flow through pipeline
- VR rendering techniques
- User experience design
- Technical architecture

---

## How to Use

### Accessing Documentation

1. **Clone this repository**:
   ```bash
   git clone https://github.com/your-org/ceng-407-408-2025-2026-MedVisVR-Documents.git
   cd ceng-407-408-2025-2026-MedVisVR-Documents
   ```

2. **View PDF reports**: All reports are in PDF format and can be opened with any PDF reader

3. **Review visual materials**: Images are in JPG/PNG format for easy viewing

4. **Execute preprocessing**: Python script is ready to run with BraTS2020 data

### Document Navigation

- **For Project Overview**: Start with `Final_report.pdf`
- **For Technical Details**: Read `Methodology_Report.pdf`
- **For Data Processing**: See `Dataset_Description_&_Preprocessing_MedVisVR.pdf`
- **For Research Context**: Review `Ceng407_Literature_Review.pdf`

---

## Project Timeline

| Phase | Deliverable | Status |
|-------|-------------|--------|
| **Phase 1** | Literature Review | ✅ Complete |
| **Phase 2** | Methodology Report | ✅ Complete |
| **Phase 3** | Dataset Preparation | ✅ Complete |
| **Phase 4** | Implementation | 🔄 In Progress |
| **Phase 5** | Testing & Validation | 📅 Planned |
| **Phase 6** | Final Report | ✅ Complete |

---

## Academic Information

**Course**: CENG 407/408 - Senior Design Project
**Academic Year**: 2025-2026
**Institution**: Çankaya University

### Team Members
| Student Number | Name |
|----------------|------|
| 202111012 | Alperen Berke Çetinkaya |
| 202211052 | Muhammed Yusuf Özcan |
| 202211011 | Sezer Ataş |
| 202211061 | Mete Serpil |

**Advisor**: Assoc. Prof. Dr. Gül Tokdemir
**Department**: Computer Engineering

---

## Additional Resources

### Related Repositories
- **Main Project**: [MedVisVR Implementation](https://github.com/your-org/ceng-407-408-2025-2026-MedVisVR)

### External References
- **BraTS Challenge**: [https://www.med.upenn.edu/cbica/brats2020/](https://www.med.upenn.edu/cbica/brats2020/)
- **MONAI Framework**: [https://monai.io/](https://monai.io/)
- **PyTorch Documentation**: [https://pytorch.org/docs/](https://pytorch.org/docs/)

---

## File Index

### PDF Documentation
| File | Size | Description |
|------|------|-------------|
| `Ceng407_Literature_Review.pdf` | - | Academic literature analysis |
| `Ceng407_Methodology_Report.pdf` | - | Technical methodology |
| `Dataset_Description_&_Preprocessing_MedVisVR.pdf` | - | Data processing guide |
| `Final_report.pdf` | - | Complete project report |

### Source Code
| File | Language | Purpose |
|------|----------|---------|
| `Preprocessing.py` | Python | Data preprocessing pipeline |

### Visual Assets
| Category | Files | Count |
|----------|-------|-------|
| Dataset Examples | `dataset_1.jpg`, `dataset_2.jpg`, `dataset_3.jpg` | 3 |
| Methodology Diagrams | `methodology_photo1-9.png` | 9 |

---

## License

This documentation is part of an academic project and is provided for educational purposes.

**Copyright** © 2025-2026 MedVisVR Team. All rights reserved.

---

<p align="center">
  <em>For the latest version of documentation, please check the main branch</em>
</p>

<p align="center">
  <strong>MedVisVR Project Documentation</strong><br>
  CENG 407/408 Senior Design Project 2025-2026
</p>

