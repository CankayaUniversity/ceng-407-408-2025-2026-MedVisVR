# CARVIS — Clinical Anatomical Rendering Visualization Intelligent System

CARVIS is a desktop application for medical image analysis. It runs segmentation models on brain, liver, and lung MRI/CT scans, generates structured clinical reports, and lets clinicians ask questions about the findings in natural language.

Everything runs locally — no data is sent to external servers.

---

## What it does

- **Segmentation** — Runs nnUNet models to segment tumors in brain MRI (BraTS format), liver CT, and lung CT scans
- **Report generation** — Produces structured radiology-style reports from segmentation results using a local LLM
- **Q&A** — Answers natural language questions about a case (tumor location, volume, affected regions, etc.)
- **3D viewer** — Renders segmentation masks as interactive 3D meshes inside the desktop app

---

## Requirements

- Windows 10 / 11
- Python 3.11+
- Node.js 18+
- NVIDIA GPU recommended (CUDA 12.x) — CPU fallback is available but slow for segmentation

The setup script handles everything else automatically.

---

## Installation

Clone or download this repository, then double-click:

```
Install_New_PC_OneClick.bat
```

The script will:

1. Verify Python 3.11+ and Node.js 18+ (installs them if missing)
2. Create a Python virtual environment and install all dependencies
3. Detect your GPU and install the correct PyTorch (CUDA or CPU)
4. Build the Electron desktop app
5. Download the Qwen2.5-7B GGUF model (~4.5 GB) from HuggingFace
6. Download the nnUNet segmentation models (~2.8 GB) from HuggingFace
7. Run verification tests
8. Create a desktop shortcut

**First-time setup takes 15–40 minutes depending on your internet speed.**

---

## Models

Models are not included in this repository. They are downloaded automatically during setup.

| Model | Source | Size |
|---|---|---|
| Qwen2.5-7B-Instruct (GGUF, q4_k_m) | [Qwen/Qwen2.5-7B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF) | ~4.5 GB |
| nnUNet Brain (BraTS 2020) | [Yusufinhoo/carvis-nnunet-models](https://huggingface.co/Yusufinhoo/carvis-nnunet-models) | ~1.3 GB |
| nnUNet Liver (Task003) | [Yusufinhoo/carvis-nnunet-models](https://huggingface.co/Yusufinhoo/carvis-nnunet-models) | ~1.25 GB |
| nnUNet Lung (Dataset601) | [Yusufinhoo/carvis-nnunet-models](https://huggingface.co/Yusufinhoo/carvis-nnunet-models) | ~0.25 GB |

Models are saved to:
- `models/Qwen2.5-7B-Instruct-GGUF/`
- `nnunet_brain/`, `nnunet_liver/`, `nnunet_lung/`

---

## Usage

After installation, launch the app using the desktop shortcut **CARVIS** or run:

```
Start_Desktop_App_OneClick.bat
```

To stop all background services:

```
Stop_OneClick.bat
```

---

## Project structure

```
ai_assistant/
  api/          FastAPI backend (routes for segmentation, Q&A, reports, viewer)
  core/         Business logic (report generation, QA engine, segmentation metrics)
  segmentation/ nnUNet inference wrappers (brain, liver, lung)
  prompts/      LLM prompt templates
  assets/       Harvard-Oxford brain atlas data

desktop_app/
  electron/     Electron main process
  src/          React frontend (App.jsx, 3D viewer, report display)

tests/          pytest test suite (77 tests)
docs/           Clinical guideline snippets used for Q&A retrieval
```

---

## Running tests

```
.venv\Scripts\python.exe -m pytest tests\test_report_generator.py tests\test_qa_engine.py -v
```

---

## License

This project was developed as part of a graduation project at Çankaya University (CENG 407-408, 2025–2026).

The source code is released for academic use. The nnUNet models are subject to their original training data licenses. The Qwen2.5 model is released under the [Qwen License](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/main/LICENSE).

---

## Notes

- Patient data (`brain_data/`, `liver_data/`, `lung_data/`) is not included in this repository
- Model files are not tracked by git — they are downloaded at setup time
- The app is designed for research and demonstration purposes, not clinical deployment
