from __future__ import annotations
import hashlib
import json
import logging
from pathlib import Path

LOGGER = logging.getLogger(__name__)

_EXPECTED_MODELS = {
    "nnunet_brain": {"description": "Brain tumor segmentation (BraTS, nnUNet v2)", "required": False},
    "nnunet_liver": {"description": "Liver tumor segmentation (Task003, nnUNet v1)", "required": False},
    "nnunet_lung":  {"description": "Lung tumor segmentation (nnUNet v2)", "required": False},
}

def check_models(workspace_root: Path) -> dict:
    results = {}
    for model_dir, info in _EXPECTED_MODELS.items():
        model_path = workspace_root / model_dir
        status = {"present": False, "description": info["description"], "required": info["required"], "files": 0, "warnings": []}
        if model_path.exists() and model_path.is_dir():
            files = list(model_path.rglob("*.pth")) + list(model_path.rglob("*.model")) + list(model_path.rglob("*.pkl"))
            status["present"] = True
            status["files"] = len(files)
            if len(files) == 0:
                status["warnings"].append("Directory exists but no .pth/.model/.pkl files found")
            plans = list(model_path.rglob("plans.json")) + list(model_path.rglob("*.json"))
            status["has_metadata"] = len(plans) > 0
        else:
            status["warnings"].append(f"Model directory not found: {model_dir}/")
        results[model_dir] = status
    return results

def log_model_status(workspace_root: Path) -> None:
    results = check_models(workspace_root)
    for name, s in results.items():
        if s["present"]:
            LOGGER.info("[model_checker] ✓ %s — %d weight files", name, s["files"])
            for w in s["warnings"]:
                LOGGER.warning("[model_checker] ⚠ %s: %s", name, w)
        else:
            level = logging.WARNING if s["required"] else logging.INFO
            LOGGER.log(level, "[model_checker] ✗ %s not found — %s", name, s["description"])
