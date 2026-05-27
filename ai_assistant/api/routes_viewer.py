from __future__ import annotations

import base64
import json as _json
import os
import shutil
import struct
import threading
import time
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Literal

import numpy as np
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response

from ai_assistant.core.paths import workspace_root, default_segmentation_root

router = APIRouter()

_VOL_CACHE: dict[str, tuple[np.ndarray, float]] = {}   
_VOL_CACHE_MAX   = 2          
_VOL_CACHE_LOCK  = threading.Lock()


def _load_volume_cached(path: Path) -> np.ndarray:
    key = str(path)

    with _VOL_CACHE_LOCK:
        if key in _VOL_CACHE:
            vol, _ = _VOL_CACHE[key]
            _VOL_CACHE[key] = (vol, time.monotonic())   
            return vol

    vol = _load_volume(path)

    with _VOL_CACHE_LOCK:
        if key not in _VOL_CACHE:                       
            if len(_VOL_CACHE) >= _VOL_CACHE_MAX:
                oldest = min(_VOL_CACHE, key=lambda k: _VOL_CACHE[k][1])
                del _VOL_CACHE[oldest]
            _VOL_CACHE[key] = (vol, time.monotonic())

    return vol



_MESH_CACHE: dict[str, tuple[dict, float]] = {} 
_MESH_CACHE_MAX  = 8
_MESH_CACHE_LOCK = threading.Lock()


def _mesh_key(organ: str, case_id: str, step: int) -> str:
    return f"{organ}:{case_id}:s{step}"


def _mesh_disk_path(organ: str, case_id: str, step: int) -> Path:
    return _ws() / f"{organ}_data" / case_id / f"_mesh_s{step}.json"


def _mesh_cache_get(organ: str, case_id: str, step: int) -> dict | None:
    key = _mesh_key(organ, case_id, step)

    with _MESH_CACHE_LOCK:
        if key in _MESH_CACHE:
            data, _ = _MESH_CACHE[key]
            _MESH_CACHE[key] = (data, time.monotonic())
            return data

    dp = _mesh_disk_path(organ, case_id, step)
    if dp.exists():
        try:
            data = _json.loads(dp.read_text(encoding="utf-8"))
            _mesh_cache_put(key, data)
            return data
        except Exception:
            pass  

    return None


def _mesh_cache_put(key: str, data: dict) -> None:
    with _MESH_CACHE_LOCK:
        if key not in _MESH_CACHE and len(_MESH_CACHE) >= _MESH_CACHE_MAX:
            oldest = min(_MESH_CACHE, key=lambda k: _MESH_CACHE[k][1])
            del _MESH_CACHE[oldest]
        _MESH_CACHE[key] = (data, time.monotonic())


def _mesh_cache_save(organ: str, case_id: str, step: int, data: dict) -> None:
    def _write():
        try:
            dp = _mesh_disk_path(organ, case_id, step)
            dp.write_text(_json.dumps(data), encoding="utf-8")
        except Exception:
            pass
    threading.Thread(target=_write, daemon=True).start()


def _ws() -> Path:
    return Path(os.environ.get("AI_WORKSPACE_ROOT", os.getcwd()))



def _load_volume(path: Path) -> np.ndarray:
    try:
        import nibabel as nib
    except ImportError:
        raise HTTPException(500, "nibabel is not installed")
    if not path.exists():
        raise HTTPException(404, f"NIfTI file not found: {path.name}")
    try:
        img = nib.load(str(path))
        return img.get_fdata(dtype=np.float32)
    except Exception as exc:
        raise HTTPException(500, f"Failed to load {path.name}: {exc}") from exc


def _extract_slice(vol: np.ndarray, plane: str, idx: int) -> np.ndarray:
    if plane == "axial":
        return np.rot90(vol[:, :, idx], k=1)
    elif plane == "coronal":
        return np.rot90(vol[:, idx, :], k=1)
    elif plane == "sagittal":
        return np.rot90(vol[idx, :, :], k=1)
    raise HTTPException(400, f"Unknown plane '{plane}'.")


def _normalise(arr: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(arr, 1), np.percentile(arr, 99)
    if hi - lo < 1e-6:
        return np.zeros_like(arr, dtype=np.uint8)
    return ((np.clip(arr, lo, hi) - lo) / (hi - lo) * 255).astype(np.uint8)


def _png_chunk(tag: bytes, data: bytes) -> bytes:
    payload = tag + data
    return struct.pack(">I", len(data)) + payload + struct.pack(">I", zlib.crc32(payload) & 0xFFFFFFFF)


def _encode_png_rgba(rgba: np.ndarray) -> bytes:
    h, w = rgba.shape[:2]
    filter_col = np.zeros((h, 1), dtype=np.uint8)
    rows = np.concatenate([filter_col, rgba.reshape(h, w * 4)], axis=1)
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", zlib.compress(rows.tobytes(), level=1))
        + _png_chunk(b"IEND", b"")
    )


def _to_png(grey: np.ndarray, mask: np.ndarray | None,
            overlay: bool, colors: dict) -> bytes:
    h, w = grey.shape
    rgba = np.stack([grey, grey, grey, np.full((h, w), 255, np.uint8)], axis=-1)
    if overlay and mask is not None:
        for label, (r, g, b, a) in colors.items():
            m = mask == label
            rgba[m, 0] = r; rgba[m, 1] = g; rgba[m, 2] = b; rgba[m, 3] = a
    return _encode_png_rgba(rgba)


def _build_mesh(binary: np.ndarray, step: int, smooth: float = 0.0) -> dict | None:
    try:
        from skimage.measure import marching_cubes
    except ImportError:
        raise HTTPException(500, "scikit-image not installed")
    if binary.sum() < 50:
        return None
    try:
        vol = binary.astype(np.float32)
        if smooth > 0:
            try:
                from scipy import ndimage as ndi
                vol = ndi.gaussian_filter(vol, sigma=smooth)
            except Exception:
                pass
        v, f, n, _ = marching_cubes(vol, level=0.5, step_size=step, allow_degenerate=False)
        return {"vertices": v.flatten().tolist(), "faces": f.flatten().tolist(), "normals": n.flatten().tolist()}
    except Exception:
        return None


def _find_mask(mask_dir: Path) -> Path | None:
    if not mask_dir.exists():
        return None
    files = sorted(mask_dir.glob("*.nii.gz"))
    return files[0] if files else None

def _nifti_spacing(path: Path) -> list[float]:
    try:
        import nibabel as nib
        zooms = nib.load(str(path)).header.get_zooms()[:3]
        return [float(v) for v in zooms]
    except Exception:
        return [1.0, 1.0, 1.0]


def _label_encoding(modality: str) -> dict[int, int]:
    if modality == "brain":
        return {1: 85, 2: 170, 4: 255}
    if modality == "liver":
        return {1: 128, 2: 255}
    return {1: 255}


def _label_legend(modality: str) -> list[dict]:
    if modality == "brain":
        return [
            {"label": 1, "value": 85,  "name": "NCR",         "color": "#e05c5c"},
            {"label": 2, "value": 170, "name": "ED",           "color": "#e8a248"},
            {"label": 4, "value": 255, "name": "ET",           "color": "#4d8ef0"},
        ]
    if modality == "liver":
        return [
            {"label": 1, "value": 128, "name": "Liver",        "color": "#50c878"},
            {"label": 2, "value": 255, "name": "Liver Tumor",  "color": "#f06464"},
        ]
    return [{"label": 1, "value": 255, "name": "Lung Tumor",   "color": "#f97316"}]


def _downsample_pair(
    vol: np.ndarray, mask: np.ndarray | None, max_dim: int
) -> tuple[np.ndarray, np.ndarray | None, int]:
    largest = max(vol.shape)
    step = max(1, int(np.ceil(largest / max_dim))) if largest > max_dim else 1
    if step > 1:
        vol  = np.ascontiguousarray(vol[::step, ::step, ::step])
        if mask is not None:
            mask = np.ascontiguousarray(mask[::step, ::step, ::step])
    return vol, mask, step


def _pack_label_volume(
    mask: np.ndarray | None, shape: tuple[int, int, int], modality: str
) -> np.ndarray:
    packed = np.zeros(shape, dtype=np.uint8)
    if mask is None:
        return packed
    mapping = _label_encoding(modality)
    mask_i  = mask.astype(np.int32, copy=False)
    try:
        from scipy import ndimage as ndi
        for label, value in mapping.items():
            smooth = ndi.gaussian_filter((mask_i == label).astype(np.float32), sigma=0.7)
            packed = np.maximum(packed, np.clip(smooth * value, 0, value).astype(np.uint8))
    except Exception:
        for label, value in mapping.items():
            packed[mask_i == label] = value
    return packed


def _vtk_order_bytes(arr: np.ndarray) -> bytes:
    return np.asarray(arr).ravel(order="F").tobytes()


def _volume_payload(
    *,
    case_id: str,
    modality: str,
    volume_path: Path,
    mask_path: Path | None,
    sequence: str = "ct",
    max_dim: int = 160,
) -> dict:
    vol  = _load_volume_cached(volume_path)
    mask = _load_volume_cached(mask_path).astype(np.int32) if mask_path and mask_path.exists() else None
    vol, mask, step = _downsample_pair(vol, mask, max_dim)
    spacing = [v * step for v in _nifti_spacing(volume_path)]

    if modality in ("lung", "liver"):
        scalar      = np.clip(np.rint(vol), -32768, 32767).astype("<i2", copy=False)
        scalar_type = "int16"
        finite      = scalar[np.isfinite(scalar)]
        scalar_range = [int(finite.min()), int(finite.max())] if finite.size else [0, 1]
        scalar_b64  = base64.b64encode(_vtk_order_bytes(scalar)).decode()
        legacy_key  = "ct_b64"
    else:
        positive = vol[vol > 0]
        lo, hi   = np.percentile(positive, [1, 99]) if positive.size else (float(np.min(vol)), float(np.max(vol)))
        if hi - lo < 1e-6:
            hi = lo + 1
        scalar      = np.clip((vol - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
        scalar_type = "uint8"
        scalar_range = [0, 255]
        scalar_b64  = base64.b64encode(_vtk_order_bytes(scalar)).decode()
        legacy_key  = "mri_b64"

    labels    = _pack_label_volume(mask, scalar.shape, modality)
    label_b64 = base64.b64encode(_vtk_order_bytes(labels)).decode()
    dims      = [int(v) for v in scalar.shape]
    return {
        "case_id":      case_id,
        "modality":     modality,
        "sequence":     sequence,
        "dims":         dims,
        "spacing":      spacing,
        "downsample":   step,
        "scalar_type":  scalar_type,
        "scalar_range": scalar_range,
        "scalar_b64":   scalar_b64,
        "label_b64":    label_b64,
        "label_legend": _label_legend(modality),
        "has_mask":     bool(mask_path and mask_path.exists()),
        legacy_key:     scalar_b64,
        "mask_b64":     label_b64,
    }




_BRAIN_COLORS: dict[int, tuple] = {
    1: (224,  92,  92, 210),
    2: (232, 162,  72, 210),
    4: ( 77, 142, 240, 210),
}
_BRAIN_MESH_META = {
    1: {"name": "NCR", "color": "#e05c5c"},
    2: {"name": "ED",  "color": "#e8a248"},
    4: {"name": "ET",  "color": "#4d8ef0"},
}
_BRAIN_SHELL = {"name": "Brain", "color": "#8fd8ff"}
SEQUENCES = ("t1", "t1ce", "t2", "flair")


def _brain_root() -> Path:
    return _ws() / "brain_data"

def _nifti_path(case_id: str, seq: str) -> Path:

    d = _brain_root() / case_id
    exact = d / f"{case_id}_{seq}.nii"
    if exact.exists():
        return exact
    for pat in (f"*_{seq}.nii.gz", f"*_{seq}.nii"):
        matches = sorted(d.glob(pat))
        if matches:
            return matches[0]
    return exact  

def _brain_mask(case_id: str) -> Path:
    return Path(default_segmentation_root()) / f"{case_id}.nii.gz"


def _build_brain_binary(vol: np.ndarray) -> np.ndarray:
    vol = np.asarray(vol, dtype=np.float32)
    pos = vol[vol > 0]
    if pos.size == 0:
        return np.zeros_like(vol, dtype=np.float32)
    try:
        from scipy import ndimage as ndi
        blurred = ndi.gaussian_filter(vol, sigma=1.15)
        thr = max(float(np.percentile(blurred[blurred > 0], 12)) * 0.42, 1e-3)
        binary = blurred > thr
        labeled, n = ndi.label(binary)
        if n > 1:
            counts = np.bincount(labeled.ravel()); counts[0] = 0
            binary = labeled == int(counts.argmax())
        binary = ndi.binary_fill_holes(binary)
        binary = ndi.binary_closing(binary, structure=np.ones((3,3,3),bool), iterations=1)
        binary = ndi.binary_opening(binary, structure=np.ones((2,2,2),bool), iterations=1)
        return binary.astype(np.float32)
    except ImportError:
        thr = max(float(np.percentile(pos, 8)) * 0.4, 1e-3)
        return (vol > thr).astype(np.float32)


@router.get("/{case_id}/meta")
def brain_meta(case_id: str):
    vol = _load_volume(_nifti_path(case_id, "t1"))
    x, y, z = vol.shape
    seqs = [s for s in SEQUENCES if _nifti_path(case_id, s).exists()]
    return {
        "case_id": case_id, "shape": {"x": x, "y": y, "z": z},
        "default_indices": {"axial": z//2, "coronal": y//2, "sagittal": x//2},
        "available_sequences": seqs, "has_mask": _brain_mask(case_id).exists(),
        "modality": "brain",
    }


@router.get("/{case_id}/slice")
def brain_slice(
    case_id: str,
    plane: Literal["axial","coronal","sagittal"] = Query("axial"),
    index: int = Query(77), sequence: str = Query("t1ce"), overlay: bool = Query(True),
):
    if sequence not in SEQUENCES:
        raise HTTPException(400, f"sequence must be one of {SEQUENCES}")
    vol = _load_volume_cached(_nifti_path(case_id, sequence))
    x, y, z = vol.shape
    idx = max(0, min(index, {"axial": z-1, "coronal": y-1, "sagittal": x-1}[plane]))
    grey = _normalise(_extract_slice(vol, plane, idx))
    mask = None
    mp = _brain_mask(case_id)
    if overlay and mp.exists():
        mask = _extract_slice(_load_volume_cached(mp), plane, idx).astype(np.int32)
    return Response(content=_to_png(grey, mask, overlay, _BRAIN_COLORS), media_type="image/png")


@router.get("/{case_id}/mesh")
def brain_mesh(case_id: str, step_size: int = Query(2, ge=1, le=4)):
    cached = _mesh_cache_get("brain", case_id, step_size)
    if cached is not None:
        return cached

    mp = _brain_mask(case_id)
    if not mp.exists():
        raise HTTPException(404, "Segmentation mask not found")

    mask = _load_volume_cached(mp).astype(np.int32)
    try:
        ref = _load_volume_cached(_nifti_path(case_id, "t1ce"))
    except HTTPException:
        ref = _load_volume_cached(_nifti_path(case_id, "t1"))

    jobs: list[tuple[str, dict, np.ndarray, int]] = []
    jobs.append(("brain", _BRAIN_SHELL, _build_brain_binary(ref).astype(np.float32), 2))
    for label, meta in _BRAIN_MESH_META.items():
        jobs.append((str(label), meta, (mask == label).astype(np.float32), step_size))

    result: dict = {}
    with ThreadPoolExecutor(max_workers=min(len(jobs), 4)) as ex:
        futs = {ex.submit(_build_mesh, arr, step): (key, meta)
                for key, meta, arr, step in jobs}
        for fut in as_completed(futs):
            key, meta = futs[fut]
            m = fut.result()
            if m:
                result[key] = {**meta, **m}

    if not result:
        raise HTTPException(404, "No tumour found in mask")

    _mesh_cache_put(_mesh_key("brain", case_id, step_size), result)
    _mesh_cache_save("brain", case_id, step_size, result)
    return result


@router.get("/{case_id}/volume")
def brain_volume(case_id: str, sequence: str = Query("t1ce"), max_dim: int = Query(160, ge=32, le=256)):
    seq = sequence if sequence in SEQUENCES else "t1ce"
    mri_path = _nifti_path(case_id, seq)
    if not mri_path.exists():
        mri_path = _nifti_path(case_id, "t1")
    mp = _brain_mask(case_id)
    return _volume_payload(
        case_id=case_id,
        modality="brain",
        volume_path=mri_path,
        mask_path=mp if mp.exists() else None,
        sequence=seq,
        max_dim=max_dim,
    )



_LUNG_COLORS = {1: (249, 115, 22, 215)}
_LUNG_MESH   = {1: {"name": "Lung Tumor", "color": "#f97316"}}
_LUNG_SHELL  = {"name": "Lung", "color": "#a8d8ea"}

_LUNG_NNUNET_ROOTS = [
    Path(r"C:\data\nnunet\output_lung"),
    Path(r"C:\data\nnunet\nnUNet_results\Dataset601_LungTumor"),
    Path(r"C:\nnunet_output\lung"),
]


def _lung_ct(case_id: str) -> Path:
    d = _ws() / "lung_data" / case_id
    if not d.exists():
        raise HTTPException(404, f"Lung case not found: lung_data/{case_id}")
    for pat in ("*.nii.gz", "*.nii"):
        f = sorted(d.glob(pat))
        if f: return f[0]
    raise HTTPException(404, f"No NIfTI in lung_data/{case_id}")


def _lung_mask(case_id: str) -> Path | None:
    ws = _ws()
    primary = ws / "outputs" / "segmentations" / "lung" / case_id
    m = _find_mask(primary)
    if m:
        return m
    for root in _LUNG_NNUNET_ROOTS:
        if not root.exists():
            continue
        for pat in (f"{case_id}.nii.gz", f"{case_id}_*.nii.gz"):
            matches = sorted(root.glob(pat))
            if matches:
                return matches[0]
    return None


def _lung_binary(vol: np.ndarray) -> np.ndarray:
    try:
        from scipy import ndimage as ndi
        binary = (vol > -950) & (vol < -300)
        labeled, n = ndi.label(binary)
        if n > 1:
            counts = np.bincount(labeled.ravel()); counts[0] = 0
            top2 = np.argsort(counts)[-2:]
            binary = np.isin(labeled, top2)
        binary = ndi.binary_fill_holes(binary)
        return binary.astype(np.float32)
    except Exception:
        return np.zeros_like(vol, dtype=np.float32)


@router.get("/lung/{case_id}/meta")
def lung_meta(case_id: str):
    ct = _lung_ct(case_id)
    vol = _load_volume(ct)
    x, y, z = vol.shape
    return {
        "case_id": case_id, "shape": {"x": x, "y": y, "z": z},
        "default_indices": {"axial": z//2, "coronal": y//2, "sagittal": x//2},
        "available_sequences": ["ct"], "has_mask": _lung_mask(case_id) is not None,
        "modality": "lung",
        "label_legend": [{"label": 1, "name": "Lung Tumor", "color": "rgb(249,115,22)"}],
    }


@router.get("/lung/{case_id}/slice")
def lung_slice(
    case_id: str,
    plane: Literal["axial","coronal","sagittal"] = Query("axial"),
    index: int = Query(0), overlay: bool = Query(True),
):
    vol = _load_volume_cached(_lung_ct(case_id))
    x, y, z = vol.shape
    idx = max(0, min(index, {"axial": z-1, "coronal": y-1, "sagittal": x-1}[plane]))
    raw = _extract_slice(vol, plane, idx)
    grey = np.clip((raw + 1350) / 1500 * 255, 0, 255).astype(np.uint8)
    mask = None
    mp = _lung_mask(case_id)
    if overlay and mp:
        mv = _load_volume_cached(mp)
        mask = _extract_slice(mv, plane, idx).astype(np.int32)
    return Response(content=_to_png(grey, mask, overlay, _LUNG_COLORS), media_type="image/png")


@router.get("/lung/{case_id}/slices")
def lung_slices_batch(
    case_id: str,
    indices: str = Query(""),           
    plane: Literal["axial","coronal","sagittal"] = Query("axial"),
    overlay: bool = Query(True),
):

    idx_list = [int(x.strip()) for x in indices.split(",") if x.strip().lstrip("-").isdigit()]
    idx_list = idx_list[:50]            
    if not idx_list:
        return {}

    vol = _load_volume_cached(_lung_ct(case_id))
    xd, yd, zd = vol.shape
    dim_max = {"axial": zd - 1, "coronal": yd - 1, "sagittal": xd - 1}

    mp = _lung_mask(case_id)
    mask_vol = _load_volume_cached(mp) if overlay and mp else None

    result: dict[str, str] = {}
    for raw_idx in idx_list:
        idx = max(0, min(raw_idx, dim_max[plane]))
        raw  = _extract_slice(vol, plane, idx)
        grey = np.clip((raw + 1350) / 1500 * 255, 0, 255).astype(np.uint8)
        mslice = _extract_slice(mask_vol, plane, idx).astype(np.int32) if mask_vol is not None else None
        png = _to_png(grey, mslice, overlay, _LUNG_COLORS)
        result[str(raw_idx)] = base64.b64encode(png).decode()

    return result


@router.get("/lung/{case_id}/mesh")
def lung_mesh(case_id: str, step_size: int = Query(2, ge=1, le=4)):
    cached = _mesh_cache_get("lung", case_id, step_size)
    if cached is not None:
        return cached

    mp = _lung_mask(case_id)
    if not mp:
        raise HTTPException(404, "Lung segmentation mask not found. Run segmentation first.")

    mask = _load_volume_cached(mp).astype(np.int32)

    jobs: list[tuple[str, dict, np.ndarray, int]] = []
    try:
        ct = _load_volume_cached(_lung_ct(case_id))
        jobs.append(("lung", _LUNG_SHELL, _lung_binary(ct).astype(np.float32), 2))
    except Exception:
        pass
    for label, meta in _LUNG_MESH.items():
        jobs.append((str(label), meta, (mask == label).astype(np.float32), step_size))

    result: dict = {}
    with ThreadPoolExecutor(max_workers=min(len(jobs), 4)) as ex:
        futs = {ex.submit(_build_mesh, arr, step): (key, meta)
                for key, meta, arr, step in jobs}
        for fut in as_completed(futs):
            key, meta = futs[fut]
            m = fut.result()
            if m:
                result[key] = {**meta, **m}

    if not result:
        raise HTTPException(404, "No tumour found in lung mask")

    _mesh_cache_put(_mesh_key("lung", case_id, step_size), result)
    _mesh_cache_save("lung", case_id, step_size, result)
    return result


@router.get("/lung/{case_id}/volume")
def lung_volume(case_id: str, max_dim: int = Query(160, ge=48, le=256)):
    mp = _lung_mask(case_id)
    return _volume_payload(
        case_id=case_id,
        modality="lung",
        volume_path=_lung_ct(case_id),
        mask_path=mp,
        sequence="ct",
        max_dim=max_dim,
    )


@router.post("/lung/{case_id}/import-mask")
def lung_import_mask(case_id: str):

    ws = _ws()
    dest_dir  = ws / "outputs" / "segmentations" / "lung" / case_id
    dest_file = dest_dir / f"{case_id}.nii.gz"

    if dest_file.exists():
        return {"status": "ok", "message": "Mask already present", "path": str(dest_file)}

    found: Path | None = None
    for root in _LUNG_NNUNET_ROOTS:
        if not root.exists():
            continue
        for pat in (f"{case_id}.nii.gz", f"{case_id}_*.nii.gz", "*.nii.gz"):
            matches = sorted(root.glob(pat))
            if matches:
                found = matches[0]
                break
        if found:
            break

    if not found:
        raise HTTPException(
            404,
            f"No nnUNet mask found for '{case_id}'. "
            f"Searched: {[str(r) for r in _LUNG_NNUNET_ROOTS]}. "
            f"Place the mask at: {dest_file}"
        )

    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(found), str(dest_file))
    return {"status": "ok", "message": f"Imported from {found.name}", "path": str(dest_file)}



_LIVER_COLORS = {
    1: ( 80, 200, 120, 140),
    2: (240, 100, 100, 220),
}
_LIVER_MESH = {
    1: {"name": "Liver",       "color": "#7a2e1c"},   
    2: {"name": "Liver Tumor", "color": "#f06464"},   
}
_LIVER_KEY_MAP = {1: "liver", 2: "2"}

_LIVER_NNUNET_ROOTS = [
    Path(r"C:\data\nnunet\output_liver"),
    Path(r"C:\data\nnunet\results_v1"),
    Path(r"C:\nnunet_output\liver"),
]


def _liver_ct(case_id: str) -> Path:
    d = _ws() / "liver_data" / case_id
    if not d.exists():
        raise HTTPException(404, f"Liver case not found: liver_data/{case_id}")
    for pat in ("*.nii.gz", "*.nii"):
        f = sorted(d.glob(pat))
        if f: return f[0]
    raise HTTPException(404, f"No NIfTI in liver_data/{case_id}")


def _liver_mask(case_id: str) -> Path | None:
    ws = _ws()
    primary = ws / "outputs" / "segmentations" / "liver" / case_id
    m = _find_mask(primary)
    if m:
        return m
    for root in _LIVER_NNUNET_ROOTS:
        if not root.exists():
            continue
        for pat in (f"{case_id}.nii.gz", f"{case_id}_*.nii.gz"):
            matches = sorted(root.glob(pat))
            if matches:
                return matches[0]
    return None


@router.get("/liver/{case_id}/meta")
def liver_meta(case_id: str):
    ct = _liver_ct(case_id)
    vol = _load_volume(ct)
    x, y, z = vol.shape
    return {
        "case_id": case_id, "shape": {"x": x, "y": y, "z": z},
        "default_indices": {"axial": z//2, "coronal": y//2, "sagittal": x//2},
        "available_sequences": ["ct"], "has_mask": _liver_mask(case_id) is not None,
        "modality": "liver",
        "label_legend": [
            {"label": 1, "name": "Liver",       "color": "rgb(80,200,120)"},
            {"label": 2, "name": "Liver Tumor",  "color": "rgb(240,100,100)"},
        ],
    }


@router.get("/liver/{case_id}/slice")
def liver_slice(
    case_id: str,
    plane: Literal["axial","coronal","sagittal"] = Query("axial"),
    index: int = Query(0), overlay: bool = Query(True),
):
    vol = _load_volume_cached(_liver_ct(case_id))
    x, y, z = vol.shape
    idx = max(0, min(index, {"axial": z-1, "coronal": y-1, "sagittal": x-1}[plane]))
    raw = _extract_slice(vol, plane, idx)
    grey = np.clip((raw + 140) / 400 * 255, 0, 255).astype(np.uint8)
    mask = None
    mp = _liver_mask(case_id)
    if overlay and mp:
        mv = _load_volume_cached(mp)
        mask = _extract_slice(mv, plane, idx).astype(np.int32)
    return Response(content=_to_png(grey, mask, overlay, _LIVER_COLORS), media_type="image/png")


@router.get("/liver/{case_id}/slices")
def liver_slices_batch(
    case_id: str,
    indices: str = Query(""),
    plane: Literal["axial","coronal","sagittal"] = Query("axial"),
    overlay: bool = Query(True),
):

    idx_list = [int(x.strip()) for x in indices.split(",") if x.strip().lstrip("-").isdigit()]
    idx_list = idx_list[:50]
    if not idx_list:
        return {}

    vol = _load_volume_cached(_liver_ct(case_id))
    xd, yd, zd = vol.shape
    dim_max = {"axial": zd - 1, "coronal": yd - 1, "sagittal": xd - 1}

    mp = _liver_mask(case_id)
    mask_vol = _load_volume_cached(mp) if overlay and mp else None

    result: dict[str, str] = {}
    for raw_idx in idx_list:
        idx = max(0, min(raw_idx, dim_max[plane]))
        raw  = _extract_slice(vol, plane, idx)
        grey = np.clip((raw + 140) / 400 * 255, 0, 255).astype(np.uint8)
        mslice = _extract_slice(mask_vol, plane, idx).astype(np.int32) if mask_vol is not None else None
        png = _to_png(grey, mslice, overlay, _LIVER_COLORS)
        result[str(raw_idx)] = base64.b64encode(png).decode()

    return result


@router.get("/liver/{case_id}/mesh")
def liver_mesh(case_id: str, step_size: int = Query(2, ge=1, le=4)):
    cached = _mesh_cache_get("liver", case_id, step_size)
    if cached is not None:
        return cached

    mp = _liver_mask(case_id)
    if not mp:
        raise HTTPException(404, "Liver segmentation mask not found. Run segmentation first.")

    mask = _load_volume_cached(mp).astype(np.int32)

    jobs = [(_LIVER_KEY_MAP.get(lbl, str(lbl)), meta, (mask == lbl).astype(np.float32), step_size, 1.8)
            for lbl, meta in _LIVER_MESH.items()]

    result: dict = {}
    with ThreadPoolExecutor(max_workers=min(len(jobs), 4)) as ex:
        futs = {ex.submit(_build_mesh, arr, step, smooth): (key, meta)
                for key, meta, arr, step, smooth in jobs}
        for fut in as_completed(futs):
            key, meta = futs[fut]
            m = fut.result()
            if m:
                result[key] = {**meta, **m}

    if not result:
        raise HTTPException(404, "No components found in liver mask")

    _mesh_cache_put(_mesh_key("liver", case_id, step_size), result)
    _mesh_cache_save("liver", case_id, step_size, result)
    return result


@router.get("/liver/{case_id}/volume")
def liver_volume(case_id: str, max_dim: int = Query(160, ge=48, le=256)):
    mp = _liver_mask(case_id)
    return _volume_payload(
        case_id=case_id,
        modality="liver",
        volume_path=_liver_ct(case_id),
        mask_path=mp,
        sequence="ct",
        max_dim=max_dim,
    )


@router.post("/liver/{case_id}/import-mask")
def liver_import_mask(case_id: str):

    ws = _ws()
    dest_dir  = ws / "outputs" / "segmentations" / "liver" / case_id
    dest_file = dest_dir / f"{case_id}.nii.gz"

    if dest_file.exists():
        return {"status": "ok", "message": "Mask already present", "path": str(dest_file)}

    found: Path | None = None
    for root in _LIVER_NNUNET_ROOTS:
        if not root.exists():
            continue
        for pat in (f"{case_id}.nii.gz", f"{case_id}_*.nii.gz", "*.nii.gz"):
            matches = sorted(root.glob(pat))
            if matches:
                found = matches[0]
                break
        if found:
            break

    if not found:
        raise HTTPException(
            404,
            f"No nnUNet mask found for '{case_id}'. "
            f"Searched: {[str(r) for r in _LIVER_NNUNET_ROOTS]}. "
            f"Place the mask at: {dest_file}"
        )

    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(found), str(dest_file))
    return {"status": "ok", "message": f"Imported from {found.name}", "path": str(dest_file)}
