import os
import sys
import shutil
import tempfile
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

WORKSPACE_ROOT = Path(os.environ.get("AI_WORKSPACE_ROOT", os.getcwd()))
LIVER_MODEL_DIR = WORKSPACE_ROOT / "nnunet_liver"

LIVER_RESULTS_STAGING = WORKSPACE_ROOT / "nnunet_liver_staging"



_POOLS_PATCHED = False


def _patch_pools() -> None:
    global _POOLS_PATCHED
    if _POOLS_PATCHED:
        return

    class _AsyncResult:
        def __init__(self, result): self._result = result
        def get(self, timeout=None): return self._result
        def wait(self, timeout=None): pass
        def ready(self): return True
        def successful(self): return True

    class _SeqPool:
        def __init__(self, *a, **kw): pass
        def map(self, fn, it, **kw): return list(map(fn, it))
        def imap(self, fn, it, **kw): return iter(list(map(fn, it)))
        def imap_unordered(self, fn, it, **kw): return iter(list(map(fn, it)))
        def starmap(self, fn, it, **kw): return [fn(*a) for a in it]
        def starmap_async(self, fn, it, **kw): return _AsyncResult([fn(*a) for a in it])
        def map_async(self, fn, it, **kw): return _AsyncResult(list(map(fn, it)))
        def apply(self, fn, args=(), kwds={}): return fn(*args, **kwds)
        def apply_async(self, fn, args=(), kwds={}, **kw): return _AsyncResult(fn(*args, **kwds))
        def close(self): pass
        def join(self): pass
        def terminate(self): pass
        def restart(self, *a, **kw): pass
        def __enter__(self): return self
        def __exit__(self, *a): pass

    for mod_name, attrs in [
        ("pathos.multiprocessing", ["Pool", "ProcessPool", "ThreadPool", "_ProcessPool"]),
        ("multiprocess",           ["Pool"]),
        ("multiprocess.pool",      ["Pool"]),
        ("multiprocessing",        ["Pool"]),
        ("multiprocessing.pool",   ["Pool"]),
    ]:
        try:
            import importlib
            mod = importlib.import_module(mod_name)
            for attr in attrs:
                if hasattr(mod, attr):
                    setattr(mod, attr, _SeqPool)
        except ImportError:
            pass

    _POOLS_PATCHED = True


def _ensure_liver_staging() -> None:

    trainer_link = (
        LIVER_RESULTS_STAGING
        / "nnUNet" / "3d_fullres" / "Task003_Liver"
        / "nnUNetTrainerV2__nnUNetPlansv2.1"
    )
    if trainer_link.exists():
        return

    trainer_link.parent.mkdir(parents=True, exist_ok=True)
    src = LIVER_MODEL_DIR / "nnUNetTrainerV2__nnUNetPlansv2.1"

    try:
        import subprocess as _sp
        _sp.run(
            ["cmd", "/c", f'mklink /J "{trainer_link}" "{src}"'],
            check=True, capture_output=True,
        )
        logger.info(f"Liver staging: junction → {trainer_link}")
        return
    except Exception as e:
        logger.debug(f"Junction failed ({e}), trying symlink...")

    try:
        os.symlink(str(src), str(trainer_link))
        logger.info(f"Liver staging: symlink → {trainer_link}")
        return
    except OSError as e:
        logger.debug(f"Symlink failed ({e}), one-time copytree...")

    shutil.copytree(str(src), str(trainer_link))
    logger.info(f"Liver staging: model copied → {trainer_link}")




def run_liver_segmentation(input_nifti_path: str, output_dir: str) -> dict:
    input_path = Path(input_nifti_path)
    if not input_path.exists():
        return {"success": False, "output_file": None,
                "error": f"Input not found: {input_nifti_path}"}

    if not LIVER_MODEL_DIR.exists():
        return {"success": False, "output_file": None,
                "error": f"Liver model dir not found: {LIVER_MODEL_DIR}"}

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
            with open(input_path, "rb") as f_in, gzip.open(dest, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

        try:
            _ensure_liver_staging()
        except Exception as e:
            return {"success": False, "output_file": None,
                    "error": f"Liver staging failed: {e}"}

        _patch_pools()

        _old_results = os.environ.get("RESULTS_FOLDER")
        os.environ["RESULTS_FOLDER"] = str(LIVER_RESULTS_STAGING)

        _old_argv = sys.argv
        sys.argv = [
            "nnUNet_predict",
            "-i",  str(tmp_input),
            "-o",  str(output_path),
            "-t",  "Task003_Liver",
            "-m",  "3d_fullres",
            "-f",  "0", "1", "2", "3", "4",
            "--num_threads_preprocessing", "1",
            "--num_threads_nifti_save",   "1",
        ]

        try:
            from nnunet.inference.predict_simple import main as _nnunet_main
            _nnunet_main()
        except SystemExit:
            pass
        except Exception as e:
            logger.error(f"Liver segmentation error: {e}", exc_info=True)
            return {"success": False, "output_file": None, "error": str(e)}
        finally:
            sys.argv = _old_argv
            if _old_results is None:
                os.environ.pop("RESULTS_FOLDER", None)
            else:
                os.environ["RESULTS_FOLDER"] = _old_results

    output_files = list(output_path.glob("*.nii.gz"))
    output_file = str(output_files[0]) if output_files else None

    if not output_file:
        return {"success": False, "output_file": None,
                "error": "nnUNet produced no output — check logs for details."}

    return {
        "success": True,
        "output_file": output_file,
        "labels": {"0": "Background", "1": "Liver", "2": "Liver Tumor"},
        "error": None,
    }
