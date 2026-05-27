from __future__ import annotations

import argparse
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import re
from typing import Any
import sys

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

import numpy as np

from ai_assistant.core.brats_feature_extractor import _configure_logging
from ai_assistant.core.paths import workspace_root


LOGGER = logging.getLogger(__name__)
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
_INDEX_CACHE: dict[str, Any] = {}
_LOBE_HINTS = ("frontal", "temporal", "parietal", "occipital", "insular", "cerebellar")
_CLINICAL_EXPANSIONS: dict[str, tuple[str, ...]] = {
    "motor cortex": ("precentral", "motor", "eloquent", "m1"),
    "brainstem": ("brainstem", "pontine", "medullary", "midbrain"),
    "midline shift": ("mls", "shift", "displaced", "displacement"),
    "enhancing tumor": ("ce", "contrast enhancing", "enhancing", "t1ce"),
    "edema": ("flair", "peritumoral", "vasogenic", "t2 signal"),
    "ventricle": ("ventricular", "hydrocephalus", "frontal horn"),
    "optic": ("optic tract", "optic radiation", "visual pathway"),
    "cisternal effacement": ("cisternal", "basal cistern", "cisternal effacement", "cisternal compression"),
    "compression": ("compression", "crowding", "mass effect", "effacement"),
}
_ANATOMY_HINTS = {
    "motor_cortex": ("motor cortex", "motor area", "precentral"),
    "brainstem": ("brainstem", "midbrain", "pons", "medulla"),
    "optic_pathway": ("optic", "optic tract", "optic pathway", "optic radiation"),
}
_ANATOMY_DISTANCE_FIELDS = {
    "motor_cortex": ("critical_proximity", "motor_cortex_distance_mm"),
    "brainstem": ("critical_proximity", "brainstem_distance_mm"),
}
_RERANK_WEIGHTS = {
    "dense_similarity": 0.20,
    "bm25_similarity": 0.15,
    "location_similarity": 0.20,
    "measurement_similarity": 0.20,
    "distortion_similarity": 0.10,
    "pathology_match": 0.10,
    "modality_match": 0.05,
}


def _default_case_index_path() -> Path:
    configured = Path(str(Path.cwd()))
    env_path = None
    try:
        import os

        env_path = os.getenv("AI_CASE_INDEX_PATH")
    except Exception:
        env_path = None
    if env_path:
        return Path(env_path)
    return workspace_root() / "output_brain" / "case_index" / "full_index.json"


def _default_retrieval_root() -> Path:
    try:
        import os

        env_root = os.getenv("AI_RETRIEVAL_INDEX_ROOT")
        if env_root:
            return Path(env_root)
    except Exception:
        pass
    return workspace_root() / "output_brain" / "retrieval_index"


def _tokenize(text: str) -> list[str]:
    return [token for token in str(text or "").lower().replace(".", " ").replace(",", " ").split() if token]


def _tokenize_phrase(text: str) -> list[str]:
    clean = (
        str(text or "")
        .lower()
        .replace(".", " ")
        .replace(",", " ")
        .replace(":", " ")
        .replace(";", " ")
        .replace("(", " ")
        .replace(")", " ")
        .replace("/", " ")
    )
    return [token for token in clean.split() if token]


def _expand_text_tokens(text: str) -> list[str]:
    text_lower = str(text or "").lower()
    expanded: list[str] = []
    for phrase, synonyms in _CLINICAL_EXPANSIONS.items():
        if phrase in text_lower or any(alias in text_lower for alias in synonyms):
            expanded.extend(_tokenize_phrase(phrase))
            for synonym in synonyms:
                expanded.extend(_tokenize_phrase(synonym))
    return expanded


def _clinical_bucket_tokens(case: dict[str, Any]) -> list[str]:
    tokens: list[str] = []

    motor_distance = _safe_float(_nested_get(case, "critical_proximity", "motor_cortex_distance_mm"))
    if motor_distance is not None:
        if motor_distance <= 5.0:
            tokens.extend(["near", "motor", "cortex", "eloquent", "highrisk", "precentral"])
        elif motor_distance <= 8.0:
            tokens.extend(["motor", "cortex", "eloquent", "caution"])

    brainstem_distance = _safe_float(_nested_get(case, "critical_proximity", "brainstem_distance_mm"))
    if brainstem_distance is not None:
        if brainstem_distance <= 5.0:
            tokens.extend(["brainstem", "compression", "crowding", "cisternal", "effacement"])
        elif brainstem_distance <= 10.0:
            tokens.extend(["brainstem", "caution", "compression"])

    midline_shift = _safe_float(_nested_get(case, "mass_effect", "midline_shift_mm"))
    if midline_shift is not None:
        if midline_shift >= 7.5:
            tokens.extend(["midline", "shift", "mls", "critical", "displacement"])
        elif midline_shift >= 5.0:
            tokens.extend(["midline", "shift", "mls", "warning"])

    ventricular_compression = _nested_get(case, "mass_effect", "ventricular_compression")
    if isinstance(ventricular_compression, bool) and ventricular_compression:
        tokens.extend(["ventricle", "ventricular", "compression", "frontal", "horn"])

    sulcal_effacement = _nested_get(case, "mass_effect", "sulcal_effacement")
    if isinstance(sulcal_effacement, bool) and sulcal_effacement:
        tokens.extend(["sulcal", "effacement", "mass", "effect"])

    enhancing_volume = _safe_float(_nested_get(case, "tumor_burden", "enhancing_volume_mm3"))
    if enhancing_volume is not None and enhancing_volume >= 15000.0:
        tokens.extend(["enhancing", "tumor", "contrast", "enhancing", "t1ce"])

    edema_volume = _safe_float(_nested_get(case, "tumor_burden", "edema_volume_mm3"))
    if edema_volume is not None and edema_volume >= 20000.0:
        tokens.extend(["edema", "peritumoral", "vasogenic", "flair"])

    return tokens


def _bm25_tokens_for_case(case: dict[str, Any]) -> list[str]:
    narrative = str(case.get("narrative_text", "") or "")
    base_tokens = _tokenize(narrative)
    expanded_tokens = _expand_text_tokens(narrative)
    bucket_tokens = _clinical_bucket_tokens(case)
    return [token for token in (base_tokens + expanded_tokens + bucket_tokens) if token]


def _bm25_query_tokens(query: str) -> list[str]:
    base_tokens = _tokenize(query)
    expanded_tokens = _expand_text_tokens(query)
    return [token for token in (base_tokens + expanded_tokens) if token]


def _normalize_scores(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    min_val = float(values.min())
    max_val = float(values.max())
    if max_val - min_val < 1e-9:
        return np.zeros_like(values, dtype=float)
    return (values - min_val) / (max_val - min_val)


def _nested_get(data: dict[str, Any], *path: str) -> Any:
    value: Any = data
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _series_range(cases: list[dict[str, Any]], path: tuple[str, ...]) -> tuple[float, float]:
    values = [_safe_float(_nested_get(case, *path)) for case in cases]
    valid = [value for value in values if value is not None]
    if not valid:
        return 0.0, 1.0
    return min(valid), max(valid)


def _percentile_target(cases: list[dict[str, Any]], path: tuple[str, ...], percentile: float) -> float | None:
    values = [_safe_float(_nested_get(case, *path)) for case in cases]
    valid = [value for value in values if value is not None]
    if not valid:
        return None
    return float(np.percentile(np.asarray(valid, dtype=float), percentile))


def _inverse_distance_similarity(value: float | None, target: float | None, min_value: float, max_value: float) -> float:
    if value is None or target is None:
        return 0.5
    span = max(max_value - min_value, 1e-6)
    normalized_distance = min(abs(value - target) / span, 1.0)
    return max(0.0, 1.0 - normalized_distance)


def _proximity_similarity(value: float | None, min_value: float, max_value: float) -> float:
    if value is None:
        return 0.5
    span = max(max_value - min_value, 1e-6)
    normalized = min(max((value - min_value) / span, 0.0), 1.0)
    return max(0.0, 1.0 - normalized)


def _mls_bin(value: float | None) -> str:
    if value is None:
        return "unknown"
    if value < 5.0:
        return "none"
    if value < 10.0:
        return "mild"
    if value < 15.0:
        return "moderate"
    return "marked"


def _extract_numeric_target(query_text: str, patterns: tuple[str, ...]) -> float | None:
    for pattern in patterns:
        match = re.search(pattern, query_text)
        if match:
            return float(match.group(1))
    return None


def _extract_query_hints(query: str, cases: list[dict[str, Any]]) -> dict[str, Any]:
    query_text = str(query or "").lower()
    hemisphere = None
    if "left" in query_text:
        hemisphere = "left"
    elif "right" in query_text:
        hemisphere = "right"

    lobe = next((candidate for candidate in _LOBE_HINTS if candidate in query_text), None)

    anatomy_target = None
    for key, aliases in _ANATOMY_HINTS.items():
        if any(alias in query_text for alias in aliases):
            anatomy_target = key
            break

    shift_target_mm = None
    if "midline shift" in query_text or "mls" in query_text or "shift" in query_text:
        shift_target_mm = _extract_numeric_target(
            query_text,
            (
                r"shift\s*(?:>|over|above|at least)?\s*(\d+(?:\.\d+)?)\s*mm",
                r"midline shift\s*(?:>|over|above|at least)?\s*(\d+(?:\.\d+)?)\s*mm",
                r"(\d+(?:\.\d+)?)\s*mm\s*midline shift",
            ),
        )

    diameter_target_mm = _extract_numeric_target(
        query_text,
        (
            r"(?:diameter|size)\s*(?:>|over|above|at least)?\s*(\d+(?:\.\d+)?)\s*mm",
            r"(\d+(?:\.\d+)?)\s*mm\s*(?:diameter|size)",
        ),
    )
    total_volume_target_mm3 = None
    volume_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:cm3|cc)\b", query_text)
    if volume_match and any(token in query_text for token in ("volume", "tumor", "lesion", "mass")):
        total_volume_target_mm3 = float(volume_match.group(1)) * 1000.0

    large_hint = any(token in query_text for token in ("large", "big", "bulky", "massive"))
    small_hint = any(token in query_text for token in ("small", "tiny", "minimal"))

    if large_hint:
        if diameter_target_mm is None:
            diameter_target_mm = _percentile_target(cases, ("tumor_burden", "max_diameter_mm"), 75)
        if total_volume_target_mm3 is None:
            total_volume_target_mm3 = _percentile_target(cases, ("tumor_burden", "total_volume_mm3"), 75)
        if shift_target_mm is None and ("shift" in query_text or "mass effect" in query_text or "compression" in query_text):
            shift_target_mm = _percentile_target(cases, ("mass_effect", "midline_shift_mm"), 75)

    if small_hint:
        if diameter_target_mm is None:
            diameter_target_mm = _percentile_target(cases, ("tumor_burden", "max_diameter_mm"), 25)
        if total_volume_target_mm3 is None:
            total_volume_target_mm3 = _percentile_target(cases, ("tumor_burden", "total_volume_mm3"), 25)
        if shift_target_mm is None and ("shift" in query_text or "mass effect" in query_text or "compression" in query_text):
            shift_target_mm = _percentile_target(cases, ("mass_effect", "midline_shift_mm"), 25)

    structured_hint = any(
        value is not None
        for value in (hemisphere, lobe, anatomy_target, shift_target_mm, diameter_target_mm, total_volume_target_mm3)
    ) or any(token in query_text for token in ("midline shift", "mls", "mass effect", "compression", "shift"))

    return {
        "hemisphere": hemisphere,
        "lobe": lobe,
        "anatomy_target": anatomy_target,
        "shift_target_mm": shift_target_mm,
        "diameter_target_mm": diameter_target_mm,
        "total_volume_target_mm3": total_volume_target_mm3,
        "shift_reference_present": any(token in query_text for token in ("midline shift", "mls", "mass effect", "compression", "shift")),
        "structured_hint": structured_hint,
    }


def _location_similarity(case: dict[str, Any], query_hints: dict[str, Any], cases: list[dict[str, Any]]) -> float:
    components: list[float] = []
    hemisphere_hint = query_hints.get("hemisphere")
    if hemisphere_hint:
        components.append(1.0 if _nested_get(case, "location", "hemisphere") == hemisphere_hint else 0.0)

    lobe_hint = query_hints.get("lobe")
    if lobe_hint:
        case_lobe = str(_nested_get(case, "location", "lobe") or "").lower()
        components.append(1.0 if case_lobe == lobe_hint else 0.0)

    anatomy_target = query_hints.get("anatomy_target")
    if anatomy_target:
        path = _ANATOMY_DISTANCE_FIELDS.get(anatomy_target)
        if path:
            min_value, max_value = _series_range(cases, path)
            components.append(_proximity_similarity(_safe_float(_nested_get(case, *path)), min_value, max_value))

    if not components:
        return 0.5
    return float(sum(components) / len(components))


def _measurement_similarity(case: dict[str, Any], query_hints: dict[str, Any], cases: list[dict[str, Any]]) -> float:
    components: list[float] = []

    shift_target = _safe_float(query_hints.get("shift_target_mm"))
    if shift_target is not None:
        min_value, max_value = _series_range(cases, ("mass_effect", "midline_shift_mm"))
        value = _safe_float(_nested_get(case, "mass_effect", "midline_shift_mm"))
        components.append(_inverse_distance_similarity(value, shift_target, min_value, max_value))

    diameter_target = _safe_float(query_hints.get("diameter_target_mm"))
    if diameter_target is not None:
        min_value, max_value = _series_range(cases, ("tumor_burden", "max_diameter_mm"))
        value = _safe_float(_nested_get(case, "tumor_burden", "max_diameter_mm"))
        components.append(_inverse_distance_similarity(value, diameter_target, min_value, max_value))

    volume_target = _safe_float(query_hints.get("total_volume_target_mm3"))
    if volume_target is not None:
        min_value, max_value = _series_range(cases, ("tumor_burden", "total_volume_mm3"))
        value = _safe_float(_nested_get(case, "tumor_burden", "total_volume_mm3"))
        components.append(_inverse_distance_similarity(value, volume_target, min_value, max_value))

    if not components:
        return 0.5
    return float(sum(components) / len(components))


def _distortion_similarity(case: dict[str, Any], query_hints: dict[str, Any]) -> float:
    if not query_hints.get("shift_reference_present"):
        return 0.5
    candidate_shift = _safe_float(_nested_get(case, "mass_effect", "midline_shift_mm"))
    candidate_bin = _mls_bin(candidate_shift)
    shift_target = _safe_float(query_hints.get("shift_target_mm"))
    if shift_target is None:
        return 1.0 if candidate_bin != "none" else 0.0
    return 1.0 if candidate_bin == _mls_bin(shift_target) else 0.0


def _structured_rerank_scores(
    cases: list[dict[str, Any]],
    bm25_norm: np.ndarray,
    dense_norm: np.ndarray,
    query_hints: dict[str, Any],
) -> dict[str, np.ndarray]:
    location_scores = np.asarray([_location_similarity(case, query_hints, cases) for case in cases], dtype=float)
    measurement_scores = np.asarray([_measurement_similarity(case, query_hints, cases) for case in cases], dtype=float)
    distortion_scores = np.asarray([_distortion_similarity(case, query_hints) for case in cases], dtype=float)
    pathology_scores = np.ones(len(cases), dtype=float)
    modality_scores = np.ones(len(cases), dtype=float)

    final_scores = (
        _RERANK_WEIGHTS["dense_similarity"] * dense_norm
        + _RERANK_WEIGHTS["bm25_similarity"] * bm25_norm
        + _RERANK_WEIGHTS["location_similarity"] * location_scores
        + _RERANK_WEIGHTS["measurement_similarity"] * measurement_scores
        + _RERANK_WEIGHTS["distortion_similarity"] * distortion_scores
        + _RERANK_WEIGHTS["pathology_match"] * pathology_scores
        + _RERANK_WEIGHTS["modality_match"] * modality_scores
    )

    return {
        "final_scores": final_scores,
        "location_scores": location_scores,
        "measurement_scores": measurement_scores,
        "distortion_scores": distortion_scores,
        "pathology_scores": pathology_scores,
        "modality_scores": modality_scores,
    }


def _load_rank_bm25():
    try:
        from rank_bm25 import BM25Okapi
    except ImportError as exc:
        raise RuntimeError("rank_bm25 is required for retrieval indexing. Install it from requirements.txt.") from exc
    return BM25Okapi


def _load_sentence_transformer(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("sentence-transformers is required for dense retrieval. Install it from requirements.txt.") from exc
    try:
        return SentenceTransformer(model_name, local_files_only=True)
    except Exception:
        return SentenceTransformer(model_name)


def build_retrieval_index(
    full_index_path: str | Path,
    output_root: str | Path,
    *,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> dict[str, Any]:
    full_index_path = Path(full_index_path)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    cases = json.loads(full_index_path.read_text(encoding="utf-8-sig"))
    if not isinstance(cases, list):
        raise ValueError(f"Expected a list in {full_index_path}")

    narratives = [str(case.get("narrative_text", "") or "") for case in cases]
    tokenized = [_bm25_tokens_for_case(case) for case in cases]

    BM25Okapi = _load_rank_bm25()
    bm25 = BM25Okapi(tokenized)
    model = _load_sentence_transformer(model_name)
    embeddings = np.asarray(model.encode(narratives, convert_to_numpy=True, normalize_embeddings=True))

    metadata = {
        "case_count": len(cases),
        "model_name": model_name,
        "full_index_path": str(full_index_path),
    }
    (output_root / "cases.json").write_text(json.dumps(cases, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    (output_root / "tokenized_corpus.json").write_text(json.dumps(tokenized, ensure_ascii=True) + "\n", encoding="utf-8")
    (output_root / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    np.save(output_root / "dense_embeddings.npy", embeddings)
    LOGGER.info("Built retrieval index with %d cases at %s", len(cases), output_root)
    return metadata


def _load_index(index_root: str | Path, model_name: str = DEFAULT_EMBEDDING_MODEL) -> dict[str, Any]:
    index_root = Path(index_root)
    cache_key = f"{index_root.resolve()}::{model_name}"
    cached = _INDEX_CACHE.get(cache_key)
    if cached is not None:
        return cached

    cases = json.loads((index_root / "cases.json").read_text(encoding="utf-8-sig"))
    tokenized = json.loads((index_root / "tokenized_corpus.json").read_text(encoding="utf-8-sig"))
    embeddings = np.load(index_root / "dense_embeddings.npy")
    metadata = json.loads((index_root / "metadata.json").read_text(encoding="utf-8-sig"))

    BM25Okapi = _load_rank_bm25()
    bm25 = BM25Okapi(tokenized)
    model = _load_sentence_transformer(metadata.get("model_name") or model_name)

    cached = {
        "cases": cases,
        "tokenized": tokenized,
        "embeddings": embeddings,
        "metadata": metadata,
        "bm25": bm25,
        "model": model,
    }
    _INDEX_CACHE[cache_key] = cached
    return cached


def search(query: str, top_k: int = 5, *, index_root: str | Path | None = None) -> list[dict[str, Any]]:
    index_root = Path(index_root) if index_root is not None else _default_retrieval_root()
    loaded = _load_index(index_root)
    cases = loaded["cases"]
    bm25 = loaded["bm25"]
    embeddings = loaded["embeddings"]
    model = loaded["model"]

    def _bm25_scores() -> np.ndarray:
        return np.asarray(bm25.get_scores(_bm25_query_tokens(query)), dtype=float)

    def _dense_scores() -> np.ndarray:
        query_embedding = np.asarray(model.encode([query], convert_to_numpy=True, normalize_embeddings=True))[0]
        return np.asarray(embeddings @ query_embedding, dtype=float)

    with ThreadPoolExecutor(max_workers=2) as executor:
        bm25_future = executor.submit(_bm25_scores)
        dense_future = executor.submit(_dense_scores)
        bm25_scores = bm25_future.result()
        dense_scores = dense_future.result()

    bm25_norm = _normalize_scores(bm25_scores)
    dense_norm = _normalize_scores(dense_scores)
    base_scores = 0.5 * bm25_norm + 0.5 * dense_norm
    query_hints = _extract_query_hints(query, cases)

    if query_hints.get("structured_hint"):
        reranked = _structured_rerank_scores(cases, bm25_norm, dense_norm, query_hints)
        final_scores = reranked["final_scores"]
    else:
        reranked = {
            "location_scores": np.full(len(cases), 0.5, dtype=float),
            "measurement_scores": np.full(len(cases), 0.5, dtype=float),
            "distortion_scores": np.full(len(cases), 0.5, dtype=float),
            "pathology_scores": np.ones(len(cases), dtype=float),
            "modality_scores": np.ones(len(cases), dtype=float),
        }
        final_scores = base_scores

    ranked_indices = np.argsort(final_scores)[::-1][:top_k]
    results: list[dict[str, Any]] = []
    for idx in ranked_indices:
        case_dict = dict(cases[int(idx)])
        case_dict["retrieval_score"] = round(float(final_scores[idx]), 6)
        case_dict["base_score"] = round(float(base_scores[idx]), 6)
        case_dict["bm25_score"] = round(float(bm25_norm[idx]), 6)
        case_dict["dense_score"] = round(float(dense_norm[idx]), 6)
        case_dict["location_similarity"] = round(float(reranked["location_scores"][idx]), 6)
        case_dict["measurement_similarity"] = round(float(reranked["measurement_scores"][idx]), 6)
        case_dict["distortion_similarity"] = round(float(reranked["distortion_scores"][idx]), 6)
        case_dict["pathology_match"] = round(float(reranked["pathology_scores"][idx]), 6)
        case_dict["modality_match"] = round(float(reranked["modality_scores"][idx]), 6)
        case_dict["query_hints"] = dict(query_hints)
        results.append(case_dict)
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Build or query the BraTS retrieval index.")
    parser.set_defaults(
        command="build",
        full_index_path=str(_default_case_index_path()),
        output_root=str(_default_retrieval_root()),
        model_name=DEFAULT_EMBEDDING_MODEL,
    )
    subparsers = parser.add_subparsers(dest="command")

    build_parser = subparsers.add_parser("build", help="Build BM25 + dense retrieval artifacts.")
    build_parser.add_argument("--full-index-path", default=str(_default_case_index_path()))
    build_parser.add_argument("--output-root", default=str(_default_retrieval_root()))
    build_parser.add_argument("--model-name", default=DEFAULT_EMBEDDING_MODEL)

    search_parser = subparsers.add_parser("search", help="Search the retrieval index.")
    search_parser.add_argument("--query", required=True)
    search_parser.add_argument("--top-k", type=int, default=5)
    search_parser.add_argument("--index-root", default=str(_default_retrieval_root()))

    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    _configure_logging(args.verbose)

    if args.command in {None, "build"}:
        build_retrieval_index(args.full_index_path, args.output_root, model_name=args.model_name)
        return 0

    results = search(args.query, top_k=args.top_k, index_root=args.index_root)
    print(json.dumps(results, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
