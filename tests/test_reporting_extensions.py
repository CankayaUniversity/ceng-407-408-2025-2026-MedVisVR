import json

from ai_assistant.core.anatomic_mapper import MNI_TEMPLATE_PATH, anatomy_from_lesion, lookup_mni_point, map_centroid_to_anatomy
from ai_assistant.core.qa_engine import answer_question
from ai_assistant.core.report_descriptor import build_report_descriptor
from ai_assistant.eval.automated_eval import run_evaluation
from ai_assistant.eval.comparative import compare_settings
from ai_assistant.eval.error_analysis import analyze_errors
from ai_assistant.eval.report_quality import calculate_quality_metrics
from ai_assistant.harness.run_case import run_case
from tests.conftest import make_context, make_qa_request


def test_build_report_descriptor_exposes_grounded_narrative_fields():
    descriptor = build_report_descriptor(make_context().model_dump())
    assert descriptor["case_overview"]["patient_id"] == "TEST_001"
    assert descriptor["segmentation_overview"]["lesion_count"] == 3
    assert descriptor["segmentation_overview"]["segmented_component_count"] == 3
    assert descriptor["component_profile"]["dominant_component"] == "non-enhancing"
    assert "spatial_distribution" in descriptor
    assert "semantic_segmentation" in descriptor
    assert descriptor["clinical_reporting_cards"]["quantitative_reliability"]
    assert "mass_effect_is_not_directly_measured" in descriptor["narrative_hints"]["heuristic_flags"]


def test_map_centroid_to_anatomy_returns_expected_keys():
    anatomy = map_centroid_to_anatomy(x=50, y=40, z=120, image_shape_zyx=(155, 240, 240))
    assert anatomy["hemisphere"] in {"Left", "Right"}
    assert anatomy["lobe"] in {"Frontal", "Parietal", "Temporal", "Occipital"}
    assert anatomy["depth"] in {"Periventricular", "Deep white matter", "Subcortical"}
    assert anatomy["level"] in {"Superior", "Middle", "Inferior"}
    assert anatomy["description"]


def test_lookup_mni_point_returns_harvard_oxford_region():
    region = lookup_mni_point((-28.0, -55.0, 61.0))
    assert region["mapping_basis"] in {"harvard_oxford_mni152", "mni152_fallback"}
    assert region["hemisphere"] in {"Left", "Right"}
    assert region["description"]


def test_anatomy_from_lesion_uses_overlap_summary_with_identity_transform(monkeypatch):
    sitk = __import__("SimpleITK")
    monkeypatch.setenv("AI_USE_ATLAS_LOCALIZATION", "1")
    source_image = sitk.ReadImage(str(MNI_TEMPLATE_PATH))
    center_mni = (-28.0, -55.0, 61.0)
    center_idx = source_image.TransformPhysicalPointToContinuousIndex(center_mni)
    lesion = {
        "centroid_xyz_voxel": [center_idx[0], center_idx[1], center_idx[2]],
        "centroid_zyx_voxel": [center_idx[2], center_idx[1], center_idx[0]],
    }
    coords = []
    for dz in range(-1, 2):
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                coords.append([int(round(center_idx[2] + dz)), int(round(center_idx[1] + dy)), int(round(center_idx[0] + dx))])
    anatomy = anatomy_from_lesion(
        lesion,
        image_shape_zyx=tuple(reversed(source_image.GetSize())),
        source_image=source_image,
        standard_space_transform={
            "transform": sitk.Euler3DTransform(),
            "summary": {
                "quality_flag": "high",
                "quality_score": 0.91,
                "normalized_cross_correlation": 0.88,
                "foreground_dice": 0.93,
                "center_distance_mm": 2.1,
            },
        },
        lesion_coords_zyx=__import__("numpy").array(coords),
    )
    assert anatomy["mapping_basis"] in {"harvard_oxford_mni152_overlap", "mni152_fallback"}
    assert "atlas_support_fraction" in anatomy
    assert anatomy["registration_quality_flag"] == "high"
    assert anatomy["description"]


def test_low_registration_quality_downgrades_atlas_confidence(monkeypatch):
    sitk = __import__("SimpleITK")
    monkeypatch.setenv("AI_USE_ATLAS_LOCALIZATION", "1")
    source_image = sitk.ReadImage(str(MNI_TEMPLATE_PATH))
    center_mni = (-28.0, -55.0, 61.0)
    center_idx = source_image.TransformPhysicalPointToContinuousIndex(center_mni)
    lesion = {
        "centroid_xyz_voxel": [center_idx[0], center_idx[1], center_idx[2]],
        "centroid_zyx_voxel": [center_idx[2], center_idx[1], center_idx[0]],
    }
    coords = __import__("numpy").array([[int(round(center_idx[2])), int(round(center_idx[1])), int(round(center_idx[0]))]])
    anatomy = anatomy_from_lesion(
        lesion,
        image_shape_zyx=tuple(reversed(source_image.GetSize())),
        source_image=source_image,
        standard_space_transform={
            "transform": sitk.Euler3DTransform(),
            "summary": {
                "quality_flag": "low",
                "quality_score": 0.22,
                "normalized_cross_correlation": 0.19,
                "foreground_dice": 0.24,
                "center_distance_mm": 24.8,
            },
        },
        lesion_coords_zyx=coords,
    )
    assert anatomy["registration_quality_flag"] == "low"
    assert anatomy["atlas_confidence"] == "low"
    assert anatomy["mapping_basis"] == "mni152_fallback"


def test_report_quality_metrics_detect_filled_sections():
    report_text = (
        "# Clinical Context\nctx\n\n# Findings\n"
        "## Localization\n- Anatomic location: Left frontal lobe.\n"
        "## Mass Effect\n- Severity: Moderate.\n- Estimate: present.\n"
        "## Midline Shift\n- Probability: Low.\n- Estimate: unlikely.\n"
        "## Enhancement Pattern\n- Dominant: Non-enhancing.\n- Percentage: 80.0%.\n- Description: infiltrative.\n"
        "## Multifocality\n- Type: Dominant+satellites.\n- Component count: 3.\n- Description: dominant lesion.\n"
        "# Impression\nimp\n# Limitations\nlim\n# Suggested Clinical Correlation\nrec\n# Disclaimer\ndis\n"
    )
    metrics = calculate_quality_metrics(report_text)
    assert metrics["information_coverage"] > 0.9
    assert metrics["lexical_diversity"] > 0


def test_automated_eval_and_error_analysis_flag_measurement_mismatch(tmp_path):
    case_dir = tmp_path / "TEST_CASE"
    case_dir.mkdir()
    (case_dir / "tumor_metrics.json").write_text(
        json.dumps(
            {
                "total_tumor_volume_mm3": 45000.0,
                "tumor_count": 3,
                "max_diameter_mm": 52.0,
                "label_volumes_mm3": {"enhancing": 12000.0, "nonenhancing": 18000.0, "edema": 15000.0},
                "lesion_list": [{"label_name": "nonenhancing", "volume_mm3": 18000.0, "anatomy": {"description": "Left frontal lobe, deep white matter, middle level"}}],
                "dominant_anatomic_location": "Left frontal lobe, deep white matter, middle level",
            },
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )
    qa_path = case_dir / "qa_results.jsonl"
    qa_path.write_text(
        json.dumps(
            {
                "qid": 1,
                "question": "What is the total tumor volume?",
                "answer": "Total segmented tumor volume is 41000.00 mm3 (41.00 cm3).",
                "evidence_ids": ["ctx:segmentation_metrics.total_tumor_volume_mm3"],
                "confidence": 0.72,
                "safety_note": "For decision support only; clinician confirmation required.",
            },
            ensure_ascii=True,
        )
        + "\n",
        encoding="utf-8",
    )

    summary = run_evaluation(qa_path)
    errors = analyze_errors(qa_path)

    assert summary["factual_accuracy"] == 0.0
    assert summary["hallucination_rate"] == 1.0
    assert errors["summary"]["measurement_error_count"] == 1


def test_automated_eval_accepts_ratio_tolerance(tmp_path):
    case_dir = tmp_path / "TEST_RATIO"
    case_dir.mkdir()
    (case_dir / "tumor_metrics.json").write_text(
        json.dumps(
            {
                "total_tumor_volume_mm3": 117932.0,
                "tumor_count": 8,
                "max_diameter_mm": 83.0,
                "label_volumes_mm3": {"enhancing": 19948.0, "nonenhancing": 97785.0, "edema": 315.0},
                "lesion_list": [],
            },
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )
    qa_path = case_dir / "qa_results.jsonl"
    qa_path.write_text(
        json.dumps(
            {
                "qid": 8,
                "question": "What is the edema-to-tumor volume ratio?",
                "answer": "Edema-to-total-tumor volume ratio is 0.003 (0.3% of total segmented burden).",
                "evidence_ids": ["ctx:segmentation_metrics.label_volumes_mm3.edema"],
                "confidence": 0.68,
                "safety_note": "For decision support only; clinician confirmation required.",
            },
            ensure_ascii=True,
        )
        + "\n",
        encoding="utf-8",
    )

    summary = run_evaluation(qa_path)
    assert summary["factual_accuracy"] == 1.0
    assert summary["hallucination_rate"] == 0.0


def test_compare_settings_prefers_balanced_or_detailed():
    comparison = compare_settings(
        case_id="TEST_001",
        settings_list=["conservative", "balanced", "detailed"],
        context=make_context(),
    )
    assert comparison["recommended"] in {"balanced", "detailed"}
    assert 0.0 <= comparison["consistency_score"] <= 1.0
    assert len(comparison["variants"]) == 3
    by_name = {item["setting"]: item for item in comparison["variants"]}
    assert by_name["detailed"]["avg_sentence_length"] >= by_name["balanced"]["avg_sentence_length"]


def test_run_case_writes_extended_eval_artifacts(tmp_path):
    contexts_root = tmp_path / "cases"
    runs_root = tmp_path / "runs"
    case_dir = contexts_root / "TEST_001"
    case_dir.mkdir(parents=True)
    context = make_context()
    (case_dir / "patient_context.json").write_text(context.model_dump_json(indent=2), encoding="utf-8")
    (case_dir / "tumor_metrics.json").write_text(
        json.dumps(context.segmentation_metrics.model_dump(), ensure_ascii=True, indent=2),
        encoding="utf-8",
    )

    out_dir = run_case(
        case_id="TEST_001",
        contexts_root=str(contexts_root),
        out_root=str(runs_root),
        seg_root=str(tmp_path / "missing_seg_root"),
    )

    assert (out_dir / "metrics.json").exists()
    assert (out_dir / "eval_summary.json").exists()
    assert (out_dir / "error_log.json").exists()
    assert (out_dir / "comparison.json").exists()


def test_qa_uses_atlas_backed_wording_when_mapping_basis_is_atlas():
    ctx = make_context()
    seg = ctx.segmentation_metrics.model_dump()
    seg["lesion_list"][1]["anatomy"] = {
        "description": "Left Parietal lobe (Left Superior Parietal Lobule), subcortical, superior level",
        "mapping_basis": "harvard_oxford_mni152",
    }
    ctx.segmentation_metrics = type(ctx.segmentation_metrics)(**seg)
    resp = answer_question(make_qa_request("What is the largest lesion and where is it located?", context=ctx))
    assert "approximate atlas-supported location" in resp.answer.lower()
