from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class FlexibleModel(BaseModel):
    model_config = ConfigDict(extra="allow")


class DicomSummary(FlexibleModel):
    dicom_root: str | None = None
    dicom_count: int = 0
    study_dirs: list[str] = Field(default_factory=list)
    series_dirs: list[str] = Field(default_factory=list)


class NiftiSummary(FlexibleModel):
    nifti_root: str | None = None
    nifti_files: list[str] = Field(default_factory=list)
    nifti_json_sidecars: list[str] = Field(default_factory=list)


class FindingsStructured(FlexibleModel):
    dicom_summary: DicomSummary | None = None
    nifti_summary: NiftiSummary | None = None
    source_evidence: list[str] = Field(default_factory=list)


class LesionMetric(FlexibleModel):
    lesion_id: int | None = None
    label_value: int | None = None
    label_name: str = ""
    voxel_count: int | None = None
    volume_mm3: float = 0.0
    max_diameter_mm: float = 0.0
    centroid_zyx_voxel: list[float] = Field(default_factory=list)
    centroid_xyz_voxel: list[float] = Field(default_factory=list)
    anatomy: dict[str, Any] = Field(default_factory=dict)


class SegmentationMetrics(FlexibleModel):
    patient_id: str | None = None
    study_id: str | None = None
    tumor_count: int = 0
    segmented_component_count: int = 0
    lesion_list: list[LesionMetric] = Field(default_factory=list)
    volume_mm3: float = 0.0
    total_tumor_volume_mm3: float = 0.0
    max_diameter_mm: float = 0.0
    location_labels: list[str] = Field(default_factory=list)
    label_volumes_mm3: dict[str, float] = Field(default_factory=dict)
    voxel_spacing_xyz_mm: list[float] = Field(default_factory=list)
    image_shape_zyx: list[int] = Field(default_factory=list)
    source_mask_path: str | None = None
    model_version: str | None = None
    dominant_lesion_id: int | None = None
    dominant_anatomic_location: str | None = None
    uncertainty: dict[str, Any] = Field(default_factory=dict)


class PatientContext(FlexibleModel):
    patient_id: str
    study_id: str
    study_date: str
    modality: str = Field(description="CT or MR")
    clinical_note: str | None = None
    findings_structured: FindingsStructured | None = None
    segmentation_metrics: SegmentationMetrics | None = None
    prior_reports: list[str] | None = None


class ReportRequest(BaseModel):
    context: PatientContext


class ReportFromCaseRequest(BaseModel):
    case_id: str


class ReportResponse(BaseModel):
    report_markdown: str
    evidence_ids: list[str]
    confidence: float
    safety_note: str


class QARequest(BaseModel):
    patient_id: str
    study_id: str
    question: str
    context: PatientContext
    session_id: str | None = None
    reset_session: bool = False


class QAFromCaseRequest(BaseModel):
    case_id: str
    question: str
    session_id: str | None = None
    reset_session: bool = False


class ClinicalImplicationItem(FlexibleModel):
    implication_type: str = ""
    source_metric: str = ""
    metric_value: float | bool | str | None = None
    unit: str | None = None
    tier: str = ""
    severity_rank: int = 0
    rationale: str = ""
    evidence_level: str = ""
    literature_anchor: list[str] = Field(default_factory=list)
    btrads_alignment: str = ""
    surfaced_in_answer: bool = False


class ClinicalImplication(FlexibleModel):
    generated: bool = False
    summary: str = ""
    overall_tier: str = ""
    triggered_by_intent: bool = False
    items: list[ClinicalImplicationItem] = Field(default_factory=list)


class QAResponse(BaseModel):
    answer: str
    evidence_ids: list[str]
    confidence: float
    context_confidence: float | None = None
    answer_confidence: float | None = None
    confidence_breakdown: dict[str, Any] = Field(default_factory=dict)
    safety_note: str
    session_id: str | None = None
    resolved_question: str | None = None
    resolved_intent: dict[str, Any] = Field(default_factory=dict)
    retrieval_trace: dict[str, Any] | None = None
    supporting_cases: list[dict[str, Any]] = Field(default_factory=list)
    clinical_implication: ClinicalImplication | None = None
    session_turns_used: int = 0


class QASessionClearRequest(BaseModel):
    session_id: str


class QASessionClearResponse(BaseModel):
    session_id: str
    cleared: bool = True


class CaseListItem(FlexibleModel):
    case_id: str
    patient_id: str
    study_id: str
    study_date: str
    modality: str
    has_report: bool = False
    has_quality_report: bool = False
    has_evidence_index: bool = False
    tumor_count: int = 0
    segmented_component_count: int = 0
    total_tumor_volume_mm3: float = 0.0
    max_diameter_mm: float = 0.0


class CaseAnswerSummary(FlexibleModel):
    qid: int | None = None
    question: str
    answer: str
    evidence_ids: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    safety_note: str = ""
    evidence_status: str = "not_available"
    insufficient: bool = False


class CaseQualityPanel(FlexibleModel):
    total_questions: int = 0
    insufficient_count: int = 0
    evidence_coverage_rate: float = 0.0
    unsupported_answer_count: int = 0
    unsupported_answer_qids: list[int] = Field(default_factory=list)


class CaseDetailResponse(FlexibleModel):
    case_id: str
    context: PatientContext
    report_markdown: str | None = None
    quality_report: dict[str, Any] = Field(default_factory=dict)
    eval_summary: dict[str, Any] = Field(default_factory=dict)
    findings_structured: dict[str, Any] = Field(default_factory=dict)
    tumor_metrics: dict[str, Any] = Field(default_factory=dict)
    question_results: list[CaseAnswerSummary] = Field(default_factory=list)
    evidence_index: dict[str, Any] = Field(default_factory=dict)
    quality_panel: CaseQualityPanel = Field(default_factory=CaseQualityPanel)
