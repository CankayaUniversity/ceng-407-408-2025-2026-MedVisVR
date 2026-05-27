const { contextBridge, ipcRenderer, webUtils } = require("electron");
const fs = require("fs");
const path = require("path");

const API_BASE = "http://127.0.0.1:8000";

function repoRoot() {
  return path.resolve(__dirname, "..", "..");
}

function contextsRoot() {
  return path.join(repoRoot(), "outputs", "cases");
}

function runsRoot() {
  const preferred = path.join(repoRoot(), "outputs", "runs_first10_reportllm_relaxed_v1");
  if (fs.existsSync(preferred)) return preferred;
  return path.join(repoRoot(), "outputs", "runs");
}

function readJson(filePath) {
  if (!fs.existsSync(filePath)) return {};
  try { return JSON.parse(fs.readFileSync(filePath, "utf8")); }
  catch { return {}; }
}

function readText(filePath) {
  if (!fs.existsSync(filePath)) return null;
  return fs.readFileSync(filePath, "utf8");
}

function readJsonl(filePath) {
  if (!fs.existsSync(filePath)) return [];
  return fs.readFileSync(filePath, "utf8")
    .split(/\r?\n/).filter(Boolean)
    .map(line => { try { return JSON.parse(line); } catch { return null; } })
    .filter(Boolean);
}

function artifactDir(caseId) {
  const preferred = path.join(runsRoot(), caseId);
  if (fs.existsSync(preferred)) return preferred;
  return path.join(contextsRoot(), caseId);
}

async function requestJson(pathname, options = {}) {
  const response = await fetch(`${API_BASE}${pathname}`, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options
  });
  if (!response.ok) {
    const payload = await response.text();
    throw new Error(payload || `Request failed with status ${response.status}`);
  }
  return response.json();
}

function listCases() {
  if (!fs.existsSync(contextsRoot())) return [];
  return fs.readdirSync(contextsRoot(), { withFileTypes: true })
    .filter(e => e.isDirectory())
    .map(e => {
      const caseId = e.name;
      const context = readJson(path.join(contextsRoot(), caseId, "patient_context.json"));
      const seg = context.segmentation_metrics || readJson(path.join(contextsRoot(), caseId, "tumor_metrics.json"));
      const artifacts = artifactDir(caseId);
      return {
        case_id: caseId,
        patient_id: context.patient_id || caseId,
        study_id: context.study_id || caseId,
        study_date: context.study_date || "unknown",
        modality: context.modality || "MR",
        has_report: fs.existsSync(path.join(artifacts, "report_draft.md")),
        has_quality_report: fs.existsSync(path.join(artifacts, "quality_report.json")),
        has_evidence_index: fs.existsSync(path.join(artifacts, "evidence.json")),
        tumor_count: Number(seg.tumor_count || 0),
        segmented_component_count: Number(seg.segmented_component_count || 0),
        total_tumor_volume_mm3: Number(seg.total_tumor_volume_mm3 || 0),
        max_diameter_mm: Number(seg.max_diameter_mm || 0)
      };
    })
    .sort((a, b) => a.case_id.localeCompare(b.case_id));
}

function loadCaseDetail(caseId) {
  const contextDir = path.join(contextsRoot(), caseId);
  const artifacts = artifactDir(caseId);
  const context = readJson(path.join(contextDir, "patient_context.json"));
  const tumorMetrics = readJson(path.join(artifacts, "tumor_metrics.json"));
  const qaResults = readJsonl(path.join(artifacts, "qa_results.jsonl"));
  const questionResults = qaResults.map(row => {
    const answer = String(row.answer || "");
    const insufficient = answer.trim().toLowerCase().startsWith("insufficient evidence");
    const evidenceIds = Array.isArray(row.evidence_ids) ? row.evidence_ids : [];
    return {
      qid: typeof row.qid === "number" ? row.qid : null,
      question: row.question || "",
      answer,
      evidence_ids: evidenceIds,
      confidence: Number(row.confidence || 0),
      safety_note: row.safety_note || "",
      evidence_status: insufficient ? "insufficient" : evidenceIds.length ? "supported" : "missing",
      insufficient
    };
  });

  const unsupported = questionResults.filter(i => i.evidence_status === "missing" && i.qid != null);
  const supportedCount = questionResults.filter(i => i.evidence_status === "supported").length;

  return {
    case_id: caseId,
    context,
    report_markdown: readText(path.join(artifacts, "report_draft.md")),
    quality_report: readJson(path.join(artifacts, "quality_report.json")),
    eval_summary: readJson(path.join(artifacts, "eval_summary.json")),
    findings_structured: readJson(path.join(artifacts, "findings_structured.json")),
    tumor_metrics: tumorMetrics,
    question_results: questionResults,
    evidence_index: readJson(path.join(artifacts, "evidence.json")),
    quality_panel: {
      total_questions: questionResults.length,
      insufficient_count: questionResults.filter(i => i.insufficient).length,
      evidence_coverage_rate: questionResults.length ? Number((supportedCount / questionResults.length).toFixed(4)) : 0,
      unsupported_answer_count: unsupported.length,
      unsupported_answer_qids: unsupported.map(i => i.qid)
    }
  };
}

contextBridge.exposeInMainWorld("desktopRuntime", {
  apiBase: API_BASE,
  appName: "NeuroLens Clinical Workspace",
  listCases,
  loadCaseDetail,
  getHealth: () => requestJson("/health"),
  askQuestion: (caseId, question) =>
    requestJson("/qa/ask-from-case", {
      method: "POST",
      body: JSON.stringify({ case_id: caseId, question })
    }),
  regenerateReport: (caseId) =>
    requestJson("/report/generate-from-case", {
      method: "POST",
      body: JSON.stringify({ case_id: caseId })
    }),
  getViewerMeta: (caseId) => requestJson(`/viewer/${caseId}/meta`),
  getSliceUrl: (caseId, plane, index, sequence, overlay) =>
    `${API_BASE}/viewer/${caseId}/slice?plane=${plane}&index=${index}&sequence=${sequence}&overlay=${overlay}`,
  getViewerMesh: (caseId, stepSize) =>
    requestJson(`/viewer/${caseId}/mesh?step_size=${stepSize || 2}`),
  getVolumeTexture: (caseId, sequence, maxDim) =>
    requestJson(`/viewer/${caseId}/volume?sequence=${sequence || "t1ce"}&max_dim=${maxDim || 64}`),

  chooseNiftiFile: () => ipcRenderer.invoke("choose-nifti-file"),
  chooseNiftiFiles: () => ipcRenderer.invoke("choose-nifti-files"),

  uploadInfer: async (filePath, organ) => {
    const fs = require("fs");
    const path = require("path");
    const formData = new FormData();
    const blob = new Blob([fs.readFileSync(filePath)]);
    const fileName = path.basename(filePath);
    formData.append("file", blob, fileName);
    formData.append("organ", organ);
    const response = await fetch(`${API_BASE}/upload/infer`, {
      method: "POST",
      body: formData,
    });
    if (!response.ok) {
      const text = await response.text();
      throw new Error(text || `Upload failed: ${response.status}`);
    }
    return response.json();
  },

  uploadInferBrain: async (filePaths) => {
    const fs = require("fs");
    const path = require("path");
    const formData = new FormData();
    for (const fp of filePaths) {
      const blob = new Blob([fs.readFileSync(fp)]);
      const fileName = path.basename(fp);
      formData.append("files", blob, fileName);
    }
    const response = await fetch(`${API_BASE}/upload/infer-brain`, {
      method: "POST",
      body: formData,
    });
    if (!response.ok) {
      const text = await response.text();
      throw new Error(text || `Upload failed: ${response.status}`);
    }
    return response.json();
  },

  uploadStatus: (jobId) => requestJson(`/upload/status/${jobId}`),

  getPathForFile: (file) => webUtils.getPathForFile(file),

  savePdf: (html, defaultName) =>
    ipcRenderer.invoke("save-pdf", { html, defaultName }),

  printPdf: (html) =>
    ipcRenderer.invoke("print-pdf", { html }),
});
