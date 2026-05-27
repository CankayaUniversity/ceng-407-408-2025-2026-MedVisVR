const API_BASE = window.desktopRuntime?.apiBase || "http://127.0.0.1:8000";

async function request(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {})
    },
    ...options
  });

  if (!response.ok) {
    const payload = await response.text();
    throw new Error(payload || `Request failed with status ${response.status}`);
  }

  return response.json();
}

export function getHealth() {
  if (window.desktopRuntime?.getHealth) {
    return Promise.resolve(window.desktopRuntime.getHealth());
  }
  return request("/health");
}

export function getCases() {
  if (window.desktopRuntime?.listCases) {
    return Promise.resolve(window.desktopRuntime.listCases());
  }
  return request("/cases");
}

export function getCaseDetail(caseId) {
  if (window.desktopRuntime?.loadCaseDetail) {
    return Promise.resolve(window.desktopRuntime.loadCaseDetail(caseId));
  }
  return request(`/cases/${encodeURIComponent(caseId)}`);
}

export function askQuestion(caseId, question, sessionId = null) {
  if (window.desktopRuntime?.askQuestion) {
    return Promise.resolve(window.desktopRuntime.askQuestion(caseId, question));
  }
  return request("/qa/ask-from-case", {
    method: "POST",
    body: JSON.stringify({
      case_id: caseId,
      question,
      ...(sessionId ? { session_id: sessionId } : {}),
    })
  });
}

export function clearQASession(sessionId) {
  return request("/qa/session/clear", {
    method: "POST",
    body: JSON.stringify({ session_id: sessionId })
  });
}

export function deleteCase(caseId) {
  return request(`/cases/${encodeURIComponent(caseId)}`, { method: "DELETE" });
}

export function regenerateReport(caseId) {
  if (window.desktopRuntime?.regenerateReport) {
    return Promise.resolve(window.desktopRuntime.regenerateReport(caseId));
  }
  return request("/report/generate-from-case", {
    method: "POST",
    body: JSON.stringify({
      case_id: caseId
    })
  });
}
