export const API_BASE_URL = "http://127.0.0.1:8000";

export function getAbsoluteUrl(url) {
  if (!url) {
    return "";
  }

  if (/^https?:\/\//i.test(url) || url.startsWith("blob:") || url.startsWith("data:")) {
    return url;
  }

  return `${API_BASE_URL}${url.startsWith("/") ? "" : "/"}${url}`;
}

export function getSvgContent(data) {
  if (typeof data === "string") {
    return data.includes("<svg") ? data : "";
  }

  const content = data?.svg_content || data?.svgContent || data?.svg || data?.svg_result;
  return typeof content === "string" && content.trim() ? content : "";
}

export function getBackendErrorMessage(data) {
  if (typeof data === "string") {
    return data || "Vectorization failed.";
  }

  const detail = data?.detail;
  if (typeof detail === "string") {
    return detail;
  }

  if (detail && typeof detail === "object") {
    return [detail.message, detail.stderr].filter(Boolean).join("\n") || "Vectorization failed.";
  }

  return data?.message || "Vectorization failed.";
}

async function parseResponse(response) {
  const contentType = response.headers.get("content-type") || "";

  if (contentType.includes("application/json")) {
    return response.json();
  }

  return response.text();
}

// Send the uploaded image and parameter values to the FastAPI /vectorize endpoint.
export async function vectorizeImage({ device = "cpu", file, pathNum, optimizeIter }) {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("path_num", String(Number(pathNum)));
  formData.append("optimize_iter", String(Number(optimizeIter)));
  formData.append("device", device);

  const response = await fetch(`${API_BASE_URL}/vectorize`, {
    method: "POST",
    body: formData,
  });
  const data = await parseResponse(response);

  if (!response.ok) {
    throw new Error(getBackendErrorMessage(data));
  }

  return data;
}

export function isBackendUnavailableError(error) {
  const message = error?.message || "";
  return error instanceof TypeError || /failed to fetch|networkerror|load failed/i.test(message);
}
