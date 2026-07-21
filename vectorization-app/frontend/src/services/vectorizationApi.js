const DEFAULT_API_BASE_URL = "http://127.0.0.1:8000";
export const DEFAULT_NUM_REGIONS = 64;
export const MIN_NUM_REGIONS = 2;
export const MAX_NUM_REGIONS = 256;

function normalizeApiBaseUrl(value) {
  const normalizedValue = String(value || "")
    .trim()
    .replace(/\/+$/, "");

  return normalizedValue || DEFAULT_API_BASE_URL;
}

export const API_BASE_URL = normalizeApiBaseUrl(import.meta.env.VITE_API_BASE_URL);

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

  if (Array.isArray(detail)) {
    const messages = detail
      .map((item) => {
        if (typeof item === "string") {
          return item;
        }

        const location = Array.isArray(item?.loc) ? item.loc.join(".") : "";
        const message = item?.msg || item?.message || "";
        return [location, message].filter(Boolean).join(": ");
      })
      .filter(Boolean);

    return messages.join("\n") || "Vectorization failed.";
  }

  if (detail && typeof detail === "object") {
    return [detail.message, detail.error, detail.model_error, detail.stderr, detail.traceback]
      .filter(Boolean)
      .join("\n") || "Vectorization failed.";
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

export function parseNumRegions(value) {
  const normalizedValue = String(value ?? "").trim();
  const parsedValue = Number(normalizedValue);

  if (
    !normalizedValue ||
    !Number.isInteger(parsedValue) ||
    parsedValue < MIN_NUM_REGIONS ||
    parsedValue > MAX_NUM_REGIONS
  ) {
    throw new Error(
      `Number of regions must be an integer between ${MIN_NUM_REGIONS} and ${MAX_NUM_REGIONS}.`,
    );
  }

  return parsedValue;
}

// Send the uploaded image and SLIC configuration to the FastAPI /vectorize endpoint.
export async function vectorizeImage({ file, numRegions = DEFAULT_NUM_REGIONS }) {
  const validatedNumRegions = parseNumRegions(numRegions);
  const formData = new FormData();
  formData.append("file", file);
  formData.append("num_regions", String(validatedNumRegions));

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
