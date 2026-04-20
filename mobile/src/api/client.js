import Constants from "expo-constants";

const extra = Constants?.expoConfig?.extra || {};

// Use your machine LAN IP for physical device testing, not localhost.
export const API_BASE_URL = extra.apiBaseUrl || "http://localhost:8000";

export async function identifyCard(imageUri) {
  const filename = imageUri.split("/").pop() || "card.jpg";
  const ext = filename.split(".").pop()?.toLowerCase();
  const mimeType = ext === "png" ? "image/png" : "image/jpeg";

  const formData = new FormData();
  formData.append("image", {
    uri: imageUri,
    name: filename,
    type: mimeType,
  });

  const response = await fetch(`${API_BASE_URL}/identify-card`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const body = await response.text();
    throw new Error(`Identify request failed (${response.status}): ${body}`);
  }

  return response.json();
}
