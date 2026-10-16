// The one fetch helper: the token rides in a header — never the URL, where
// it would land in the server's access logs (and any proxy's). 204 maps to
// null, and FastAPI's `detail` field is extracted for error text.
const TOKEN_KEY = "aether_token";

export const token = () => localStorage.getItem(TOKEN_KEY) || "";
export const hasToken = () => Boolean(token());

export async function api(path, opts) {
  const { headers, ...rest } = opts || {};
  const res = await fetch(path, {
    ...rest,
    headers: { "x-aether-token": token(), ...headers },
  });
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch { /* not json */ }
    throw new Error(detail || `request failed (${res.status})`);
  }
  if (res.status === 204) return null;
  return res.json();
}
