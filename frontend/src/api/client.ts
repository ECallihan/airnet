// frontend/src/api/client.ts
import type {
  SearchNodesResponse,
  PaperWithConceptsResponse,
  GraphStatsResponse,
  IngestArxivRequest,
  IngestArxivResponse,
  PaperInfluenceResponse,
} from "./types";


const API_BASE = "/api"; 

/**
 * Low-level GET wrapper with typed response.
 */
async function apiGet<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "GET",
    headers: {
      "Accept": "application/json",
    },
    ...init,
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(
      `GET ${path} failed with ${res.status}: ${text || res.statusText}`,
    );
  }

  return res.json() as Promise<T>;
}

/**
 * Low-level POST wrapper with typed response.
 */
async function apiPost<TReq, TRes>(
  path: string,
  body: TReq,
  init?: RequestInit,
): Promise<TRes> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Accept": "application/json",
    },
    body: JSON.stringify(body),
    ...init,
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(
      `POST ${path} failed with ${res.status}: ${text || res.statusText}`,
    );
  }

  return res.json() as Promise<TRes>;
}

/* ========================================================================== */
/*  Specific API calls                                                        */
/* ========================================================================== */

export function fetchSearchNodes(
  q: string,
  kind?: string,
): Promise<SearchNodesResponse> {
  const params = new URLSearchParams({ q });
  if (kind) params.set("kind", kind);

  return apiGet<SearchNodesResponse>(`/search/nodes?${params.toString()}`);
}

export function fetchPaper(
  arxivId: string,
): Promise<PaperWithConceptsResponse> {
  return apiGet<PaperWithConceptsResponse>(`/papers/${encodeURIComponent(arxivId)}`);
}

export function fetchGraphStats(): Promise<GraphStatsResponse> {
  return apiGet<GraphStatsResponse>("/graph/stats");
}

export function postIngestArxiv(
  payload: IngestArxivRequest,
): Promise<IngestArxivResponse> {
  return apiPost<IngestArxivRequest, IngestArxivResponse>(
    "/ingest/arxiv",
    payload,
  );
}

export function fetchPaperInfluence(
  arxivId: string,
): Promise<PaperInfluenceResponse> {
  return apiGet<PaperInfluenceResponse>(
    `/papers/${encodeURIComponent(arxivId)}/influence`,
  );
}
