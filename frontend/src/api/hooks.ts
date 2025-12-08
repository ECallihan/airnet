// frontend/src/api/hooks.ts
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import type { QueryKey } from "@tanstack/react-query";


import type {
  SearchNodesResponse,
  PaperWithConceptsResponse,
  GraphStatsResponse,
  IngestArxivRequest,
  IngestArxivResponse,
  PaperInfluenceResponse,
} from "./types";


import {
  fetchSearchNodes,
  fetchPaper,
  fetchGraphStats,
  postIngestArxiv,
  fetchPaperInfluence,
} from "./client";


/* ========================================================================== */
/*  Query keys                                                                */
/* ========================================================================== */

const qk = {
  searchNodes: (q: string, kind?: string): QueryKey => ["searchNodes", { q, kind }],
  paper: (arxivId: string): QueryKey => ["paper", { arxivId }],
  paperInfluence: (arxivId: string): QueryKey => ["paperInfluence", { arxivId }],
  graphStats: (): QueryKey => ["graphStats"],
};


/* ========================================================================== */
/*  Hooks                                                                     */
/* ========================================================================== */

/**
 * Search for nodes (papers / concepts) by substring.
 *
 * Usage:
 *   const { data, isLoading } = useSearchNodes(searchText, kind);
 */
export function useSearchNodes(q: string, kind?: string) {
  const enabled = q.trim().length > 0;

  return useQuery<SearchNodesResponse>({
    queryKey: qk.searchNodes(q, kind),
    queryFn: () => fetchSearchNodes(q, kind),
    enabled,
    staleTime: 30_000, // 30s
  });
}

/**
 * Fetch a single paper with its concepts & neighbors.
 *
 * Usage:
 *   const { data, isLoading } = usePaper(arxivId);
 */
export function usePaper(arxivId: string | null | undefined) {
  const enabled = !!arxivId && arxivId.trim().length > 0;

  return useQuery<PaperWithConceptsResponse>({
    queryKey: enabled ? qk.paper(arxivId!) : ["paper", { arxivId: null }],
    queryFn: () => fetchPaper(arxivId!),
    enabled,
  });
}

/**
 * Fetch influence graph (incoming/outgoing papers) for a given paper.
 *
 * Usage:
 *   const { data, isLoading, error } = usePaperInfluence(arxivId);
 */
export function usePaperInfluence(
  arxivId: string | null | undefined,
) {
  const enabled = !!arxivId && arxivId.trim().length > 0;

  return useQuery<PaperInfluenceResponse>({
    queryKey: enabled
      ? qk.paperInfluence(arxivId!)
      : ["paperInfluence", { arxivId: null }],
    queryFn: () => fetchPaperInfluence(arxivId!),
    enabled,
  });
}


/**
 * Fetch global graph stats (node/edge counts, top concepts).
 *
 * Usage:
 *   const { data, isLoading } = useGraphStats();
 */
export function useGraphStats() {
  return useQuery<GraphStatsResponse>({
    queryKey: qk.graphStats(),
    queryFn: () => fetchGraphStats(),
    staleTime: 60_000, // 1 minute
  });
}

/**
 * Ingest a set of arXiv ids.
 *
 * Usage:
 *   const ingestMutation = useIngestArxiv();
 *   ingestMutation.mutate({ ids: ["2401.12345"] });
 */
export function useIngestArxiv() {
  const queryClient = useQueryClient();

  return useMutation<IngestArxivResponse, Error, IngestArxivRequest>({
    mutationFn: (payload) => postIngestArxiv(payload),
    onSuccess: (_data, _vars) => {
      // After ingest, stats might change (more papers, more concepts),
      // and some paper detail might now exist.
      queryClient.invalidateQueries({ queryKey: qk.graphStats() });
      // Up to you if you want to aggressively invalidate paper queries too.
      // queryClient.invalidateQueries({ queryKey: ["paper"] });
    },
  });
}
