// Generic helpers
export type JsonObject = Record<string, unknown>;

/**
 * Default FastAPI-style error response.
 */
export interface ApiError {
  detail: string;
}

/* ========================================================================== */
/*  /search/nodes                                                             */
/* ========================================================================== */

export type NodeKind = "paper" | "concept" | string;

export interface SearchNodeResult {
  /** Internal graph node id, e.g. "paper:2401.12345" or "concept:gnn" */
  node_id: string;
  /** Semantic kind, usually "paper" or "concept" */
  kind: NodeKind | null;
  /** Human-readable label/title/name */
  label: string;
}

export interface SearchNodesResponse {
  /** Raw query string the user entered */
  query: string;
  /** Matching nodes */
  hits: SearchNodeResult[];
}

/* ========================================================================== */
/*  /papers/{arxiv_id}                                                        */
/* ========================================================================== */

export interface PaperModel {
  /** Internal graph node id, e.g. "paper:2401.12345" */
  id: string;
  /** arXiv id or canonical paper identifier */
  arxiv_id: string;
  title: string | null;
  abstract: string | null;
  /** Extra metadata from the graph node (year, venue, etc.) */
  attributes: JsonObject;
}

export interface ConceptSummary {
  /** Internal concept node id, e.g. "concept:graph-neural-networks" */
  id: string;
  /** Stable key/slug for the concept */
  key: string;
  /** Human-readable label */
  label: string;
  /** Relation type from paper → concept (e.g. "HAS_CONCEPT") */
  relation?: string | null;
  /** Optional weight/score (importance, tf-idf, etc.) */
  weight?: number | null;
  /** Extra concept metadata */
  attributes: JsonObject;
}

export type NeighborDirection = "in" | "out";

export interface Neighbor {
  /** Internal neighbour node id (usually another paper) */
  id: string;
  /** Node kind, typically "paper" (concepts are *not* included here) */
  kind: NodeKind | null;
  /** Human-readable label/title */
  label: string | null;
  /** Relation type on the edge (e.g. "CITES", "COCITED", etc.) */
  relation?: string | null;
  /**
   * Direction of the edge *relative to the center paper*:
   * - "out": paper → neighbor
   * - "in" : neighbor → paper
   */
  direction: NeighborDirection;
  /** Optional numeric weight from edge metadata */
  weight?: number | null;
  /** Extra neighbour metadata */
  attributes: JsonObject;
}

/**
 * Response from GET /papers/{arxiv_id}
 */
export interface PaperWithConceptsResponse {
  paper: PaperModel;
  /** Direct concept neighbours of the paper */
  concepts: ConceptSummary[];
  /**
   * Non-concept neighbours (usually other papers), directly connected
   * via incoming or outgoing edges.
   */
  neighbors: Neighbor[];
}

/* ========================================================================== */
/*  /graph/stats                                                              */
/* ========================================================================== */

export interface ConceptStats {
  /** Internal concept node id */
  id: string;
  /** Stable concept key/slug */
  key: string;
  /** Human-readable label */
  label: string;
  /** Graph degree (total edges incident on this concept) */
  degree: number;
}

/**
 * Response from GET /graph/stats
 */
export interface GraphStatsResponse {
  num_nodes: number;
  num_edges: number;
  num_papers: number;
  num_concepts: number;
  /** Top concepts sorted by degree */
  top_concepts: ConceptStats[];
}

/* ========================================================================== */
/*  /ingest/arxiv                                                             */
/* ========================================================================== */

/**
 * Request body for POST /ingest/arxiv
 */
export interface IngestArxivRequest {
  /** List of arXiv ids to ingest (duplicates are handled server-side) */
  ids: string[];
}

/**
 * Response body for POST /ingest/arxiv
 */
export interface IngestArxivResponse {
  /** IDs successfully ingested/updated in the graph */
  ingested: string[];
  /** IDs that were skipped (already present, invalid, etc.) */
  skipped: string[];
  /**
   * Map from arXiv id → error message for failures.
   * If empty, there were no ingestion errors.
   */
  errors: Record<string, string>;
}

/* ========================================================================== */
/*  /papers/{arxiv_id}/influence                                              */
/* ========================================================================== */

/**
 * One neighbor in the paper influence graph.
 *
 * Direction is relative to the *focal* paper:
 *  - "in"  : neighbor → focal paper (neighbor cites focal)
 *  - "out" : focal paper → neighbor (focal cites neighbor)
 *  - "both": bidirectional
 */
export interface InfluenceNeighbor {
  paper: PaperModel;
  direction: "in" | "out" | "both" | "unknown" | string;
  relation?: string | null;
  weight?: number | null;
}

/**
 * Response body for GET /papers/{arxiv_id}/influence
 */
export interface PaperInfluenceResponse {
  paper: PaperModel;
  incoming: InfluenceNeighbor[];
  outgoing: InfluenceNeighbor[];
}
