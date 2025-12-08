import React from "react";
import { useSearchNodes } from "../api/hooks";
import type { NodeKind } from "../api/types";

type SortMode = "relevance" | "label";

interface SearchResultsListProps {
  query: string;
  kindFilter?: NodeKind | "all";
  sortMode?: SortMode;
  onSelectPaper: (arxivId: string) => void;
  onSelectConcept?: (conceptKey: string) => void;
}

export const SearchResultsList: React.FC<SearchResultsListProps> = ({
  query,
  kindFilter = "all",
  sortMode = "relevance",
  onSelectPaper,
  onSelectConcept,
}) => {
  // Map "all" -> undefined so backend returns all node kinds,
  // matching CLI behavior.
  const kindParam =
    kindFilter === "all" ? undefined : (kindFilter as string | undefined);

  const { data, isLoading, error } = useSearchNodes(query, kindParam);

  const trimmed = query.trim();

  if (!trimmed) {
    return <div style={{ color: "#777" }}>Type to search…</div>;
  }

  if (isLoading) {
    return <div>Searching…</div>;
  }

  if (error) {
    return (
      <div style={{ color: "red" }}>
        Error: {(error as Error).message}
      </div>
    );
  }

  const rawHits = data?.hits ?? [];

  // Apply sort mode:
  // - "relevance": preserve backend order.
  // - "label": sort by label (case-insensitive), then node_id.
  const hits =
    sortMode === "label"
      ? [...rawHits].sort((a, b) => {
          const la = (a.label ?? "").toLowerCase();
          const lb = (b.label ?? "").toLowerCase();
          if (la < lb) return -1;
          if (la > lb) return 1;
          const ia = a.node_id.toLowerCase();
          const ib = b.node_id.toLowerCase();
          if (ia < ib) return -1;
          if (ia > ib) return 1;
          return 0;
        })
      : rawHits;

  if (hits.length === 0) {
    return (
      <div style={{ color: "#6b7280" }}>
        No results for “{trimmed}”
        {kindFilter !== "all" && <> in {kindFilter}s</>}.
      </div>
    );
  }

  return (
    <div
      style={{
        maxHeight: "60vh",
        overflowY: "auto",
        border: "1px solid #eee",
        borderRadius: 4,
      }}
    >
      {hits.map((hit) => {
        const isPaper = hit.kind === "paper";
        const isConcept = hit.kind === "concept";

        const handleClick = () => {
          if (isPaper) {
            const arxivId = hit.node_id.startsWith("paper:")
              ? hit.node_id.substring("paper:".length)
              : hit.node_id;
            onSelectPaper(arxivId);
          } else if (isConcept && onSelectConcept) {
            // Normalize to concept "key" (strip 'concept:' prefix if present)
            const key = hit.node_id.startsWith("concept:")
              ? hit.node_id.substring("concept:".length)
              : hit.node_id;
            onSelectConcept(key);
          }
        };

        const kindLabel =
          hit.kind === "paper"
            ? "paper"
            : hit.kind === "concept"
            ? "concept"
            : hit.kind || "";

        return (
          <div
            key={hit.node_id}
            onClick={handleClick}
            style={{
              padding: "0.5rem 0.75rem",
              borderBottom: "1px solid #f0f0f0",
              cursor: isPaper || isConcept ? "pointer" : "default",
              backgroundColor: isPaper ? "#fdfdfd" : "#fafafa",
            }}
          >
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                gap: "0.5rem",
              }}
            >
              <div
                style={{
                  fontWeight: 500,
                  overflow: "hidden",
                  textOverflow: "ellipsis",
                  whiteSpace: "nowrap",
                  flex: 1,
                }}
              >
                {hit.label}
              </div>
              {kindLabel && (
                <span
                  style={{
                    fontSize: "0.7rem",
                    textTransform: "uppercase",
                    letterSpacing: "0.05em",
                    padding: "0.1rem 0.4rem",
                    borderRadius: 999,
                    border: "1px solid #e5e7eb",
                    color: "#6b7280",
                    backgroundColor: "#f9fafb",
                    whiteSpace: "nowrap",
                    marginLeft: "0.5rem",
                    flexShrink: 0,
                  }}
                >
                  {kindLabel}
                </span>
              )}
            </div>
            <div
              style={{
                fontSize: "0.8rem",
                color: "#555",
                marginTop: "0.15rem",
              }}
            >
              <code>{hit.node_id}</code>
            </div>
          </div>
        );
      })}
    </div>
  );
};
