import React, { useState } from "react";
import { SearchBar } from "./components/SearchBar";
import { SearchResultsList } from "./components/SearchResultsList";
import { PaperDetail } from "./components/PaperDetail";
import { PaperInfluenceExplorer } from "./components/PaperInfluenceExplorer";
import { ConceptDetail } from "./components/ConceptDetail";

import { useDebouncedValue } from "./hooks/useDebouncedValue";
import type { NodeKind } from "./api/types";

type SortMode = "relevance" | "label";

export const App: React.FC = () => {
  const [rawQuery, setRawQuery] = useState("");
  const [selectedArxivId, setSelectedArxivId] = useState<string | null>(null);
  const [selectedConceptKey, setSelectedConceptKey] = useState<string | null>(
    null
  );

  // Default to "all" so UI matches CLI search behavior
  const [filterKind, setFilterKind] = useState<"all" | NodeKind>("all");

  // Sort toggle: backend order (relevance-ish) vs label A→Z
  const [sortMode, setSortMode] = useState<SortMode>("relevance");

  const debouncedQuery = useDebouncedValue(rawQuery, 250);

  const handleSelectPaper = (arxivId: string) => {
    setSelectedArxivId(arxivId);
  };

  const handleSelectConcept = (conceptKey: string) => {
    setSelectedConceptKey(conceptKey);
  };

  const renderFilterButton = (value: "all" | NodeKind, label: string) => {
    const isActive = filterKind === value;
    return (
      <button
        key={value}
        type="button"
        onClick={() => setFilterKind(value)}
        style={{
          padding: "0.2rem 0.6rem",
          borderRadius: 999,
          border: isActive ? "1px solid #2563eb" : "1px solid #ddd",
          backgroundColor: isActive ? "#2563eb" : "#f9fafb",
          color: isActive ? "#ffffff" : "#111827",
          fontSize: "0.8rem",
          cursor: "pointer",
        }}
      >
        {label}
      </button>
    );
  };

  const renderSortButton = (value: SortMode, label: string) => {
    const isActive = sortMode === value;
    return (
      <button
        key={value}
        type="button"
        onClick={() => setSortMode(value)}
        style={{
          padding: "0.2rem 0.55rem",
          borderRadius: 999,
          border: isActive ? "1px solid #4b5563" : "1px solid #ddd",
          backgroundColor: isActive ? "#4b5563" : "#f9fafb",
          color: isActive ? "#ffffff" : "#111827",
          fontSize: "0.75rem",
          cursor: "pointer",
        }}
      >
        {label}
      </button>
    );
  };

  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "minmax(0, 1.4fr) minmax(0, 1.6fr)",
        gap: "1.5rem",
        padding: "1.5rem",
        fontFamily:
          '-apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif',
        fontSize: "14px",
      }}
    >
      {/* Left column: search + results */}
      <div>
        <h1 style={{ margin: 0, marginBottom: "0.75rem", fontSize: "1.25rem" }}>
          AirNet search
        </h1>
        <p style={{ margin: 0, marginBottom: "0.75rem", color: "#4b5563" }}>
          Search across papers and concepts in the knowledge graph. Select a
          paper to inspect its concepts and influence neighborhood, or a concept
          to inspect it in the concept pane.
        </p>

        <SearchBar value={rawQuery} onChange={setRawQuery} />

        {/* Filter + sort controls */}
        <div
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            gap: "0.75rem",
            marginTop: "0.5rem",
            marginBottom: "0.5rem",
            flexWrap: "wrap",
          }}
        >
          {/* Kind filter chips */}
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: "0.5rem",
              flexWrap: "wrap",
            }}
          >
            <span style={{ fontSize: "0.8rem", color: "#6b7280" }}>
              Filter:
            </span>
            {renderFilterButton("all", "All")}
            {renderFilterButton("paper", "Papers")}
            {renderFilterButton("concept", "Concepts")}
          </div>

          {/* Sort toggle */}
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: "0.4rem",
              flexWrap: "wrap",
            }}
          >
            <span style={{ fontSize: "0.8rem", color: "#6b7280" }}>Sort:</span>
            {renderSortButton("relevance", "Relevance")}
            {renderSortButton("label", "Label A→Z")}
          </div>
        </div>

        <SearchResultsList
          query={debouncedQuery}
          kindFilter={filterKind}
          sortMode={sortMode}
          onSelectPaper={handleSelectPaper}
          onSelectConcept={handleSelectConcept}
        />
      </div>

      {/* Right column: paper detail + concept inspector */}
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          gap: "1rem",
          maxHeight: "calc(100vh - 3rem)",
        }}
      >
        <div
          style={{
            flex: 1,
            minHeight: 0,
            display: "flex",
            flexDirection: "column",
            gap: "1rem",
            overflow: "auto",
          }}
        >
        <PaperDetail arxivId={selectedArxivId} />
        {selectedArxivId && (
          <PaperInfluenceExplorer
            arxivId={selectedArxivId}
            onSelectArxivId={setSelectedArxivId}
          />
        )}
        </div>
        <div
          style={{
            flexShrink: 0,
          }}
        >
          <ConceptDetail conceptKey={selectedConceptKey} />
        </div>
      </div>
    </div>
  );
};
