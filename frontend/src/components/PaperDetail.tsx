import React from "react";
import { usePaper } from "../api/hooks";

interface PaperDetailProps {
  arxivId: string | null;
}

export const PaperDetail: React.FC<PaperDetailProps> = ({ arxivId }) => {
  const { data, isLoading, error } = usePaper(arxivId);

  if (!arxivId) return <div>Select a paper to see details.</div>;
  if (isLoading) return <div>Loading paper…</div>;
  if (error) return <div style={{ color: "red" }}>Error: {(error as Error).message}</div>;
  if (!data) return <div>No data.</div>;

  const { paper, concepts, neighbors } = data;

  return (
    <div>
      <h2 style={{ marginTop: 0 }}>
        {paper.title ?? paper.arxiv_id}
      </h2>
      <div style={{ marginBottom: "0.5rem", color: "#555" }}>
        <strong>arXiv:</strong> {paper.arxiv_id}
      </div>
      {paper.abstract && (
        <p style={{ fontSize: "0.9rem", lineHeight: 1.5 }}>{paper.abstract}</p>
      )}

      <h3>Concepts</h3>
      {concepts.length === 0 ? (
        <div style={{ color: "#777" }}>No concepts attached.</div>
      ) : (
        <ul style={{ paddingLeft: "1.25rem" }}>
          {concepts.map((c) => (
            <li key={c.id}>
              {c.label}
              {c.weight != null && (
                <span style={{ marginLeft: 4, fontSize: "0.8rem", color: "#555" }}>
                  ({c.weight.toFixed(2)})
                </span>
              )}
            </li>
          ))}
        </ul>
      )}

      <h3>Neighbors</h3>
      {neighbors.length === 0 ? (
        <div style={{ color: "#777" }}>No neighbors at this depth.</div>
      ) : (
        <ul style={{ paddingLeft: "1.25rem" }}>
          {neighbors.map((n) => (
            <li key={n.id}>
              {n.label ?? n.id}{" "}
              <span style={{ fontSize: "0.8rem", color: "#555" }}>
                [{n.direction}
                {n.relation && ` · ${n.relation}`}]
              </span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
};
