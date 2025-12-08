import React, { useState } from "react";
import { usePaperInfluence } from "../api/hooks";
import type { InfluenceNeighbor, PaperInfluenceResponse } from "../api/types";

type PaperInfluenceExplorerProps = {
  arxivId: string;
  onSelectArxivId?: (nextArxivId: string) => void;
};


export const PaperInfluenceExplorer: React.FC<PaperInfluenceExplorerProps> = ({
  arxivId,
  onSelectArxivId,
}) => {

  const [activeTab, setActiveTab] = useState<"incoming" | "outgoing">(
    "incoming",
  );

  const { data, isLoading, error } = usePaperInfluence(arxivId);

  const neighbors: InfluenceNeighbor[] =
    activeTab === "incoming"
      ? data?.incoming ?? []
      : data?.outgoing ?? [];

  const paper = data?.paper;

  return (
    <div
      style={{
        borderRadius: 12,
        border: "1px solid #e2e8f0",
        padding: 16,
        backgroundColor: "#ffffff",
        boxShadow: "0 4px 10px rgba(15, 23, 42, 0.06)",
      }}
    >
      {/* Header */}
      <div style={{ marginBottom: 16 }}>
        <div style={{ fontSize: 12, color: "#64748b", marginBottom: 4 }}>
          Paper Influence Explorer
        </div>
        <div style={{ display: "flex", alignItems: "baseline", gap: 8 }}>
          <h3 style={{ fontSize: 18, fontWeight: 600, margin: 0 }}>
            {paper?.title || `arXiv:${arxivId}`}
          </h3>
          <span
            style={{
              fontSize: 12,
              padding: "2px 6px",
              borderRadius: 999,
              backgroundColor: "#eff6ff",
              color: "#1d4ed8",
            }}
          >
            {paper?.arxiv_id ?? arxivId}
          </span>
        </div>
        {paper?.abstract && (
          <p
            style={{
              marginTop: 8,
              marginBottom: 0,
              fontSize: 13,
              color: "#475569",
            }}
          >
            {truncate(paper.abstract, 260)}
          </p>
        )}
      </div>

      {/* Loading / error states */}
      {isLoading && (
        <div style={{ padding: 8, fontSize: 13, color: "#64748b" }}>
          Loading influence graph…
        </div>
      )}

      {error && !isLoading && (
        <div
          style={{
            padding: 12,
            marginBottom: 12,
            borderRadius: 8,
            backgroundColor: "#fef2f2",
            color: "#b91c1c",
            fontSize: 13,
          }}
        >
          Error: {(error as Error).message}
        </div>
      )}

      {!isLoading && !error && data && (
        <>
          {/* Tabs */}
          <div
            style={{
              display: "flex",
              gap: 8,
              marginBottom: 12,
              borderBottom: "1px solid #e2e8f0",
            }}
          >
            <TabButton
              active={activeTab === "incoming"}
              label={`Incoming (${data.incoming.length})`}
              onClick={() => setActiveTab("incoming")}
            />
            <TabButton
              active={activeTab === "outgoing"}
              label={`Outgoing (${data.outgoing.length})`}
              onClick={() => setActiveTab("outgoing")}
            />
          </div>

          {/* Empty state */}
          {neighbors.length === 0 && (
            <div
              style={{
                padding: 16,
                fontSize: 13,
                color: "#64748b",
                fontStyle: "italic",
              }}
            >
              No {activeTab} influence edges found for this paper.
            </div>
          )}

          {/* Table of neighbors */}
          {neighbors.length > 0 && (
            <div style={{ overflowX: "auto" }}>
              <table
                style={{
                  width: "100%",
                  borderCollapse: "collapse",
                  fontSize: 13,
                }}
              >
                <thead>
                  <tr
                    style={{
                      textAlign: "left",
                      borderBottom: "1px solid #e2e8f0",
                    }}
                  >
                    <th style={thStyle}>Title</th>
                    <th style={thStyle}>arXiv ID</th>
                    <th style={thStyle}>Direction</th>
                    <th style={thStyle}>Relation</th>
                    <th style={thStyle}>Weight</th>
                  </tr>
                </thead>
                <tbody>
                {neighbors.map((n) => {
                    const clickable = !!onSelectArxivId && !!n.paper.arxiv_id;

                    const handleClick = () => {
                    if (clickable) {
                        onSelectArxivId!(n.paper.arxiv_id);
                    }
                    };

                    return (
                    <tr
                        key={n.paper.id}
                        onClick={handleClick}
                        title={
                        clickable
                            ? `Explore ${n.paper.arxiv_id} in the influence explorer`
                            : undefined
                        }
                        style={{
                        borderBottom: "1px solid #f1f5f9",
                        cursor: clickable ? "pointer" : "default",
                        backgroundColor: clickable ? "#ffffff" : "#ffffff",
                        }}
                    >
                        <td style={tdStyle}>
                        <div
                            style={{
                            fontWeight: 500,
                            textDecoration: clickable ? "underline" : "none",
                            textDecorationStyle: clickable ? "dotted" : "solid",
                            }}
                        >
                            {n.paper.title || "(untitled)"}
                        </div>
                        <div
                            style={{
                            fontSize: 11,
                            color: "#94a3b8",
                            marginTop: 2,
                            }}
                        >
                            {n.paper.id}
                        </div>
                        </td>
                        <td style={tdStyle}>{n.paper.arxiv_id}</td>
                        <td style={tdStyle}>
                        <DirectionBadge direction={n.direction} />
                        </td>
                        <td style={tdStyle}>
                        {n.relation ? (
                            <span
                            style={{
                                fontSize: 11,
                                padding: "2px 6px",
                                borderRadius: 999,
                                backgroundColor: "#f1f5f9",
                                color: "#0f172a",
                            }}
                            >
                            {n.relation}
                            </span>
                        ) : (
                            <span style={{ fontSize: 11, color: "#cbd5e1" }}>—</span>
                        )}
                        </td>
                        <td style={tdStyle}>
                        {typeof n.weight === "number" ? (
                            <WeightBadge weight={n.weight} />
                        ) : (
                            <span style={{ fontSize: 11, color: "#cbd5e1" }}>n/a</span>
                        )}
                        </td>
                    </tr>
                    );
                })}
                </tbody>
              </table>
            </div>
          )}
        </>
      )}
    </div>
  );
};

// ---------- Small helper subcomponents & utils ----------

const thStyle: React.CSSProperties = {
  padding: "8px 6px",
  fontSize: 11,
  textTransform: "uppercase",
  letterSpacing: 0.5,
  color: "#94a3b8",
};

const tdStyle: React.CSSProperties = {
  padding: "8px 6px",
  verticalAlign: "top",
};

type TabButtonProps = {
  active: boolean;
  label: string;
  onClick: () => void;
};

const TabButton: React.FC<TabButtonProps> = ({ active, label, onClick }) => (
  <button
    type="button"
    onClick={onClick}
    style={{
      border: "none",
      backgroundColor: "transparent",
      padding: "8px 10px",
      fontSize: 13,
      cursor: "pointer",
      borderBottom: active ? "2px solid #1d4ed8" : "2px solid transparent",
      color: active ? "#1d4ed8" : "#64748b",
      fontWeight: active ? 600 : 500,
    }}
  >
    {label}
  </button>
);

type DirectionBadgeProps = {
  direction: InfluenceNeighbor["direction"];
};

const DirectionBadge: React.FC<DirectionBadgeProps> = ({ direction }) => {
  let label = direction;
  let icon = "↔";
  if (direction === "in") {
    label = "Influences this paper";
    icon = "⬅";
  } else if (direction === "out") {
    label = "Cited by this paper";
    icon = "➡";
  } else if (direction === "both") {
    label = "Mutual";
    icon = "↔";
  } else if (direction === "unknown") {
    label = "Unknown";
    icon = "?";
  }

  return (
    <span
      title={label}
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: 4,
        fontSize: 11,
        padding: "2px 6px",
        borderRadius: 999,
        backgroundColor: "#eff6ff",
        color: "#1d4ed8",
      }}
    >
      <span>{icon}</span>
      <span>{direction}</span>
    </span>
  );
};

type WeightBadgeProps = {
  weight: number;
};

const WeightBadge: React.FC<WeightBadgeProps> = ({ weight }) => {
  const clamped = Math.max(0, Math.min(1, weight));
  const percent = Math.round(clamped * 100);

  return (
    <div style={{ minWidth: 80 }}>
      <div
        style={{
          height: 6,
          borderRadius: 999,
          backgroundColor: "#e2e8f0",
          overflow: "hidden",
          marginBottom: 2,
        }}
      >
        <div
          style={{
            width: `${percent}%`,
            height: "100%",
            borderRadius: 999,
            background:
              "linear-gradient(90deg, rgba(59,130,246,1), rgba(45,212,191,1))",
          }}
        />
      </div>
      <div style={{ fontSize: 11, color: "#64748b" }}>{weight.toFixed(3)}</div>
    </div>
  );
};

function truncate(text: string, max: number): string {
  if (text.length <= max) return text;
  return text.slice(0, max - 1) + "…";
}
