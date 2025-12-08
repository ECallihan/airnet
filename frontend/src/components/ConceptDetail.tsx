import React, { useEffect, useState } from "react";

interface ConceptDetailProps {
  conceptKey: string | null;
}

interface ConceptApiResponse {
  id?: string;
  key?: string;
  label?: string;
  description?: string;
  summary?: string;
  attributes?: Record<string, any>;
  // Best-effort guesses for related papers fields:
  papers?: any[];
  connected_papers?: any[];
  neighbor_papers?: any[];
}

export const ConceptDetail: React.FC<ConceptDetailProps> = ({
  conceptKey,
}) => {
  const [data, setData] = useState<ConceptApiResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    if (!conceptKey) {
      setData(null);
      setError(null);
      setIsLoading(false);
      return;
    }

    let cancelled = false;

    const fetchConcept = async () => {
      setIsLoading(true);
      setError(null);

      try {
        const res = await fetch(
          `/api/concepts/${encodeURIComponent(conceptKey)}`
        );
        if (!res.ok) {
          const text = await res.text();
          throw new Error(
            `GET /concepts/${conceptKey} failed: ${res.status} ${text}`
          );
        }
        const json = (await res.json()) as ConceptApiResponse;
        if (!cancelled) {
          setData(json);
        }
      } catch (err) {
        if (!cancelled) {
          setError(err as Error);
          setData(null);
        }
      } finally {
        if (!cancelled) {
          setIsLoading(false);
        }
      }
    };

    fetchConcept();

    return () => {
      cancelled = true;
    };
  }, [conceptKey]);

  const cardStyle: React.CSSProperties = {
    border: "1px solid #e5e7eb",
    borderRadius: 6,
    padding: "0.75rem 0.9rem",
    backgroundColor: "#f9fafb",
  };

  if (!conceptKey) {
    return (
      <div style={cardStyle}>
        <h2
          style={{
            margin: 0,
            marginBottom: "0.35rem",
            fontSize: "0.95rem",
            fontWeight: 600,
          }}
        >
          Concept inspector
        </h2>
        <p style={{ margin: 0, color: "#6b7280", fontSize: "0.8rem" }}>
          Click a concept result on the left to inspect it here.
        </p>
      </div>
    );
  }

  if (isLoading) {
    return (
      <div style={cardStyle}>
        <h2
          style={{
            margin: 0,
            marginBottom: "0.35rem",
            fontSize: "0.95rem",
            fontWeight: 600,
          }}
        >
          Concept inspector
        </h2>
        <p style={{ margin: 0, fontSize: "0.8rem" }}>Loading concept…</p>
      </div>
    );
  }

  if (error) {
    return (
      <div style={cardStyle}>
        <h2
          style={{
            margin: 0,
            marginBottom: "0.35rem",
            fontSize: "0.95rem",
            fontWeight: 600,
          }}
        >
          Concept inspector
        </h2>
        <p style={{ margin: 0, color: "red", fontSize: "0.8rem" }}>
          Error loading concept “{conceptKey}”: {error.message}
        </p>
      </div>
    );
  }

  if (!data) {
    return (
      <div style={cardStyle}>
        <h2
          style={{
            margin: 0,
            marginBottom: "0.35rem",
            fontSize: "0.95rem",
            fontWeight: 600,
          }}
        >
          Concept inspector
        </h2>
        <p style={{ margin: 0, fontSize: "0.8rem" }}>
          No data for concept “{conceptKey}”.
        </p>
      </div>
    );
  }

  const name =
    data.label || data.key || data.id || conceptKey || "(unnamed concept)";
  const description = data.description || data.summary;
  const attributes = data.attributes || {};

  const papers =
    data.papers || data.connected_papers || data.neighbor_papers || [];

  return (
    <div style={cardStyle}>
      <h2
        style={{
          margin: 0,
          marginBottom: "0.35rem",
          fontSize: "0.95rem",
          fontWeight: 600,
        }}
      >
        Concept: {name}
      </h2>

      {description && (
        <p
          style={{
            marginTop: "0.15rem",
            marginBottom: "0.45rem",
            fontSize: "0.8rem",
            color: "#374151",
          }}
        >
          {description}
        </p>
      )}

      {Object.keys(attributes).length > 0 && (
        <div
          style={{
            marginBottom: "0.4rem",
          }}
        >
          <div
            style={{
              fontSize: "0.75rem",
              fontWeight: 600,
              marginBottom: "0.15rem",
              color: "#4b5563",
            }}
          >
            Attributes
          </div>
          <dl
            style={{
              margin: 0,
              display: "grid",
              gridTemplateColumns: "auto 1fr",
              columnGap: "0.5rem",
              rowGap: "0.1rem",
              fontSize: "0.75rem",
            }}
          >
            {Object.entries(attributes).map(([key, value]) => (
              <React.Fragment key={key}>
                <dt style={{ fontWeight: 500, color: "#6b7280" }}>{key}</dt>
                <dd style={{ margin: 0, color: "#374151" }}>
                  {typeof value === "number" || typeof value === "boolean"
                    ? String(value)
                    : typeof value === "string"
                    ? value
                    : JSON.stringify(value)}
                </dd>
              </React.Fragment>
            ))}
          </dl>
        </div>
      )}

      {papers.length > 0 && (
        <div>
          <div
            style={{
              fontSize: "0.75rem",
              fontWeight: 600,
              marginBottom: "0.2rem",
              color: "#4b5563",
            }}
          >
            Connected papers
          </div>
          <ul
            style={{
              listStyle: "disc",
              paddingLeft: "1.1rem",
              margin: 0,
              display: "flex",
              flexDirection: "column",
              gap: "0.1rem",
              fontSize: "0.75rem",
              color: "#374151",
            }}
          >
            {papers.slice(0, 8).map((p, idx) => {
              const pid =
                p.arxiv_id || p.paper_id || p.id || p.node_id || `paper-${idx}`;
              const title = p.title || "";
              return (
                <li key={pid}>
                  <span style={{ fontWeight: 500 }}>{pid}</span>
                  {title && <> — {title}</>}
                </li>
              );
            })}
            {papers.length > 8 && (
              <li style={{ color: "#6b7280" }}>
                …and {papers.length - 8} more
              </li>
            )}
          </ul>
        </div>
      )}
    </div>
  );
};
