import React from "react";

interface SearchBarProps {
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
}

export const SearchBar: React.FC<SearchBarProps> = ({
  value,
  onChange,
  placeholder = "Search papers or concepts…",
}) => {
  return (
    <div style={{ marginBottom: "0.75rem" }}>
      <input
        type="text"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        style={{
          width: "100%",
          padding: "0.5rem 0.75rem",
          borderRadius: 4,
          border: "1px solid #ccc",
          fontSize: "0.95rem",
        }}
      />
    </div>
  );
};
