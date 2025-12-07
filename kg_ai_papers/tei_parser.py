# kg_ai_papers/tei_parser.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import xml.etree.ElementTree as ET


TEI_NS = {"tei": "http://www.tei-c.org/ns/1.0"}


@dataclass
class PaperSection:
    """
    Represents a logical section in a paper extracted from TEI.

    Attributes:
        id: Optional TEI xml:id or a synthetic ID.
        title: Section title (from <head>).
        level: Section level (1 = top-level, 2 = subsection, ...).
        text: Plain text contents of the section (excluding the <head>).
        path: Hierarchical path of titles from root to this section.
    """
    id: Optional[str]
    title: str
    level: int
    text: str
    path: List[str]


def _load_tei(tei_source: Path | str) -> ET.Element:
    """
    Load TEI XML from a file path or string containing XML.
    """
    if isinstance(tei_source, Path) or (isinstance(tei_source, str) and Path(tei_source).exists()):
        tree = ET.parse(str(tei_source))
        return tree.getroot()

    # Assume tei_source is a raw XML string
    return ET.fromstring(tei_source)


def _get_div_level(div_elem: ET.Element) -> int:
    """
    Infer a section level from TEI <div> nesting.
    Top-level <div> directly under <body> is level 1.
    """
    level = 0
    parent = div_elem
    while parent is not None:
        parent = parent.getparent() if hasattr(parent, "getparent") else None
        # xml.etree.ElementTree in stdlib does not support getparent(),
        # so as a fallback we approximate level using the @n attribute if present.
        # For now, we return 1 as a default and let callers refine if needed.
    # Fallback: try @n
    n_attr = div_elem.attrib.get("n")
    if n_attr and n_attr.isdigit():
        return int(n_attr)
    return 1


def extract_sections_from_tei(tei_source: Path | str) -> List[PaperSection]:
    """
    Extract logical sections from a TEI document produced by GROBID.

    Strategy (simple but robust enough for now):
      - Look under //tei:body//tei:div
      - For each <div>:
          * title = text of first <head> child (if any)
          * text = concatenated text of the <div>, excluding <head>
          * id = @xml:id if present, else None
          * level = 1 for now (we can refine later)
          * path = [title] (we can refine later)
    """
    root = _load_tei(tei_source)

    body = root.find(".//tei:body", TEI_NS)
    if body is None:
        return []

    sections: List[PaperSection] = []

    # xml.etree doesn't preserve parent links, so we won't try to compute
    # precise levels yet; we'll keep it simple.
    for div in body.findall(".//tei:div", TEI_NS):
        head = div.find("tei:head", TEI_NS)
        title = (head.text or "").strip() if head is not None else ""

        # Build text excluding <head>
        parts: List[str] = []
        for elem in div.iter():
            # Skip the head element and its descendants
            if elem is head:
                continue
            if elem.text:
                parts.append(elem.text.strip())
            if elem.tail:
                parts.append(elem.tail.strip())

        text = " ".join(p for p in parts if p)  # normalize spaces

        # GROBID typically uses xml:id, but xml.etree doesn't auto-handle XML NS
        sec_id = div.attrib.get("{http://www.w3.org/XML/1998/namespace}id")

        if not title and not text:
            continue  # skip empty divs

        section = PaperSection(
            id=sec_id,
            title=title,
            level=1,         # TODO: refine when we add better hierarchy
            text=text,
            path=[title] if title else [],
        )
        sections.append(section)

    return sections
