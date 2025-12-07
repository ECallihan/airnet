from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Set
import re
import xml.etree.ElementTree as ET


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


@dataclass
class PaperSection:
    """
    Simple representation of a logical section extracted from a TEI document.

    Attributes
    ----------
    id:
        Optional xml:id of the <div> in TEI (if present).
    title:
        Section title, typically from a <head> element.
    level:
        Nesting level (1 for top-level section, 2 for subsections, etc.).
    text:
        Plain text content of the section (all descendant text concatenated).
    path:
        Hierarchical path of titles leading to this section, e.g.
        ["Introduction"], ["Methods", "Data"], etc.
    """

    id: Optional[str]
    title: str
    level: int
    text: str
    path: List[str]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_tei(tei_path: Path | str) -> ET.Element:
    """Parse a TEI XML file and return the root element."""
    tei_path = Path(tei_path)
    tree = ET.parse(tei_path)
    return tree.getroot()


def _get_ns(root: ET.Element) -> str:
    """
    Extract the TEI namespace URI from the root tag.

    TEI elements usually look like '{http://www.tei-c.org/ns/1.0}TEI'.
    """
    if root.tag.startswith("{"):
        return root.tag.split("}", 1)[0][1:]
    # Fallback: no namespace
    return ""


def _iter_divs_with_heads(root: ET.Element) -> Iterable[tuple[ET.Element, List[str]]]:
    """
    Walk the TEI body <div> hierarchy and yield (div_element, path_of_titles).

    path_of_titles is the list of titles (heads) from the root down to this div.
    """
    ns = _get_ns(root)
    ns_prefix = f"{{{ns}}}" if ns else ""

    body = root.find(f".//{ns_prefix}text/{ns_prefix}body")
    if body is None:
        return  # type: ignore[return-value]

    def walk(div: ET.Element, parent_path: List[str]):
        # Try to find a <head> as the section title
        head = div.find(f"{ns_prefix}head")
        if head is not None and (head.text or "").strip():
            title = (head.text or "").strip()
        else:
            # Fallback if no <head> – treat as "Untitled"
            title = "Untitled section"

        path = parent_path + [title]
        yield div, path

        # Recurse into nested <div>
        for child_div in div.findall(f"{ns_prefix}div"):
            yield from walk(child_div, path)

    for top_div in body.findall(f"{ns_prefix}div"):
        yield from walk(top_div, [])


def _normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def _extract_arxiv_id(text: str) -> Optional[str]:
    """
    Extract a normalized arXiv ID from a string, if present.

    Handles strings like:
      - "arXiv:2401.01234"
      - "arxiv:2401.01234v2 [cs.LG]"
      - "https://arxiv.org/abs/2401.01234v1"

    Returns the base ID without version, e.g. "2401.01234", or None.
    """
    if not text:
        return None

    # Look for arXiv-style IDs
    # New style: 4 digits '.' 4–5 digits
    m = re.search(r"(\d{4}\.\d{4,5})(v\d+)?", text)
    if not m:
        return None
    return m.group(1)


# ---------------------------------------------------------------------------
# Public API: Section extraction
# ---------------------------------------------------------------------------


def extract_sections_from_tei(tei_path: Path | str) -> List[PaperSection]:
    """
    Extract logical sections from a TEI file as a flat list of PaperSection.

    This is intentionally simple and robust rather than TEI-spec-perfect:
    - It looks under <text>/<body> for nested <div> elements.
    - Each <div> with a <head> becomes a section.
    - The section text is all descendant text of that <div>.
    """
    root = _parse_tei(tei_path)
    ns = _get_ns(root)
    ns_prefix = f"{{{ns}}}" if ns else ""

    sections: List[PaperSection] = []

    for div, path in _iter_divs_with_heads(root):
        # xml:id, if present
        xml_id = div.get("{http://www.w3.org/XML/1998/namespace}id")

        # Grab all descendant text within this div
        text_chunks = [t for t in div.itertext()]
        full_text = _normalize_whitespace(" ".join(text_chunks))

        section = PaperSection(
            id=xml_id,
            title=path[-1],
            level=len(path),
            text=full_text,
            path=path,
        )
        sections.append(section)

    return sections


# ---------------------------------------------------------------------------
# Public API: Reference / influence extraction
# ---------------------------------------------------------------------------


def extract_references_from_tei(tei_path: Path | str) -> List[str]:
    """
    Extract a list of *normalized arXiv IDs* for references from a TEI file.

    Strategy:
    - Look under <listBibl>/<biblStruct>.
    - Within each <biblStruct>, prefer <idno type="arXiv"> or similar.
    - As a fallback, look for URLs in @target (e.g. ptr/@target) that contain
      arxiv.org/abs/..., and parse the ID from there.
    - Normalize IDs to the base arXiv form without version (e.g. "2401.01234").
    - Return a de-duplicated list (order of first appearance is preserved).
    """
    root = _parse_tei(tei_path)
    ns = _get_ns(root)
    ns_prefix = f"{{{ns}}}" if ns else ""

    # Find all bibliographic structures
    bibl_structs = root.findall(f".//{ns_prefix}listBibl//{ns_prefix}biblStruct")

    seen: Set[str] = set()
    results: List[str] = []

    for bibl in bibl_structs:
        arxiv_id: Optional[str] = None

        # 1) Look for <idno type="arXiv"> or similar
        for idno in bibl.findall(f".//{ns_prefix}idno"):
            id_type = (idno.get("type") or "").lower()
            text = (idno.text or "").strip()
            if "arxiv" in id_type or "arxiv" in text.lower():
                arxiv_id = _extract_arxiv_id(text)
                if arxiv_id:
                    break

        # 2) Fallback: look at ptr/@target for arxiv.org/abs/...
        if not arxiv_id:
            for ptr in bibl.findall(f".//{ns_prefix}ptr"):
                target = ptr.get("target") or ""
                if "arxiv.org" in target.lower():
                    arxiv_id = _extract_arxiv_id(target)
                    if arxiv_id:
                        break

        if arxiv_id and arxiv_id not in seen:
            seen.add(arxiv_id)
            results.append(arxiv_id)

    return results
