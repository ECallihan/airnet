# kg_ai_papers/tei_parser.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Union
import re
import xml.etree.ElementTree as ET

from kg_ai_papers.models.reference import Reference

# Namespaces
TEI_NS_DEFAULT = "http://www.tei-c.org/ns/1.0"
TEI_NS = {"tei": TEI_NS_DEFAULT}
XML_NS = "{http://www.w3.org/XML/1998/namespace}"


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


@dataclass
class PaperSection:
    """
    Lightweight representation of a logical section in a TEI document.

    This is intentionally minimal – we only capture what the ingestion /
    concept-extraction pipeline currently needs.
    """

    id: Optional[str]
    title: Optional[str]
    level: int
    text: str
    path: List[str]


# ---------------------------------------------------------------------------
# TEI parsing helpers
# ---------------------------------------------------------------------------


def _load_tei_content(tei: Union[str, Path]) -> str:
    """
    Accept either a TEI XML string or a filesystem path and return the XML text.

    Heuristics:
    - If `tei` is a Path, read it.
    - If `tei` is a string that contains '<TEI', treat it as XML.
    - Otherwise, if it looks like a path on disk, read it.
    - As a last resort, treat the string as XML.
    """
    if isinstance(tei, Path):
        return tei.read_text(encoding="utf-8")

    s = str(tei)
    if "<TEI" in s or "<tei:" in s:
        return s

    p = Path(s)
    if p.exists():
        return p.read_text(encoding="utf-8")

    return s


def _parse_tei_any(tei: Union[str, Path]) -> Tuple[ET.Element, Optional[dict]]:
    """
    Parse TEI from XML string or path and detect the namespace.

    Returns (root_element, namespace_dict_or_None).
    """
    xml_text = _load_tei_content(tei)
    root = ET.fromstring(xml_text)

    if root.tag.startswith("{"):
        uri = root.tag.split("}")[0].strip("{")
        ns = {"tei": uri}
    else:
        ns = None

    return root, ns


def _iter_section_divs(root: ET.Element, ns: Optional[dict]) -> Iterable[ET.Element]:
    """
    Iterate over section-like <div> elements in the TEI <body>.

    We support both namespaced and non-namespaced TEI.
    """
    if ns:
        # Any <div> inside <text>/<body>
        return root.findall(".//tei:text/tei:body//tei:div", ns)
    return root.findall(".//text/body//div")


def _div_title(div: ET.Element, ns: Optional[dict]) -> Optional[str]:
    """
    Extract the title of a section from a <div>, typically from its <head>.
    """
    if ns:
        head = div.find("tei:head", ns)
    else:
        head = div.find("head")

    if head is not None and head.text:
        return head.text.strip()
    return None


def _normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def _div_text(div: ET.Element) -> str:
    """
    Aggregate all descendant text from a <div> into a single normalized string.
    """
    pieces: List[str] = []
    for t in div.itertext():
        if t:
            pieces.append(t)
    return _normalize_whitespace(" ".join(pieces))


# ---------------------------------------------------------------------------
# Section extraction
# ---------------------------------------------------------------------------


def extract_sections_from_tei(tei_path_or_xml: Union[str, Path]) -> List[PaperSection]:
    """
    Extract a flat list of top-level sections from a TEI XML document or file.

    Parameters
    ----------
    tei_path_or_xml:
        Either a TEI XML string or a path to the TEI XML produced by GROBID.

    Returns
    -------
    List[PaperSection]
        One PaperSection per <div> in the TEI body that has any textual
        content. For now we treat everything as level=1 and use a simple
        one-element `path` based on the section title.
    """
    root, ns = _parse_tei_any(tei_path_or_xml)
    sections: List[PaperSection] = []

    for div in _iter_section_divs(root, ns):
        # xml:id or plain id
        sec_id = div.get(f"{XML_NS}id") or div.get("id")
        title = _div_title(div, ns)
        text = _div_text(div)

        if not title and not text:
            # Skip completely empty structures
            continue

        section = PaperSection(
            id=sec_id,
            title=title,
            level=1,  # Flat for now; can refine hierarchy later
            text=text,
            path=[title] if title else [],
        )
        sections.append(section)

    return sections


# ---------------------------------------------------------------------------
# Reference extraction
# ---------------------------------------------------------------------------

_ARXIV_ID_RE = re.compile(r"(\d{4}\.\d{4,5})(v\d+)?", re.IGNORECASE)


def _findall_bibl_structs(root: ET.Element, ns: Optional[dict]) -> List[ET.Element]:
    if ns:
        return root.findall(".//tei:listBibl/tei:biblStruct", ns)
    return root.findall(".//listBibl/biblStruct")


def _get_bibl_title(bibl: ET.Element, ns: Optional[dict]) -> Optional[str]:
    # Prefer analytic/title, then monogr/title
    if ns:
        el = bibl.find(".//tei:analytic/tei:title", ns)
        if el is None:
            el = bibl.find(".//tei:monogr/tei:title", ns)
    else:
        el = bibl.find(".//analytic/title")
        if el is None:
            el = bibl.find(".//monogr/title")

    if el is not None and el.text:
        return el.text.strip()
    return None


def _get_bibl_authors(bibl: ET.Element, ns: Optional[dict]) -> List[str]:
    """
    Collect authors as "Surname, Forename" strings.
    """
    authors: List[str] = []

    if ns:
        pers_elems: List[ET.Element] = []
        pers_elems.extend(bibl.findall(".//tei:analytic/tei:author/tei:persName", ns))
        pers_elems.extend(bibl.findall(".//tei:monogr/tei:author/tei:persName", ns))
    else:
        pers_elems = []
        pers_elems.extend(bibl.findall(".//analytic/author/persName"))
        pers_elems.extend(bibl.findall(".//monogr/author/persName"))

    for pers in pers_elems:
        if ns:
            fn = pers.find("tei:forename", ns)
            sn = pers.find("tei:surname", ns)
        else:
            fn = pers.find("forename")
            sn = pers.find("surname")

        forename = fn.text.strip() if (fn is not None and fn.text) else None
        surname = sn.text.strip() if (sn is not None and sn.text) else None

        # Format as "Surname, Forename" to match tests
        if surname and forename:
            authors.append(f"{surname}, {forename}")
        elif surname:
            authors.append(surname)
        elif forename:
            authors.append(forename)

    return authors


def _get_bibl_year(bibl: ET.Element, ns: Optional[dict]) -> Optional[int]:
    """
    Extract a 4-digit publication year from imprint/date or date elements.
    """
    if ns:
        date_el = bibl.find(".//tei:imprint/tei:date", ns)
        if date_el is None:
            date_el = bibl.find(".//tei:date", ns)
    else:
        date_el = bibl.find(".//imprint/date") or bibl.find(".//date")

    if date_el is None:
        return None

    val = (
        date_el.get("when")
        or date_el.get("when-iso")
        or (date_el.text.strip() if date_el.text else None)
    )
    if not val:
        return None

    m = re.search(r"(\d{4})", val)
    return int(m.group(1)) if m else None


def _get_bibl_ids(bibl: ET.Element, ns: Optional[dict]) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract (doi, arxiv_id) from <idno> elements, normalizing arXiv IDs.
    """
    if ns:
        idnos = bibl.findall(".//tei:idno", ns)
    else:
        idnos = bibl.findall(".//idno")

    doi: Optional[str] = None
    arxiv_id: Optional[str] = None

    for idno in idnos:
        id_type = (idno.get("type") or "").lower()
        text = (idno.text or "").strip()
        if not text:
            continue

        if id_type == "doi" and doi is None:
            doi = text

        elif id_type.startswith("arxiv") and arxiv_id is None:
            # Normalize "arXiv:2101.00001" → "2101.00001"
            m = _ARXIV_ID_RE.search(text)
            arxiv_id = m.group(1) if m else text

    return doi, arxiv_id


def extract_references_from_tei(tei_xml_or_path: Union[str, Path]) -> List[Reference]:
    """
    Parse TEI XML and extract a list of Reference objects from <listBibl>/<biblStruct>.

    We try to be robust to:
    - TEI default namespaces (xmlns="http://www.tei-c.org/ns/1.0")
    - Titles in <analytic><title> or <monogr><title>
    - Authors encoded as <persName><forename>/<surname>
    - Dates with @when/@when-iso or text
    - DOIs and arXiv IDs inside <idno type="DOI"> / <idno type="arXiv">
    """
    root, ns = _parse_tei_any(tei_xml_or_path)

    references: List[Reference] = []

    for idx, bibl in enumerate(_findall_bibl_structs(root, ns)):
        title = _get_bibl_title(bibl, ns)
        authors = _get_bibl_authors(bibl, ns)
        year = _get_bibl_year(bibl, ns)
        doi, arxiv_id = _get_bibl_ids(bibl, ns)

        # Raw TEI snippet and a normalized raw-text version
        raw_xml = ET.tostring(bibl, encoding="unicode")
        raw_text = _normalize_whitespace(" ".join(bibl.itertext())) or raw_xml

        # Derive a stable key:
        # 1. arXiv ID if present
        # 2. title (+ year) if present
        # 3. DOI
        # 4. synthetic fallback
        if arxiv_id:
            key = arxiv_id
        elif title:
            if year is not None:
                key = f"{title} ({year})"
            else:
                key = title
        elif doi:
            key = doi
        else:
            key = f"ref-{idx + 1}"

        ref = Reference(
            key=key,
            raw=raw_text,
            title=title,
            year=year,
            doi=doi,
            arxiv_id=arxiv_id,
            authors=authors,
        )
        references.append(ref)

    return references


# ---------------------------------------------------------------------------
# Backwards-compatible aliases expected by other modules
# ---------------------------------------------------------------------------


def parse_sections(tei_path: Path | str) -> List[PaperSection]:
    """
    Compatibility wrapper around extract_sections_from_tei.

    Accepts either a TEI XML string or a path.
    """
    return extract_sections_from_tei(tei_path)


def parse_references(tei_path: Path | str) -> List[str]:
    """
    Compatibility wrapper around extract_references_from_tei.

    Returns a simple list of identifier strings, preferring:
    - arxiv_id
    - then DOI
    - then Reference.key
    """
    refs = extract_references_from_tei(tei_path)
    ids: List[str] = []
    for r in refs:
        if r.arxiv_id:
            ids.append(r.arxiv_id)
        elif r.doi:
            ids.append(r.doi)
        else:
            ids.append(r.key)
    return ids
