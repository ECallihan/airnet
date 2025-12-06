# kg_ai_papers/parsing/tei_parser.py

from dataclasses import dataclass
from typing import List, Optional
from lxml import etree

from kg_ai_papers.models.section import Section
from kg_ai_papers.models.reference import Reference


NS = {"tei": "http://www.tei-c.org/ns/1.0"}


def parse_sections(tei_xml: str) -> List[Section]:
    """
    Extract sections from TEI XML produced by Grobid.
    """
    root = etree.fromstring(tei_xml.encode("utf-8"))
    sections: List[Section] = []

    # TEI structure: /TEI/text/body/div
    for div in root.xpath(".//tei:text/tei:body/tei:div", namespaces=NS):
        head_el = div.find("tei:head", namespaces=NS)
        title = head_el.text.strip() if head_el is not None and head_el.text else ""
        # level attribute often indicates section level
        level = div.get("{http://www.tei-c.org/ns/1.0}level") or div.get("n") or "1"

        # gather all paragraph text
        paras = div.findall(".//tei:p", namespaces=NS)
        text_parts = []
        for p in paras:
            t = "".join(p.itertext()).strip()
            if t:
                text_parts.append(t)
        text = "\n".join(text_parts)

        if text.strip():
            sections.append(Section(title=title, level=level, text=text))

    return sections


def parse_references(tei_xml: str) -> List[Reference]:
    root = etree.fromstring(tei_xml.encode("utf-8"))
    refs: List[Reference] = []

    # refs typically under back/listBibl/biblStruct
    bibl_structs = root.xpath(".//tei:text/tei:back//tei:listBibl/tei:biblStruct",
                              namespaces=NS)

    for bibl in bibl_structs:
        # Title
        title_el = bibl.find(".//tei:title", namespaces=NS)
        title = title_el.text.strip() if title_el is not None and title_el.text else ""

        # Authors
        authors = []
        for pers in bibl.findall(".//tei:author/tei:persName", namespaces=NS):
            first = " ".join([x for x in [
                pers.findtext("tei:forename", namespaces=NS),
                pers.findtext("tei:surname", namespaces=NS)
            ] if x])
            if first:
                authors.append(first)

        # Year
        date_el = bibl.find(".//tei:date", namespaces=NS)
        year = None
        if date_el is not None:
            when = date_el.get("when")
            if when and len(when) >= 4 and when[:4].isdigit():
                year = int(when[:4])

        # DOI
        doi = None
        idno_doi = bibl.find(".//tei:idno[@type='DOI']", namespaces=NS)
        if idno_doi is not None and idno_doi.text:
            doi = idno_doi.text.strip()

        # Raw reference text
        raw_text_parts = []
        for t in bibl.itertext():
            raw_text_parts.append(t.strip())
        raw = " ".join([p for p in raw_text_parts if p])

        refs.append(
            Reference(
                title=title,
                authors=authors,
                year=year,
                doi=doi,
                raw=raw,
            )
        )

    return refs
