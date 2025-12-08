# tests/test_tei_references.py

from kg_ai_papers.tei_parser import extract_references_from_tei
from kg_ai_papers.models.reference import Reference


def test_extract_references_basic():
    # Minimal synthetic TEI with two references
    tei_xml = """<?xml version="1.0" encoding="UTF-8"?>
    <TEI xmlns="http://www.tei-c.org/ns/1.0">
      <text>
        <back>
          <listBibl>
            <biblStruct>
              <analytic>
                <title>First cited paper</title>
                <author>
                  <persName>
                    <forename>Alice</forename>
                    <surname>Smith</surname>
                  </persName>
                </author>
              </analytic>
              <monogr>
                <imprint>
                  <date when="2020-05-01"/>
                </imprint>
              </monogr>
              <idno type="DOI">10.1000/xyz123</idno>
            </biblStruct>

            <biblStruct>
              <analytic>
                <title>Second cited paper</title>
              </analytic>
              <idno type="arXiv">arXiv:2101.00001</idno>
            </biblStruct>
          </listBibl>
        </back>
      </text>
    </TEI>
    """

    refs = extract_references_from_tei(tei_xml)

    # We should get exactly two references
    assert len(refs) == 2
    assert all(isinstance(r, Reference) for r in refs)

    first, second = refs

    # First reference: DOI + year + author formatting
    assert first.title == "First cited paper"
    assert first.doi == "10.1000/xyz123"
    assert first.year == 2020
    assert any("Smith, Alice" == a for a in first.authors)

    # Second reference: arXiv ID normalized
    assert second.title == "Second cited paper"
    assert second.arxiv_id == "2101.00001"
    assert second.doi is None

    # Keys: should be derived from identifiers
    keys = {r.key for r in refs}
    assert "2101.00001" in keys
    assert any(k.startswith("First cited paper") or k == "First cited paper (2020)" for k in keys)
