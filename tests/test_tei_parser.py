# tests/test_tei_parser.py

from kg_ai_papers.tei_parser import extract_sections_from_tei, PaperSection


def test_extract_sections_from_minimal_tei():
    tei_snippet = """
    <TEI xmlns="http://www.tei-c.org/ns/1.0">
      <text>
        <body>
          <div xml:id="sec1">
            <head>Introduction</head>
            <p>This is the introduction text.</p>
          </div>
          <div xml:id="sec2">
            <head>Methods</head>
            <p>We propose a new method.</p>
          </div>
        </body>
      </text>
    </TEI>
    """

    sections = extract_sections_from_tei(tei_snippet)
    assert len(sections) == 2

    intro, methods = sections

    assert isinstance(intro, PaperSection)
    assert intro.id == "sec1"
    assert intro.title == "Introduction"
    assert "introduction text" in intro.text
    assert intro.level == 1

    assert methods.id == "sec2"
    assert methods.title == "Methods"
    assert "new method" in methods.text
