"""Guards on the generated description document.

The PDF is submission collateral, and it carries the same numbers as the deck and
the README. Two ways it can go wrong, both of which are silent:

  1. A figure drifts from the source it is generated from.
  2. A character is missing from base-14 Helvetica and prints as a hollow box.

The second one is the nastier of the two. reportlab emits a notdef silently -- no
warning, no exception -- and a PDF full of replacement boxes still opens, still
paginates, and still looks plausible in a thumbnail. It was found by inspecting a
content stream, not by reading the file, which is why it is worth a test.

Everything here skips when reportlab/pypdf are absent, so a clean clone with only
numpy still runs the suite green.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

# The generators derive their figures rather than hardcoding them, so the
# tests do too -- otherwise the guard hardcodes the very number it guards.
sys.path.insert(0, str(ROOT / "scripts"))
from _testcount import count_tests  # noqa: E402
SCRIPT = ROOT / "scripts" / "make_description_doc.py"
PDF = ROOT / "docs" / "Snapdragoon-description.pdf"
DOCX = ROOT / "docs" / "Snapdragoon-description.docx"

# reportlab draws with base-14 Helvetica. These are outside it: U+2022 and
# U+25A0 come out as 0x7F notdef, and the emoji is silently swapped into
# ZapfDingbats where it lands on a filled black square. Each of these shipped in
# an earlier draft of the file.
BANNED_IN_PDF = ("\u2022", "\u25a0", "\U0001f409", "\u2248", "\u25cf")


@pytest.fixture(scope="module")
def built(tmp_path_factory) -> Path:
    """Build the document fresh, so the test never reads a stale artefact."""
    pytest.importorskip("reportlab", reason="reportlab is not a project dependency")
    pytest.importorskip("docx", reason="python-docx is not a project dependency")
    out = tmp_path_factory.mktemp("desc")
    # The script writes to a fixed path, so build then copy out of it.
    subprocess.run([sys.executable, str(SCRIPT)], cwd=ROOT, check=True,
                   capture_output=True)
    assert PDF.is_file(), f"{PDF.name} was not produced"
    local = out / PDF.name
    local.write_bytes(PDF.read_bytes())
    return local


def _streams(pdf: Path) -> str:
    """Decoded content streams. A notdef appears as the escape sequence \\177."""
    pypdf = pytest.importorskip("pypdf")
    reader = pypdf.PdfReader(str(pdf))
    return "".join(p.get_contents().get_data().decode("latin-1")
                   for p in reader.pages)


def test_pdf_has_no_notdef_glyphs(built: Path) -> None:
    """Every character must resolve to a real glyph.

    A PDF string escapes 0x7F as the four characters backslash-1-7-7, so that
    literal in a content stream is precisely "a character the font lacks".
    """
    streams = _streams(built)
    count = streams.count("\\177")
    assert count == 0, (
        f"{count} notdef glyph(s) in the PDF. reportlab does not warn about "
        f"these; they render as hollow boxes. Replace the character with one "
        f"base-14 Helvetica actually has, or a ZapfDingbats substitution."
    )


def test_pdf_source_avoids_known_unsupported_characters() -> None:
    """Check the generator's source, not the extracted text.

    Extraction cannot make this call: a broken U+25A0 and the deliberate
    ZapfDingbats bullet both come back as U+25A0, so a text-level ban would
    reject the fix it was written to enforce. The source is unambiguous -- and it
    is where the mistake is actually made.

    Only the PDF half of the file is scanned. The dragon is allowed below
    build_docx, because Word and Google Docs fall back to Segoe UI Emoji and draw
    it correctly there.
    """
    source = SCRIPT.read_text()
    pdf_half = source.split("def build_docx")[0]
    for ch in BANNED_IN_PDF:
        assert ch not in pdf_half, (
            f"{ch!r} (U+{ord(ch):04X}) appears in the PDF half of "
            f"{SCRIPT.name}. It is not in base-14 Helvetica: it renders as a "
            f"hollow box, or -- for the emoji -- as a ZapfDingbats filled "
            f"square, since reportlab silently substitutes the font."
        )


def test_symbols_that_do_work_are_kept(built: Path) -> None:
    """The other direction: a guard that bans everything is not a good guard.

    U+2713, U+2717 and U+2192 are outside Helvetica too, but reportlab
    substitutes ZapfDingbats and the correct glyph appears. They carry meaning
    here, so a future edit should not remove them on a false alarm.
    """
    pytest.importorskip("pypdf")
    from pypdf import PdfReader

    text = "".join(p.extract_text() or "" for p in PdfReader(str(built)).pages)
    for ch in ("\u2713", "\u2717", "\u2192"):
        assert ch in text, (
            f"{ch!r} (U+{ord(ch):04X}) went missing. It is safe: reportlab "
            f"substitutes ZapfDingbats, where it maps to a real glyph."
        )


def test_figures_match_the_audit_source(built: Path) -> None:
    """The document's headline numbers must still come from the audit files."""
    pytest.importorskip("pypdf")
    from pypdf import PdfReader

    text = "".join(p.extract_text() or "" for p in PdfReader(str(built)).pages)

    total = 0
    for name in ("accessibility-audit-qualcomm-ai-lab.md",
                 "accessibility-audit-unstop-ai-lab-challenge.md"):
        path = ROOT / name
        if not path.is_file():
            continue
        total += len(re.findall(
            r"^\|\s*\d+\s*\|\s*(?:Blocker|Serious|Moderate|Minor)\s*\|",
            path.read_text(), re.M))

    assert total > 0, "no audit findings parsed; the checks below would be vacuous"
    assert f"{total} issues" in text, (
        f"the audits contain {total} issues but the document does not say so")
    assert "190" in text and "400" in text, "the ASR range is missing"
    assert str(count_tests()) in text, (
        f"the suite collects {count_tests()} tests but the document does not "
        f"say so -- the generator derives this, so a mismatch means collection "
        f"failed and the fallback was used")


def test_docx_is_readable_and_carries_the_numbers(built: Path) -> None:
    pytest.importorskip("docx")
    from docx import Document

    doc = Document(str(DOCX))
    text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    assert "Snapdragoon" in text
    assert "challalokesh08" in text, "the repository URL is missing"
    assert str(count_tests()) in text
    assert "190" in text and "400" in text
    assert doc.tables, "the measured-results table is missing"


def test_docx_keeps_the_dragon_the_pdf_drops(built: Path) -> None:
    """The two files disagree on the emoji on purpose.

    The PDF cannot draw it; Word and Google Docs both fall back to Segoe UI Emoji
    and can. Asserting the asymmetry keeps a future edit from 'fixing' the DOCX
    in the wrong direction.
    """
    pytest.importorskip("docx")
    from docx import Document

    doc = Document(str(DOCX))
    text = "\n".join(p.text for p in doc.paragraphs)
    assert "\U0001f409" in text, "the DOCX should still carry the dragon"


# -- the hand-written prose -------------------------------------------------
# The generators derive their figures, which removes the drift risk from the PDF,
# the DOCX and the deck. The markdown is still typed by hand, so it is guarded
# here instead: a stale test count in a README is the same defect as a stale one
# in a slide, and nobody can see either from the outside.
PROSE = (
    "README.md",
    "SUBMISSION.md",
    "docs/ARCHITECTURE.md",
    "docs/ACCESSIBILITY.md",
    "docs/SNAPDRAGON_DEPLOYMENT.md",
    "docs/DESCRIPTION.md",
)


def test_prose_test_counts_are_current() -> None:
    """Every 'N tests' claim in the markdown must equal the collected count.

    Some counts are legitimately scenario-specific -- the suite skips tests whose
    model weights are absent, so "N passed" from a bare checkout is a different
    and equally real figure. Those are listed rather than waved through, so a
    reader knows which is which and a stale one still gets noticed.
    """
    n = count_tests()
    # Counts that are correct for a narrower run than "the whole suite".
    ALTERNATE = {
        # Measured in a venv with requirements.txt only: no onnxruntime, no
        # tiktoken, no model weights, and no reportlab/python-docx/pypdf either,
        # which are build-time-only and deliberately not project dependencies.
        (122, "passed, 12 skipped"),
    }

    stale = []
    for name in PROSE:
        path = ROOT / name
        if not path.is_file():
            continue
        body = path.read_text()
        for m in re.finditer(r"(?<![\d.])(\d+)\s+(tests?\b|passed\b[^\n|]*)",
                             body):
            claimed = int(m.group(1))
            trailing = m.group(2)
            if claimed == n:
                continue
            # An alternate is acceptable only if its qualifier travels with it.
            if any(claimed == c and tail in trailing for c, tail in ALTERNATE):
                continue
            line = body[:m.start()].count("\n") + 1
            stale.append(f"{name}:{line} claims {claimed}, suite has {n}")
    assert not stale, (
        "stale test counts in the documentation:\n    "
        + "\n    ".join(stale)
        + "\n  Update them, or delete the claim. If it is a genuine narrower "
        "run, add it to ALTERNATE with its qualifier."
    )


def test_generators_derive_the_test_count() -> None:
    """The derived count has to stay derived.

    Easy to undo by accident: someone tidying a literal back into a generator,
    not realising it was the thing keeping the figure honest. The check is
    deliberately narrow -- an earlier version banned every three-digit string and
    flagged the ASR latency, which is a real number in its own right.
    """
    deck = (ROOT / "scripts" / "make_deck.py").read_text()
    desc = (ROOT / "scripts" / "make_description_doc.py").read_text()

    assert "count_tests()" in deck, "the deck no longer derives its test count"
    assert "str(count_tests())" in deck, (
        "the deck should pass the derived count to stat(), not a literal")
    assert "TESTS = str(count_tests())" in desc, (
        "the description should derive TESTS from the suite, not hardcode it")
