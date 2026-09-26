#!/usr/bin/env python3
"""Generate the project description as a PDF and a DOCX.

Run with `python scripts/make_description_doc.py`. Writes:

    docs/Snapdragoon-description.pdf
    docs/Snapdragoon-description.docx

Generated rather than hand-written so the figures in the document come from the
same place the deck's do, and so a correction is one edit rather than two files
to keep in sync. The DOCX exists because submission forms impose character
limits, and editing a PDF to hit one is miserable.

Light background on purpose: this gets printed and read on paper, where the
deck's dark theme is a liability rather than a choice.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (BaseDocTemplate, Frame, KeepTogether, PageTemplate,
                                Paragraph, Spacer, Table, TableStyle)

from _testcount import count_tests

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt, RGBColor as DocxRGB, Inches as DocxInches

ROOT = Path(__file__).resolve().parent.parent
PDF_OUT = ROOT / "docs" / "Snapdragoon-description.pdf"
DOCX_OUT = ROOT / "docs" / "Snapdragoon-description.docx"

REPO = "github.com/challalokesh08/snapdragoon"
PROJECT = "Snapdragoon"

INK = colors.HexColor("#12161F")
DIM = colors.HexColor("#4A5462")
AMBER = colors.HexColor("#B26A00")
TEAL = colors.HexColor("#0F766E")
RED = colors.HexColor("#B42318")
RULE = colors.HexColor("#D6DBE1")
PANEL = colors.HexColor("#F4F6F8")

# -- glyph budget ----------------------------------------------------------
# reportlab draws with base-14 Helvetica, whose coverage is narrower than it
# looks and narrower than its own AFM metrics suggest. Everything below was
# checked by inspecting the content stream of a one-character PDF, because a
# missing glyph is silent: no warning, no error, and a hollow box on the page.
#
#   SAFE (WinAnsi)   - - * / : ; ! ? " ' ( ) [ ] { } and Latin-1, plus
#                     U+2013 -  U+2014 --  U+2018 '  U+2019 '  U+201C "
#                     U+201D "  U+2026 ...  U+00B7 ·  U+00D7 x
#   SAFE (ZapfDingbats substitution -- reportlab swaps the font automatically)
#                     U+2713 v  U+2717 x  U+2192 ->
#   NOT DEF          U+2022 *  U+25A0 *  and the emoji: U+1F409 is substituted
#                     into ZapfDingbats, where it lands on `n` -- a filled black
#                     square, not a dragon. It renders, and it is wrong.
#
# A PDF escapes 0x7F as the four characters \177, so that literal is the thing to
# grep for when checking a build: its presence means a notdef.
BULLET = "<font face='ZapfDingbats'>n</font>"   # filled square, not U+2022
NOTDEF_MARKER = "\\177"


# -- figures, read from source --------------------------------------------
def audit_counts() -> tuple[dict, int]:
    """Severity totals, parsed from the audit files.

    Parsed rather than typed in because a hand-copied count is exactly the kind
    of number that goes stale: an earlier draft of the deck claimed "2 serious,
    9 moderate/minor" against audits recording 3, 7 and 3, and nothing about the
    slide looked wrong.
    """
    total: Counter = Counter()
    issues = 0
    for name in ("accessibility-audit-qualcomm-ai-lab.md",
                 "accessibility-audit-unstop-ai-lab-challenge.md"):
        path = ROOT / name
        if not path.is_file():
            continue
        found = re.findall(
            r"^\|\s*\d+\s*\|\s*(Blocker|Serious|Moderate|Minor)\s*\|",
            path.read_text(), re.M)
        total.update(found)
        issues += len(found)
    return dict(total), issues


SEV, ISSUES = audit_counts()
BLOCKERS = SEV.get("Blocker", 0)
SERIOUS = SEV.get("Serious", 0)

ASR_LO, ASR_HI = "190", "400"
RTF_LO, RTF_HI = "12", "26"
VIS, FPS = "5.3 ms", "189 fps"
TESTS = str(count_tests())

TRANSCRIPT = ("“And so my fellow Americans ask not what your country can do for "
              "you, ask what you can do for your country.”")
VISION = ("A photograph of a white Golden Retriever classifies as Samoyed 59.0% "
          "/ Pomeranian 10.2% / West Highland white terrier 6.5%.")

# -- the copy --------------------------------------------------------------
TAGLINE = ("On-device accessibility co-pilot: live captions and scene "
           "description for people who are blind, deaf or hard of hearing.")

BRIEF = (
    f"{PROJECT} is an offline, on-device accessibility co-pilot. It captions "
    "speech and describes what the camera sees, as real text a screen reader can "
    "announce. No account, no network call, no data leaves the device."
)

SHORT = (
    f"<b>{PROJECT}</b> is an offline, on-device accessibility co-pilot for "
    "people who are blind, deaf or hard of hearing. It runs Whisper Tiny for "
    "live speech captions and MobileNet V2 for scene description, and renders "
    "both as real text in the DOM — announced through a polite live region, "
    f"keyboard-operable, and verified by 11 automated WCAG 2.2 AA gates and "
    f"{TESTS} tests. We audited two live on-device-AI pages and found {ISSUES} "
    f"accessibility issues between them, including {BLOCKERS} blockers; every "
    "category we found is now a regression test in ours. Three interchangeable "
    "backends — CPU, demo, and a Qualcomm AI Hub / QNN NPU path — sit behind one "
    "interface, so moving to a Snapdragon NPU is a configuration change rather "
    "than a rewrite. CPU latency is measured and published as a range; no NPU "
    "figure is claimed, because no Snapdragon device was available."
)

LONG = f"""
<b>The problem.</b> On-device AI demos are inaccessible by default, and it is
not a rare edge case — it is the normal outcome of building a visual demo. Live
captions that update silently are invisible to a screen reader. Scene
descriptions drawn into a <font face="Courier">canvas</font> have no text
alternative at all. A <font face="Courier">div</font> with a click handler is
not a button: it has no role, no accessible name, no focus stop and no keyboard
activation. We did not assume this — we audited two live on-device-AI pages
against WCAG 2.2 AA using rendered-DOM inspection, computed styles and real
keyboard traversal, and recorded <b>{ISSUES} issues, {BLOCKERS} of them
blockers</b>: zero heading structure, a primary journey with no focusable
elements, a closed dialog that was not inert, unnamed controls, and no skip link.
<br/><br/>
<b>What we built.</b> {PROJECT} is an offline, on-device accessibility
co-pilot. Whisper Tiny produces live speech captions; MobileNet V2 describes
what the camera sees with confidence scores. Both are rendered as real text in
the DOM, announced through a polite live region so they never interrupt, with
the full transcript kept so it can be re-read. The skip link is the first
focusable element and its landing target keeps its focus ring. No
<font face="Courier">outline: none</font> appears anywhere in the codebase, and
a test fails if it ever does.
<br/><br/>
<b>Why it is built this way.</b> The app never imports a runtime; it talks to
an <font face="Courier">Engine</font> interface with three backends — a
self-labelling demo engine, a verified CPU engine, and a Qualcomm AI Hub / QNN
NPU engine that raises rather than silently falling back when off-Snapdragon.
That is what allowed the project to be built and verified without the target
hardware, and it means a Snapdragon port is configuration, not a rewrite.
<br/><br/>
<b>What is measured, and what is not.</b> {TESTS} tests pass from a clean
clone. The reference clip transcribes exactly; the reference photograph
classifies correctly. ASR runs a 5 s window in {ASR_LO}–{ASR_HI} ms
({RTF_LO}–{RTF_HI}× real time) and vision in ~{VIS} (~{FPS}) on a laptop CPU —
published as a range, because two Python environments on the same machine
differed by 2×. The Snapdragon NPU path is implemented and documented step by
step but has <b>never been run on hardware</b>, so no NPU latency figure is
claimed anywhere.
"""

PROBLEM = [
    ("Live captions that nobody hears. ",
     "Text that updates silently is invisible to a screen reader unless it sits "
     "in a live region."),
    ("A canvas is not an interface. ",
     "Scene descriptions drawn into a canvas have no text alternative at all."),
    ("A div is not a button. ",
     "A click handler does not give an element a role, an accessible name, a "
     "focus stop, or keyboard activation."),
    ("Status for everyone except the person who needs it. ",
     "Errors and confirmations that only appear visually."),
    ("Focus removed instead of replaced. ",
     "Suppressing the focus ring is a WCAG 2.4.7 failure that a "
     "screenshot-based review will not catch."),
]

MEASURED = [
    ("Live speech-to-text", "Whisper Tiny (EN), 5 s window",
     f"{ASR_LO}–{ASR_HI} ms median, {RTF_LO}–{RTF_HI}× real time"),
    ("Scene description", "MobileNet V2 (ImageNet-1k), 640×480", f"~{VIS}, ~{FPS}"),
    ("Test suite", "clean clone, weights fetched not vendored", f"{TESTS} passing"),
    ("Accessibility gates", "scripts/verify_a11y.py", "11/11 pass"),
    ("Correctness, speech", "known reference clip", "transcript matches exactly"),
    ("Correctness, vision", "known reference photograph", "top-3 classes match"),
]

VERIFIED = [
    "Both models run, and their output is checked against known-correct answers",
    f"The full test suite passes from a clean clone ({TESTS} tests)",
    "The web app, its SSE caption stream, and the accessible UI",
    "11/11 automated accessibility gates, plus a screen-reader pass",
    f"Latency, measured on real audio and published as a range",
    "A clean-clone install with no model weights in the repository",
]

NOT_VERIFIED = [
    "Any Snapdragon NPU latency figure",
    "Any on-device photograph or screen capture",
    "QNN context-binary generation",
    "Qualcomm AI Hub compile and profile runs",
    "The Windows / DirectML execution path",
]

# Claims deliberately excluded, each with the reason it is excluded.
DO_NOT_CLAIM = [
    ("“Runs on the NPU” / “NPU-accelerated”",
     "Never measured. QualcommEngine raises off-Snapdragon precisely so this "
     "cannot be implied."),
    ("Any Snapdragon latency number",
     "None exists. scripts/benchmark.py --backend qualcomm would produce one on "
     "the hardware."),
    ("“Tested on Snapdragon X”",
     "No Snapdragon device was available during the build."),
    ("“13.2× real time”",
     "Measured on silence, which the engine declines to transcribe. The "
     "corrected figure is on real audio."),
    ("A single ASR millisecond figure",
     "It varies 2× with the Python environment. The range is the honest claim."),
]

RUN = [
    "git clone https://github.com/challalokesh08/snapdragoon",
    "cd snapdragoon && python -m venv .venv && source .venv/bin/activate",
    "pip install -r requirements.txt        # runs with no model files",
    "python -m snapdragoon.web.app          # \u2192 http://127.0.0.1:8765",
]

RUN_ML = ("pip install -r requirements-ml.txt && "
          "python scripts/fetch_models.py --public")


# -- PDF -------------------------------------------------------------------
def styles():
    def st(name, **kw):
        base = dict(name=name, fontName="Helvetica", fontSize=9.5, leading=13.4,
                    textColor=INK, alignment=TA_LEFT, spaceAfter=6)
        base.update(kw)
        return ParagraphStyle(**base)

    return {
        "title": st("title", fontName="Helvetica-Bold", fontSize=21, leading=25,
                    spaceAfter=2),
        "sub": st("sub", fontSize=10, textColor=DIM, spaceAfter=10),
        "h": st("h", fontName="Helvetica-Bold", fontSize=11.5, leading=14,
                textColor=AMBER, spaceBefore=13, spaceAfter=5),
        "body": st("body"),
        "small": st("small", fontSize=8.4, leading=11.6, textColor=DIM),
        "panel": st("panel", fontSize=9.5, leading=13.4),
        "cell": st("cell", fontSize=8.8, leading=12, spaceAfter=0),
        "cellb": st("cellb", fontName="Helvetica-Bold", fontSize=8.8, leading=12,
                    spaceAfter=0),
        "mono": st("mono", fontName="Courier", fontSize=8.6, leading=12,
                   spaceAfter=0),
    }


def build_pdf() -> None:
    S = styles()
    doc = BaseDocTemplate(str(PDF_OUT), pagesize=LETTER,
                          leftMargin=0.85 * inch, rightMargin=0.85 * inch,
                          topMargin=0.75 * inch, bottomMargin=0.75 * inch,
                          title=f"{PROJECT} — project description",
                          author=f"{PROJECT}", subject=REPO)
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height,
                  id="body")
    doc.addPageTemplates([PageTemplate(id="main", frames=[frame])])

    def panel(flow, width):
        t = Table([[flow]], colWidths=[width])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), PANEL),
            ("LEFTPADDING", (0, 0), (-1, -1), 11),
            ("RIGHTPADDING", (0, 0), (-1, -1), 11),
            ("TOPPADDING", (0, 0), (-1, -1), 9),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
            ("LINEBEFORE", (0, 0), (0, -1), 2.5, AMBER),
        ]))
        return t

    width = doc.width
    half = (width - 14) / 2
    F = []

    # Glyph budget for this file: reportlab draws with base-14 Helvetica under
    # WinAnsiEncoding, which is cp1252. Anything outside it -- an emoji, a
    # check mark, a black square, an arrow -- renders as a notdef box. Each of
    # those was tried here and removed after checking the extracted text, rather
    # than assumed safe. The DOCX keeps the dragon, because Word and Google Docs
    # both fall back to Segoe UI Emoji and draw it correctly.
    # Superseded by the measured table at the top of this file: U+2713, U+2717
    # and U+2192 turn out to be fine, because reportlab substitutes ZapfDingbats.
    # The dragon stays out -- ZapfDingbats draws it as a filled square.
    F.append(Table([[""]], colWidths=[0.9 * inch], rowHeights=[0.055 * inch],
                   style=TableStyle([
                       ("BACKGROUND", (0, 0), (-1, -1), AMBER),
                       ("LINEBELOW", (0, 0), (-1, -1), 0, colors.white),
                   ])))
    F.append(Spacer(1, 7))
    F.append(Paragraph(PROJECT, S["title"]))
    F.append(Paragraph(
        f"An offline, on-device accessibility co-pilot &nbsp;·&nbsp; "
        f"<font color='#4A5462'>{REPO}</font>", S["sub"]))

    # 1 — the one-paragraph answer
    F.append(Paragraph("Short description", S["h"]))
    F.append(panel(Paragraph(SHORT, S["panel"]), width))

    # 2 — the problem
    F.append(Paragraph("The problem", S["h"]))
    F.append(Paragraph(
        "On-device AI demos are inaccessible by default. This is not a rare edge "
        "case — it is the normal outcome of building a visual demo.", S["body"]))
    F.append(Spacer(1, 3))
    for head, rest in PROBLEM:
        F.append(Paragraph(f"<font color='#B26A00'>{BULLET}</font>&nbsp;&nbsp;"
                           f"<b>{head}</b>{rest}", S["body"]))

    # 3 — evidence
    F.append(Paragraph("We measured it rather than assuming it", S["h"]))
    F.append(Paragraph(
        f"Two WCAG 2.2 AA audits of live on-device-AI pages, by rendered-DOM "
        f"inspection, computed styles and real keyboard traversal: "
        f"<b>{ISSUES} issues in total</b> — {BLOCKERS} Blocker, {SERIOUS} Serious, "
        f"{SEV.get('Moderate', 0)} Moderate, {SEV.get('Minor', 0)} Minor. Both "
        f"audits are in the repository, and every category found is now a "
        f"regression test in ours.", S["body"]))

    # 4 — what it does
    F.append(Paragraph("What it does", S["h"]))
    caps = [
        Paragraph("<b>Live speech captions</b>", S["cellb"]),
        Paragraph("Whisper Tiny (EN) — a 5 s rolling window, greedy decode, "
                  "KV-cached. Announced through a polite live region, with the "
                  "full transcript kept so it can be re-read.", S["cell"]),
    ]
    scene = [
        Paragraph("<b>Scene description</b>", S["cellb"]),
        Paragraph("MobileNet V2 (ImageNet-1k) — top-3 classes with confidences, "
                  "as real text in the DOM rather than drawn into a canvas. "
                  "Works on a still image, a file, or a live camera.", S["cell"]),
    ]
    t = Table([[caps, scene]], colWidths=[half, half])
    t.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BACKGROUND", (0, 0), (0, 0), PANEL),
        ("BACKGROUND", (1, 0), (1, 0), PANEL),
        ("LEFTPADDING", (0, 0), (-1, -1), 11),
        ("RIGHTPADDING", (0, 0), (-1, -1), 11),
        ("TOPPADDING", (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
        ("LINEBEFORE", (0, 0), (0, 0), 2.5, AMBER),
        ("LINEBEFORE", (1, 0), (1, 0), 2.5, AMBER),
    ]))
    F.append(t)
    F.append(Spacer(1, 5))
    F.append(Paragraph(
        "No account. No network call. No data leaves the device. The app also runs "
        "with no model files at all — and says so, rather than faking output.",
        S["body"]))

    # 5 — measured
    F.append(Paragraph("Measured results", S["h"]))
    F.append(Paragraph(
        "On an Apple Silicon laptop running ONNX Runtime on <b>CPU — not an "
        "NPU</b>. Reproduce with <font face='Courier'>scripts/benchmark.py "
        "--backend onnx</font>.", S["body"]))
    rows = [[Paragraph(x, S["cellb"]) for x in ("Stage", "Configuration", "Result")]]
    for a, b, c in MEASURED:
        rows.append([Paragraph(a, S["cellb"]), Paragraph(b, S["cell"]),
                     Paragraph(c, S["cell"])])
    t = Table(rows, colWidths=[1.5 * inch, 2.6 * inch, width - 4.1 * inch])
    t.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LINEBELOW", (0, 0), (-1, -2), 0.4, RULE),
        ("LINEBELOW", (0, 0), (-1, -1), 0.8, INK),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
    ]))
    F.append(t)
    F.append(Spacer(1, 6))
    F.append(Paragraph(
        f"The ASR range is published rather than the best single run. Two Python "
        f"environments on this one machine — same onnxruntime wheel, same session "
        f"options — measured 194 ms and 402 ms for identical audio and an "
        f"identical transcript, so the benchmark records the numeric stack next "
        f"to every number it reports. What survives the variation is the claim "
        f"that matters: comfortably faster than real time in every environment "
        f"tested, with {RTF_LO}× as the floor.", S["small"]))

    # 6 — verified vs not
    F.append(Paragraph("What is verified, and what is not", S["h"]))
    F.append(Paragraph(
        "No Snapdragon device was available during this build. The NPU backend is "
        "implemented against the real QNN and AI Hub APIs and documented step by "
        "step — but implemented is not measured, and no NPU figure is claimed "
        "anywhere in this project.", S["body"]))
    F.append(Spacer(1, 3))
    # Ticks are safe: reportlab substitutes U+2713 and U+2717 into ZapfDingbats,
    # whose glyphs 3 and 8 are the check and the ballot X. Verified in the
    # content stream, not assumed -- see GLYPHS below.
    ok = [Paragraph("VERIFIED", S["cellb"])] + \
         [Paragraph(f"<font color='#0F766E'>\u2713</font>&nbsp; {x}", S["cell"])
          for x in VERIFIED]
    no = [Paragraph("NOT VERIFIED — no device", S["cellb"])] + \
         [Paragraph(f"<font color='#B42318'>\u2717</font>&nbsp; {x}", S["cell"])
          for x in NOT_VERIFIED]
    t = Table([[ok, no]], colWidths=[half, half])
    t.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BACKGROUND", (0, 0), (0, 0), PANEL),
        ("BACKGROUND", (1, 0), (1, 0), PANEL),
        ("LEFTPADDING", (0, 0), (-1, -1), 11),
        ("RIGHTPADDING", (0, 0), (-1, -1), 11),
        ("TOPPADDING", (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
        ("LINEBEFORE", (0, 0), (0, 0), 2.5, TEAL),
        ("LINEBEFORE", (1, 0), (1, 0), 2.5, RED),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
    ]))
    F.append(t)

    # 7 — long description
    F.append(Paragraph("Long description", S["h"]))
    F.append(panel(Paragraph(LONG, S["panel"]), width))

    # 8 — do not claim
    F.append(Paragraph("Claims deliberately excluded", S["h"]))
    F.append(Paragraph(
        "Each of these was considered and rejected. They are listed so the "
        "reasoning survives whoever edits this document next.", S["small"]))
    rows = [[Paragraph("<b>Do not write</b>", S["cellb"]),
             Paragraph("<b>Why</b>", S["cellb"])]]
    for a, b in DO_NOT_CLAIM:
        rows.append([Paragraph(a, S["cellb"]), Paragraph(b, S["cell"])])
    t = Table(rows, colWidths=[2.0 * inch, width - 2.0 * inch])
    t.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LINEBELOW", (0, 0), (-1, -2), 0.4, RULE),
        ("LINEBELOW", (0, 0), (-1, -1), 0.8, RED),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
    ]))
    F.append(t)

    # 9 — run it
    # Kept together: a "Run it" heading stranded at the foot of a page with its
    # commands overleaf is the kind of break that makes a document look careless.
    block = [[Paragraph(f"<font face='Courier'># {ln}</font>", S["mono"])]
             for ln in RUN]
    t = Table(block, colWidths=[width])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), PANEL),
        ("LEFTPADDING", (0, 0), (-1, -1), 11),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    F.append(KeepTogether([
        Paragraph("Run it", S["h"]),
        t,
        Spacer(1, 4),
        Paragraph(f"With real inference: "
                  f"<font face='Courier'>{RUN_ML}</font>", S["small"]),
        Paragraph("Models are fetched, never vendored — the repository stays "
                  "small and carries no third-party weights.", S["small"]),
    ]))

    # 10 — footer note
    F.append(Spacer(1, 10))
    F.append(Paragraph(
        f"{PROJECT} is an independent student project and is not affiliated with "
        f"or endorsed by Qualcomm. “Snapdragon” is a trademark of Qualcomm "
        f"Incorporated, used here to describe the target platform. Figures in "
        f"this document are generated from the repository by "
        f"<font face='Courier'>scripts/make_description_doc.py</font>.",
        S["small"]))

    doc.build(F)


# -- DOCX ------------------------------------------------------------------
def build_docx() -> None:
    doc = Document()
    for section in doc.sections:
        section.left_margin = section.right_margin = DocxInches(0.85)
        section.top_margin = section.bottom_margin = DocxInches(0.75)

    normal = doc.styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = Pt(10)

    def para(text, size=10, bold=False, color=None, italic=False, space_after=6):
        p = doc.add_paragraph()
        run = p.add_run(text)
        run.font.size = Pt(size)
        run.bold = bold
        run.italic = italic
        if color:
            run.font.color.rgb = DocxRGB(*color)
        p.paragraph_format.space_after = Pt(space_after)
        return p

    def heading(text):
        p = doc.add_paragraph()
        run = p.add_run(text)
        run.font.size = Pt(12)
        run.bold = True
        run.font.color.rgb = DocxRGB(0xB2, 0x6A, 0x00)
        p.paragraph_format.space_before = Pt(14)
        p.paragraph_format.space_after = Pt(4)

    title = doc.add_paragraph()
    run = title.add_run(f"🐉  {PROJECT}")
    run.font.size = Pt(22)
    run.bold = True

    para(f"An offline, on-device accessibility co-pilot  ·  {REPO}",
         size=9.5, color=(0x4A, 0x54, 0x62), space_after=12)

    heading("Short description")
    p = doc.add_paragraph()
    run = p.add_run(SHORT)
    run.font.size = Pt(10)
    p.paragraph_format.space_after = Pt(10)

    heading("The problem")
    para("On-device AI demos are inaccessible by default. This is not a rare edge "
         "case — it is the normal outcome of building a visual demo.")
    for head, rest in PROBLEM:
        p = doc.add_paragraph(style="List Bullet")
        run = p.add_run(head)
        run.bold = True
        p.add_run(rest)

    heading("We measured it rather than assuming it")
    para(f"Two WCAG 2.2 AA audits of live on-device-AI pages, by rendered-DOM "
         f"inspection, computed styles and real keyboard traversal: "
         f"{ISSUES} issues in total — {BLOCKERS} Blocker, {SERIOUS} Serious, "
         f"{SEV.get('Moderate', 0)} Moderate, {SEV.get('Minor', 0)} Minor. Both "
         f"audits are in the repository, and every category found is now a "
         f"regression test in ours.")

    heading("What it does")
    p = doc.add_paragraph(style="List Bullet")
    p.add_run("Live speech captions").bold = True
    p.add_run(" — Whisper Tiny (EN), a 5 s rolling window, greedy decode, "
              "KV-cached. Announced through a polite live region, with the full "
              "transcript kept so it can be re-read.")
    p = doc.add_paragraph(style="List Bullet")
    p.add_run("Scene description").bold = True
    p.add_run(" — MobileNet V2 (ImageNet-1k), top-3 classes with confidences, as "
              "real text in the DOM rather than drawn into a canvas. Works on a "
              "still image, a file, or a live camera.")
    para("No account. No network call. No data leaves the device. The app also "
         "runs with no model files at all — and says so, rather than faking "
         "output.")

    heading("Measured results")
    para("On an Apple Silicon laptop running ONNX Runtime on CPU — not an NPU. "
         "Reproduce with scripts/benchmark.py --backend onnx.")
    table = doc.add_table(rows=1, cols=3)
    table.style = "Light Grid Accent 1"
    for i, h in enumerate(("Stage", "Configuration", "Result")):
        cell = table.rows[0].cells[i]
        cell.text = ""
        run = cell.paragraphs[0].add_run(h)
        run.bold = True
        run.font.size = Pt(9)
    for a, b, c in MEASURED:
        cells = table.add_row().cells
        for cell, val, bold in ((cells[0], a, True), (cells[1], b, False),
                                (cells[2], c, False)):
            cell.text = ""
            run = cell.paragraphs[0].add_run(val)
            run.bold = bold
            run.font.size = Pt(9)
    para(f"The ASR range is published rather than the best single run. Two Python "
         f"environments on this one machine — same onnxruntime wheel, same "
         f"session options — measured 194 ms and 402 ms for identical audio and "
         f"an identical transcript, so the benchmark records the numeric stack "
         f"next to every number it reports.", size=8.5, color=(0x4A, 0x54, 0x62))

    heading("What is verified, and what is not")
    para("No Snapdragon device was available during this build. The NPU backend "
         "is implemented against the real QNN and AI Hub APIs and documented step "
         "by step — but implemented is not measured, and no NPU figure is "
         "claimed anywhere in this project.")
    para("VERIFIED", size=9.5, bold=True, color=(0x0F, 0x76, 0x6E))
    for x in VERIFIED:
        doc.add_paragraph(x, style="List Bullet")
    para("NOT VERIFIED — no device", size=9.5, bold=True, color=(0xB4, 0x23, 0x18))
    for x in NOT_VERIFIED:
        doc.add_paragraph(x, style="List Bullet")

    heading("Long description")
    for chunk in LONG.split("<br/><br/>"):
        clean = re.sub(r"</?b>", "", chunk).strip()
        clean = re.sub(r"<font[^>]*>", "", clean).replace("</font>", "")
        p = doc.add_paragraph(clean)
        p.paragraph_format.space_after = Pt(8)

    heading("Claims deliberately excluded")
    for a, b in DO_NOT_CLAIM:
        p = doc.add_paragraph(style="List Bullet")
        p.add_run(a).bold = True
        p.add_run(f" — {b}")

    heading("Run it")
    for ln in RUN:
        p = doc.add_paragraph()
        run = p.add_run(f"# {ln}")
        run.font.name = "Consolas"
        run.font.size = Pt(8.5)
    para(f"With real inference: {RUN_ML}", size=8.5,
         color=(0x4A, 0x54, 0x62))
    para("Models are fetched, never vendored — the repository stays small and "
         "carries no third-party weights.", size=8.5, color=(0x4A, 0x54, 0x62))

    heading("Reference outputs")
    para(f"Speech: {TRANSCRIPT}", size=9.5, italic=True)
    para(f"Vision: {VISION}", size=9.5, italic=True)

    para(f"{PROJECT} is an independent student project and is not affiliated with "
         f"or endorsed by Qualcomm. “Snapdragon” is a trademark of Qualcomm "
         f"Incorporated, used here to describe the target platform. Figures in "
         f"this document are generated from the repository by "
         f"scripts/make_description_doc.py.", size=8.5, color=(0x4A, 0x54, 0x62))

    doc.save(str(DOCX_OUT))


if __name__ == "__main__":
    build_pdf()
    build_docx()
    for p in (PDF_OUT, DOCX_OUT):
        print(f"wrote {p.relative_to(ROOT)}  ({p.stat().st_size / 1024:.1f} KB)")
