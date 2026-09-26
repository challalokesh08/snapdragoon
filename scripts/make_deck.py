#!/usr/bin/env python3
"""Generate the presentation deck.

Run with `python scripts/make_deck.py`. Writes `docs/Snapdragoon-deck.pptx`.

Kept as a script rather than a committed binary so every number on a slide can be
traced to the file it came from, and so a correction is a one-line edit instead
of a re-record. The measured figures live in BENCH below; they are the values
`scripts/benchmark.py` produced on an Apple Silicon laptop, and they are
reproduced with the honest range rather than the best single run.
"""

from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Emu, Inches, Pt

OUT = Path(__file__).resolve().parent.parent / "docs" / "Snapdragoon-deck.pptx"

# -- palette ---------------------------------------------------------------
# Dark, because a projector in a competition room is usually a washed-out
# bright panel, and light-on-dark survives that better than dark-on-light.
BG = RGBColor(0x0C, 0x10, 0x18)
PANEL = RGBColor(0x15, 0x1C, 0x28)
INK = RGBColor(0xF2, 0xF5, 0xF9)
DIM = RGBColor(0x9A, 0xA7, 0xB8)
AMBER = RGBColor(0xF5, 0xA6, 0x23)   # dragon / accent
TEAL = RGBColor(0x3D, 0xD6, 0xC0)    # verified
RED = RGBColor(0xFF, 0x6B, 0x6B)     # unverified / blockers
RULE = RGBColor(0x27, 0x31, 0x42)

HEAD = "Calibri"     # bundled with Office on Windows and macOS
BODY = "Calibri"
MONO = "Consolas"

W, H = Inches(13.333), Inches(7.5)
MARGIN = Inches(0.85)
CONTENT_W = W - 2 * MARGIN

# -- measured figures ------------------------------------------------------
# From scripts/benchmark.py --backend onnx on an Apple Silicon laptop, and
# scripts/fetch_models.py --verify. The range is real: two Python environments
# on the same machine differed by 2x, so the range is what gets presented.
BENCH = {
    "asr_lo": "190 ms", "asr_hi": "400 ms", "rtf_lo": "12x", "rtf_hi": "26x",
    "vis": "5.3 ms", "fps": "189 fps",
    "encoder": "125 ms", "decode": "~50 ms / 12 tokens",
}

ROOT = Path(__file__).resolve().parent.parent
AUDITS = ("accessibility-audit-qualcomm-ai-lab.md",
          "accessibility-audit-unstop-ai-lab-challenge.md")


def audit_counts() -> tuple[dict, int]:
    """Count audit findings by severity, straight from the audit files.

    Parsed rather than typed in because a hand-copied severity count is exactly
    the kind of number that goes stale: an earlier draft of this deck claimed
    "2 serious, 9 moderate/minor" against audits recording 3, 7 and 3. Nothing
    about the slide looks wrong when a count is wrong, which is the problem.
    """
    import re
    from collections import Counter

    total = Counter()
    issues = 0
    for name in AUDITS:
        path = ROOT / name
        if not path.is_file():
            continue
        found = re.findall(r"^\|\s*\d+\s*\|\s*(Blocker|Serious|Moderate|Minor)\s*\|",
                           path.read_text(), re.M)
        total.update(found)
        issues += len(found)
    return dict(total), issues


def slide(prs: Presentation, title: str | None = None, kicker: str | None = None):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    bg = s.shapes.add_shape(1, 0, 0, W, H)
    bg.fill.solid()
    bg.fill.fore_color.rgb = BG
    bg.line.fill.background()
    bg.shadow.inherit = False

    y = Inches(0.62)
    if kicker:
        text(s, kicker.upper(), MARGIN, y, CONTENT_W, Pt(12), AMBER, HEAD, bold=True,
             space=1.6)
        y += Inches(0.34)
    if title:
        text(s, title, MARGIN, y, CONTENT_W, Pt(34), INK, HEAD, bold=True, space=1.0)
        y += Inches(0.92)
        rule(s, MARGIN, y - Inches(0.26), Inches(1.5), AMBER)
    return s, y


def text(s, body, x, y, w, size, colour, font=BODY, bold=False, align=None,
         space=1.25, italic=False):
    """Add a textbox. `body` may be a str or a list of (text, colour) runs."""
    box = s.shapes.add_textbox(x, y, w, Inches(0.4))
    tf = box.text_frame
    tf.word_wrap = True
    lines = body if isinstance(body, list) else [(body, colour)]
    for i, (line, col) in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(space * 6)
        p.line_spacing = space
        if align is not None:
            p.alignment = align
        run = p.add_run()
        run.text = line
        run.font.size = size
        run.font.color.rgb = col
        run.font.name = font
        run.font.bold = bold
        run.font.italic = italic
    return box


def bullets(s, items, x, y, w, size=Pt(17), gap=Inches(0.52), colour=INK,
            bullet_colour=AMBER):
    for i, item in enumerate(items):
        yy = y + i * gap
        dot = s.shapes.add_shape(9, x, yy + Inches(0.10), Inches(0.10), Inches(0.10))
        dot.fill.solid()
        dot.fill.fore_color.rgb = bullet_colour
        dot.line.fill.background()
        dot.shadow.inherit = False
        if isinstance(item, tuple):
            head, rest = item
            box = s.shapes.add_textbox(x + Inches(0.30), yy - Inches(0.03),
                                       w - Inches(0.30), Inches(0.5))
            tf = box.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            p.line_spacing = 1.2
            r1 = p.add_run()
            r1.text = head
            r1.font.size = size
            r1.font.bold = True
            r1.font.color.rgb = INK
            r1.font.name = BODY
            r2 = p.add_run()
            r2.text = rest
            r2.font.size = size
            r2.font.color.rgb = DIM
            r2.font.name = BODY
        else:
            text(s, item, x + Inches(0.30), yy - Inches(0.03), w - Inches(0.30),
                 size, colour, BODY)


def rule(s, x, y, w, colour=RULE, h=Pt(2)):
    bar = s.shapes.add_shape(1, x, y, w, h)
    bar.fill.solid()
    bar.fill.fore_color.rgb = colour
    bar.line.fill.background()
    bar.shadow.inherit = False


def panel(s, x, y, w, h, fill=PANEL, line=None):
    box = s.shapes.add_shape(1, x, y, w, h)
    box.fill.solid()
    box.fill.fore_color.rgb = fill
    if line is None:
        box.line.fill.background()
    else:
        box.line.color.rgb = line
        box.line.width = Pt(1)
    box.shadow.inherit = False
    return box


def stat(s, x, y, w, value, label, colour=AMBER, vsize=Pt(38)):
    text(s, value, x, y, w, vsize, colour, HEAD, bold=True, space=1.0)
    text(s, label, x, y + Inches(0.62), w, Pt(12.5), DIM, BODY, space=1.15)


def footer(s, n):
    text(s, "Snapdragoon — on-device accessibility co-pilot", MARGIN,
         H - Inches(0.52), Inches(6), Pt(10), RULE, BODY)
    text(s, str(n), W - MARGIN - Inches(0.6), H - Inches(0.52), Inches(0.6),
         Pt(10), RULE, BODY, align=PP_ALIGN.RIGHT)


# -- slides ----------------------------------------------------------------
def build() -> Presentation:
    prs = Presentation()
    prs.slide_width, prs.slide_height = W, H
    n = 0

    # 1 — title
    s, _ = slide(prs)
    n += 1
    text(s, "🐉", MARGIN, Inches(1.55), Inches(1.2), Pt(54), AMBER, BODY)
    text(s, "Snapdragoon", MARGIN, Inches(2.45), CONTENT_W, Pt(62), INK, HEAD,
         bold=True, space=1.0)
    rule(s, MARGIN, Inches(3.42), Inches(2.2), AMBER)
    text(s, "An offline, on-device accessibility co-pilot", MARGIN, Inches(3.72),
         CONTENT_W, Pt(25), INK, HEAD, space=1.15)
    text(s, "Live captions and scene description for people who are blind, "
            "deaf or hard of hearing — running entirely on your own hardware.",
         MARGIN, Inches(4.22), Inches(8.6), Pt(16), DIM, BODY, space=1.3)
    text(s, "Qualcomm Snapdragon AI Lab  ·  Build & Present Challenge  ·  "
            "Python + ONNX Runtime, with a QNN / AI Hub NPU path",
         MARGIN, Inches(5.35), Inches(10.5), Pt(12.5), RULE, BODY)
    footer(s, n)

    # 2 — the problem
    s, y = slide(prs, "On-device AI demos are inaccessible by default", "The problem")
    n += 1
    bullets(s, [
        ("Captions nobody hears. ", "Live text that updates silently is invisible to "
         "a screen reader unless it is in a live region."),
        ("A canvas is not an interface. ", "Scene descriptions drawn into a <canvas> "
         "have no text alternative at all."),
        ("A div is not a button. ", "role=\"button\" on a click handler does not give "
         "it a role, a name, a focus stop or a keyboard activation."),
        ("Status for everyone except the person who needs it. ", "Errors and "
         "confirmations that only appear visually."),
    ], MARGIN, y + Inches(0.30), Inches(11.4))
    text(s, [("This is not a rare edge case. It is the default outcome of building a "
              "visual demo, and it is invisible to automated checks that only look at "
              "contrast or ARIA attributes.", DIM)], MARGIN, Inches(5.62),
         Inches(11.4), Pt(14), DIM, BODY, italic=True)
    footer(s, n)

    # 3 — the evidence (the audit)
    s, y = slide(prs, "We measured it rather than assuming it", "The problem, evidenced")
    n += 1
    text(s, "Two WCAG 2.2 AA audits of live on-device-AI pages, by rendered-DOM "
            "inspection, computed styles and real keyboard traversal.",
         MARGIN, y + Inches(0.06), Inches(11.4), Pt(15), DIM, BODY)
    sev, issues = audit_counts()
    text(s, f"{issues} issues in total across the two pages:  "
            f"{sev.get('Blocker', 0)} Blocker  ·  {sev.get('Serious', 0)} Serious  ·  "
            f"{sev.get('Moderate', 0)} Moderate  ·  {sev.get('Minor', 0)} Minor",
         MARGIN, y + Inches(0.50), Inches(11.4), Pt(14), AMBER, HEAD, bold=True)
    rows = [
        (RED, "Zero headings in the page content — every title is a <span>"),
        (RED, "Primary journey is static text: <main> has 0 focusable elements"),
        (RED, "Closed off-canvas dialog is not inert; its Close button is tab stop #5"),
        (RED, "6 links and 2 buttons with no accessible name"),
        (AMBER, "No skip link — 35 tab stops from the header into 29 footer links"),
        (AMBER, "A feedback widget's close button is tabindex=\"-1\": keyboard cannot dismiss it"),
    ]
    yy = y + Inches(1.00)
    for col, desc in rows:
        bar = s.shapes.add_shape(1, MARGIN, yy + Inches(0.09), Inches(0.22), Pt(3.5))
        bar.fill.solid()
        bar.fill.fore_color.rgb = col
        bar.line.fill.background()
        bar.shadow.inherit = False
        text(s, desc, MARGIN + Inches(0.44), yy - Inches(0.02), Inches(11.0),
             Pt(14.5), INK, BODY)
        yy += Inches(0.56)
    panel(s, MARGIN, Inches(5.72), CONTENT_W, Inches(0.80), PANEL, AMBER)
    text(s, "A sample of the Qualcomm AI Lab findings; both full audits are in the "
            "repository, and every category above has a corresponding regression "
            "test in ours.",
         MARGIN + Inches(0.28), Inches(5.90), CONTENT_W - Inches(0.5), Pt(13.5),
         INK, BODY, space=1.25)
    footer(s, n)

    # 4 — what it does
    s, y = slide(prs, "Two capabilities, both on-device", "What it is")
    n += 1
    panel(s, MARGIN, y + Inches(0.24), Inches(5.5), Inches(2.5), PANEL)
    text(s, "Live captions", MARGIN + Inches(0.35), y + Inches(0.50), Inches(4.8),
         Pt(22), INK, HEAD, bold=True)
    text(s, "Whisper Tiny (EN) — 5 s rolling window, greedy decode, KV-cached. "
            "Announced through a polite live region, with full history kept so it "
            "can be re-read.", MARGIN + Inches(0.35), y + Inches(1.05),
         Inches(4.8), Pt(14), DIM, BODY, space=1.28)

    panel(s, MARGIN + Inches(5.9), y + Inches(0.24), Inches(5.5), Inches(2.5), PANEL)
    text(s, "Scene description", MARGIN + Inches(6.25), y + Inches(0.50),
         Inches(4.8), Pt(22), INK, HEAD, bold=True)
    text(s, "MobileNet V2 (ImageNet-1k) — top-3 classes with confidences, as real "
            "text in the DOM, not drawn into a canvas. Works on a still image, a "
            "file, or a live camera.", MARGIN + Inches(6.25), y + Inches(1.05),
         Inches(4.8), Pt(14), DIM, BODY, space=1.28)

    text(s, "No account. No network call. No data leaves the device. The app works "
            "with no model files at all — and says so, rather than faking output.",
         MARGIN, Inches(6.10), Inches(11.4), Pt(14.5), DIM, BODY)
    footer(s, n)

    # 5 — architecture
    s, y = slide(prs, "One interface, three backends", "How it works")
    n += 1
    text(s, "The central design decision: the app never imports a runtime. It talks "
            "to an Engine, and the Engine is configuration.",
         MARGIN, y + Inches(0.04), Inches(11.4), Pt(15), DIM, BODY)

    box_y = y + Inches(0.66)
    panel(s, MARGIN, box_y, Inches(11.6), Inches(0.86), PANEL, AMBER)
    text(s, "Engine  (abstract)   ·   transcribe()  ·   classify()  ·   describe()",
         MARGIN + Inches(0.30), box_y + Inches(0.26), Inches(11.0), Pt(18), INK,
         MONO, bold=True)

    eng_w, gap = Inches(3.6), Inches(0.4)
    for i, (name, sub, state, col) in enumerate([
        ("DemoEngine", "No model files. Self-labelling: every caption it emits is "
         "prefixed so it can never be mistaken for inference.",
         "runs anywhere", DIM),
        ("OnnxEngine", "ONNX Runtime on CPU. Whisper + MobileNet, the development "
         "and CI baseline.", "VERIFIED", TEAL),
        ("QualcommEngine", "QNN / Qualcomm AI Hub / LiteRT. Raises "
         "QualcommUnavailable off-Snapdragon rather than silently falling back.",
         "unverified", RED),
    ]):
        x = MARGIN + i * (eng_w + gap)
        panel(s, x, box_y + Inches(1.22), eng_w, Inches(2.30), PANEL, col)
        text(s, name, x + Inches(0.24), box_y + Inches(1.42), eng_w - Inches(0.5),
             Pt(17), INK, HEAD, bold=True)
        text(s, state, x + Inches(0.24), box_y + Inches(1.74), eng_w - Inches(0.5),
             Pt(11), col, HEAD, bold=True)
        text(s, sub, x + Inches(0.24), box_y + Inches(2.06), eng_w - Inches(0.5),
             Pt(12.5), DIM, BODY, space=1.22)

    text(s, "Mac → Snapdragon is a config change, not a rewrite. That is what let "
            "this be built and verified without the target hardware.",
         MARGIN, Inches(6.28), Inches(11.4), Pt(14.5), DIM, BODY, italic=True)
    footer(s, n)

    # 6 — the audio front-end
    s, y = slide(prs, "The audio front-end is part of the model", "Technical depth")
    n += 1
    bullets(s, [
        ("Whisper's expected mel spectrogram is reproduced exactly. ",
         "Slaney-style mel filterbank with area normalisation, and a periodic "
         "Hann window — not a symmetric one."),
        ("Power spectrum accumulated in float64. ",
         "float32 silently overflows at 30 s of input and returns NaN, which looks "
         "like a model failure rather than a DSP bug."),
        ("Normalisation was measured, not assumed. ",
         "Filterbank area spread across bands: 7.96× un-normalised, 1.16× with "
         "Slaney normalisation."),
        ("Short windows are refused, in content seconds — not padding ratio. ",
         "5 s padded to 30 s is 83% padding and transcribes correctly. 1 s is 97% "
         "padding and hallucinates. The engine knows its own padding, so the gate "
         "lives there."),
    ], MARGIN, y + Inches(0.34), Inches(11.4), gap=Inches(1.02))
    panel(s, MARGIN, Inches(6.02), CONTENT_W, Inches(0.78), PANEL, AMBER)
    text(s, "A wrong front-end still produces fluent, confident, wrong text — so it "
            "is checked against a known transcript, every run.",
         MARGIN + Inches(0.28), Inches(6.24), CONTENT_W - Inches(0.5), Pt(14), INK, BODY)
    footer(s, n)

    # 7 — ONNX traps
    s, y = slide(prs, "Seven ways a Whisper ONNX export fails silently", "Technical depth")
    n += 1
    text(s, "None of these raise. Each produces plausible output that is quietly "
            "wrong, which is why every one is now a named test.",
         MARGIN, y + Inches(0.04), Inches(11.4), Pt(15), DIM, BODY)
    traps = [
        ("get_inputs()[0] ordering", "A different export order feeds the encoder "
         "the token ids. No error, total garbage."),
        ("present.N.* vs past_key_values.N.*", "Same tensors, two names. Matched "
         "naively, the KV cache is never used: every step re-decodes from empty "
         "context — fluent, degenerate repetition."),
        ("Encoder present on the cached path", "Returns a (0, 6, 1, 64) zero-batch "
         "placeholder. Feed it back and encoder attention dies from token two: "
         "right for a few words, then stops for no visible reason."),
        ("Symbolic dims in a declared shape", "['batch_size', 6, 'past_seq', 64] "
         "filtered to ints becomes [6, 64] and reads as heads=64."),
        ("logits axis order", "Exports differ between (b, seq, vocab) and "
         "(b, vocab, seq). Resolved against declared vocab_size."),
        ("Hard-coded special-token ids", "SOT/EOT differ per checkpoint — and "
         "tiktoken.encode(\"<|startoftranscript|>\")[0] is 27, not the special id."),
        ("<|en|> on an English-only checkpoint", "Costs accuracy for a token with "
         "no meaning there. Switched on is_multilingual from config.json."),
    ]
    yy = y + Inches(0.60)
    for i, (head, body) in enumerate(traps):
        col = AMBER if i % 2 == 0 else TEAL
        panel(s, MARGIN, yy, Inches(0.30), Inches(0.30), col)
        text(s, str(i + 1), MARGIN, yy + Inches(0.035), Inches(0.30), Pt(12), BG,
             HEAD, bold=True, align=PP_ALIGN.CENTER)
        text(s, [(f"{head} — {body}", INK)], MARGIN + Inches(0.52), yy - Inches(0.01),
             Inches(11.0), Pt(13), INK, BODY, space=1.18)
        yy += Inches(0.585)
    footer(s, n)

    # 8 — accessibility as a contract
    s, y = slide(prs, "Accessibility as a contract, not a checklist", "The differentiator")
    n += 1
    stat(s, MARGIN, y + Inches(0.26), Inches(2.4), "11/11", "automated a11y gates pass", TEAL, Pt(34))
    stat(s, MARGIN + Inches(2.7), y + Inches(0.26), Inches(2.4), "18", "assertions in the web test suite", AMBER, Pt(34))
    stat(s, MARGIN + Inches(5.4), y + Inches(0.26), Inches(2.4), str(audit_counts()[1]),
         "issues found auditing 2 live AI-lab pages", AMBER, Pt(34))
    stat(s, MARGIN + Inches(8.1), y + Inches(0.26), Inches(2.4), "0", "suppressed focus outlines", TEAL, Pt(34))

    bullets(s, [
        "A skip link is the first focusable element — and its landing target keeps its outline.",
        "Captions go to a polite live region, so they never interrupt; the full transcript stays re-readable.",
        "Every control is a real <button>, operable by keyboard alone. No role=\"button\" on a div.",
        "Scene output is real text in the DOM, so it is selectable, zoomable and reachable.",
        "Contrast ≥ 4.5:1 everywhere, and the live region respects prefers-reduced-motion.",
    ], MARGIN, y + Inches(1.68), Inches(11.4), size=Pt(15), gap=Inches(0.58))
    panel(s, MARGIN, Inches(6.22), CONTENT_W, Inches(0.62), PANEL, TEAL)
    text(s, "Regression-tested, not asserted: outline: none appears nowhere in the "
            "codebase, and a test fails if it ever does.",
         MARGIN + Inches(0.28), Inches(6.38), CONTENT_W - Inches(0.5), Pt(13.5), INK, BODY)
    footer(s, n)

    # 9 — measured results
    s, y = slide(prs, "Measured on real audio, on a laptop CPU", "Results")
    n += 1
    stat(s, MARGIN, y + Inches(0.22), Inches(3.5), f"{BENCH['asr_lo']}–{BENCH['asr_hi']}",
         f"Whisper Tiny, 5 s window\n{BENCH['rtf_lo']}–{BENCH['rtf_hi']} real time", AMBER, Pt(33))
    stat(s, MARGIN + Inches(4.0), y + Inches(0.22), Inches(3.0), BENCH["vis"],
         f"MobileNet V2, 640×480\n{BENCH['fps']}", TEAL, Pt(33))
    stat(s, MARGIN + Inches(7.5), y + Inches(0.22), Inches(3.0), "126", "tests passing, weights present", TEAL, Pt(33))

    yy = y + Inches(1.60)
    panel(s, MARGIN, yy, Inches(11.6), Inches(1.28), PANEL)
    text(s, "Where the time goes  (measured, not asserted)", MARGIN + Inches(0.28),
         yy + Inches(0.16), Inches(11.0), Pt(13.5), AMBER, HEAD, bold=True)
    text(s, f"Encoder forward  {BENCH['encoder']} — a fixed cost of the graph, "
            f"identical in every environment tested.        Decode loop  "
            f"{BENCH['decode']} — dominated by re-feeding the encoder KV cache, "
            f"which is a property of the export.",
         MARGIN + Inches(0.28), yy + Inches(0.52), Inches(11.0), Pt(13), DIM, BODY, space=1.25)

    text(s, "The range is published rather than the best run. Two Python environments "
            "on this one machine — same onnxruntime wheel, same session options — gave "
            "194 ms and 402 ms for identical audio and an identical transcript, so the "
            "benchmark now records the numeric stack next to every number.",
         MARGIN, Inches(6.30), Inches(11.4), Pt(13.5), DIM, BODY, italic=True)
    footer(s, n)

    # 10 — correctness
    s, y = slide(prs, "Fast and wrong is worthless", "Correctness")
    n += 1
    panel(s, MARGIN, y + Inches(0.26), Inches(11.6), Inches(1.42), PANEL, TEAL)
    text(s, "Speech, verified", MARGIN + Inches(0.30), y + Inches(0.44), Inches(5.0),
         Pt(13), TEAL, HEAD, bold=True)
    text(s, "\u201cAnd so my fellow Americans ask not what your country can do for "
            "you, ask what you can do for your country.\u201d",
         MARGIN + Inches(0.30), y + Inches(0.76), Inches(11.0), Pt(16), INK, BODY,
         italic=True)

    panel(s, MARGIN, y + Inches(1.94), Inches(11.6), Inches(1.42), PANEL, TEAL)
    text(s, "Vision, verified", MARGIN + Inches(0.30), y + Inches(2.12), Inches(5.0),
         Pt(13), TEAL, HEAD, bold=True)
    text(s, "A photograph of a white Golden Retriever →  Samoyed 59.0%  /  "
            "Pomeranian 10.2%  /  West Highland white terrier 6.5%",
         MARGIN + Inches(0.30), y + Inches(2.44), Inches(11.0), Pt(16), INK, BODY)

    text(s, "Both run on every clean clone via scripts/fetch_models.py --verify, which "
            "checks the transcript and the classification rather than merely confirming "
            "the files downloaded.",
         MARGIN, Inches(6.28), Inches(11.4), Pt(14), DIM, BODY)
    footer(s, n)

    # 11 — verified vs not
    s, y = slide(prs, "What is verified, and what is not", "The honest slide")
    n += 1
    text(s, "This is the slide that decides whether a technical reviewer trusts "
            "everything else on the deck.",
         MARGIN, y + Inches(0.02), Inches(11.4), Pt(15), DIM, BODY, italic=True)

    col_w = Inches(5.6)
    yy = y + Inches(0.60)
    panel(s, MARGIN, yy, col_w, Inches(4.10), PANEL, TEAL)
    text(s, "VERIFIED", MARGIN + Inches(0.30), yy + Inches(0.22), Inches(4.0),
         Pt(14), TEAL, HEAD, bold=True)
    bullets(s, [
        "Both models run and are correct",
        "Full test suite, from a clean clone",
        "Web app, SSE caption stream, accessible UI",
        "11/11 automated accessibility gates",
        "Latency, measured on real audio",
        "Clean-clone install with no weights in git",
    ], MARGIN + Inches(0.30), yy + Inches(0.66), col_w - Inches(0.6), size=Pt(13.5),
        gap=Inches(0.52), bullet_colour=TEAL)

    x2 = MARGIN + col_w + Inches(0.4)
    panel(s, x2, yy, col_w, Inches(4.10), PANEL, RED)
    text(s, "NOT VERIFIED — no device", x2 + Inches(0.30), yy + Inches(0.22),
         Inches(4.4), Pt(14), RED, HEAD, bold=True)
    bullets(s, [
        "Any Snapdragon NPU latency figure",
        "Any on-device photograph or capture",
        "QNN context-binary generation",
        "AI Hub compile and profile runs",
        "Windows / DirectML execution path",
    ], x2 + Inches(0.30), yy + Inches(0.66), col_w - Inches(0.6), size=Pt(13.5),
        gap=Inches(0.52), bullet_colour=RED)

    text(s, "No Snapdragon device was available during this build. The NPU backend "
            "is implemented against the real QNN and AI Hub APIs and documented step "
            "by step — but implemented is not measured, and no NPU number is claimed "
            "anywhere in this project.",
         MARGIN, Inches(6.36), Inches(11.4), Pt(14), INK, BODY)
    footer(s, n)

    # 12 — snapdragon path
    s, y = slide(prs, "The Snapdragon path", "Platform relevance")
    n += 1
    bullets(s, [
        ("The models are already the right ones. ", "Both are published by Qualcomm "
         "on AI Hub in QNN-ready form, so the deployment is a compile step rather "
         "than a model swap."),
        ("QualcommEngine targets QNN, AI Hub DirectML and LiteRT. ",
         "Off-Snapdragon it raises QualcommUnavailable rather than silently "
         "degrading to CPU — so a demo can never claim an NPU it is not using."),
        ("The status endpoint reports the truth. ",
         "/api/status states which engine is live and whether it is NPU-accelerated, "
         "and the UI shows it. Showing that proactively is worth more than being asked."),
        ("Conversion is documented command by command. ",
         "Expected QNN tensor names, KV-cache handling, and a troubleshooting table "
         "in docs/SNAPDRAGON_DEPLOYMENT.md."),
        ("And then run the benchmark. ", "scripts/benchmark.py --backend qualcomm "
         "produces the NPU figures. It has never been run, because there was no "
         "device to run it on."),
    ], MARGIN, y + Inches(0.30), Inches(11.4), size=Pt(15), gap=Inches(0.94))
    footer(s, n)

    # 13 — run it
    s, y = slide(prs, "Run it in four commands", "Try it")
    n += 1
    code = [
        ("git clone https://github.com/challalokesh08/snapdragoon", DIM),
        ("cd snapdragoon && python -m venv .venv && source .venv/bin/activate", DIM),
        ("pip install -r requirements.txt        # runs with no model files", INK),
        ("python -m snapdragoon.web.app          # → http://127.0.0.1:8765", INK),
    ]
    yy = y + Inches(0.34)
    for line, col in code:
        panel(s, MARGIN, yy, Inches(11.6), Inches(0.56), PANEL)
        text(s, "$ " + line, MARGIN + Inches(0.28), yy + Inches(0.13), Inches(11.0),
             Pt(13.5), col, MONO)
        yy += Inches(0.68)

    text(s, "With real inference:", MARGIN, yy + Inches(0.22), Inches(11.4),
         Pt(14), AMBER, HEAD, bold=True)
    text(s, "$ pip install -r requirements-ml.txt && python scripts/fetch_models.py --public",
         MARGIN + Inches(0.28), yy + Inches(0.56), Inches(11.0), Pt(13.5), INK, MONO)
    text(s, "Models are fetched, never vendored — the repository stays small and "
            "carries no third-party weights.",
         MARGIN, Inches(6.40), Inches(11.4), Pt(13.5), DIM, BODY, italic=True)
    footer(s, n)

    # 14 — close
    s, _ = slide(prs)
    n += 1
    text(s, "🐉", MARGIN, Inches(1.35), Inches(1.2), Pt(48), AMBER, BODY)
    text(s, "On-device AI should not require sighted, hearing,\nkeyboard-and-mouse "
            "users to operate it.",
         MARGIN, Inches(2.35), CONTENT_W, Pt(34), INK, HEAD, bold=True, space=1.18)
    rule(s, MARGIN, Inches(3.62), Inches(2.2), AMBER)
    text(s, "Snapdragoon is a working, tested, honestly-measured answer — with an "
            "NPU path ready for the hardware it was built for.",
         MARGIN, Inches(3.92), Inches(9.6), Pt(17), DIM, BODY, space=1.3)
    text(s, [("github.com/challalokesh08/snapdragoon", TEAL),
             ("      Independent student project — not affiliated with or endorsed "
              "by Qualcomm. “Snapdragon” is a trademark of Qualcomm Incorporated.",
              DIM)],
         MARGIN, Inches(5.65), Inches(11.6), Pt(12.5), BODY, space=1.3)
    footer(s, n)

    return prs


if __name__ == "__main__":
    deck = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    deck.save(str(OUT))
    print(f"wrote {OUT}  ({len(deck.slides.__iter__.__self__._sldIdLst)} slides)")
