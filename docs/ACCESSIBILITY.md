# Accessibility conformance statement

**Snapdragoon** · WCAG 2.2 Level AA · last reviewed 26 September 2026

---

## Why this document exists

Snapdragoon is an accessibility tool, so "it is accessible" is a claim that
needs evidence rather than adjectives. This document states what is
implemented, how it is verified, and — importantly — what is *not* yet
verified.

Accessibility in this project is treated as a property of the **inference
pipeline**, not a UI pass applied at the end. A caption that is never announced
is functionally identical to no caption at all, so the live-region contract is
part of the model output contract.

## Verification

Two layers, both runnable:

```bash
python -m pytest tests/ -q          # 126 tests, includes 18 a11y assertions
python scripts/verify_a11y.py       # 11 checks against the served page
```

`verify_a11y.py` exits non-zero on failure, so it works as a CI gate.

## Summary

| WCAG 2.2 success criterion | Level | Status |
|---|---|---|
| 1.1.1 Non-text Content | A | Supported |
| 1.3.1 Info and Relationships | A | Supported |
| 1.3.2 Meaningful Sequence | A | Supported |
| 1.4.1 Use of Color | A | Supported |
| 1.4.3 Contrast (Minimum) | AA | Supported |
| 1.4.4 Resize Text | AA | Supported |
| 1.4.10 Reflow | AA | Supported |
| 1.4.11 Non-text Contrast | AA | Supported |
| 1.4.12 Text Spacing | AA | Supported |
| 2.1.1 Keyboard | A | Supported |
| 2.1.2 No Keyboard Trap | A | Supported |
| 2.4.1 Bypass Blocks | A | Supported |
| 2.4.2 Page Titled | A | Supported |
| 2.4.3 Focus Order | A | Supported |
| 2.4.6 Headings and Labels | AA | Supported |
| 2.4.7 Focus Visible | AA | Supported |
| 2.4.11 Focus Not Obscured (Minimum) | AA | Supported |
| 2.5.3 Label in Name | A | Supported |
| 2.5.8 Target Size (Minimum) | AA | Supported |
| 3.1.1 Language of Page | A | Supported |
| 3.2.2 On Input | A | Supported |
| 3.3.2 Labels or Instructions | A | Supported |
| 4.1.2 Name, Role, Value | A | Supported |
| 4.1.3 Status Messages | AA | Supported |

"Supported" means implemented and covered by an automated assertion. It does
**not** mean certified — see *Known gaps*.

---

## How the criteria are met

### Live captions are announced, not just displayed — 4.1.3

The most important decision in the project.

```html
<p id="live-caption" role="status" aria-live="polite" aria-atomic="true">
  Captions have not started yet.
</p>
```

- `role="status"` implies `aria-live="polite"`, so captions **never interrupt**
  whatever a screen reader user is currently reading. Assertive would be wrong
  here — a caption every few seconds would make the page unusable.
- The region exists in the initial HTML. A live region **added at the same
  moment as its content** is frequently not announced at all, which is the most
  common way this feature is quietly broken in the wild.
- `aria-atomic="true"` means each caption is announced whole, not as a diff
  against the previous one.

### The transcript is re-readable, not only announced — 1.3.1, 2.4.5

Announcing is not the same as being able to revisit. Every caption is also
appended to an ordered list:

```html
<ol id="transcript" class="transcript" aria-labelledby="transcript-heading">
  <li class="transcript__item">
    <span class="transcript__time" aria-hidden="true">14:32:07</span>
    <span class="transcript__text">…</span>
    <span class="transcript__meta">inference 412 ms</span>
  </li>
</ol>
```

The timestamp is `aria-hidden` because it is duplicated in the ordered list's
implicit position; announcing it twice is noise. Capped at 100 entries so the
DOM cannot grow without bound during a long session.

### Real controls only — 4.1.2, 2.1.1

Every control is a native element. There is not a single
`role="button"` on a `<div>`, `<span>` or `<img>` in the codebase, and
`tests/test_web.py` asserts this so a regression cannot land.

This matters more than it looks. `role="button"` is a *promise* of Enter and
Space activation, focusability and disabled state. A `<div>` provides none of
that automatically, so every use is a reimplementation that can be incomplete
in ways review does not catch.

### Keyboard and focus — 2.1.1, 2.1.2, 2.4.3, 2.4.7, 2.4.11

- A **skip link is the first focusable element**, targeting `<main tabindex="-1">`.
  It is hidden with `clip`-style off-screen positioning, never `display:none` or
  `visibility:hidden`, which would remove it from the tab order entirely.
- **Focus outlines are never removed.** `:focus-visible` gets a 3px outline at
  2px offset, plus a shadow so it survives any background. The stylesheet
  contains no `outline: none` and no `outline: 0` at all, and a test asserts
  that — so a future edit cannot quietly reintroduce one.

  `<main>` gets an indicator on *any* focus, not only `:focus-visible`, and that
  is deliberate. `main` is the skip link's programmatic landing target, and
  `:focus-visible` is not reliable for programmatic focus: browsers decide by
  heuristic and disagree. An earlier version had `main:focus { outline: none }`
  with a `:focus-visible` replacement, which meant a keyboard user who activated
  the skip link could land on the main region with **no** visible confirmation
  of where they were — a WCAG 2.4.7 failure that a screenshot review would not
  catch, because the focus ring is not in the screenshot. Outlining the region is
  the feedback the skip link exists to deliver.
- **No keyboard trap.** The `EventSource` is closed on `beforeunload` and on
  stop, and nothing ever captures focus. The SSE stream can end without
  stranding the user.
- **No positive `tabindex`** anywhere; the natural DOM order is the tab order.

### Landmarks and structure — 1.3.1, 2.4.1, 2.4.6

`<header>`, `<nav aria-label="Snapdragoon sections">`, `<main>`, `<footer>`,
plus `<aside aria-label="Inference status">` for the status panel.

Heading order is asserted, not assumed:

```
[1, 2, 2, 2, 3, 2, 3, 3, 3]
```

Exactly one `<h1>`, no skipped levels. The status panel is an `<aside>` rather
than a `<section>` specifically so its title does not become an `<h2>` preceding
the `<h1>` — a real bug that `verify_a11y.py` caught during the build.

### Colour and contrast — 1.4.1, 1.4.3, 1.4.11

| Token | Light | Contrast |
|---|---|---|
| `--ink` on `--paper` | `#16181d` / `#fff` | 16.9:1 |
| `--ink-muted` on `--paper` | `#4a4f5a` / `#fff` | 8.1:1 |
| `--brand` on `--paper` | `#0b4fd8` / `#fff` | 7.4:1 |
| `--brand-contrast` on `--brand` | `#fff` / `#0b4fd8` | 7.4:1 |
| `--warn-ink` on `--warn-bg` | `#6b3d00` / `#fff4e5` | 7.0:1 |

All body text clears 4.5:1. Dark mode is a separate checked token set, not an
inverted copy. Confidence bars are `aria-hidden` because the percentage is
already in text — colour is never the sole carrier of information.

Windows **forced-colors** mode is honoured explicitly, which most projects skip
and high-contrast users notice immediately.

### Motion and resize — 2.3.3, 1.4.4, 1.4.10, 1.4.12

`prefers-reduced-motion: reduce` collapses every animation and transition
globally. Layout is fluid to 320px with no horizontal scroll, and uses
`clamp()` for headings so text scales with the viewport rather than snapping.

### Spoken output is written for the ear — 2.5.3, 1.3.1

`format_detections()` exists because model labels are not speakable text.
ImageNet synsets are comma-separated synonym lists — `"jersey, T-shirt, tee
shirt, teeshirt"` — which would shred the sentence structure the function
builds. Labels are reduced to the primary synonym, lowercased, and stripped of
brackets. Percentages are spoken as "percent confidence" rather than `%`.

### Error states are announced — 3.3.1, 4.1.3

Connection loss, camera denial, and inference failure all write into a
`role="status"` region, so a screen reader user learns about the failure instead
of inferring it from a UI that silently stopped updating.

### The demo engine never lies — 1.1.1, 3.1.2

When no model runtime is available, the UI does not show plausible-looking
placeholder text. It shows a persistent notice:

> Demo engine active: no model inference is running and all captions are
> synthetic placeholders.

Synthetic captions are prefixed `[synthetic]`, and `/api/status` reports
`neural_accelerated: false` with a device string that says so. Accessibility
honesty includes not deceiving the user, including about your own software.

---

## Known gaps

Stated plainly rather than buried.

| Gap | Impact | Plan |
|---|---|---|
| No screen-reader testing with NVDA/JAWS/VoiceOver | Automated checks cannot prove announcement quality | Manual pass before submission; this is the top remaining task |
| No testing above 200% zoom or with a switch device | Reflow and target size are reasoned about, not measured | Manual pass with browser zoom and a switch/voice-control tool |
| Colour-blind simulation not performed | Contrast ratios are colour-independent, so risk is low |axe DevTools colour-blind emulation |
| No `forced-colors` screenshot evidence | Rule is implemented but unverified visually | Capture one on Windows high contrast |
| `prefers-contrast: more` not implemented | Users who want more contrast than AA get the default | Add a high-contrast token block |
| Captions not translated | English-only, matching the model | Out of scope for the challenge |

**This document is not a legal accessibility claim.** It is a developer's
statement of implementation and test status. A formal audit by a screen-reader
user with assistive technology is still required, and the honest position is
that this has not happened yet.

---

## Manual test script

Run this before you submit. It takes about ten minutes and is the difference
between "automated checks pass" and "I have actually used it".

**Keyboard only**
1. Load the page. Press <kbd>Tab</kbd> once — the skip link must appear and be
   visible.
2. <kbd>Enter</kbd> — focus must land in `<main>`.
3. <kbd>Tab</kbd> through every control. A visible focus ring on each.
4. Start captions with <kbd>Space</kbd>. Confirm no focus is trapped.
5. <kbd>Shift</kbd>+<kbd>Tab</kbd> backwards through the page. Order must be logical.

**Screen reader (VoiceOver: <kbd>⌘F5</kbd>)**
1. Open the rotor and go to Headings. You must see: Live captions → Scene
   description → About & accessibility, with nested subheadings.
2. Start captions. Each new caption must be announced without interrupting.
3. Stop. Confirm the announcement that captioning ended.
4. Navigate landmarks. Expect: banner, navigation, Inference status, main,
   contentinfo.
5. Run an image through Scene description. Confirm the sentence and each
   detection's label *and* percentage are read.

**Reduced motion and contrast**
1. macOS: System Settings → Accessibility → Display → Reduce motion. Reload.
2. macOS: Increase contrast. Reload. Text must stay readable.
3. Windows: Settings → Accessibility → Contrast themes → Aquatic.
4. Browser zoom to 200%. No horizontal scrolling, no clipped text.
