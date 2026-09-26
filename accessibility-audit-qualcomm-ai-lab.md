# Accessibility Audit — Snapdragon AI Lab

**URL:** https://www.qualcomm.com/snapdragon/ai-lab
**Date:** 2026-09-26
**Method:** Live rendered DOM inspection + computed styles + real keyboard (Tab) traversal in Chromium.
**Standard:** WCAG 2.2 Level AA

> The page is a client-rendered React SPA. The static HTML ships an empty
> `<div id="spa-root">`, so all findings below come from the **hydrated** DOM, not
> from the initial payload. Findings marked *unverified* need a human/screen-reader pass.

---

## Summary

The page has **no heading structure at all** and its **entire primary user journey is
non-interactive text**. Both are structural, not cosmetic, and both are invisible to
automated-only checks that only look at contrast or ARIA attributes.

| # | Severity | Issue | WCAG |
|---|----------|-------|------|
| 1 | Blocker | Zero headings in page content; all titles are `<span>` | 1.3.1, 2.4.6 |
| 2 | Blocker | The only 7 `role="heading"` elements are inside the cookie modal, starting at level 2 | 1.3.1, 4.1.2 |
| 3 | Blocker | All 4 steps are static text — `<main>` has **0** focusable elements | 2.1.1, 4.1.2 |
| 4 | Blocker | 6 links + 2 buttons with no accessible name | 4.1.2, 2.4.4 |
| 5 | Blocker | Closed off-canvas dialog is not `aria-hidden`/`inert`; its Close button is tab stop #5 | 2.4.3, 1.3.1 |
| 6 | Blocker | Footnote 1.69:1 and trademark line 1.83:1 contrast | 1.4.3 |
| 7 | Serious | Feedback widget's close button is `tabindex="-1"` — keyboard cannot dismiss it | 2.1.1, 2.1.2 |
| 8 | Serious | No skip link; 35 tab stops from header straight into 29 footer links | 2.4.1 |
| 9 | Serious | `aria-label` placed on `<svg>` instead of the `<button>` | 4.1.2 |
| 10 | Moderate | `title="Close"` is the only accessible name for the drawer close button | 4.1.2 |
| 11 | Moderate | Primary `<nav>` landmark has no accessible name | 1.3.1 |
| 12 | Moderate | No `aria-current` on any nav item | 1.3.1, 2.4.8 |
| 13 | Moderate | Language links missing `hreflang` / `lang`; `html lang="en"` not `en-US` | 3.1.1, 3.1.2 |
| 14 | Moderate | `role="dialog"` without `aria-modal`; no focus management | 4.1.2 |
| 15 | Moderate | 3 decorative images have verbose alt text duplicating adjacent copy | 1.1.1 |
| 16 | Moderate | 40 keyframe animations but only 2 `prefers-reduced-motion` rules | 2.3.3 |
| 17 | Minor | `<title>AI Lab</title>` — no brand or site context | 2.4.2 |
| 18 | Minor | No `width`/`height`/`loading` on any of 11 images | (CLS) |
| 19 | Minor | Footer links are 14–16px tall vs. 24×24 minimum | 2.5.8 |

---

## Blockers

### 1. The page has no headings whatsoever

```
document.querySelectorAll('h1,h2,h3,h4,h5,h6').length  ->  0
```

Every visual title is a bare `<span>` carrying hierarchy purely through `font-size`:

| Visual text | Actual element | Size | Semantics |
|---|---|---|---|
| Snapdragon AI Lab | `<span>` | 44px | none |
| Hands-on AI Learning Program… | `<span>` | 34px | none |
| Program overview | `<span>` | 34px | none |
| STEP 1 / 2 / 3 / 4 | `<span>` | 34px | none |
| Who can participate | `<span>` | 34px | none |
| Exciting rewards | `<span>` | 17px | none |

All are also `font-weight: 400`, so the visual hierarchy rests on size alone.

**Impact:** a screen reader user pressing <kbd>H</kbd>, or opening the headings rotor,
gets an empty list on this page. There is no way to skim or navigate by section.

**Fix** — use real native headings:

```html
<h1>Snapdragon AI Lab</h1>
<p class="subtitle">Hands-on AI Learning Program for tomorrow's innovators</p>

<h2>Program overview</h2>
<p>Snapdragon AI Lab enables students and builders…</p>

<h2>How it works</h2>
<ol>
  <li>
    <h3>Step 1</h3>
    <p><a href="/signup">Sign up for a Snapdragon AI Lab workshop</a></p>
  </li>
  <!-- … -->
</ol>

<h2>Exciting rewards</h2>
<h2>Who can participate</h2>
```

`<ol>` also conveys the sequence semantically, replacing the visual "STEP 1…4" labels.

### 2. The only headings on the page are inside the cookie modal

All 7 `role="heading"` elements live under `#onetrust-pc-sdk` (OneTrust Privacy
Preference Center), not in the content:

```
P[role=heading][aria-level=2] "Privacy Preference Center"
P[role=heading][aria-level=3] "Manage Consent Preferences"
P[role=heading][aria-level=4] "Strictly Necessary Cookies"
P[role=heading][aria-level=4] "Analytics Cookies"
P[role=heading][aria-level=4] "Personalization Cookies"
P[role=heading][aria-level=4] "Targeting Cookies"
P[role=heading][aria-level=3] "Cookie List"
```

Two problems: the hierarchy **starts at level 2** with no level 1, and the real page
contributes nothing — so a screen reader encounters a level-2 heading as the very
first heading on the site.

**Fix:** use native `<h2>`/`<h3>` in the OneTrust template config. This is a
third-party component, so fix it in the embed configuration, not in app code.

### 3. The entire primary journey is non-interactive

The four steps are plain text. The ancestor chain of "STEP 1":

```
SPAN > SPAN > STRONG > P > DIV > DIV > DIV
```

No `<a>`, no `<button>`, no `onclick`, no `role`, no `tabindex`, `cursor: auto`.

Counting focusable elements by landmark region:

```
{ "HEADER": 4, "DIALOG": 1, "FOOTER": 29, "OTHER": 1 }   // MAIN: absent
```

**`<main>` contains zero links and zero buttons.** "Sign up", "Buy a Snapdragon PC",
"present to Qualcomm mentors" — the three actions the page exists to drive — cannot be
reached by keyboard, are not announced as actionable, and are indistinguishable from
body copy in a screen reader's flat reading.

**Fix:** wrap each step's call to action in a real link (see the `<h1>`/`<ol>` markup
above). If these are meant to be non-interactive marketing copy, the step titles still
need to be `<h3>`s.

### 4. Eight controls have no accessible name

```
HEADER  <a href="https://www.qualcomm.com">   >>> UNNAMED <<<   (logo, SVG only)
FOOTER  <a href="https://www.qualcomm.com">   >>> UNNAMED <<<   (logo, SVG only)
FOOTER  <a href="linkedin.com/company/qualcomm/">   >>> UNNAMED <<<
FOOTER  <a href="x.com/qualcomm">                    >>> UNNAMED <<<
FOOTER  <a href="youtube.com/qualcomm">              >>> UNNAMED <<<
FOOTER  <a href="instagram.com/qualcomm/">           >>> UNNAMED <<<
HEADER  <button>  hamburger  — no name, no aria-expanded
HEADER  <button>  <svg aria-label="Go to workspace">  — label on the wrong element
```

Screen readers announce the first six simply as "link". The two header buttons
announce as "button".

**Fix:**

```html
<a href="https://www.qualcomm.com" aria-label="Qualcomm home">
  <svg aria-hidden="true" focusable="false" …>…</svg>
</a>

<a href="https://www.linkedin.com/company/qualcomm/" aria-label="Qualcomm on LinkedIn">…</a>
<a href="https://x.com/qualcomm"                   aria-label="Qualcomm on X">…</a>
<a href="https://www.youtube.com/qualcomm"         aria-label="Qualcomm on YouTube">…</a>
<a href="https://www.instagram.com/qualcomm/"      aria-label="Qualcomm on Instagram">…</a>

<button aria-label="Open menu" aria-expanded="false" aria-controls="primary-nav">…</button>
<button aria-label="Go to workspace">              <!-- on the button, not the svg -->
  <svg aria-hidden="true" focusable="false" …>…</svg>
</button>
```

Note `aria-label` on the workspace `<svg>` is also invalid ARIA — the element has no
`role="img"`, so per spec the label may be dropped entirely.

### 5. Closed off-canvas dialog is exposed and focusable

The "Developer Workspace" drawer is translated off-screen but never hidden from AT:

```js
{ role: "dialog", "aria-label": "developer-workspace" }   // aria-modal: null
transform: matrix(1, 0, 0, 1, 846, 0)                      // pushed 846px right
offsetParent: null                                         // visually closed
aria-hidden: null      // <-- not set
inert:        false     // <-- not set
```

Its **Close** button renders at `x=1628` in a 782px viewport and was **tab stop #5**,
right after the header. A keyboard user tabs into an invisible panel and lands on a
close button for a drawer they cannot see.

**Fix:**

```html
<!-- closed -->
<div id="workspace" role="dialog" aria-label="Developer workspace"
     aria-modal="true" aria-hidden="true" inert>…</div>
```

Remove `inert` on open, move focus into the panel, trap it while open, and restore
focus to the trigger on close.

### 6. Contrast failures in the legal/eligibility text

| Text | Colour | Background | Ratio | Required | Result |
|---|---|---|---|---|---|
| `*Internship at Qualcomm is subject to eligibility criteria…` | `rgb(187,192,200)` | `rgb(245,246,247)` | **1.69:1** | 4.5:1 | **FAIL** |
| `Arduino and UNO are trademarks or registered trademarks…` | `rgb(187,192,200)` | `#fff` | **1.83:1** | 4.5:1 | **FAIL** |

The eligibility disclaimer is the text that explains the internship's terms, and it
is effectively invisible — 2.7× below the minimum.

Passing, for context: `STEP n` red `rgb(231,19,36)` on `#f5f6f7` = 4.3:1 (OK as large
text); body copy `rgba(0,0,0,0.55)` composites to ≈`rgb(110,111,111)` ≈ **4.7:1** —
passes, but with almost no margin, so any darkening of the background will break it.

**Fix:** darken to at least `#767676` on `#f5f6f7` (4.54:1) — or `#595959` for
comfortable headroom on 12px text.

### 7. Feedback widget cannot be dismissed by keyboard

```html
<button role="button" id="QSIFeedbackButton-close-btn" tabindex="-1" …>
```

The trigger ("Feedback", tab stop #35) is reachable, but its close button is removed
from the tab order and has no accessible name. A keyboard-only user can open the
widget and cannot close it without a pointer.

**Fix:** drop `tabindex="-1"`, add `aria-label="Close feedback"`, and move focus to
the close button on open.

---

## Serious

### 8. No skip link

35 tab stops: 4 header items → 1 phantom dialog button → 29 footer links. There is no
"skip to main content" link, and no focusable content in `<main>` to skip *to*.

**Fix:** make the first focusable element a skip link.

```html
<a href="#main" class="skip-link">Skip to main content</a>
…
<main id="main" tabindex="-1">…</main>
```

```css
.skip-link {
  position: absolute; left: -9999px;
  /* NOT display:none / visibility:hidden — those remove it from tab order */
}
.skip-link:focus { left: 1rem; top: 1rem; z-index: 100; }
```

### 9–10. Fragile accessible names

The drawer's close button relies solely on `title="Close"`:

```html
<button class="… w-10 h-10 …" title="Close" type="button">
  <svg role="presentation" …>…</svg>
</button>
```

`title` is a weak fallback — inconsistently announced by VoiceOver and several
NVDA/JAWS configurations. `role="presentation"` on the SVG is correct, but add
`aria-hidden="true"` as well. Prefer a real label:

```html
<button aria-label="Close developer workspace" type="button">
  <svg aria-hidden="true" focusable="false" …>…</svg>
</button>
```

---

## Moderate

### 11–12. Landmarks and current page

Four `<nav>` elements; three are named, the primary one is not:

| nav | Accessible name |
|---|---|
| 0 — header | **none** |
| 1 | `aria-labelledby="quickLinks"` |
| 2 | `aria-labelledby="companyInfo"` |
| 3 | `aria-label="legal links"` |

```html
<nav aria-label="Primary">…</nav>
```

`aria-current` appears **zero** times across all navs. On a page reached from
"Developer", nothing tells a screen reader user where they are:

```html
<a href="/developer" aria-current="page">Developer</a>
```

### 13. Language attributes

`<html lang="en">` while the UI states "English (US)" — use `lang="en-US"`.
Neither language link carries `hreflang` or `lang`:

```html
<a href="http://www.qualcomm.cn/" hreflang="zh-Hans-CN" lang="zh-Hans">简体中文 (China)</a>
<a href="https://www.qualcomm.com/"   hreflang="en-US"      lang="en-US">English (United States)</a>
```

Without `lang="zh-Hans"`, Chinese text is announced with an English voice synthesiser.

### 14. Dialog semantics

`role="dialog"` without `aria-modal="true"`, and no focus trap or focus restoration
observed. If the drawer is modal, add `aria-modal="true"`, trap <kbd>Tab</kbd> while
open, close on <kbd>Esc</kbd>, and return focus to the trigger.

### 15. Inconsistent decorative-image policy

Eight images correctly use `alt=""`. Three purely illustrative ones instead carry
sentences — and the third duplicates the copy beside it:

```
alt="Person working on a laptop at a desk in a study environment."
alt="Laptop with transparent overlay connected to a development board."
alt="Illustration of two tickets labeled “Internship” and “Rewards” emerging from a box."
                                        ^ duplicates adjacent "Internship" / "Rewards" text
```

None convey information absent from the surrounding copy, so all three should be
`alt=""`, consistent with the other eight.

### 16. Reduced motion

**40** `@keyframes` animations; only **2** `prefers-reduced-motion` rules. Any
scroll-triggered or parallax animation on the steps/rewards sections is very likely
unaware of the preference.

```css
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {
    animation-duration: 0.01ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: 0.01ms !important;
    scroll-behavior: auto !important;
  }
}
```

---

## Minor / verify

- **17. Page title** — `<title>AI Lab</title>`. Multiple Qualcomm tabs are
  indistinguishable. Use `Snapdragon AI Lab | Qualcomm`.
- **18. Images** — none of the 11 carry `width`/`height`, and none use
  `loading="lazy"`. Causes layout shift, which is a cognitive and motor-access
  concern. Add intrinsic dimensions; lazy-load below-the-fold art.
- **19. Target size** — footer links measure 14–16px tall, under the WCAG 2.2
  24×24px minimum (2.5.8). The spacing exception may apply for links in a
  sentence, but these are stacked list items and likely fail. Needs manual
  spacing measurement.
- **20. Outline suppression** — 90 CSS rules set `outline: none`, including
  `:focus-visible` on real controls:
  ```css
  .GlobalFooter-module__manageSubscriptionLink:focus-visible { outline: none }
  .SimpleRTE-module__simpleRTE a:focus-visible,
  .SimpleRTE-module__simpleRTE a:focus-visible * { outline: none }
  ```
  **I tested this with real Tab keypresses and focus indicators are currently
  visible** (`outline-style: auto` on every stop, and the overlay link falls back
  to `box-shadow`). So this is not a current failure — but it is one CSS refactor
  away from becoming one. Add a visible replacement to any selector that zeroes
  outlines.
- **21. Copy error** — the Step 1 CTA reads "Sign up for a Snapdragon AI Lab
  **workshops**" (should be "workshop"). This string is the accessible name of the
  primary CTA, and it also degrades voice-control matching.
- **22. JS dependency** — all content is client-rendered, and a pre-hiding script
  applies `body { opacity: 0 !important }` for 3000ms. If JS fails or the timeout
  never fires, the page is blank (the `<noscript>` message is the only fallback).
- **23. Hero contrast** — *unverified.* The page title and subtitle are
  `rgb(255,255,255)` over a photographic hero. Contrast depends entirely on the
  image luminance, which I could not sample. If any part of the image is light, the
  title fails. Verify, and consider a solid or scrimmed overlay.

---

## The "Enable accessibility" link is an overlay, not a fix

Worth addressing directly, since it is the first thing in the page text:

```html
<a href="#" id="usntA40Link" onclick="return enableUsableNetAssistive()"
   aria-label="Enable accessibility">
  <div id="usntA40Txt" role="presentation" aria-hidden="true">Enable accessibility</div>
```

This is a third-party widget (`a40.usablenet.com`). Overlays like this **do not
remediate accessibility defects** and are widely discouraged — W3C and multiple
national enforcement bodies (e.g. the 2023 Target case) have taken the position that
they can create false confidence while leaving the underlying issues in place.

Concretely: every blocker above — no headings, static-text steps, unnamed links,
1.69:1 contrast — would be reported as **fully accessible** by such a widget while
remaining exactly as broken. It also injects a third-party script and an
`onclick`/`href="#"` control into the tab order.

Recommended action: do not treat it as mitigation. Fix the underlying issues, and
route genuine user reports to the accessibility contact in the footer.

---

## Fix order

1. **Headings** — convert the visual titles to native `h1`/`h2`/`h3`. Largest
   screen-reader payoff, lowest risk.
2. **Make the 4 steps real links** — restores the page's entire purpose for keyboard
   and screen reader users.
3. **Name the 8 unnamed controls** — mechanical, one-line changes.
4. **Contrast on the footnote + trademark line** — one colour token.
5. **`inert` + `aria-hidden` on the closed drawer**; remove `tabindex="-1"` from the
   feedback close button.
6. **Skip link**, nav naming, `aria-current`, language attributes.
7. Reduced motion, image alt/dimensions, dialog focus management, target sizes.
