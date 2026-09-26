# Accessibility Audit — Snapdragon® AI Lab Build & Present Challenge

**URL:** https://unstop.com/competitions/crp-snapdragon-ai-lab-build-present-challenge-qualcomm-1748893
**Date:** 2026-09-26
**Method:** Live rendered DOM inspection, computed styles, real keyboard (Tab / Arrow / Enter) traversal in Chromium.
**Standard:** WCAG 2.2 Level AA

Substantially healthier than the Qualcomm AI Lab page. It has real headings, a
properly named nav landmark, alt text on all images, and passing contrast. The
dominant problem is a **half-implemented ARIA tab widget that misrepresents itself
to assistive technology**.

---

## Summary

| # | Severity | Issue | WCAG |
|---|----------|-------|------|
| 1 | Blocker | Tablist is non-functional: arrow keys dead, no `aria-selected`, zero tabpanels | 4.1.2, 2.1.1 |
| 2 | Blocker | 9 controls are `role="button"` on `<div>`/`<img>`/`<un-icon>` — no native keyboard activation | 4.1.2, 2.1.1 |
| 3 | Blocker | Broken accessible names: `aria-label="calendar"`, unnamed first tab, 2 phantom tab stops | 4.1.2, 2.4.4 |
| 4 | Serious | "Read More" disclosure never sets `aria-expanded` | 4.1.2 |
| 5 | Serious | Zero live regions despite a live countdown and mutable registration state | 4.1.2 |
| 6 | Moderate | Countdown number and label are separate, ungrouped elements | 1.3.1 |
| 7 | Moderate | No `aria-current` anywhere | 1.3.1, 2.4.8 |
| 8 | Moderate | No `<header>`/`<footer>` landmarks; page is one `<app-public-competition>` element | 1.3.1 |
| 9 | Moderate | Toggle buttons (Watchlist, Share) expose no `aria-pressed`/`aria-expanded` | 4.1.2 |
| 10 | Minor | Several controls are 16–20px tall vs. 24×24 minimum | 2.5.8 |
| 11 | Minor | Nav promo links concatenate headline + description into one long name | 2.4.4 |

### Verified as passing

- Real `<h1>` ("Snapdragon® AI Lab Build & Present Challenge") plus five `<h2>`s —
  correct document structure.
- `<nav aria-label="Main Navigation">` is named.
- All 9 `<img>` elements have `alt` attributes.
- Contrast passes on every sampled string: 6.58:1, 11.03:1, 11.73:1, 21:1
  (required 4.5:1).
- Focus indicators visible on all 10 probed tab stops (`outline` or `box-shadow`
  present under real Tab traversal).
- Only 2 of 32 focusable elements lack a name — versus 8 of 35 on the Qualcomm page.
- The two submenu triggers correctly pair `aria-expanded="false"` with their trigger.

---

## Blockers

### 1. The tablist does not work, but claims to be a tablist

```html
<ul role="tablist">
  <li role="tab" tabindex="0"></li>                    <!-- unnamed -->
  <li role="tab" tabindex="0">Stages & Timeline</li>
  <li role="tab" tabindex="0">Details</li>
  <li role="tab" tabindex="0">Prizes</li>
  <li role="tab" tabindex="0">Reviews</li>
  <li role="tab" tabindex="0">FAQs & Discussions</li>
</ul>
```

Measured behaviour — focus started on "Details", then:

```
start (Details)       LI role=tab "Details"
after ArrowRight      LI role=tab "Details"     <- no movement
after ArrowRight      LI role=tab "Details"     <- no movement
after ArrowLeft       LI role=tab "Details"     <- no movement
after End             LI role=tab "Details"     <- no movement
after Home            LI role=tab "Details"     <- no movement

TABS: [ {"sel":null,"ti":"0"}, {"sel":null,"ti":"0"}, {"sel":null,"ti":"0"},
        {"sel":null,"ti":"0"}, {"sel":null,"ti":"0"}, {"sel":null,"ti":"0"} ]
```

Four compounding defects:

- **Arrow keys do nothing.** Arrow navigation is the defining keyboard interaction of
  the tabs pattern; it is absent.
- **`aria-selected` is `null` on all six tabs.** A screen reader announces "Details,
  tab" with no way to tell it is the current one.
- **`role="tabpanel"` count is `0`.** There are tabs but no panels, and no
  `aria-controls` on any tab. The relationship is entirely unexpressed.
- **All six tabs carry `tabindex="0"`.** The pattern requires a *roving* tabindex —
  the selected tab at `0`, the rest at `-1`. As built, a keyboard user tabs six times
  to cross the tab strip instead of pressing Right once.

This is worse than shipping no ARIA at all: the markup advertises a tab widget, so a
screen reader user commits to the arrow-key interaction, and it silently does nothing.

The first tab is also **unnamed** — it is almost certainly a logo/breadcrumb fragment
that should not be a tab (`talent-color` / `Go to Unstop Home page` in the text dump).

**Fix** — use real buttons, implement the pattern properly:

```html
<ul role="tablist" aria-label="Competition sections">
  <li role="presentation">
    <button role="tab" id="tab-details" aria-controls="panel-details"
            aria-selected="true" tabindex="0">Details</button>
  </li>
  <li role="presentation">
    <button role="tab" id="tab-prizes" aria-controls="panel-prizes"
            aria-selected="false" tabindex="-1">Prizes</button>
  </li>
  <!-- … -->
</ul>

<section role="tabpanel" id="panel-details" aria-labelledby="tab-details" tabindex="0">…</section>
```

Required behaviour: ↑/↓/←/→ move focus, `Home`/`End` jump to first/last, `Enter`/`Space`
activate, and the tablist holds a single tab stop.

### 2. Nine controls are fake buttons

`role="button"` applied to elements that have no native button behaviour:

```
DIV      role=button  aria-label="Unstop Logo"
DIV      role=button  aria-label="Login"
DIV      role=button  aria-label="Register"
DIV      role=button  aria-label="Report An Issue"    h=16px
IMG      role=button  aria-label="Unstop"
UN-ICON  role=button  aria-label="View Website"
UN-ICON  role=button  aria-label="calendar"
UN-ICON  role=button  aria-label="Watchlist"
UN-ICON  role=button  aria-label="Share"
```

`role="button"` is a *promise* of Enter and Space activation, focusability, and
disabled state. A `<div>` or custom element provides none of that automatically —
each must be reimplemented in JavaScript, and any omission is invisible in review.

Two bare focusable elements have **no role and no name at all** — they are announced
as nothing while still consuming a tab stop:

```
SPAN    tabindex=0  role=null  aria-label=""
UN-ICON tabindex=0  role=null  aria-label=""
```

**Fix:** use `<button type="button">` and put `aria-label` on the button, not on an
inner `<un-icon>`. Drop `tabindex="0"` from non-interactive spans.

### 3. Broken accessible names

**`aria-label="calendar"`** — the icon's internal name is being used as the button's
accessible name. A screen reader announces literally *"calendar, button"*. The action
is "add to calendar" or "add reminder".

**Unnamed first tab** in the tablist (see #1).

**Verbose nav promo names** — headline and description concatenate into one string:

```
"Post a Job or Internship Hire through jobs & …"
"Become a Mentor Shape the next generation an…"
```

The visible label and the accessible name should match, separated deliberately.

**Fix:**

```html
<a href="/post">Post a Job or Internship</a>
<button aria-label="Add competition deadline to calendar">
  <un-icon name="calendar" aria-hidden="true"></un-icon>
</button>
```

### 4. "Read More" never exposes its state

```html
<button>Read More</button>   <!-- aria-expanded: null, aria-controls: null -->
```

Verified by activating it:

```
before:       { text: "Read More",  exp: null, ctrl: null }
after click:  { text: "Read Less",  exp: null, ctrl: null }
body length:  1754 -> 5485 chars
```

The disclosure works visually and the body text grows by ~3,700 characters, but
`aria-expanded` is never written. A screen reader user hears no state change and
cannot tell whether the eligibility criteria are expanded or collapsed.

**Fix:**

```html
<button aria-expanded="false" aria-controls="eligibility-full">Read More</button>
<div id="eligibility-full" hidden>…</div>
```

Toggle both the attribute and `hidden` on activation.

### 5. No live regions at all

```
document.querySelectorAll('[aria-live],[role="status"],[role="alert"],[role="timer"]').length  ->  0
document.querySelectorAll('[aria-labelledby]').length                                          ->  0
```

This page has genuinely dynamic, consequential state: a deadline countdown, a
registration state that flips to "You've Registered", and a notice that "data on this
page gets updated every 15 minutes". None of it is announced, and nothing on the page
uses `aria-labelledby` at all.

**Fix** — announce discrete events, not a per-second tick:

```html
<p role="status" aria-live="polite">4 days left to submit</p>
<div role="status" aria-live="polite">Registration confirmed</div>
```

Update the countdown text only when the displayed value changes (e.g. on the day
boundary), not every second.

---

## Moderate

### 6. Countdown number and label are ungrouped

The markup nests as `<span>Days Left</span>` inside `<div>4 Days Left</div>`, with
the digit in a sibling element. There is no `role="timer"` and no grouping label, so
the number and its unit are not programmatically associated. Give the container a
single name:

```html
<div role="timer" aria-label="4 days left to submit">…</div>
```

### 7. No `aria-current`

Zero occurrences page-wide. The active section among Details / Prizes / Reviews is
not indicated. (This is the same gap the `aria-selected` omission in #1 creates.)

### 8. Landmark and component structure

No `<header>` and no `<footer>`. The entire page is a single custom element:

```
<app-public-competition>   ← contains everything
  <nav aria-label="Main Navigation">
  <main>
```

Widen the component's internal structure so the banner, the competition summary, and
the legal footer are real landmarks. The "Powered By", "Best Viewed in Chrome…", and
"Copyright © 2026 FLIVE Consulting" text currently sit inside the page body with no
`<footer>` wrapper.

### 9. Toggle state unexposed

`Watchlist` and `Share` are toggles but carry no `aria-pressed` or `aria-expanded`, so
their state is invisible. The submenu triggers do this correctly — apply the same
pattern here.

### 10. Target size

```
DIV  role=button  "Report An Issue"   h=16px
UN-ICON role=button "View Website"    h=32px
UN-ICON role=button "Watchlist"/"Share" h=32px
submenu expanders                      h=20px
```

"Report An Issue" at 16px and the submenu expanders at 20px fall below the WCAG 2.2
minimum of 24×24 CSS px (2.5.8). Needs manual verification against the spacing
exception.

---

## Fix order

1. **Rebuild the tablist** with `<button role="tab">`, roving tabindex, arrow-key
   handling, `aria-selected`, `aria-controls`, and real `role="tabpanel"` containers.
   Remove the unnamed first tab.
2. **Convert the nine `role="button"` divs/images/custom elements to `<button>`**, and
   remove `tabindex="0"` from the two unlabelled non-interactive spans.
3. **`aria-expanded` on "Read More"**, wired to the panel it controls.
4. **Fix the names**: `aria-label="calendar"` → a real action label; split the nav
   promo names.
5. **`role="status"` live regions** for countdown day changes and registration state;
   `aria-current` on the active tab.
6. Landmarks, target sizes, `aria-pressed` on Watchlist/Share.

---

## Practical notes (not accessibility)

- **Deadline:** 30 Sep 2026, 11:59 PM IST (`2026-09-30T23:59:00+05:30`). The
  Solution Submission Round opened 04 Sep 2026, 12:00 PM IST. **4 days remaining**
  as of this audit.
- **Eligibility:** residents of the Republic of India, aged 18 or above. You must own a
  Snapdragon-powered laptop. Not open to people residing outside India, or those
  affiliated with government agencies.
- **Format:** individual participation. Prize listed as pre-placement interviews.
  12,320 registered. Already registered on this account.
- **Requirement:** the solution must be designed, developed, or intended to be
  optimised for Snapdragon-powered HP PCs. New submissions, or significant
  modifications to pre-existing ones adding Qualcomm AI Hub / open-source models.
- **Worth clarifying:** the Qualcomm campaign page carries a footnote that a
  *Qualcomm internship* requires BTech/MTech in CS or ECE with CGPA 7.5+. The Unstop
  challenge eligibility is separate and much looser — Indian residency, 18+, own a
  Snapdragon laptop. The CGPA and stream conditions attach to the internship award,
  not to entering or submitting to the challenge. The page's own teaser text
  ("Stand a chance to win an Internship at Qualcomm*") is what makes this easy to
  misread as an entry requirement.
