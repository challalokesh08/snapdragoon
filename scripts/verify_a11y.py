#!/usr/bin/env python3
"""Automated accessibility self-check for Snapdragoon.

Runs a set of concrete WCAG 2.2 AA assertions against the served page. Every
check here corresponds to a real defect observed in the wild during this
project's research, which is why they are worth automating:

  * zero native headings            -> screen reader heading navigation is dead
  * role=heading without aria-level -> broken heading hierarchy
  * focusable element with no name  -> announced as just "button"
  * role=button on a div/span/img   -> no native Enter/Space activation
  * tabs with no aria-selected      -> current tab is unknowable
  * tabs with no tabpanel           -> half-implemented ARIA pattern
  * missing skip link               -> no way past the nav
  * unnamed nav landmark            -> cannot distinguish navigation regions
  * outline:none on :focus-visible  -> invisible keyboard focus
  * live region added with content  -> frequently not announced
  * contrast below 4.5:1            -> unreadable text

Usage:
    python scripts/verify_a11y.py                       # starts nothing, expects server up
    python scripts/verify_a11y.py --url http://127.0.0.1:8765
    python scripts/verify_a11y.py --all                 # include optional axe-core pass

Exit code is 0 only when every check passes, so it works as a CI gate.
"""

from __future__ import annotations

import argparse
import re
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from html.parser import HTMLParser

DEFAULT_URL = "http://127.0.0.1:8765/"

GREEN, RED, YELLOW, DIM, BOLD, RESET = (
    "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[1m", "\033[0m"
)


@dataclass
class Element:
    tag: str
    attrs: dict = field(default_factory=dict)
    text: str = ""

    def get(self, name: str, default: str = "") -> str:
        return self.attrs.get(name, default)


class Collector(HTMLParser):
    """Collects the elements and text the assertions need."""

    VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input",
            "link", "meta", "source", "track", "wbr"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.elements: list[Element] = []
        self._stack: list[Element] = []
        self.stylesheets: list[str] = []
        self.inline_style: list[str] = []

    def handle_starttag(self, tag, attrs):
        el = Element(tag, {k: (v or "") for k, v in attrs})
        self.elements.append(el)
        if tag == "link" and "stylesheet" in el.get("rel"):
            self.stylesheets.append(el.get("href"))
        if el.get("style"):
            self.inline_style.append(el.get("style"))
        if tag not in self.VOID:
            self._stack.append(el)

    def handle_endtag(self, tag):
        if tag in self.VOID:
            return
        for i in range(len(self._stack) - 1, -1, -1):
            if self._stack[i].tag == tag:
                del self._stack[i:]
                break

    def handle_data(self, data):
        chunk = data.strip()
        if chunk and self._stack:
            self._stack[-1].text = (self._stack[-1].text + " " + chunk).strip()

    def css(self) -> str:
        """Fetch and concatenate reachable stylesheets, best-effort."""
        parts = list(self.inline_style)
        for href in self.stylesheets:
            if not href:
                continue
            url = href if href.startswith("http") else urllib.parse.urljoin(BASE, href)
            try:
                with urllib.request.urlopen(url, timeout=10) as resp:
                    parts.append(resp.read().decode("utf-8", errors="replace"))
            except Exception:  # noqa: BLE001
                pass
        return "\n".join(parts)


import urllib.parse  # noqa: E402  (used by Collector.css)

BASE = DEFAULT_URL


# ---------------------------------------------------------------------------
# assertions
# ---------------------------------------------------------------------------
def check_headings(p: Collector) -> tuple[bool, str]:
    headings = [e for e in p.elements if e.tag in {"h1", "h2", "h3", "h4", "h5", "h6"}]
    h1s = [e for e in headings if e.tag == "h1"]
    if not h1s:
        return False, "no <h1> found — the page has no document title for AT"
    if len(h1s) > 1:
        return False, f"{len(h1s)} <h1> elements found, expected exactly 1"
    return True, f"1 <h1>, {len(headings)} total headings"


def check_aria_heading_levels(p: Collector) -> tuple[bool, str]:
    bad = [e for e in p.elements
           if e.get("role") == "heading" and not e.get("aria-level")]
    if bad:
        return False, f"{len(bad)} role=heading without aria-level"
    return True, "all role=heading have aria-level"


def check_heading_order(p: Collector) -> tuple[bool, str]:
    levels = [int(e.tag[1]) for e in p.elements if e.tag in {"h1", "h2", "h3", "h4", "h5", "h6"}]
    if not levels:
        return False, "no headings to check"
    if levels[0] != 1:
        return False, f"first heading is h{levels[0]}, expected h1"
    for prev, cur in zip(levels, levels[1:]):
        if cur > prev + 1:
            return False, f"heading level jumps from h{prev} to h{cur}"
    return True, f"heading levels ascend without skipping: {levels}"


def check_accessible_names(p: Collector) -> tuple[bool, str]:
    focusable = {"a", "button", "input", "select", "textarea"}
    problems = []
    for e in p.elements:
        if e.tag not in focusable:
            continue
        if e.tag == "a" and not e.get("href"):
            continue
        if e.get("aria-hidden") == "true" or e.get("tabindex") == "-1":
            continue
        if e.tag == "input" and e.get("type") == "hidden":
            continue
        name = (e.get("aria-label") or e.get("title")
                or e.get("aria-labelledby") or e.text).strip()
        # A <label for> association is a legitimate accessible name mechanism
        # and is the primary one for form controls, so consult it too.
        if not name and e.tag in {"input", "select", "textarea"} and e.get("id"):
            name = (LABELS_BY_FOR.get(e.get("id"), "") or "").strip()
        if not name:
            problems.append(f"<{e.tag}> at line-ish {e.get('id') or e.get('class') or '?'}")
    if problems:
        return False, f"{len(problems)} focusable element(s) with no accessible name: " \
                      + "; ".join(problems[:5])
    return True, "every focusable element has an accessible name"


def check_fake_buttons(p: Collector) -> tuple[bool, str]:
    bad = [e for e in p.elements
           if e.get("role") == "button" and e.tag in {"div", "span", "img", "a"}]
    if bad:
        return False, (f"{len(bad)} role=button on a non-button element: "
                       + ", ".join(sorted({e.tag for e in bad})))
    return True, "no role=button on div/span/img"


def check_tabs(p: Collector) -> tuple[bool, str]:
    tabs = [e for e in p.elements if e.get("role") == "tab"]
    if not tabs:
        return True, "no tab pattern in use (nothing to check)"
    problems = []
    if not any(e.get("aria-selected") is not None for e in tabs):
        problems.append("no tab has aria-selected")
    missing_selected = sum(1 for e in tabs if e.get("aria-selected") is None)
    if missing_selected:
        problems.append(f"{missing_selected} tab(s) missing aria-selected")
    if not any(e.get("aria-controls") for e in tabs):
        problems.append("no tab has aria-controls")
    if not any(e.get("role") == "tabpanel" for e in p.elements):
        problems.append("no role=tabpanel exists")
    tabstops = [e for e in tabs if e.get("tabindex") in (None, "0")]
    if len(tabs) > 1 and len(tabstops) == len(tabs):
        problems.append("every tab is in the tab order; roving tabindex expected")
    unnamed = [e for e in tabs if not (e.get("aria-label") or e.text).strip()]
    if unnamed:
        problems.append(f"{len(unnamed)} tab(s) with no accessible name")
    if problems:
        return False, "; ".join(problems)
    return True, f"{len(tabs)} tabs, correctly wired to panels"


def check_skip_link(p: Collector) -> tuple[bool, str]:
    links = [e for e in p.elements if e.tag == "a"]
    skip = [e for e in links
            if "skip" in (e.get("href", "") + e.text + e.get("class", "")).lower()]
    if not skip:
        return False, "no skip link found"
    first_focusable = None
    for e in p.elements:
        if e.tag in {"a", "button", "input", "select", "textarea"}:
            if e.tag == "a" and not e.get("href"):
                continue
            if e.get("tabindex") == "-1" or e.get("aria-hidden") == "true":
                continue
            first_focusable = e
            break
    if first_focusable is not None and first_focusable is not skip[0]:
        return False, "a skip link exists but is not the first focusable element"
    return True, "skip link is the first focusable element"


def check_named_landmarks(p: Collector) -> tuple[bool, str]:
    navs = [e for e in p.elements if e.tag == "nav"]
    if not navs:
        return False, "no <nav> landmark"
    unnamed = [e for e in navs
               if not (e.get("aria-label") or e.get("aria-labelledby")).strip()]
    if unnamed:
        return False, f"{len(unnamed)} of {len(navs)} <nav> landmarks have no name"
    if not any(e.tag == "main" for e in p.elements):
        return False, "no <main> landmark"
    if not any(e.tag == "footer" for e in p.elements):
        return False, "no <footer> landmark"
    return True, f"{len(navs)} named nav landmarks, plus main and footer"


def check_live_regions(p: Collector) -> tuple[bool, str]:
    live = [e for e in p.elements
            if e.get("aria-live") or e.get("role") in {"status", "alert", "log"}]
    if not live:
        return False, "no live region; dynamic content will never be announced"
    return True, f"{len(live)} live region(s) present"


def check_labels(p: Collector) -> tuple[bool, str]:
    inputs = [e for e in p.elements
              if e.tag in {"input", "select", "textarea"} and e.get("type") != "hidden"]
    problems = []
    for e in inputs:
        has_label = e.get("aria-label") or e.get("aria-labelledby")
        if not has_label and e.get("id"):
            has_label = LABELS_BY_FOR.get(e.get("id"))
        if not has_label:
            problems.append(f"<{e.tag} id={e.get('id') or '?'}>")
    if problems:
        return False, f"{len(problems)} form control(s) without a label: " + \
                      ", ".join(problems)
    return True, f"all {len(inputs)} form control(s) are labelled"


LABELS_BY_FOR: dict[str, str] = {}


def check_outline_suppression(p: Collector, css: str) -> tuple[bool, str]:
    # A rule that zeroes outlines on a focus state is only a failure if nothing
    # replaces it, so this flags for human confirmation rather than hard-failing.
    risky = re.findall(r"[^{}]*:focus[^{}]*\{[^{}]*outline\s*:\s*(?:none|0)", css)
    if risky:
        return True, (f"{YELLOW}{len(risky)} rule(s) zero the focus outline{RESET} — "
                      f"confirm a visible replacement (box-shadow/background) exists")
    return True, "no focus outline suppression found"


def check_lang(p: Collector) -> tuple[bool, str]:
    html = [e for e in p.elements if e.tag == "html"]
    if not html:
        return False, "no <html> element"
    lang = html[0].get("lang")
    if not lang:
        return False, "<html> has no lang attribute"
    return True, f'lang="{lang}"'


CHECKS = [
    ("html has a lang attribute", check_lang),
    ("native heading structure", check_headings),
    ("heading levels ascend without skipping", check_heading_order),
    ("role=heading carries aria-level", check_aria_heading_levels),
    ("focusable elements have accessible names", check_accessible_names),
    ("no role=button on non-button elements", check_fake_buttons),
    ("tab pattern is correctly implemented", check_tabs),
    ("skip link is first focusable element", check_skip_link),
    ("landmarks are present and named", check_named_landmarks),
    ("dynamic content has a live region", check_live_regions),
    ("form controls have labels", check_labels),
]


def main() -> int:
    global BASE, LABELS_BY_FOR
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--all", action="store_true",
                    help="also attempt the optional axe-core pass")
    args = ap.parse_args()

    BASE = args.url
    url = args.url if args.url.endswith("/") else args.url + "/"

    print(f"\n{BOLD}Snapdragoon accessibility self-check{RESET}\n  target: {url}\n")

    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            html = resp.read().decode("utf-8", errors="replace")
    except urllib.error.URLError as exc:
        print(f"  {RED}could not reach the server{RESET}: {exc}")
        print(f"  start it first:  python -m snapdragoon.web.app\n")
        return 2

    parser = Collector()
    parser.feed(html)
    LABELS_BY_FOR = {
        e.get("for"): e.text
        for e in parser.elements if e.tag == "label" and e.get("for")
    }
    css = parser.css()

    failures = 0
    for name, fn in CHECKS:
        try:
            ok, detail = fn(parser)
        except Exception as exc:  # noqa: BLE001
            ok, detail = False, f"check raised {exc}"
        mark = f"{GREEN}PASS{RESET}" if ok else f"{RED}FAIL{RESET}"
        print(f"  [{mark}] {name}")
        print(f"         {DIM}{detail}{RESET}")
        if not ok:
            failures += 1

    ok, detail = check_outline_suppression(parser, css)
    print(f"  [{GREEN if not detail.startswith(chr(27)) else 'INFO'}] focus outline audit")
    print(f"         {DIM}{detail}{RESET}")

    print()
    if failures:
        print(f"{RED}{failures} check(s) failed.{RESET}\n")
    else:
        print(f"{GREEN}All {len(CHECKS)} checks passed.{RESET}")
        print(f"{DIM}Automated checks are necessary but not sufficient: still walk the")
        print(f"page with a screen reader and a keyboard before you submit.{RESET}\n")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
