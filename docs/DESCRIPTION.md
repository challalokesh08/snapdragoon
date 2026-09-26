# Project description

Copy-paste blocks for the submission form, shortest first. Pick the one that
fits the character limit you are given; nothing here claims anything that is not
in the repository.

---

## Tagline (≈60 chars — title field, if there is one)

> On-device accessibility co-pilot: live captions and scene description.

## Very short (≈180 chars — where a form says "brief")

> Snapdragoon is an offline, on-device accessibility co-pilot. It captions
> speech and describes what the camera sees, as real text a screen reader can
> announce. No account, no network call, no data leaves the device.

## Short (≈420 chars — most likely size for a project abstract)

> **Snapdragoon** is an offline, on-device accessibility co-pilot for people who
> are blind, deaf or hard of hearing. It runs Whisper Tiny for live speech
> captions and MobileNet V2 for scene description, and renders both as real text
> in the DOM — announced through a polite live region, keyboard-operable, and
> verified by 11 automated WCAG 2.2 AA gates and 134 tests. We audited two live
> on-device-AI pages and found 30 accessibility issues between them, including
> 9 blockers; every category we found is now a regression test in ours. Three
> interchangeable backends — CPU, demo, and a Qualcomm AI Hub / QNN NPU path —
> sit behind one interface, so moving to a Snapdragon NPU is a configuration
> change rather than a rewrite. CPU latency is measured and published as a range;
> no NPU figure is claimed, because no Snapdragon device was available.

## Medium (≈900 chars — long abstract, or the "problem/solution" field)

> **The problem.** On-device AI demos are inaccessible by default, and it is not
> a rare edge case — it is the normal outcome of building a visual demo. Live
> captions that update silently are invisible to a screen reader. Scene
> descriptions drawn into a `<canvas>` have no text alternative at all. A `div`
> with a click handler is not a button: it has no role, no accessible name, no
> focus stop and no keyboard activation. We did not assume this — we audited two
> live on-device-AI pages against WCAG 2.2 AA using rendered-DOM inspection,
> computed styles and real keyboard traversal, and recorded **30 issues, 9 of
> them blockers**: zero heading structure, a primary journey with no focusable
> elements, a closed dialog that was not inert, unnamed controls, and no skip
> link.
>
> **What we built.** Snapdragoon is an offline, on-device accessibility co-pilot.
> Whisper Tiny produces live speech captions; MobileNet V2 describes what the
> camera sees with confidence scores. Both are rendered as real text in the DOM,
> announced through a polite live region so they never interrupt, with the full
> transcript kept so it can be re-read. The skip link is the first focusable
> element and its landing target keeps its focus ring. No `outline: none` appears
> anywhere in the codebase, and a test fails if it ever does.
>
> **Why it is built this way.** The app never imports a runtime; it talks to an
> `Engine` interface with three backends — a self-labelling demo engine, a
> verified CPU engine, and a Qualcomm AI Hub / QNN NPU engine that raises rather
> than silently falling back when off-Snapdragon. That is what allowed the
> project to be built and verified without the target hardware, and it means a
> Snapdragon port is configuration, not a rewrite.
>
> **What is measured, and what is not.** 134 tests pass from a clean clone. The
> reference clip transcribes exactly; the reference photograph classifies
> correctly. ASR runs a 5 s window in 190–400 ms (12–26× real time) and vision in
> ~5 ms (~190 fps) on a laptop CPU — published as a range, because two Python
> environments on the same machine differed by 2×. The Snapdragon NPU path is
> implemented and documented step by step but has **never been run on hardware**,
> so no NPU latency figure is claimed anywhere.

## One-line pitch (for a tweet, a banner, or a slide)

> On-device AI should not require sighted, hearing, keyboard-and-mouse users to
> operate it.

---

## Notes on what is deliberately absent

Do not add any of the following to the form without a device to back it up. Each
was considered and rejected:

| Do not write | Why |
|---|---|
| "Runs on the NPU" / "NPU-accelerated" | Never measured. `QualcommEngine` raises off-Snapdragon precisely so this cannot be implied. |
| Any Snapdragon latency number | None exists. `scripts/benchmark.py --backend qualcomm` would produce one on the hardware. |
| "Tested on Snapdragon X" | No Snapdragon device was available during the build. |
| "13.2× real time" | Measured on silence, which the engine declines to transcribe. The corrected figure is on real audio. |
| A single ASR millisecond figure | It varies 2× with the environment. The range is the honest claim. |

The eligibility question — whether "intended to be optimised for Snapdragon" is
sufficient without owning the hardware — is drafted separately in
`ELIGIBILITY_QUESTION.md`. It is worth asking before submitting, because it is
the one thing that can invalidate the entry.
