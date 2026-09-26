# Submission checklist

**Deadline: 30 September 2026, 23:59 IST.** The whole checklist below is
designed to fit in the time remaining, ordered so that the items which are
cheapest to do and most likely to be forgotten come first.

---

## The one thing to resolve first

> **Eligibility.** The rules say participants *"should own a Snapdragon-powered
> laptop"*. No device was available for this build.

This is the only open risk in the project, and no amount of code fixes it.
Options, in order of preference:

1. **Ask.** Post on the challenge Unstop page or contact Qualcomm asking whether
   *"intended to be optimised for Snapdragon"* is sufficient without hardware. A
   one-line question costs nothing and the answer is authoritative.
2. **Borrow or rent** a Snapdragon X / X Elite Windows laptop for a few hours. This
   also unblocks the NPU benchmark, which is the single strongest thing you could
   add to the writeup.
3. If neither works, **submit anyway and be explicit** in the writeup. A clearly
   stated limitation is a far smaller risk than an eligibility disqualification
   discovered later — but this is a judgement call, so make it deliberately.

Do this **today**. Everything else here can be finished in an evening.

---

## 1. Repository hygiene (15 minutes)

- [ ] `git init`, add, commit. No remote yet — see the note below.
- [ ] Confirm `models/` is not committed (it is ~160 MB and is fetched, not vendored).
- [ ] Confirm the repo has no secrets and no `.env`.
- [ ] `LICENSE` present (MIT), with the Qualcomm trademark non-affiliation note.

**On the remote.** The submission form will want a repository URL. Create the GitHub
repo only when you are ready for it to be public, then push. Do not push to a
public remote and then keep changing things — the deadline snapshot should be
the final state.

## 2. Prove it works from a clean checkout (30 minutes)

This is the highest-value check in the list, because it is the exact path a judge
takes. Do it in a fresh directory, not your working copy:

```bash
git clone <your-repo> snapdragoon-clean
cd snapdragoon-clean

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m pytest tests/ -q          # must pass with no ML extras installed
python -m snapdragoon.web.app       # must serve on 127.0.0.1:8765
python scripts/verify_a11y.py       # must report 11/11
```

Then, with real inference:

```bash
pip install -r requirements-ml.txt
python scripts/fetch_models.py --public
python scripts/fetch_models.py --verify   # must report both models correct
python -m pytest tests/ -q                # including the -m slow accuracy tests
python scripts/benchmark.py --backend onnx --runs 20
```

- [ ] Every command above passes from a clean clone, in that order.
- [ ] `--verify` reports the reference transcript and the correct image class.

## 3. Record the demo (1 hour)

Screen recording beats slides, and the terminal demo is the easiest thing to
capture well. 90 seconds is plenty.

```bash
python scripts/demo.py --image assets/reference-photo.jpg
python scripts/demo.py --file assets/reference-speech.wav
```

- [ ] Terminal demo recorded, showing a real transcript and real confidences.
- [ ] Browser demo recorded: open <http://127.0.0.1:8765>, click **Start captions**,
      then **Use bundled sample image**.
- [ ] Record the screen reader pass (VoiceOver on macOS, NVDA on Windows). This
      is the part that proves the accessibility claim, and a reviewer who reads
      the proposal will ask about it.
- [ ] Show `/api/status` once. It states plainly which engine is running and
      whether it is NPU-accelerated. Showing it *proactively* is worth more than
      being asked.

**Do not edit the benchmark numbers by hand anywhere.** They come from
`scripts/benchmark.py`, and the JSON is proof.

## 4. Write the proposal

Map each section to the evaluation criteria you are actually scored on. For this
challenge they are roughly: technical depth, Snapdragon relevance, accessibility,
and the quality of the demonstration.

- [ ] Problem statement — who it helps and why the current answer is inadequate.
- [ ] What it does — the two models, the pipeline, the on-device claim.
- [ ] **Snapdragon relevance** — see the honest framing below. This is the
      criterion where overclaiming is fatal.
- [ ] **Measured results** — the table already in the README. CPU, labelled as CPU.
- [ ] Accessibility — point at `docs/ACCESSIBILITY.md` and the 11 assertions. This
      is the differentiator; give it real space rather than a bullet.
- [ ] What is next, including the NPU run.

### The one paragraph that matters most

State the hardware situation plainly, early, and without apology:

> *The CPU baseline is measured and asserted against known-correct answers. The
> Qualcomm NPU execution path is implemented against the real QNN and LiteRT APIs
> and documented step by step, but no Snapdragon device was available during the
> build, so no NPU latency figure is claimed. `python scripts/benchmark.py
> --backend qualcomm --runs 50` produces it on hardware and takes about a minute.*

That paragraph converts the project's biggest weakness into evidence of
engineering judgement. A qualified claim survives a technical interview with
Qualcomm engineers. An unqualified one that gets caught ends the conversation —
and the prize here *is* the conversation.

- [ ] Re-read the submission form for anything missed (video length, file format,
      team size, consent forms).
- [ ] Proofread the proposal against [`README.md`](README.md) for any number that
      appears in both.

## 5. Final 24 hours

- [ ] Re-run the clean-checkout check from §2. Do not skip this last time.
- [ ] Re-read [`docs/SNAPDRAGON_DEPLOYMENT.md`](docs/SNAPDRAGON_DEPLOYMENT.md)
      §"what is verified and what is not" and make sure the proposal agrees with
      it. They must not contradict each other.
- [ ] Submit at least **6 hours** before the deadline. The form rejecting a 160 MB
      upload at 23:58 is a real and entirely avoidable outcome.
- [ ] Screenshot the confirmation page.

---

## Judging-proof checklist

Claims that are actually *demonstrable*, in case you are asked in the interview:

| Claim | How it is proven |
|---|---|
| The models really run | `fetch_models.py --verify` checks the transcript against a known sentence |
| The captions are accessible | `verify_a11y.py`, 11 assertions, plus the recorded screen-reader pass |
| The DSP is correct | `tests/test_whisper_onnx.py` asserts the Whisper mel contract, filterbank area normalisation, and window periodicity |
| It is not a stub | 126 tests; `-m "not slow"` still passes with no weights present |
| Portability is real | one `Engine` ABC, three backends, benchmarked with one flag |
| The Snapdragon path is not vapourware | `docs/SNAPDRAGON_DEPLOYMENT.md` gives the exact QNN context build, the tensor names, and the three failure modes |

## Things to avoid

- **Do not claim an NPU measurement you did not make.** One caught claim voids
  everything else you say.
- **Do not describe the demo engine's output as inference.** It labels itself
  everywhere; keep it that way, and mention it before a reviewer finds it.
- **Do not claim WCAG conformance you have not tested manually.** The automated
  checks are necessary and not sufficient, and the docs say so.
- **Do not commit model weights.** Fetched, not vendored — see `models/README.md`.
