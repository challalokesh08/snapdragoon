# Eligibility question for the organisers

Post this on the challenge page or send it to the Qualcomm contact. One question,
cheap to ask, and the answer is authoritative. Send it **today** — every other
item on the checklist can be finished in an evening, but this one gates
everything.

---

## Short version (for a comment on the Unstop page)

> **Question on the "should own a Snapdragon-powered laptop" guideline.**
>
> The submission guidelines say participants *"should own a Snapdragon-powered
> laptop"*, and they also say projects must be *"designed, developed, **or
> intended to be optimised for"* Snapdragon hardware. I did not have access to a
> Snapdragon device for this build.
>
> My project is built so that the hardware question is a configuration change
> rather than a rewrite: the CPU and NPU paths sit behind one `Engine` interface,
> and the NPU backend is implemented against the real QNN and Qualcomm AI Hub
> APIs. The CPU baseline is measured and checked against known-correct answers;
> **no NPU latency figure is claimed anywhere**, because I could not measure one.
>
> Is a project that is *intended to be optimised for Snapdragon* — with the
> optimisation path implemented and the hardware limitation stated explicitly —
> eligible without owning a Snapdragon laptop? Happy to provide any further
> detail. Thank you.

## Longer version (if they ask for detail, or by email)

Add this only if they respond asking what "intended to be optimised" means in
practice:

> Concretely, what is implemented rather than asserted:
>
> - A single `Engine` interface with three backends — `DemoEngine` (runs with no
>   model files and labels its own output as not-inference), `OnnxEngine` (CPU,
>   verified), and `QualcommEngine` (QNN / Qualcomm AI Hub / LiteRT). On
>   non-Snapdragon hardware `QualcommEngine` raises rather than silently falling
>   back, so a demo can never claim to be on an NPU when it is not.
> - The NPU conversion steps are documented command by command in
>   `docs/SNAPDRAGON_DEPLOYMENT.md`, including the expected QNN tensor names and
>   the KV-cache handling the Whisper export requires.
> - `scripts/benchmark.py --backend qualcomm` exists and prints the NPU figures.
>   It has never been run, because there was no device. That is stated in the
>   README rather than left for a judge to discover.
>
> What is **not** claimed: any Snapdragon latency number, any on-device test
> result, or any photograph of the app running on Snapdragon hardware.

## If the answer is no

Two fallbacks, in order of preference:

1. **Borrow or rent** a Snapdragon X / X Elite Windows laptop for a few hours.
   This is the better outcome than it looks: it unblocks the NPU benchmark, which
   is the single strongest thing that could be added to the writeup, and it
   removes the eligibility question entirely.
2. **Submit with the limitation stated.** The writeup already contains the
   paragraph that does this, in `SUBMISSION.md` under "The one paragraph that
   matters most", and the verified-vs-unverified table in
   `docs/SNAPDRAGON_DEPLOYMENT.md`. A clearly stated limitation is a far smaller
   risk than a disqualification discovered late.

## If there is no reply before 30 Sep

Do not wait. Submit, with the limitation stated in the proposal itself. The
guideline's own wording — *"or intended to be optimised for"* — is the argument,
and it is a reasonable one to have made in writing before submitting.
