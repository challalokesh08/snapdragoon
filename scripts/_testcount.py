"""Count the test suite, so the number in the collateral cannot be a guess.

Six figures in the deck and the description document are the same figure: the
size of the test suite. Typed in by hand, it is wrong within one commit --
`tests/test_description_doc.py` adding six tests is enough, and nothing about a
126-looking slide looks wrong when it should say 132.

So the generators ask pytest instead. The cost is one subprocess per build, which
is nothing next to rendering a 14-slide deck, and it removes a whole class of
"did I remember to update that?" from the release checklist.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Used only if pytest cannot be invoked at all, so a build still produces a
# document rather than crashing. Printed by verify_test_count() below.
FALLBACK = 132


def count_tests(tests_dir: Path | None = None) -> int:
    """Number of tests pytest would collect, or FALLBACK if it cannot be asked.

    Counted with --collect-only rather than by running them: the figure should
    not depend on whether model weights happen to be present, since some tests
    skip without them and a skip-dependent count would wobble between a judge's
    machine and the author's.
    """
    target = tests_dir or (ROOT / "tests")
    try:
        out = subprocess.run(
            [sys.executable, "-m", "pytest", str(target), "--collect-only", "-q"],
            cwd=ROOT, capture_output=True, text=True, timeout=300,
        )
    except (OSError, subprocess.SubprocessError):
        return FALLBACK

    # The tail is either "N tests collected in Xs" or "no tests ran".
    m = re.search(r"(\d+)\s+tests?\s+collected", out.stdout)
    if m:
        return int(m.group(1))
    return FALLBACK


def verify_test_count() -> int:
    """count_tests(), but say so on stderr when the count is not trustworthy."""
    n = count_tests()
    if n == FALLBACK:
        print(f"  ! could not collect tests; using fallback count {FALLBACK}",
              file=sys.stderr)
    return n


if __name__ == "__main__":
    print(count_tests())
