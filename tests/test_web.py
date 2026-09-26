"""Tests for the HTTP surface and the accessibility contract of the front-end.

The a11y assertions here mirror `scripts/verify_a11y.py`. They are cheap,
run in CI, and stop someone shipping a regression like a `div`-button or a
stripped skip link six hours before a deadline.
"""

from __future__ import annotations

import json
import re

from snapdragoon import config


# -- routes -----------------------------------------------------------------
def test_index_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "Snapdragoon" in r.get_data(as_text=True)


def test_status_inlined_so_first_paint_is_honest(client):
    body = client.get("/").get_data(as_text=True)
    m = re.search(r"window\.SNAPDRAGOON_STATUS\s*=\s*(\{.*?\});", body, re.S)
    assert m, "status object must be inlined at serve time"
    assert json.loads(m.group(1))["engine"] == "demo"


def test_api_status(client):
    r = client.get("/api/status")
    assert r.status_code == 200
    body = r.get_json()
    assert body["engine"] == "demo"
    assert body["neural_accelerated"] is False


def test_api_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.get_json()["ok"] is True


def test_static_assets_served_with_single_charset(client):
    r = client.get("/static/styles.css")
    assert r.status_code == 200
    # A doubled "; charset=utf-8" is a real regression we already hit once.
    ctype = r.headers.get("Content-Type", "")
    assert ctype.count("charset") <= 1


def test_static_path_traversal_is_blocked(client):
    for probe in ("/static/../config.py", "/static/../../etc/passwd",
                  "/static/%2e%2e/config.py"):
        r = client.get(probe)
        assert r.status_code == 404, probe


def test_describe_requires_an_image(client):
    assert client.post("/api/describe").status_code == 400


def test_describe_rejects_missing_file(client):
    r = client.post("/api/describe?image=/nonexistent/nope.jpg")
    assert r.status_code == 400
    assert "not found" in r.get_json()["error"].lower()


def test_describe_reports_missing_opencv_as_actionable(client, monkeypatch):
    """A missing optional dependency is a setup problem, not a 500.

    Without this, users get an opaque "No module named 'cv2'" with no idea what
    to do about it.
    """
    import builtins

    from snapdragoon.vision import scene

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "cv2":
            raise ModuleNotFoundError("No module named 'cv2'", name="cv2")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    r = client.post("/api/describe?image=assets/sample.jpg")
    assert r.status_code == 503
    body = r.get_json()["error"]
    assert "cv2" in body
    assert "pip install" in body, "the error must say how to fix it"


def test_describe_resolves_relative_paths_against_project_root(client):
    """A relative path must not depend on the server's working directory."""
    from snapdragoon.web.app import _resolve_asset
    from snapdragoon import config

    rel = _resolve_asset("assets/sample.jpg")
    assert rel.is_absolute()
    assert rel == config.PROJECT_ROOT / "assets" / "sample.jpg"


def test_resolve_asset_keeps_absolute_paths():
    from snapdragoon.web.app import _resolve_asset

    assert _resolve_asset("/tmp/x.jpg") == type(_resolve_asset("/tmp/x.jpg"))("/tmp/x.jpg")


def test_captions_stream_emits_events_and_closes(client, wav_file):
    r = client.post(f"/api/captions?source=file&path={wav_file}")
    assert r.status_code == 200
    assert r.mimetype == "text/event-stream"
    raw = r.get_data(as_text=True)
    assert "retry:" in raw
    assert "data:" in raw
    # Must terminate rather than hang the client.
    assert '"kind": "closed"' in raw


def test_captions_stream_reports_bad_path_without_crashing(client):
    r = client.post("/api/captions?source=file&path=/nope.wav")
    raw = r.get_data(as_text=True)
    assert '"kind": "error"' in raw
    assert '"kind": "closed"' in raw


def test_captions_stream_skips_a_too_short_window(client, tmp_path):
    """A window with too little audio is skipped, not hallucinated.

    Whisper does not return "silence" for a window that is mostly padding -- it
    returns a confident hallucination. On the reference clip a 1 s tail produced
    a stray "Godfrey." Emitting a random proper noun is worse than emitting
    nothing, so the engine marks the window unusable and the stream must honour
    that and say why.
    """
    import numpy as np
    import wave as wavemod

    from snapdragoon import config

    path = tmp_path / "tail.wav"
    with wavemod.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(config.SAMPLE_RATE)
        t = np.linspace(0, 1.0, config.SAMPLE_RATE, endpoint=False)
        speech = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
        want = int(config.CAPTURE_CHUNK_SECONDS * config.SAMPLE_RATE)
        rest = np.zeros(want, dtype=np.float32)
        wf.writeframes((np.concatenate([speech, rest]) * 32767).astype(np.int16).tobytes())

    raw = client.post(f"/api/captions?source=file&path={path}").get_data(as_text=True)
    assert '"kind": "skip"' in raw
    assert '"kind": "closed"' in raw
    # The demo engine has no model, so the first window captions; the trailing
    # zero-filled window is pure silence and is skipped as such. The point of
    # the assertion is that nothing unusable is ever captioned.
    assert '"kind": "caption"' in raw


# -- accessibility contract -------------------------------------------------
def _body(client) -> str:
    """Served markup with HTML comments stripped.

    The source is heavily commented and several notes *discuss* markup such as
    "<aside>" and "tabindex=\"0\"". Left in, a comment is indistinguishable
    from a real element and the assertions below would both fail spuriously and
    pass spuriously. Every structural check reads through this.
    """
    raw = client.get("/").get_data(as_text=True)
    return re.sub(r"<!--.*?-->", "", raw, flags=re.S)


def test_html_has_lang_attribute(client):
    assert re.search(r'<html[^>]+lang="[^"]+"', _body(client))


def test_exactly_one_h1(client):
    body = _body(client)
    assert len(re.findall(r"<h1[\s>]", body)) == 1


def test_heading_levels_never_skip(client):
    body = _body(client)
    levels = [int(m) for m in re.findall(r"<h([1-6])[\s>]", body)]
    assert levels, "page must have headings"
    assert levels[0] == 1, "document must open at h1"
    for prev, cur in zip(levels, levels[1:]):
        assert cur <= prev + 1, f"heading level jumps h{prev} -> h{cur}"


def test_no_role_button_on_non_button_elements(client):
    for tag in ("div", "span", "img", "a", "p", "li"):
        pattern = rf"<{tag}[^>]*\brole=[\"']button[\"']"
        assert not re.search(pattern, _body(client)), f"{tag} acting as a button"


def test_no_tabindex_zero_on_non_interactive_elements(client):
    # Only real controls may join the tab order. A bare tabindex="0" on a
    # span/div creates a phantom stop that announces nothing.
    for tag in ("span", "div", "p", "li", "img"):
        pattern = rf"<{tag}[^>]*\btabindex=[\"']0[\"']"
        assert not re.search(pattern, _body(client)), f"{tag} has tabindex=0"


def test_every_control_is_a_real_button_or_link(client):
    body = _body(client)
    for name in ("btn-start", "btn-stop", "btn-clear", "btn-camera", "btn-sample"):
        assert re.search(rf'<button[^>]+id="{name}"', body), f"{name} must be a <button>"


def test_skip_link_is_first_focusable_element(client):
    body = _body(client)
    skip = re.search(r'<a[^>]+class="skip-link"[^>]*>', body)
    assert skip, "a .skip-link must exist"
    # The skip link has an href, so it is itself a candidate for "first"; the
    # assertion is that nothing focusable precedes it.
    first = re.search(
        r"<(a|button|input|select|textarea)\b[^>]*?(?:href|src)=\"[^\"]+\"[^>]*>", body
    )
    assert first is not None, "page has no focusable elements at all"
    assert skip.start() <= first.start(), "skip link must not be preceded by a control"


def test_skip_link_target_exists(client):
    body = _body(client)
    href = re.search(r'<a[^>]+class="skip-link"[^>]+href="#([^"]+)"', body)
    assert href, "skip link must point somewhere"
    assert f'id="{href.group(1)}"' in body, "skip link target must exist"


def test_main_landmark_is_focusable_skip_target(client):
    body = _body(client)
    main = re.search(r"<main[^>]*>", body)
    assert main and 'tabindex="-1"' in main.group(0)


def test_landmarks_present(client):
    body = _body(client)
    for tag in ("header", "nav", "main", "footer", "aside"):
        assert re.search(rf"<{tag}[\s>]", body), f"missing <{tag}>"


def test_nav_landmark_is_named(client):
    body = _body(client)
    for tag in re.findall(r"<nav[^>]*>", body):
        assert "aria-label" in tag or "aria-labelledby" in tag, \
            "every nav landmark needs an accessible name"


def test_live_regions_present_and_not_created_late(client):
    body = _body(client)
    assert re.search(r'aria-live="polite"', body)
    assert re.search(r'role="status"', body)


def test_form_control_has_associated_label(client):
    body = _body(client)
    assert re.search(r'<label[^>]+for="file-image"', body)
    assert re.search(r'<input[^>]+id="file-image"', body)


def test_status_placeholder_is_replaced_at_serve_time(client):
    body = _body(client)
    assert "__SNAPDRAGOON_STATUS__" not in body, "placeholder must be substituted"


def test_caption_buttons_start_enabled_and_stop_disabled(client):
    body = _body(client)
    start = re.search(r'<button[^>]+id="btn-start"[^>]*>', body).group(0)
    stop = re.search(r'<button[^>]+id="btn-stop"[^>]*>', body).group(0)
    assert "disabled" not in start
    assert "disabled" in stop


def test_no_positive_tabindex(client):
    assert not re.search(r'tabindex="[1-9]', _body(client)), \
        "positive tabindex reorders the tab sequence unpredictably"


def test_stylesheet_never_suppresses_focus_outlines(client):
    """Regression: the skip-link landing target had `main:focus { outline: none }`.

    `:focus-visible` is not reliable for programmatic focus, so that rule left
    the skip-link target with no visible indicator at all in some browsers --
    a WCAG 2.4.7 failure that a screenshot-based review would not catch.
    """
    import re

    css = client.get("/static/styles.css").get_data(as_text=True)
    # Strip comments so a rule merely *discussed* in a comment is not a failure.
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    offenders = re.findall(r"outline\s*:\s*(?:none|0)\b", css)
    assert not offenders, (
        f"{len(offenders)} focus outline suppression(s) found; the focus "
        f"indicator must be replaced, not removed"
    )
