/* Snapdragoon front-end.
 *
 * Accessibility contract honoured here:
 *  - No innerHTML with model output. All text goes in via textContent, so a
 *    mis-transcribed string can never inject markup.
 *  - Announcements go through the two existing live regions. We never create or
 *    destroy the live region itself, because a region added at the same moment
 *    as its content is frequently not announced.
 *  - Button enable/disable is mirrored in aria-disabled as well, so the state
 *    is exposed even if the disabled attribute is mis-read.
 *  - The EventSource is always closed on stop, so focus and tab order are
 *    unaffected and no keyboard trap is possible.
 */
(function () {
  "use strict";

  var $ = function (id) { return document.getElementById(id); };

  var els = {
    engine: $("stat-engine"),
    device: $("stat-device"),
    npu: $("stat-npu"),
    warning: $("engine-warning"),
    start: $("btn-start"),
    stop: $("btn-stop"),
    clear: $("btn-clear"),
    live: $("live-caption"),
    transcript: $("transcript"),
    transcriptEmpty: $("transcript-empty"),
    camera: $("btn-camera"),
    sample: $("btn-sample"),
    file: $("file-image"),
    sceneResult: $("scene-result"),
    detections: $("scene-detections")
  };

  var source = null;          // EventSource
  var entries = [];           // transcript history
  var MAX_ENTRIES = 100;

  /* ---- status ---------------------------------------------------------- */
  function renderStatus(status) {
    if (!status) return;
    els.engine.textContent = status.engine || "unknown";
    els.device.textContent = status.device || "unknown";
    els.npu.textContent = status.neural_accelerated ? "Yes — NPU" : "No — CPU";

    // Be explicit when captions are synthetic. Quietly showing a placeholder
    // as if it were a real result would be the worst possible failure mode.
    if (status.engine === "demo") {
      els.warning.hidden = false;
      els.warning.textContent =
        "Demo engine active: no model inference is running and all captions are " +
        "synthetic placeholders. Install the onnxruntime extra and run " +
        "scripts/fetch_models.py to enable real speech recognition.";
    } else if (!status.asr_loaded && status.engine === "onnx") {
      els.warning.hidden = false;
      els.warning.textContent =
        "ONNX Runtime is active but the speech model is not loaded. Run " +
        "scripts/fetch_models.py to download the models.";
    }
  }

  function refreshStatus() {
    return fetch("/api/status")
      .then(function (r) { return r.json(); })
      .then(renderStatus)
      .catch(function () {
        els.warning.hidden = false;
        els.warning.textContent =
          "Could not reach the Snapdragoon server. Is it still running?";
      });
  }

  /* ---- captions -------------------------------------------------------- */
  function setCaptionState(state) {
    // state: "idle" | "running"
    var running = state === "running";
    els.start.disabled = running;
    els.start.setAttribute("aria-disabled", String(running));
    els.stop.disabled = !running;
    els.stop.setAttribute("aria-disabled", String(!running));
    var hasEntries = entries.length > 0;
    els.clear.disabled = !hasEntries;
    els.clear.setAttribute("aria-disabled", String(!hasEntries));
  }

  function appendEntry(text, latency) {
    var when = new Date();
    var stamp = when.toTimeString().slice(0, 8);
    var li = document.createElement("li");
    li.className = "transcript__item";

    var time = document.createElement("span");
    time.className = "transcript__time";
    // Machine-readable, so a screen reader does not read "03:14:07" oddly.
    time.setAttribute("aria-hidden", "true");
    time.textContent = stamp;

    var body = document.createElement("span");
    body.className = "transcript__text";
    body.textContent = text;

    li.appendChild(time);
    li.appendChild(body);

    if (typeof latency === "number") {
      var meta = document.createElement("span");
      meta.className = "transcript__meta";
      meta.textContent = "inference " + latency.toFixed(0) + " ms";
      li.appendChild(meta);
    }

    els.transcript.appendChild(li);
    entries.push(li);

    while (entries.length > MAX_ENTRIES) {
      var old = entries.shift();
      if (old && old.parentNode) old.parentNode.removeChild(old);
    }
    els.transcriptEmpty.hidden = true;
    setCaptionState(els.start.disabled ? "running" : "idle");
  }

  function announce(text) {
    els.live.textContent = text;
  }

  function startCaptions() {
    if (source) return;
    els.live.classList.remove("caption--empty");
    announce("Listening. Captions will appear here as speech is detected.");

    source = new EventSource("/api/captions?source=mic");

    source.onmessage = function (event) {
      var payload;
      try { payload = JSON.parse(event.data); } catch (err) { return; }

      if (payload.kind === "caption") {
        announce(payload.text);
        appendEntry(payload.text, payload.latency_ms);
      } else if (payload.kind === "skip") {
        // Intentionally not announced: a stream of "silence" messages would
        // make the live region unusable.
      } else if (payload.kind === "error") {
        announce("Captioning stopped: " + payload.message);
        stopCaptions();
      } else if (payload.kind === "end" || payload.kind === "closed") {
        stopCaptions();
        announce("Captioning ended.");
      }
    };

    source.onerror = function () {
      // EventSource auto-reconnects; surface it once rather than looping.
      if (source && source.readyState === 2) {
        stopCaptions();
        announce("Lost connection to the caption stream.");
      }
    };

    setCaptionState("running");
  }

  function stopCaptions() {
    if (source) {
      source.close();
      source = null;
    }
    setCaptionState("idle");
  }

  function clearTranscript() {
    while (els.transcript.firstChild) {
      els.transcript.removeChild(els.transcript.firstChild);
    }
    entries = [];
    els.transcriptEmpty.hidden = false;
    announce("Transcript cleared.");
    setCaptionState(els.start.disabled ? "running" : "idle");
  }

  /* ---- scene ----------------------------------------------------------- */
  function renderDetections(list) {
    els.detections.textContent = "";
    (list || []).forEach(function (d) {
      var li = document.createElement("li");
      li.className = "detection";

      var label = document.createElement("span");
      label.className = "detection__label";
      label.textContent = d.label;

      var score = document.createElement("span");
      score.className = "detection__score";
      score.textContent = Math.round(d.score * 100) + "%";

      var bar = document.createElement("span");
      bar.className = "detection__bar";
      // The percentage is already in the accessible text, so the bar is
      // decorative and must be hidden from assistive technology.
      bar.setAttribute("aria-hidden", "true");

      var fill = document.createElement("span");
      fill.className = "detection__fill";
      fill.style.width = Math.round(d.score * 100) + "%";

      bar.appendChild(fill);
      li.appendChild(label);
      li.appendChild(score);
      li.appendChild(bar);
      els.detections.appendChild(li);
    });
  }

  function describe(url, options) {
    els.sceneResult.textContent = "Analysing…";
    els.sceneResult.classList.remove("caption--empty");
    return fetch("/api/describe", options || { method: "POST" })
      .then(function (r) { return r.json().then(function (b) { return { ok: r.ok, body: b }; }); })
      .then(function (res) {
        if (!res.ok) {
          throw new Error(res.body.error || "request failed");
        }
        els.sceneResult.textContent = res.body.text;
        renderDetections(res.body.detections);
      })
      .catch(function (err) {
        els.sceneResult.textContent = "Could not describe the scene: " + err.message;
        els.sceneResult.classList.add("caption--empty");
        renderDetections([]);
      });
  }

  /* ---- wiring ---------------------------------------------------------- */
  els.start.addEventListener("click", startCaptions);
  els.stop.addEventListener("click", stopCaptions);
  els.clear.addEventListener("click", clearTranscript);

  els.camera.addEventListener("click", function () {
    els.sceneResult.textContent =
      "Camera capture is available from the command line via " +
      "scripts/demo.py --camera. Upload an image or use the sample instead.";
  });

  els.sample.addEventListener("click", function () {
    describe("/api/describe?image=assets/sample.jpg", { method: "POST" });
  });

  els.file.addEventListener("change", function () {
    var file = els.file.files && els.file.files[0];
    if (!file) return;
    var form = new FormData();
    form.append("image", file);
    describe("/api/describe", { method: "POST", body: form });
  });

  // Release the microphone if the tab is closed mid-stream.
  window.addEventListener("beforeunload", stopCaptions);

  renderStatus(window.SNAPDRAGOON_STATUS);
  setCaptionState("idle");
  refreshStatus();
})();
