"""Whisper ONNX inference: encoder/decoder and merged-export support.

Why this module exists
----------------------
Whisper has no single ONNX export format. Two shapes are common in the wild and
they are *not* interchangeable:

**A. Encoder + decoder with an explicit KV cache** (what
``onnx-community/whisper-*`` and most HF Optimum exports produce)

    encoder:  input_features (b, 80, 3000)  ->  last_hidden_state (b, 1500, 384)
    decoder:  input_ids, encoder_hidden_states,
              past_key_values.{0..23}.{decoder,encoder}.{key,value},
              use_cache_branch                       ->  logits (b, seq, 51864)
                                                                + present.*

The 24 cache tensors and the ``use_cache_branch`` boolean are **required
inputs**, not optional. The first call passes empty decoder caches and
``use_cache_branch=False``; every later call passes the previous ``present.*``
tensors and ``use_cache_branch=True``.

**B. A single merged graph** taking mel features and token ids together.

The two differ in input order, in tensor names, and in the axis order of
``logits`` -- ``(batch, vocab, tokens)`` in some merged exports,
``(batch, tokens, vocab)`` in the Optimum exports above.

That combination is a trap. An engine that reads inputs **positionally**
(``get_inputs()[0]``) and indexes ``logits[0, :, -1]`` will, against a
format-A export, feed decoder token ids into the mel slot and encoder hidden
states into the token slot, then read the wrong axis of the result. None of that
raises. It just returns confident-looking garbage.

So this module is **name-driven and shape-driven**: it locates tensors by name,
confirms each role by dtype and rank, and determines the logits axis from the
declared vocabulary size rather than assuming one.

Special-token ids come from the model config, never from a BPE tokenizer.
``tiktoken.get_encoding("gpt2").encode("<|startoftranscript|>")[0]`` returns
``27`` -- the first token of a BPE spelling of that string -- not the special
id ``50257``. Getting that wrong produces fluent, meaningless text.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .. import config

# Whisper's audio front-end is fixed by the model, not tunable.
N_FFT = 400
HOP = 160
N_MELS = 80
N_SAMPLES = 30 * config.SAMPLE_RATE  # 30 s context
N_FRAMES = N_SAMPLES // HOP          # 3000

# Defaults for whisper-tiny.en / whisper-base.en, used only when the model
# ships no config. Documented rather than guessed: if you point this at a
# different checkpoint, ship its config.json alongside it.
_DEFAULT_SPECIAL = {
    "sot": 50257,               # <|startoftranscript|> / decoder_start_token_id
    "eot": 50256,               # <|endoftext|>        / eos_token_id
    "no_timestamps": 50362,     # <|notimestamps|>
    "language": 50258,          # <|en|>
    "transcribe": 50358,        # <|transcribe|>
}

_FEATURE_NAMES = ("input_features", "features", "mel", "input")
_TOKEN_NAMES = ("input_ids", "decoder_input_ids", "tokens")
_LOGITS_NAMES = ("logits", "output")


class AsrUnavailable(RuntimeError):
    """No usable ASR graph, or the model files are missing."""


@dataclass
class SpecialTokens:
    sot: int
    eot: int
    no_timestamps: int
    language: int
    transcribe: int
    vocab_size: int = 51864
    source: str = "defaults"
    multilingual: bool = False

    def prompt(self) -> list[int]:
        """The conditioning prefix for a transcription.

        Timestamps are suppressed deliberately: a caption stream wants text, and
        timestamp tokens make the output awkward to both read and speak.

        The language and task tokens are included only for *multilingual*
        checkpoints. An English-only model (``whisper-tiny.en``) was never
        trained with them in this position, and feeding them costs accuracy --
        measured on the canonical JFK clip, ``<|startoftranscript|>
        <|en|> <|notimestamps|>`` drops the comma before "ask" and truncates the
        final clause, while ``<|startoftranscript|> <|notimestamps|>`` returns
        the reference transcript verbatim.
        """
        if self.multilingual:
            return [self.sot, self.language, self.transcribe, self.no_timestamps]
        return [self.sot, self.no_timestamps]


def load_special_tokens(model_dir: Path) -> SpecialTokens:
    """Read special-token ids from the model's own config.

    Prefers ``config.json`` (``decoder_start_token_id`` / ``eos_token_id``),
    then ``generation_config.json``, then ``added_tokens.json`` by name. Falls
    back to documented defaults and records which source was used, so a
    mismatch is visible rather than silent.
    """
    merged = dict(_DEFAULT_SPECIAL)
    sources: list[str] = []
    vocab_size = 51864
    multilingual = False

    # Config files may sit beside the graph or one level up, depending on how
    # the export was laid out. Check both rather than guessing.
    def _find(name: str) -> Path | None:
        for candidate in (model_dir / name, model_dir.parent / name):
            if candidate.is_file():
                return candidate
        return None

    cfg_path = _find("config.json")
    cfg = _read_json(cfg_path) if cfg_path else None
    if cfg:
        sources.append(cfg_path.name)
        if isinstance(cfg.get("decoder_start_token_id"), int):
            merged["sot"] = cfg["decoder_start_token_id"]
        if isinstance(cfg.get("eos_token_id"), int):
            merged["eot"] = cfg["eos_token_id"]
        if isinstance(cfg.get("vocab_size"), int):
            vocab_size = cfg["vocab_size"]
        # English-only checkpoints omit is_multilingual (or set it false).
        multilingual = bool(cfg.get("is_multilingual", False))

    gen_path = _find("generation_config.json")
    gen = _read_json(gen_path) if gen_path else None
    if gen and isinstance(gen.get("decoder_start_token_id"), int):
        sources.append(gen_path.name)
        merged["sot"] = gen["decoder_start_token_id"]
        if isinstance(gen.get("eos_token_id"), int):
            merged["eot"] = gen["eos_token_id"]

    added_path = _find("added_tokens.json")
    added = _read_json(added_path) if added_path else None
    if added:
        for name, key in (
            ("<|startoftext|>", "sot"),
            ("<|startoftranscript|>", "sot"),
            ("<|endoftext|>", "eot"),
            ("<|notimestamps|>", "no_timestamps"),
            ("<|en|>", "language"),
            ("<|transcribe|>", "transcribe"),
        ):
            if isinstance(added.get(name), int):
                merged[key] = added[name]
        if any(isinstance(added.get(n), int) for n, _ in (
            ("<|startoftranscript|>", "sot"), ("<|endoftext|>", "eot"),
            ("<|notimestamps|>", "no_timestamps"), ("<|en|>", "language"),
            ("<|transcribe|>", "transcribe"))):
            sources.append(added_path.name)

    return SpecialTokens(vocab_size=vocab_size, source="+".join(sources) or "defaults",
                         multilingual=multilingual, **merged)


def _read_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


# ---------------------------------------------------------------------------
# graph introspection
# ---------------------------------------------------------------------------
def _pick(session, candidates: tuple[str, ...], predicate=None):
    """Return ``(input_meta, name)`` for the first input matching a candidate
    name or the predicate. Name matching wins; the predicate is the fallback for
    exports that use uninformative names like ``input``/``input1``."""
    by_name = {i.name: i for i in session.get_inputs()}
    for cand in candidates:
        if cand in by_name:
            return by_name[cand], cand
    if predicate is not None:
        for meta in session.get_inputs():
            if predicate(meta):
                return meta, meta.name
    return None, None


def _is_mel(meta) -> bool:
    """A 3-D float input whose last dim is the 3000-frame mel axis."""
    shape = [d for d in meta.shape if isinstance(d, int)]
    if len(meta.shape) != 3 or "float" not in meta.type:
        return False
    return not shape or shape[-1] in (N_FRAMES, 0) or N_FRAMES in shape


def _is_token_ids(meta) -> bool:
    """A 2-D int64 input -- decoder token ids."""
    return len(meta.shape) == 2 and "int64" in meta.type


def cache_shape(declared: list, is_encoder: bool, encoder_len: int) -> tuple[int, ...]:
    """Concrete shape for an empty KV-cache input, from a declared shape.

    Walks the declared shape **positionally**. Symbolic dimensions must never
    be filtered out before indexing: for
    ``['batch_size', 6, 'past_decoder_sequence_length', 64]`` dropping the two
    strings leaves ``[6, 64]``, which then reads as ``heads=64`` and fails with
    ``INVALID_ARGUMENT ... index: 1 Got: 64 Expected: 6``.

    Axes are ``(batch, heads, seq, head_dim)``. The sequence axis is always the
    *true* length -- 0 for an empty decoder cache, ``encoder_len`` for the
    encoder cache -- never a symbolic name. Both are ignored while
    ``use_cache_branch`` is false, but the graph still requires real tensors.
    """
    declared = list(declared)
    while len(declared) < 4:  # tolerate a short/odd declaration
        declared.append(64)

    shape: list[int] = []
    for axis, value in enumerate(declared):
        if axis == 0:
            shape.append(value if isinstance(value, int) and value > 0 else 1)
        elif axis == 1:
            shape.append(value if isinstance(value, int) and value > 0 else 6)
        elif axis == 2:
            shape.append(encoder_len if is_encoder else 0)
        else:
            shape.append(value if isinstance(value, int) and value > 0 else 64)
    return tuple(shape)


class WhisperOnnxRunner:
    """Greedy Whisper decoding over ONNX Runtime, format-A and format-B aware."""

    def __init__(
        self,
        model_dir: Path,
        encoder_path: Path | None = None,
        max_new_tokens: int = 120,
    ) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover
            raise AsrUnavailable("onnxruntime is not installed") from exc

        model_dir = Path(model_dir)
        decoder_path = _find_decoder(model_dir)
        if decoder_path is None:
            raise AsrUnavailable(f"no Whisper ONNX graph found in {model_dir}")
        encoder_path = encoder_path or _find_encoder(model_dir)

        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        self._ort = ort
        self._max_new_tokens = max_new_tokens
        self.special = load_special_tokens(model_dir)

        decoder = ort.InferenceSession(str(decoder_path), opts,
                                       providers=["CPUExecutionProvider"])
        _tok, tok_name = _pick(decoder, _TOKEN_NAMES, _is_token_ids)
        _mel, mel_name = _pick(decoder, _FEATURE_NAMES, _is_mel)

        self._encoder = None
        if tok_name is not None and mel_name is None:
            # Format A: decoder-only graph, needs a separate encoder.
            if encoder_path is None:
                raise AsrUnavailable(
                    f"{decoder_path.name} is a decoder-only export but no encoder "
                    f"graph was found. Place encoder_model.onnx next to it."
                )
            self._encoder = ort.InferenceSession(str(encoder_path), opts,
                                                 providers=["CPUExecutionProvider"])
            self._enc_in = self._encoder.get_inputs()[0].name
            self._enc_out = self._encoder.get_outputs()[0].name
        elif tok_name is not None and mel_name is not None:
            # Format B: one merged graph, mel + tokens in the same session.
            self._merged = True
            self._merged_mel = mel_name
            self._merged_tok = tok_name
        else:
            raise AsrUnavailable(
                f"could not identify a Whisper graph in {decoder_path.name}: "
                f"inputs are {[i.name for i in decoder.get_inputs()]}"
            )

        self._decoder = decoder
        self._tok_name = tok_name
        _lg, _ = _pick(decoder, _LOGITS_NAMES)
        self._logits_name = _lg.name if _lg else decoder.get_outputs()[0].name
        self._logits_shape = list(_lg.shape) if _lg else []
        self._present_names = [o.name for o in decoder.get_outputs()
                               if o.name.startswith("present")]
        self._past_names = [i.name for i in decoder.get_inputs()
                            if i.name.startswith("past_key_values")]
        self._use_cache_name = next(
            (i.name for i in decoder.get_inputs() if "use_cache" in i.name), None
        )
        self._vocab_axis = self._resolve_vocab_axis()

    # -- introspection ----------------------------------------------------
    def _resolve_vocab_axis(self) -> int:
        """Which axis of ``logits`` is the vocabulary.

        ``-1`` for ``(batch, tokens, vocab)`` (the Optimum export) and ``-2`` for
        ``(batch, vocab, tokens)`` (several merged exports). Decided by
        comparing the declared shape against the model's vocab size, and only
        when a declared dimension actually matches -- never by guessing.
        """
        shape = [d for d in self._logits_shape if isinstance(d, int) and d > 0]
        v = self.special.vocab_size
        if len(shape) == 3 and shape[-1] == v:
            return -1
        if len(shape) == 3 and shape[-2] == v:
            return -2
        # Nothing matched (fully dynamic shape). Default to the Optimum layout
        # and let the first call confirm: a mismatch shows up as an immediate
        # EOT or garbage, not as a crash.
        return -1

    @property
    def format_name(self) -> str:
        return "merged" if getattr(self, "_merged", False) else "encoder+decoder"

    def describe(self) -> dict:
        return {
            "format": self.format_name,
            "logits_vocab_axis": self._vocab_axis,
            "special_tokens": {
                "sot": self.special.sot,
                "eot": self.special.eot,
                "vocab_size": self.special.vocab_size,
                "source": self.special.source,
            },
        }

    # -- inference --------------------------------------------------------
    def transcribe(self, features: np.ndarray) -> tuple[str, int]:
        """Decode ``features`` of shape (1, 80, 3000). Returns ``(text, steps)``."""
        if getattr(self, "_merged", False):
            ids = self._greedy_merged(features)
        else:
            ids = self._greedy_encoder_decoder(features)
        return self._detokenize(ids), len(ids)

    def _greedy_encoder_decoder(self, features: np.ndarray) -> list[int]:
        hidden = self._encoder.run([self._enc_out],
                                   {self._enc_in: features})[0]
        out_ids: list[int] = []

        # First pass: seed both caches.
        feeds = self._decoder_feeds(self.special.prompt(), hidden, past=None,
                                    use_cache_branch=False)
        outputs = self._decoder.run(None, feeds)
        enc_past = self._split_present(self._collect_present(outputs), "encoder")
        dec_past = self._split_present(self._collect_present(outputs), "decoder")

        while len(out_ids) < self._max_new_tokens:
            nxt = self._argmax(outputs, self._logits_name)
            if nxt == self.special.eot:
                break
            out_ids.append(nxt)
            feeds = self._decoder_feeds([nxt], hidden,
                                        past={**enc_past, **dec_past},
                                        use_cache_branch=True)
            outputs = self._decoder.run(None, feeds)
            # Only the decoder cache advances. The encoder runs exactly once, so
            # its KV is captured from the first pass and held for the rest.
            #
            # Re-reading it from later passes is a real trap: on the
            # use_cache_branch=True path the graph returns *placeholder*
            # encoder tensors shaped (0, 6, 1, 64) -- batch dimension zero.
            # Feeding those back silently destroys encoder attention from the
            # second token onward, which shows up as a correct first few words
            # followed by a spurious end-of-sequence. No error, no warning.
            dec_past = self._split_present(self._collect_present(outputs), "decoder")

        return out_ids

    def _greedy_merged(self, features: np.ndarray) -> list[int]:
        """Format B: re-feed the growing token sequence each step."""
        seq = list(self.special.prompt())
        out_ids: list[int] = []
        for _ in range(self._max_new_tokens):
            outputs = self._decoder.run(None, {
                self._merged_mel: features,
                self._merged_tok: np.array([seq], dtype=np.int64),
            })
            nxt = self._argmax(outputs, self._logits_name)
            if nxt == self.special.eot:
                break
            out_ids.append(nxt)
            seq.append(nxt)
        return out_ids

    def _decoder_feeds(self, ids, hidden, past, use_cache_branch: bool) -> dict:
        feeds: dict[str, np.ndarray] = {
            self._tok_name: np.array([list(ids)], dtype=np.int64),
        }
        if getattr(self, "_encoder", None) is not None:
            feeds["encoder_hidden_states"] = hidden

        for name in self._past_names:
            # The graph names its inputs `past_key_values.N.<block>.<key|value>`
            # and its outputs `present.N.<block>.<key|value>`. These are the
            # same tensors under two names. Matching them naively never fires,
            # so the cache is silently never used and every step re-decodes
            # from an empty context -- which shows up as fluent-looking but
            # degenerate repetition rather than as an error.
            key = name.replace("past_key_values.", "present.")
            if past is not None and key in past:
                feeds[name] = past[key]
            else:
                feeds[name] = self._empty_past(name, hidden)

        if self._use_cache_name is not None:
            feeds[self._use_cache_name] = np.array([use_cache_branch],
                                                   dtype=np.bool_)
        return feeds

    def _empty_past(self, name: str, hidden: np.ndarray) -> np.ndarray:
        """Zeros shaped to the declared input shape.

        The cache axes are ``(batch, heads, seq, head_dim)``. Decoder caches are
        empty on the first pass; encoder caches span the encoder output length.
        Both are ignored when ``use_cache_branch=False``, but the graph still
        requires correctly-shaped tensors.
        """
        meta = next((i for i in self._decoder.get_inputs() if i.name == name), None)
        declared = list(meta.shape) if meta is not None else [1, 6, 0, 64]
        enc_len = int(hidden.shape[1])
        return np.zeros(
            cache_shape(declared, is_encoder="encoder" in name, encoder_len=enc_len),
            dtype=np.float32,
        )

    def _collect_present(self, outputs: list) -> dict:
        names = [o.name for o in self._decoder.get_outputs()]
        return {n: o for n, o in zip(names, outputs) if n.startswith("present")}

    @staticmethod
    def _split_present(present: dict, which: str) -> dict:
        """Keep only the encoder or only the decoder half of the cache."""
        return {n: v for n, v in present.items() if f".{which}." in n}

    def _argmax(self, outputs: list, logits_name: str) -> int:
        names = [o.name for o in self._decoder.get_outputs()]
        logits = outputs[names.index(logits_name)]
        arr = np.asarray(logits)
        if arr.ndim == 3:
            # Take the newest position along the token axis.
            return int(np.argmax(arr[0, -1, :])) if self._vocab_axis == -1 \
                else int(np.argmax(arr[0, :, -1]))
        return int(np.argmax(arr[0]))

    def _detokenize(self, ids: list[int]) -> str:
        """GPT-2 byte-level BPE decode, with a documented ASCII fallback.

        Whisper's English vocabulary *is* GPT-2 BPE, so ``tiktoken`` decodes it
        correctly. Without it, printable-ASCII projection is lossy but better
        than nothing -- and it is reported rather than passed off as correct.
        """
        if not ids:
            return ""
        try:
            import tiktoken

            enc = tiktoken.get_encoding("gpt2")
            return enc.decode([i for i in ids if i < enc.n_vocab])
        except Exception:  # noqa: BLE001
            return "".join(chr(i) for i in ids if 32 <= i < 127)


# ---------------------------------------------------------------------------
# model discovery
# ---------------------------------------------------------------------------
def _find_decoder(model_dir: Path) -> Path | None:
    """Prefer a merged export, then a decoder, then anything ONNX."""
    order = [
        "decoder_model_merged.onnx", "model.onnx", "whisper.onnx",
        "decoder_model.onnx", "decoder_with_past_model.onnx",
    ]
    for name in order:
        if (model_dir / name).is_file():
            return model_dir / name
    found = sorted(model_dir.glob("*.onnx"))
    found = [p for p in found if "encoder" not in p.name.lower()]
    return found[0] if found else None


def _find_encoder(model_dir: Path) -> Path | None:
    for name in ("encoder_model.onnx", "encoder.onnx"):
        if (model_dir / name).is_file():
            return model_dir / name
    found = sorted(p for p in model_dir.glob("*encoder*.onnx"))
    return found[0] if found else None
