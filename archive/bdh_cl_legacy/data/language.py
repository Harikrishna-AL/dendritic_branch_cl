"""Step 2A: domain-incremental language streams.

Four byte streams from genuinely different distributions:

``english``   tiny Shakespeare -- the corpus upstream already trains on
``python``    the local Python standard library, concatenated (works offline)
``german``    public-domain German prose/verse from Project Gutenberg
``hindi``     optional non-Latin script; multi-byte UTF-8 stresses a byte model

The non-Latin stream matters for a byte-level model in a way it would not for
a subword one: Devanagari codepoints are 3 bytes each in UTF-8, so the same
"amount of text" is three times the sequence length, and the model has to learn
multi-byte code structure it never needed for the Latin domains.

**Equal token budget.** Every domain is truncated to the same number of bytes
(the brief requires it), so no domain gets more gradient signal than another.
Since the vocabulary is bytes, equal bytes *is* equal tokens. Note the
consequence for the non-Latin stream: equal bytes means roughly a third as
much *text*, which is the honest reading of "equal token budget" for a byte
model and is recorded in NOTES.md.

Held-out split is the last ``val_frac`` of each stream by position, so train
and val never share a window.
"""

from __future__ import annotations

import sysconfig
from pathlib import Path

import numpy as np

from .bytes import ByteStream, encode

CACHE = Path(__file__).resolve().parents[1] / "data_cache"

SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/"
    "tinyshakespeare/input.txt"
)

#: Project Gutenberg ids, concatenated until the byte budget is met.
GUTENBERG: dict[str, list[int]] = {
    # Faust I, Faust II, Die Verwandlung, Grimms Märchen
    "german": [2229, 2230, 22367, 22555],
}

#: Unicode ranges a corpus must actually contain, checked by
#: :func:`require_script`. This exists because guessing Gutenberg ids for a
#: non-Latin corpus silently produced an *Esperanto* text labelled "hindi" --
#: a mislabelled corpus is worse than a missing one, because it would have
#: been reported as a non-Latin result.
SCRIPT_RANGES: dict[str, tuple[str, str]] = {
    "devanagari": ("\u0900", "\u097f"),
    "malayalam": ("\u0d00", "\u0d7f"),
    "latin": ("\u0000", "\u024f"),
}

#: Which script each named corpus must be in, and the minimum share of
#: characters that must fall inside it.
CORPUS_SCRIPT: dict[str, tuple[str, float]] = {
    "hindi": ("devanagari", 0.30),
    "malayalam": ("malayalam", 0.30),
}


def _download(url: str, dest: Path, timeout: int = 60) -> Path:
    import requests

    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    dest.write_bytes(r.content)
    return dest


def _strip_gutenberg(text: str) -> str:
    """Drop the licence header and footer, keeping only the work itself."""
    start = text.find("*** START OF THE PROJECT GUTENBERG")
    if start != -1:
        start = text.find("\n", start) + 1
    else:
        start = 0
    end = text.find("*** END OF THE PROJECT GUTENBERG")
    if end == -1:
        end = len(text)
    return text[start:end].strip()


# --------------------------------------------------------------------------
# Loaders -> raw bytes
# --------------------------------------------------------------------------


def load_english() -> np.ndarray:
    """Tiny Shakespeare. Reuses the upstream clone's copy if it is there."""
    upstream = Path(__file__).resolve().parents[3] / "upstream_bdh" / "input.txt"
    if upstream.exists():
        return np.frombuffer(upstream.read_bytes(), dtype=np.uint8)
    p = _download(SHAKESPEARE_URL, CACHE / "shakespeare.txt")
    return np.frombuffer(p.read_bytes(), dtype=np.uint8)


def load_python(budget_bytes: int | None = None) -> np.ndarray:
    """The local Python standard library, concatenated. Fully offline.

    Files are taken in sorted order so the corpus is reproducible, and test
    directories are skipped -- they are full of repetitive generated code that
    would make the domain artificially easy.
    """
    lib = Path(sysconfig.get_paths()["stdlib"])
    files = sorted(
        p
        for p in lib.rglob("*.py")
        if "test" not in p.parts and "site-packages" not in p.parts
    )
    chunks, total = [], 0
    for f in files:
        try:
            b = f.read_bytes()
        except OSError:
            continue
        chunks.append(b)
        total += len(b)
        if budget_bytes is not None and total >= budget_bytes:
            break
    if not chunks:
        raise RuntimeError(f"no stdlib .py files found under {lib}")
    return np.frombuffer(b"\n".join(chunks), dtype=np.uint8)


def load_gutenberg(name: str, budget_bytes: int | None = None) -> np.ndarray:
    """Concatenate public-domain Gutenberg texts until the budget is met."""
    ids = GUTENBERG.get(name)
    if not ids:
        raise KeyError(f"no Gutenberg ids registered for {name!r}")
    parts, total, failures = [], 0, []
    for gid in ids:
        url = f"https://www.gutenberg.org/cache/epub/{gid}/pg{gid}.txt"
        try:
            p = _download(url, CACHE / f"{name}_{gid}.txt")
            text = _strip_gutenberg(p.read_text(encoding="utf-8", errors="replace"))
        except Exception as e:  # network or decode failure
            failures.append(f"{gid}: {type(e).__name__}")
            continue
        if not text:
            failures.append(f"{gid}: empty after stripping boilerplate")
            continue
        parts.append(text)
        total += len(encode(text))
        if budget_bytes is not None and total >= budget_bytes:
            break
    if not parts:
        raise RuntimeError(
            f"could not fetch any {name} corpus (tried {ids}): {'; '.join(failures)}"
        )
    return encode("\n\n".join(parts))


def load_local(name: str, budget_bytes: int | None = None) -> np.ndarray:
    """A corpus the user drops into ``data_cache/<name>.txt``.

    This is how the optional non-Latin stream gets added: no Gutenberg id
    guessing, and :func:`require_script` still validates what lands here.
    """
    p = CACHE / f"{name}.txt"
    if not p.exists():
        raise RuntimeError(
            f"no corpus for {name!r}: put a UTF-8 text file at {p} "
            f"(Project Gutenberg has no plain-text Devanagari or Malayalam)"
        )
    return np.frombuffer(p.read_bytes(), dtype=np.uint8)


LOADERS = {
    "english": lambda budget=None: load_english(),
    "python": load_python,
    "german": lambda budget=None: load_gutenberg("german", budget),
    # Optional non-Latin streams: user-supplied, script-validated on load.
    "hindi": lambda budget=None: load_local("hindi", budget),
    "malayalam": lambda budget=None: load_local("malayalam", budget),
}


def require_script(name: str, data: np.ndarray) -> None:
    """Fail loudly if a corpus is not in the script its name promises."""
    rule = CORPUS_SCRIPT.get(name)
    if rule is None:
        return
    script, min_frac = rule
    lo, hi = SCRIPT_RANGES[script]
    text = bytes(data).decode("utf-8", errors="replace")
    if not text:
        raise RuntimeError(f"corpus {name!r} is empty")
    frac = sum(1 for c in text if lo <= c <= hi) / len(text)
    if frac < min_frac:
        raise RuntimeError(
            f"corpus {name!r} is only {frac:.1%} {script} characters "
            f"(need >= {min_frac:.0%}). It is probably the wrong text."
        )

#: The default Step 2A sequence: Latin prose -> code -> another Latin language.
DEFAULT_SEQUENCE = ["english", "python", "german"]


def load_corpus(name: str, budget_bytes: int | None = None) -> np.ndarray:
    if name not in LOADERS:
        raise KeyError(f"unknown corpus {name!r}; have {sorted(LOADERS)}")
    data = LOADERS[name](budget_bytes)
    require_script(name, data)
    return data


def corpus_stats(name: str, data: np.ndarray) -> dict:
    """Byte-level statistics, including how much of it is non-ASCII."""
    non_ascii = int((data >= 128).sum())
    text = bytes(data).decode("utf-8", errors="replace")
    return {
        "name": name,
        "n_bytes": int(len(data)),
        "n_chars": len(text),
        "bytes_per_char": len(data) / max(len(text), 1),
        "non_ascii_byte_frac": non_ascii / max(len(data), 1),
        "distinct_bytes": int(len(np.unique(data))),
    }


def build_language_tasks(
    names: list[str] | None = None,
    *,
    block_size: int = 256,
    budget_bytes: int | None = None,
    val_frac: float = 0.1,
    batch_size: int = 8,
    eval_batches_max: int = 24,
):
    """Load each domain, truncate all to one equal byte budget, wrap as tasks.

    With ``budget_bytes=None`` the budget is the size of the smallest available
    corpus, which is the largest budget under which all domains are equal.
    """
    from ..cl.tasks import LanguageCLTask

    names = names or DEFAULT_SEQUENCE
    raw = {n: load_corpus(n, budget_bytes) for n in names}

    budget = budget_bytes or min(len(v) for v in raw.values())
    short = {n: len(v) for n, v in raw.items() if len(v) < budget}
    if short:
        raise RuntimeError(
            f"corpora smaller than the requested budget {budget}: {short}"
        )

    tasks, stats = [], {}
    for n in names:
        data = raw[n][:budget]
        stats[n] = corpus_stats(n, data)
        tasks.append(
            LanguageCLTask(
                name=n,
                stream=ByteStream(data, block_size=block_size, val_frac=val_frac),
                batch_size=batch_size,
                eval_batches_max=eval_batches_max,
            )
        )
    return tasks, {"budget_bytes": int(budget), "corpora": stats}
