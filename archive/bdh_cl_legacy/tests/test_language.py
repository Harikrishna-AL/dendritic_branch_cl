"""Step 2A language streams.

The script guard is the important one here: guessing Gutenberg ids for a
non-Latin corpus once produced an *Esperanto* text that would have been
reported as Hindi. A mislabelled corpus is worse than a missing one.

Network-dependent corpora are skipped when offline rather than failing the
suite.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bdh_cl.data import language as L  # noqa: E402
from bdh_cl.data.bytes import encode  # noqa: E402


def test_script_guard_rejects_wrong_script():
    esperanto = encode("Stranga heredaĵo: Romano originale verkita en Esperanto")
    with pytest.raises(RuntimeError, match="devanagari"):
        L.require_script("hindi", esperanto)

    with pytest.raises(RuntimeError, match="malayalam"):
        L.require_script("malayalam", encode("plain ascii text"))


def test_script_guard_accepts_the_right_script():
    L.require_script("hindi", encode("यह हिन्दी में लिखा गया एक वाक्य है"))
    L.require_script("malayalam", encode("ഇത് മലയാളത്തിൽ എഴുതിയ ഒരു വാക്യമാണ്"))


def test_script_guard_is_a_noop_for_unconstrained_corpora():
    for name in ("english", "python", "german"):
        L.require_script(name, encode("anything at all"))


def test_script_guard_rejects_empty():
    with pytest.raises(RuntimeError, match="empty"):
        L.require_script("hindi", np.zeros(0, dtype=np.uint8))


def test_missing_local_corpus_gives_an_actionable_error():
    with pytest.raises(RuntimeError, match="put a UTF-8 text file"):
        L.load_local("definitely_not_a_corpus")


def test_python_corpus_loads_offline():
    """The code domain must work with no network at all."""
    data = L.load_python(budget_bytes=200_000)
    assert len(data) >= 200_000
    text = bytes(data[:20_000]).decode("utf-8", errors="replace")
    assert "import" in text or "def " in text


def test_corpus_stats_detects_multibyte():
    ascii_stats = L.corpus_stats("x", encode("hello world"))
    assert ascii_stats["bytes_per_char"] == pytest.approx(1.0)
    assert ascii_stats["non_ascii_byte_frac"] == 0.0

    dev = L.corpus_stats("x", encode("हिन्दी"))
    assert dev["bytes_per_char"] > 2.5  # Devanagari is 3 bytes per codepoint
    assert dev["non_ascii_byte_frac"] > 0.9


def test_equal_token_budget_across_domains():
    """Every domain must get exactly the same number of bytes."""
    try:
        tasks, meta = L.build_language_tasks(["english", "python"], block_size=64)
    except Exception as e:  # pragma: no cover - offline
        pytest.skip(f"corpus unavailable: {e}")

    sizes = {n: s["n_bytes"] for n, s in meta["corpora"].items()}
    assert len(set(sizes.values())) == 1, sizes
    assert set(sizes.values()) == {meta["budget_bytes"]}

    for t in tasks:
        assert t.metric_name == "bits_per_byte"
        assert t.higher_is_better is False
        total = t.stream.n_bytes("train") + t.stream.n_bytes("val")
        assert total == meta["budget_bytes"]
        assert t.stream.n_bytes("val") > 0


def test_budget_larger_than_a_corpus_is_refused():
    with pytest.raises(RuntimeError, match="smaller than the requested budget"):
        L.build_language_tasks(["english"], block_size=64, budget_bytes=10**9)


def test_gutenberg_boilerplate_is_stripped():
    raw = (
        "legal header blah\n"
        "*** START OF THE PROJECT GUTENBERG EBOOK FAUST ***\n"
        "Habe nun, ach! Philosophie\n"
        "*** END OF THE PROJECT GUTENBERG EBOOK FAUST ***\n"
        "licence footer blah\n"
    )
    out = L._strip_gutenberg(raw)
    assert out == "Habe nun, ach! Philosophie"
    assert "PROJECT GUTENBERG" not in out
    # Text with no markers passes through unchanged.
    assert L._strip_gutenberg("just a poem") == "just a poem"
