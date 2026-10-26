"""Shared state, cached models and rendering helpers for the Streamlit demo."""
from __future__ import annotations

import html
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.corpus import load_test_texts  # noqa: E402
from src.hmm import Transitions  # noqa: E402
from src.lm import ALPHABET, BigramLM, TrigramLM, get_bigram, get_trigram  # noqa: E402
from src.zodiac import Zodiac408, load_z408  # noqa: E402

RESULTS = ROOT / "results"
# Memory cap per EM batch: hosted demos (e.g. Streamlit Community Cloud) have ~1 GB of RAM.
BATCH_BYTES = 2e8

# Validated palette (same as the experiment plots).
GOOD, BAD = "#006300", "#d03b3b"         # text colours for correct / wrong letters
TEXT, TEXT_2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BLUE, ORANGE = "#2a78d6", "#eb6834"
SURFACE = "#fcfcfb"
SEQ_CMAP = LinearSegmentedColormap.from_list(
    "seq_blue", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])


# ---------------------------------------------------------------- cached data


@st.cache_resource(show_spinner="Loading the bigram language model…")
def bigram() -> BigramLM:
    return get_bigram()


@st.cache_resource(show_spinner="Loading the trigram language model…")
def trigram() -> TrigramLM:
    return get_trigram()


@st.cache_resource(show_spinner="Loading the Zodiac 408…")
def z408() -> Zodiac408:
    return load_z408()


@st.cache_resource(show_spinner="Loading held-out books…")
def test_texts() -> dict[int, str]:
    return load_test_texts()


@st.cache_data
def zodiac_alpha(default: float = 0.02) -> float:
    """Alpha chosen on synthetic Zodiac-like ciphers in experiment 3 (never on the Z408)."""
    path = RESULTS / "exp3_map_summary.csv"
    if not path.exists():
        return default
    b = pd.read_csv(path).query("part == 'B'")
    return float(b.loc[b.acc_key_mean.idxmax(), "alpha"])


# ---------------------------------------------------------------- session state


@dataclass
class CipherState:
    """The ciphertext currently loaded in the demo, plus the truth when known."""

    cipher: np.ndarray            # symbol ids
    n_symbols: int
    glyphs: str                   # glyphs[s] = display character of symbol s
    plain: np.ndarray | None      # true plaintext letters, if known
    true_key: np.ndarray | None   # true symbol -> letter key, if known
    label: str

    def text(self) -> str:
        return "".join(self.glyphs[s] for s in self.cipher)

    def fingerprint(self) -> int:
        """Identifies the ciphertext content (stable across Streamlit reruns)."""
        return hash((self.cipher.tobytes(), self.n_symbols, self.glyphs))


def set_cipher(state: CipherState) -> None:
    st.session_state["cipher"] = state


def get_cipher() -> CipherState | None:
    return st.session_state.get("cipher")


# ---------------------------------------------------------------- rendering

_MONO = ("font-family: 'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, monospace;"
         "font-size: 15px; line-height: 1.7; word-break: break-all; letter-spacing: 0.06em;")


class Throttle:
    """Decides when a live view may redraw, so drawing takes at most ~25% of the time.

    Waits at least `min_interval` seconds, and at least three times the cost of the
    previous draw, between the end of one draw and the start of the next.
    """

    def __init__(self, min_interval: float = 0.35):
        self.min_interval = min_interval
        self.cost = 0.0
        self.last_end = -1e9

    def ready(self, force: bool = False) -> bool:
        wait = max(self.min_interval, 3.0 * self.cost)
        return force or time.perf_counter() - self.last_end >= wait

    def draw(self, fn, *args) -> None:
        start = time.perf_counter()
        fn(*args)
        self.last_end = time.perf_counter()
        self.cost = self.last_end - start


def mono_html(text: str) -> str:
    """Plain monospace block (wraps anywhere)."""
    return f"<div style=\"{_MONO}\">{html.escape(text)}</div>"


def colored_html(decoded: str, truth: str | None) -> str:
    """Decoded text, correct letters green and wrong ones red (when the truth is known)."""
    if truth is None:
        return mono_html(decoded)
    spans = []
    for d, t in zip(decoded, truth):
        color, weight = (GOOD, 400) if d == t else (BAD, 700)
        spans.append(f"<span style='color:{color};font-weight:{weight}'>{html.escape(d)}</span>")
    return f"<div style=\"{_MONO}\">{''.join(spans)}</div>"


def grid_html(text: str, cols: int, ref: str | None = None, upper: bool = True) -> str:
    """Text laid out as a grid; cells differing from `ref` are red and bold."""
    cell = "padding:1px 7px;text-align:center;"
    rows = []
    for r in range(0, len(text), cols):
        cells = []
        for i in range(r, min(r + cols, len(text))):
            ch = text[i].upper() if upper else text[i]
            if ref is None:
                color = f"color:{TEXT}"
            elif text[i] == ref[i]:
                color = f"color:{GOOD}"
            else:
                color = f"color:{BAD};font-weight:700"
            cells.append(f"<td style='{cell}{color}'>{html.escape(ch)}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return ("<table style='border-collapse:collapse;font-family:ui-monospace,Menlo,monospace;"
            f"font-size:17px'>{''.join(rows)}</table>")


def legend_html() -> str:
    return (f"<span style='color:{GOOD}'>■ correct letter</span> &nbsp; "
            f"<span style='color:{BAD};font-weight:700'>■ wrong letter</span>")


def symbol_order(state: CipherState, B: np.ndarray | None = None) -> np.ndarray:
    """Column order for the emission heatmap.

    With a known key, symbols are grouped by their true letter, so a learned key
    shows up as blocks on the diagonal. Otherwise by frequency in the ciphertext.
    """
    if state.true_key is not None:
        freq = np.bincount(state.cipher, minlength=state.n_symbols)
        return np.lexsort((-freq, state.true_key))
    return np.argsort(-np.bincount(state.cipher, minlength=state.n_symbols), kind="stable")


def emission_heatmap(B: np.ndarray, state: CipherState, title: str):
    """Heatmap of P(symbol | letter); returns a matplotlib figure."""
    order = symbol_order(state)
    S = B.shape[1]
    fig, ax = plt.subplots(figsize=(max(6.0, 0.17 * S + 1.5), 5.0), facecolor=SURFACE)
    im = ax.imshow(B[:, order], aspect="auto", cmap=SEQ_CMAP, vmin=0, vmax=1,
                   interpolation="nearest")
    ax.set_yticks(range(26), list(ALPHABET), fontsize=7.5, color=TEXT_2)
    ax.set_xticks(range(S), [state.glyphs[s] for s in order], fontsize=7.5, color=TEXT_2,
                  family="monospace")
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_xlabel("cipher symbol" + (" (grouped by true letter)" if state.true_key is not None
                                     else " (most frequent first)"), color=TEXT_2, fontsize=9)
    ax.set_ylabel("plaintext letter", color=TEXT_2, fontsize=9)
    ax.set_title(title, loc="left", fontsize=10, color=TEXT)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.ax.tick_params(labelsize=7, colors=TEXT_2)
    cb.outline.set_visible(False)
    cb.set_label("P(symbol | letter)", fontsize=8, color=TEXT_2)
    fig.tight_layout()
    return fig


def loglik_figure(history: list[float]):
    """Log-likelihood of the best restart per EM iteration (y-axis zoomed to the data)."""
    fig, ax = plt.subplots(figsize=(5.2, 3.4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    h = np.asarray(history)
    ax.plot(np.arange(len(h)), h, color=BLUE, linewidth=2)
    tail = h[min(len(h) - 1, 2):]  # skip the first steps, which dwarf the rest
    lo, hi = tail.min(), tail.max()
    ax.set_ylim(lo - 0.08 * (hi - lo + 1), hi + 0.08 * (hi - lo + 1))
    ax.set_xlabel("EM iteration", color=TEXT_2, fontsize=9)
    ax.set_ylabel("log-likelihood (best restart)", color=TEXT_2, fontsize=9)
    ax.tick_params(colors=MUTED, labelcolor=TEXT_2, labelsize=8)
    ax.grid(color=GRID, linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    fig.tight_layout()
    return fig


def restarts_figure(logliks: np.ndarray, best: int):
    """Final log-likelihood of every restart, sorted; the selected one in orange."""
    order = np.argsort(-logliks)
    fig, ax = plt.subplots(figsize=(5.2, 3.4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.scatter(np.arange(len(order)), logliks[order], s=28, color=BLUE, zorder=2)
    ax.scatter([0], [logliks[best]], s=70, color=ORANGE, zorder=3, label="selected restart")
    ax.set_xlabel("restart, sorted by final log-likelihood", color=TEXT_2, fontsize=9)
    ax.set_ylabel("final log-likelihood", color=TEXT_2, fontsize=9)
    ax.tick_params(colors=MUTED, labelcolor=TEXT_2, labelsize=8)
    ax.grid(color=GRID, linewidth=0.6)
    ax.legend(frameon=False, fontsize=8, labelcolor=TEXT_2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    return fig


def key_table(key: np.ndarray, state: CipherState) -> pd.DataFrame:
    """Symbol -> letter table for symbols present in the ciphertext."""
    counts = np.bincount(state.cipher, minlength=state.n_symbols)
    rows = []
    for s in np.argsort(-counts, kind="stable"):
        if counts[s] == 0:
            continue
        row = {"symbol": state.glyphs[s], "count": int(counts[s]), "decoded as": ALPHABET[key[s]]}
        if state.true_key is not None:
            row["true letter"] = ALPHABET[state.true_key[s]]
            row["correct"] = "✓" if key[s] == state.true_key[s] else "✗"
        rows.append(row)
    return pd.DataFrame(rows)


def transitions_for(model: str, power: float = 1.0) -> Transitions:
    """HMM transitions for 'bigram' (26 states) or 'trigram' (676 pair states)."""
    from src.decipher import bigram_transitions
    from src.hmm_trigram import trigram_transitions
    if model == "bigram":
        return bigram_transitions(bigram(), power)
    return trigram_transitions(trigram(), power)
