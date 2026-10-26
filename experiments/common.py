"""Shared harness for the experiment scripts: cipher instances, method runners,
parallel execution, CSV output and a consistent plot style.

Each experiment script calls `run_tasks` with a list of task dicts; every task
builds its own cipher instance from a deterministic seed, so results are
reproducible and independent of the number of worker processes.
"""
from __future__ import annotations

import os

# One BLAS thread per worker: the per-step matrices are tiny, so processes
# scale far better than threads. Must be set before numpy is imported.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.baseline import frequency_analysis
from src.ciphers import CipherKey, encrypt, homophonic_key, simple_key
from src.corpus import sample_passages
from src.decipher import (DecipherConfig, DecipherResult, bigram_transitions, decipher,
                          run_em, run_em_from, select_and_decode)
from src.hmm_trigram import trigram_transitions
from src.lm import get_bigram, get_trigram, text_to_indices
from src.metrics import key_accuracy, symbol_accuracy

RESULTS = ROOT / "results"
MAX_BATCH_BYTES = 3e8  # per worker, keeps 7 trigram workers within 8 GB

# --------------------------------------------------------------------------- data


@dataclass
class Instance:
    """One synthetic cipher: held-out plaintext, random key, ciphertext."""

    length: int
    n_symbols: int
    index: int
    plain: np.ndarray
    key: CipherKey
    cipher: np.ndarray


def instance_seed(length: int, n_symbols: int, index: int, mode: str = "random") -> int:
    """Deterministic seed shared by all experiments for the same cipher setting."""
    return 1_000_003 * length + 1009 * n_symbols + index + (0 if mode == "random" else 7)


def make_instance(length: int, n_symbols: int, index: int, mode: str = "random") -> Instance:
    """Build cipher instance `index` for a (length, n_symbols) setting."""
    seed = instance_seed(length, n_symbols, index, mode)
    rng = np.random.default_rng(seed)
    plain = text_to_indices(sample_passages(1, length, seed=seed)[0])
    key = simple_key(rng) if n_symbols == 26 else homophonic_key(n_symbols, get_bigram().pi, rng)
    return Instance(length, n_symbols, index, plain, key, encrypt(plain, key, rng, mode=mode))


# ------------------------------------------------------------------------ methods


def emission_tv(B: np.ndarray, plain: np.ndarray, cipher: np.ndarray) -> float:
    """Mean total-variation distance between learned and true emission rows.

    The true row of letter k is the empirical distribution of the symbols that
    encrypt k in this ciphertext. Averaged over the letters that occur, weighted
    by their counts (so it measures the estimator, not the decoding).
    """
    N = np.zeros_like(B)
    np.add.at(N, (plain, cipher), 1.0)
    counts = N.sum(axis=1)
    present = counts > 0
    true_rows = N[present] / counts[present, None]
    tv = 0.5 * np.abs(B[present] - true_rows).sum(axis=1)
    return float(np.average(tv, weights=counts[present]))


def score(result: DecipherResult, plain: np.ndarray, true_key: np.ndarray,
          cipher: np.ndarray) -> dict:
    """Metrics of a decipherment: three decodings, key accuracy, likelihood, runtime."""
    return dict(acc_key=symbol_accuracy(result.decoded, plain),
                acc_viterbi=symbol_accuracy(result.viterbi, plain),
                acc_posterior=symbol_accuracy(result.posterior, plain),
                key_acc=key_accuracy(result.key, true_key, cipher),
                emission_tv=emission_tv(result.B, plain, cipher),
                loglik=result.loglik, n_iter=result.n_iter, runtime=result.runtime)


def run_freq(cipher: np.ndarray, n_symbols: int, plain: np.ndarray, true_key: np.ndarray) -> dict:
    """Frequency-analysis baseline (its single decoding fills all accuracy columns)."""
    t0 = time.perf_counter()
    key = frequency_analysis(cipher, n_symbols, get_bigram().pi)
    acc = symbol_accuracy(key[cipher], plain)
    return dict(acc_key=acc, acc_viterbi=acc, acc_posterior=acc,
                key_acc=key_accuracy(key, true_key, cipher), emission_tv=np.nan,
                loglik=np.nan, n_iter=0,
                runtime=time.perf_counter() - t0)


def staged_pool(cipher: np.ndarray, n_symbols: int, n_pool: int, seed: int):
    """Stage 1 of the staged method: bigram restarts (ML), reusable across settings."""
    t0 = time.perf_counter()
    cfg = DecipherConfig(n_restarts=n_pool, seed=seed, max_batch_bytes=MAX_BATCH_BYTES)
    pool = run_em(bigram_transitions(get_bigram()), cipher, n_symbols, cfg)
    return pool, time.perf_counter() - t0


def staged_from_pool(cipher: np.ndarray, n_symbols: int, pool, pool_time: float,
                     n_keep: int, config: DecipherConfig,
                     init_smoothing: float = 0.01) -> DecipherResult:
    """Stage 2: trigram EM from the n_keep best bigram solutions of `pool`.

    Equivalent to src.decipher.decipher_staged, but lets one pool serve many
    stage-2 settings. The reported runtime includes the pool's runtime.
    """
    t0 = time.perf_counter() - pool_time
    B0 = pool.B[np.argsort(-pool.loglik)[:n_keep]] + init_smoothing
    B0 /= B0.sum(axis=-1, keepdims=True)
    trans = trigram_transitions(get_trigram(), config.lm_power)
    res = run_em_from(trans, cipher, B0, config)
    return select_and_decode(trans, cipher, n_symbols, config, res, t0)


def run_method(method: str, cipher: np.ndarray, n_symbols: int, plain: np.ndarray,
               true_key: np.ndarray, seed: int = 0, n_restarts: int = 50,
               alpha: float = 0.0, lm_power: float = 1.0, n_pool: int = 1000,
               n_keep: int = 50) -> dict:
    """Run one method on one ciphertext and return its metrics.

    method: "freq", "bigram", "trigram" (random restarts) or "staged".
    """
    if method == "freq":
        return run_freq(cipher, n_symbols, plain, true_key)
    cfg = DecipherConfig(n_restarts=n_restarts, alpha=alpha, lm_power=lm_power, seed=seed,
                         max_batch_bytes=MAX_BATCH_BYTES)
    if method == "staged":
        pool, pool_time = staged_pool(cipher, n_symbols, n_pool, seed)
        result = staged_from_pool(cipher, n_symbols, pool, pool_time, n_keep, cfg)
    else:
        lm = get_bigram() if method == "bigram" else get_trigram()
        result = decipher(cipher, n_symbols, lm, cfg)
    return score(result, plain, true_key, cipher)


# ----------------------------------------------------------------------- running


def run_tasks(fn: Callable[[dict], list[dict]], tasks: list[dict], workers: int) -> pd.DataFrame:
    """Run fn(task) -> rows for every task (in parallel) and collect a DataFrame."""
    rows: list[dict] = []
    if workers <= 1:
        for t in tqdm(tasks):
            rows.extend(fn(t))
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for out in tqdm(ex.map(fn, tasks), total=len(tasks)):
                rows.extend(out)
    return pd.DataFrame(rows)


def parse_args(description: str) -> argparse.Namespace:
    """Common CLI: --quick for a smoke test, --workers for parallelism."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--quick", action="store_true", help="tiny run to check the script works")
    p.add_argument("--plot-only", action="store_true",
                   help="redraw the plots from the saved CSVs without running anything")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--instances", type=int, default=None,
                   help="ciphers per setting (overrides the script default)")
    p.add_argument("--pool", type=int, default=None,
                   help="bigram restarts in the staged pool (overrides 1000)")
    p.add_argument("--keep", type=int, default=None,
                   help="bigram solutions passed to the trigram stage (overrides 50)")
    return p.parse_args()


def output_name(name: str, quick: bool) -> str:
    """Quick runs write to *_quick files so they never overwrite real results."""
    return f"{name}_quick" if quick else name


def load_csv(name: str) -> pd.DataFrame:
    """Read results/<name>.csv."""
    return pd.read_csv(RESULTS / f"{name}.csv")


def save_csv(df: pd.DataFrame, name: str) -> Path:
    """Write a CSV to results/ and return its path."""
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"{name}.csv"
    df.to_csv(path, index=False)
    print(f"wrote {path.relative_to(ROOT)}")
    return path


def summarize(df: pd.DataFrame, by: list[str], cols: list[str]) -> pd.DataFrame:
    """Mean, std and count of `cols` grouped by `by`, flattened column names."""
    g = df.groupby(by)[cols].agg(["mean", "std", "count"])
    g.columns = [f"{c}_{s}" for c, s in g.columns]
    return g.reset_index()


# ------------------------------------------------------------------------- plots

# Validated reference palette (dataviz skill), fixed slot order, light surface.
SURFACE = "#fcfcfb"
TEXT, TEXT_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
CRITICAL = "#d03b3b"
SEQ_BLUE = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]  # ordinal ramp


def ordinal_colors(n: int) -> list[str]:
    """Steps of the blue ramp for n ordered groups (the light/mid/dark set is validated)."""
    return {1: [SEQ_BLUE[2]], 2: [SEQ_BLUE[2], SEQ_BLUE[4]],
            3: [SEQ_BLUE[0], SEQ_BLUE[2], SEQ_BLUE[4]]}.get(n, SEQ_BLUE[:n])

METHODS = {  # entity -> (label, color, marker, linestyle); colour follows the entity
    "bigram": ("HMM bigram", SLOTS[0], "o", "-"),
    "staged": ("HMM trigram (staged)", SLOTS[1], "s", "-"),
    "trigram": ("HMM trigram (random restarts)", SLOTS[2], "^", "-"),
    "freq": ("Frequency analysis", SLOTS[3], "D", "--"),
}
DECODINGS = {
    "acc_key": ("Consistent key", SLOTS[0], "o", "-"),
    "acc_viterbi": ("Viterbi", SLOTS[1], "s", "-"),
    "acc_posterior": ("Posterior (MPM)", SLOTS[2], "^", "--"),
}


def setup_style() -> None:
    """Thin marks, recessive grid and axes, text in text colours."""
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "figure.dpi": 110, "savefig.dpi": 160, "font.size": 10,
        "axes.edgecolor": AXIS, "axes.labelcolor": TEXT_2, "axes.titlecolor": TEXT,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": TEXT_2,
        "ytick.labelcolor": TEXT_2, "lines.linewidth": 2.0, "lines.markersize": 5.5,
        "legend.frameon": False, "legend.labelcolor": TEXT_2,
    })


def line_with_band(ax, x, mean, std, style: tuple, label_end: bool = True) -> None:
    """Mean line with markers, a +-1 std band, and an optional direct end label.

    End labels are queued on the axes and placed by `place_end_labels`, which
    `finish_figure` calls, so that labels of lines ending close together never overlap.
    """
    label, color, marker, ls = style
    x, mean, std = np.asarray(x, float), np.asarray(mean, float), np.nan_to_num(np.asarray(std, float))
    ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.14, linewidth=0)
    ax.plot(x, mean, color=color, marker=marker, linestyle=ls, label=label,
            markeredgecolor=SURFACE, markeredgewidth=1.2)
    if label_end:
        if not hasattr(ax, "_end_labels"):
            ax._end_labels = []
        ax._end_labels.append((x[-1], mean[-1], label))


def place_end_labels(ax, min_gap: float = 0.065) -> None:
    """Draw queued end labels, pushed apart to at least `min_gap` of the y-range."""
    items = sorted(getattr(ax, "_end_labels", []), key=lambda t: t[1])
    if not items:
        return
    lo, hi = ax.get_ylim()
    log = ax.get_yscale() == "log"
    to_frac = (lambda y: (np.log(y) - np.log(lo)) / (np.log(hi) - np.log(lo))) if log else \
        (lambda y: (y - lo) / (hi - lo))
    ys = [to_frac(y) for _, y, _ in items]
    for i in range(1, len(ys)):          # push up
        ys[i] = max(ys[i], ys[i - 1] + min_gap)
    if ys[-1] > 1.0:                     # top label left the axes: push down only as needed
        ys[-1] = 1.0
        for i in range(len(ys) - 2, -1, -1):
            ys[i] = min(ys[i], ys[i + 1] - min_gap)
    for (x, _, label), yf in zip(items, ys):
        ax.annotate(label, xy=(x, yf), xycoords=("data", "axes fraction"), xytext=(7, 0),
                    textcoords="offset points", va="center", fontsize=8.5, color=TEXT_2,
                    annotation_clip=False)
    ax._end_labels = []


def top_legend(fig, ax, title: str) -> None:
    """Figure title with a one-row legend under it, clear of all plotted data."""
    fig.suptitle(title, x=0.01, y=1.0, ha="left", va="bottom", fontsize=12, fontweight="bold")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower left", bbox_to_anchor=(0.01, 0.94), ncol=len(labels),
               fontsize=8.5, handlelength=2.6, columnspacing=1.6)
    fig.subplots_adjust(top=0.86)


def finish_figure(fig, name: str, note: str | None = None) -> Path:
    """Add an optional footnote, save results/<name>.png and close."""
    for ax in fig.axes:
        place_end_labels(ax)
    if note:  # below the lowest element of any axes (tick labels, x label)
        renderer = fig.canvas.get_renderer()
        bottom = min(ax.get_tightbbox(renderer).y0 for ax in fig.axes) / fig.bbox.height
        fig.text(0.01, bottom - 0.015, note, fontsize=7.5, color=MUTED, ha="left", va="top")
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"{name}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path.relative_to(ROOT)}")
    return path


def selected_alpha(default: float = 0.02) -> float:
    """Alpha for Zodiac-like ciphers, chosen on synthetic data in experiment 3 (part B).

    Falls back to `default` (the Phase 4 synthetic choice) if experiment 3 has not
    been run yet. Never tuned on the Zodiac 408 itself.
    """
    path = RESULTS / "exp3_map_summary.csv"
    if not path.exists():
        print(f"experiment 3 results missing, using alpha = {default}")
        return default
    s = pd.read_csv(path)
    b = s[s.part == "B"]
    return float(b.loc[b.acc_key_mean.idxmax(), "alpha"])
