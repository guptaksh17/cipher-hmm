"""Key refinement after EM: hill-climbing on a hard (deterministic) key.

EM leaves a soft emission matrix and, through it, a key that is usually right for
frequent symbols but wrong for a few rare ones. This module polishes that key by
coordinate ascent on the joint log-likelihood of a deterministic decoding,

    score(key) = log P_LM(x) + log P(o | x),    x = key[o],

where P_LM is an n-gram letter model (order 4 by default) and the emission term
uses the maximum-likelihood homophone distribution implied by the key,
log P(o | x) = sum_k sum_s n_ks log(n_ks / n_k). The emission term is zero for a
one-to-one key; for homophonic keys it stops the search from collapsing many
symbols onto a few frequent letters.

Moves: for a homophonic key, reassign one symbol to another letter; for a
one-to-one key, swap the letters of two symbols. Each pass scores all candidate
moves for a symbol at once (vectorised) and accepts the best improving move; it
stops when a full pass finds no improvement.
"""
from __future__ import annotations

import math

from pathlib import Path

import numpy as np

from src.corpus import DATA_DIR, load_train_corpus
from scipy.optimize import linear_sum_assignment

from src.lm import ALPHABET, get_bigram, get_trigram, ngram_counts, text_to_indices

K = len(ALPHABET)


# ---------------------------------------------------------------- 4-gram model

def train_fourgram(text: str, beta: float = 4.0) -> np.ndarray:
    """log P(d | a, b, c) as a (26, 26, 26, 26) float32 array.

    Dirichlet smoothing towards the trigram model: P4 = (n(abcd) + beta P3) / (n(abc) + beta),
    so frequent contexts trust their own counts and rare ones fall back to trigrams.
    """
    idx = text_to_indices(text)
    n4 = ngram_counts(idx, 4, K)
    p3 = get_trigram().T                              # P(d | b, c), shape (b, c, d)
    num = n4 + beta * p3[None, :, :, :]
    p4 = num / num.sum(axis=-1, keepdims=True)
    return np.log(p4).astype(np.float32)


def get_fourgram(beta: float = 4.0) -> np.ndarray:
    """Load the 4-gram log-probability table from data/, training it if missing."""
    path = Path(DATA_DIR) / f"lm_order4_beta{beta:g}.npz"
    if path.exists():
        return np.load(path)["logp"]
    logp = train_fourgram(load_train_corpus(), beta)
    np.savez_compressed(path, logp=logp)
    return logp


def lm_tables(order: int = 4) -> np.ndarray:
    """Log-probability table of the requested order (3 or 4)."""
    if order == 3:
        return np.log(get_trigram().T).astype(np.float32)
    return get_fourgram()


# ---------------------------------------------------------------- scoring

def lm_score(X: np.ndarray, logp: np.ndarray) -> np.ndarray:
    """Sum of n-gram log-probabilities for each row of X (R, T) of letter indices."""
    n = logp.ndim
    T = X.shape[1]
    parts = tuple(X[:, j:T - n + 1 + j] for j in range(n))
    return logp[parts].sum(axis=1)


def emission_score(key: np.ndarray, counts: np.ndarray) -> float:
    """sum_k sum_s n_s log(n_s / n_k) for a deterministic symbol -> letter key."""
    n_k = np.bincount(key, weights=counts, minlength=K)
    used = counts > 0
    return float(np.sum(counts[used] * np.log(counts[used] / n_k[key[used]])))


# ---------------------------------------------------------------- hill-climbing

def refine_key(cipher: np.ndarray, key: np.ndarray, n_symbols: int, one_to_one: bool,
               order: int = 4, max_passes: int = 30) -> np.ndarray:
    """Improve a symbol -> letter key by hill-climbing on the joint log-likelihood."""
    logp = lm_tables(order)
    key = key.copy()
    counts = np.bincount(cipher, minlength=n_symbols).astype(float)
    symbols = np.argsort(-counts)                    # frequent symbols first
    symbols = symbols[counts[symbols] > 0]
    positions = {s: np.flatnonzero(cipher == s) for s in symbols}

    def total(k: np.ndarray) -> float:
        return float(lm_score(k[cipher][None], logp)[0]) + (0.0 if one_to_one else emission_score(k, counts))

    best = total(key)
    for _ in range(max_passes):
        improved = False
        for s in symbols:
            if one_to_one:
                others = [t for t in range(n_symbols) if t != s]
                cands = np.repeat(key[None], len(others), axis=0)
                cands[np.arange(len(others)), s] = key[others]
                cands[np.arange(len(others)), others] = key[s]
                scores = lm_score(cands[:, cipher], logp)
            else:
                letters = np.array([c for c in range(K) if c != key[s]])
                X = np.repeat(key[cipher][None], len(letters), axis=0)
                X[:, positions[s]] = letters[:, None]
                scores = lm_score(X, logp)
                cands = np.repeat(key[None], len(letters), axis=0)
                cands[:, s] = letters
                scores = scores + np.array([emission_score(c, counts) for c in cands])
            j = int(np.argmax(scores))
            if scores[j] > best + 1e-9:
                best, key = float(scores[j]), cands[j].copy()
                improved = True
        if not improved:
            break
    return key


def key_score(cipher: np.ndarray, key: np.ndarray, n_symbols: int, one_to_one: bool,
              order: int = 4) -> float:
    """Joint log-likelihood used to compare keys (higher is better)."""
    counts = np.bincount(cipher, minlength=n_symbols).astype(float)
    lm = float(lm_score(key[cipher][None], lm_tables(order))[0])
    return lm + (0.0 if one_to_one else emission_score(key, counts))


def _xlogx(v: float) -> float:
    return v * math.log(v) if v > 0 else 0.0


def anneal_key(cipher: np.ndarray, key: np.ndarray, n_symbols: int, iters: int,
               rng: np.random.Generator, order: int = 4, t_start: float = 8.0,
               t_end: float = 0.2) -> np.ndarray:
    """Simulated annealing over many-to-one (homophonic) keys.

    A move reassigns one symbol to another letter. Its score change touches only the
    n-gram windows that contain that symbol and the emission terms of the two letters
    involved, so each move costs a few microseconds. Worse moves are accepted with
    probability exp(delta / temperature); the temperature falls geometrically from
    `t_start` to `t_end`, which lets the search escape the traps that stop greedy
    hill-climbing on short homophonic ciphers. Returns the best key visited.
    """
    logp = lm_tables(order)
    n = logp.ndim
    key = key.copy()
    x = key[cipher].copy()
    T = len(x)
    counts = np.bincount(cipher, minlength=n_symbols).astype(float)
    present = np.flatnonzero(counts > 0)
    pos = {s: np.flatnonzero(cipher == s) for s in present}
    wins = {s: np.unique(np.clip((pos[s][:, None] - np.arange(n)).ravel(), 0, T - n)) for s in present}
    letter_n = np.bincount(key, weights=counts, minlength=K)

    def window_sum(w: np.ndarray) -> float:
        return float(logp[tuple(x[w + j] for j in range(n))].sum())

    current = float(lm_score(x[None], logp)[0]) + float(sum(_xlogx(c) for c in counts)) \
        - float(sum(_xlogx(v) for v in letter_n))
    best, best_key = current, key.copy()
    ratio = t_end / t_start
    for i in range(iters):
        temp = t_start * ratio ** (i / iters)
        s = present[rng.integers(len(present))]
        a = key[s]
        c = int(rng.integers(K - 1))
        c += c >= a
        w = wins[s]
        old = window_sum(w)
        x[pos[s]] = c
        cs = counts[s]
        delta = window_sum(w) - old + (_xlogx(letter_n[a]) + _xlogx(letter_n[c])
                                       - _xlogx(letter_n[a] - cs) - _xlogx(letter_n[c] + cs))
        if delta >= 0 or rng.random() < math.exp(delta / temp):
            key[s] = c
            letter_n[a] -= cs
            letter_n[c] += cs
            current += delta
            if current > best:
                best, best_key = current, key.copy()
        else:
            x[pos[s]] = a
    return best_key


def search_keys(cipher: np.ndarray, candidates: list[np.ndarray], n_symbols: int, one_to_one: bool,
                order: int = 4, n_kicks: int = 30, kick_size: int = 3, seed: int = 0,
                anneal_runs: int = 0, anneal_iters: int = 200_000) -> tuple[np.ndarray, float]:
    """Refine every candidate key, keep the best, then search further.

    For homophonic keys, simulated-annealing runs start from the best refined
    candidate and then from random keys drawn from English letter frequencies
    (random starts escape the regions where EM keys get stuck); each result is
    polished by hill-climbing. Runs continue until two of them agree on the best
    score (a sign the global optimum was found), up to `anneal_runs` runs. Finally an iterated local
    search kicks the best key: each kick randomly reassigns `kick_size` symbols (or
    swaps that many letter pairs for a one-to-one key), climbs again, and is kept
    only if the score improves. Selection uses the model score alone, never the true key.
    """
    rng = np.random.default_rng(seed)
    refined = [refine_key(cipher, k, n_symbols, one_to_one, order) for k in candidates]
    scores = [key_score(cipher, k, n_symbols, one_to_one, order) for k in refined]
    if anneal_runs > 0 and not one_to_one:
        pi = get_bigram().pi
        start = refined[int(np.argmax(scores))]
        annealed: list[float] = []
        for run in range(anneal_runs):
            if run > 0:
                start = rng.choice(K, size=n_symbols, p=pi)
            k = refine_key(cipher, anneal_key(cipher, start, n_symbols, anneal_iters, rng, order),
                           n_symbols, one_to_one, order)
            sc = key_score(cipher, k, n_symbols, one_to_one, order)
            refined.append(k)
            scores.append(sc)
            annealed.append(sc)
            top = max(annealed)
            if sum(abs(a - top) < 0.5 for a in annealed) >= 2:
                break
    best_i = int(np.argmax(scores))
    best, best_score = refined[best_i], scores[best_i]
    present = np.flatnonzero(np.bincount(cipher, minlength=n_symbols) > 0)
    for _ in range(n_kicks):
        kicked = best.copy()
        for _ in range(kick_size):
            if one_to_one:
                a, b = rng.choice(n_symbols, size=2, replace=False)
                kicked[a], kicked[b] = kicked[b], kicked[a]
            else:
                kicked[rng.choice(present)] = rng.integers(K)
        cand = refine_key(cipher, kicked, n_symbols, one_to_one, order)
        s = key_score(cipher, cand, n_symbols, one_to_one, order)
        if s > best_score + 1e-9:
            best, best_score = cand, s
    return best, best_score


def key_from_emissions(B: np.ndarray, one_to_one: bool) -> np.ndarray:
    """A key from an emission matrix alone, using P(letter, symbol) ∝ P(letter) B[letter, symbol]."""
    joint = B * get_bigram().pi[:, None]
    if one_to_one:
        letters, symbols = linear_sum_assignment(-np.log(joint + 1e-300))
        key = np.zeros(B.shape[1], dtype=np.int64)
        key[symbols] = letters
        return key
    return joint.argmax(axis=0)
