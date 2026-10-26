"""Frequency-analysis baseline.

Simple substitution: match cipher symbols to letters by unigram frequency rank.
Homophonic (more symbols than letters): greedy "remaining mass" matching. Each
cipher symbol, from most to least frequent, goes to the letter whose expected
frequency is least covered by the symbols already assigned to it. With S equal
to the number of letters and one symbol per letter this is rank matching.
"""
from __future__ import annotations

import numpy as np


def symbol_frequencies(cipher: np.ndarray, n_symbols: int) -> np.ndarray:
    """Relative frequency of each cipher symbol."""
    return np.bincount(cipher, minlength=n_symbols) / len(cipher)


def rank_match(cipher: np.ndarray, n_symbols: int, unigram: np.ndarray) -> np.ndarray:
    """One-to-one mapping by frequency rank (requires n_symbols == n_letters)."""
    if n_symbols != len(unigram):
        raise ValueError("rank matching needs as many symbols as letters")
    sym_order = np.argsort(-symbol_frequencies(cipher, n_symbols), kind="stable")
    letter_order = np.argsort(-unigram, kind="stable")
    key = np.empty(n_symbols, dtype=np.int64)
    key[sym_order] = letter_order
    return key


def greedy_mass_match(cipher: np.ndarray, n_symbols: int, unigram: np.ndarray) -> np.ndarray:
    """Many-to-one mapping: give each symbol to the letter with most uncovered mass."""
    freq = symbol_frequencies(cipher, n_symbols)
    remaining = unigram / unigram.sum()
    key = np.empty(n_symbols, dtype=np.int64)
    for s in np.argsort(-freq, kind="stable"):
        letter = int(np.argmax(remaining))
        key[s] = letter
        remaining[letter] -= freq[s]
    return key


def frequency_analysis(cipher: np.ndarray, n_symbols: int, unigram: np.ndarray) -> np.ndarray:
    """Baseline key (symbol -> letter) from unigram frequencies alone."""
    if n_symbols == len(unigram):
        return rank_match(cipher, n_symbols, unigram)
    return greedy_mass_match(cipher, n_symbols, unigram)
