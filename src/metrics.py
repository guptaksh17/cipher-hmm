"""Evaluation metrics for decipherment."""
from __future__ import annotations

import numpy as np


def symbol_accuracy(decoded: np.ndarray, plain: np.ndarray) -> float:
    """Fraction of ciphertext positions decoded to the correct letter."""
    return float(np.mean(decoded == plain))


def key_accuracy(pred_key: np.ndarray, true_key: np.ndarray, cipher: np.ndarray) -> float:
    """Fraction of distinct cipher symbols (present in the ciphertext) mapped correctly."""
    present = np.unique(cipher)
    return float(np.mean(pred_key[present] == true_key[present]))


def key_from_decoding(cipher: np.ndarray, decoded: np.ndarray, n_symbols: int,
                      n_letters: int = 26) -> np.ndarray:
    """Symbol -> letter key implied by a decoding (majority vote per symbol).

    Symbols absent from the ciphertext map to letter 0; they never enter
    `key_accuracy`, which only scores symbols that occur.
    """
    votes = np.zeros((n_symbols, n_letters), dtype=np.int64)
    np.add.at(votes, (cipher, decoded), 1)
    return votes.argmax(axis=1)
