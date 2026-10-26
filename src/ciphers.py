"""Key generation and encryption for simple and homophonic substitution ciphers.

Plaintext and ciphertext are int arrays: plaintext letters 0..25, cipher symbols
0..S-1. A key is stored as `sym_to_letter` (the true decryption of each symbol)
plus the homophone groups `letter_to_syms`. A simple substitution key is the
special case S = 26 with one symbol per letter.
"""
from __future__ import annotations

import string
from dataclasses import dataclass

import numpy as np

# Printable display glyphs for cipher symbols (no space; avoids confusing pairs).
GLYPHS = (string.ascii_uppercase + string.digits + "!#$%&()*+/<=>?@[]^{|}~"
          + "αβγδεζηθλμξπρστφχψωΓΔΘΛΞΠΣΦΨΩ")


@dataclass
class CipherKey:
    """A (possibly homophonic) substitution key."""

    sym_to_letter: np.ndarray        # (S,) plaintext letter of each cipher symbol
    letter_to_syms: list[np.ndarray]  # homophones of each plaintext letter

    @property
    def n_symbols(self) -> int:
        return len(self.sym_to_letter)

    @property
    def n_letters(self) -> int:
        return len(self.letter_to_syms)


def _key_from_groups(groups: list[np.ndarray], n_symbols: int) -> CipherKey:
    sym_to_letter = np.empty(n_symbols, dtype=np.int64)
    for letter, syms in enumerate(groups):
        sym_to_letter[syms] = letter
    return CipherKey(sym_to_letter, groups)


def simple_key(rng: np.random.Generator, n_letters: int = 26) -> CipherKey:
    """Random one-to-one substitution key."""
    perm = rng.permutation(n_letters)
    return _key_from_groups([perm[i : i + 1] for i in range(n_letters)], n_letters)


def homophone_counts(freqs: np.ndarray, n_symbols: int) -> np.ndarray:
    """Number of homophones per letter, roughly proportional to `freqs`.

    Every letter gets at least one symbol; the rest are allocated with the
    largest-remainder method so that the counts sum to `n_symbols`.
    """
    n_letters = len(freqs)
    if n_symbols < n_letters:
        raise ValueError("need at least one symbol per letter")
    extra = n_symbols - n_letters
    share = freqs / freqs.sum() * extra
    counts = np.floor(share).astype(np.int64)
    remainder_order = np.argsort(-(share - counts), kind="stable")
    counts[remainder_order[: extra - counts.sum()]] += 1
    return counts + 1


def homophonic_key(n_symbols: int, freqs: np.ndarray, rng: np.random.Generator) -> CipherKey:
    """Random homophonic key; letter homophone counts follow `freqs`."""
    counts = homophone_counts(freqs, n_symbols)
    perm = rng.permutation(n_symbols)
    bounds = np.concatenate([[0], np.cumsum(counts)])
    groups = [np.sort(perm[bounds[i] : bounds[i + 1]]) for i in range(len(counts))]
    return _key_from_groups(groups, n_symbols)


def encrypt(plain: np.ndarray, key: CipherKey, rng: np.random.Generator,
            mode: str = "random") -> np.ndarray:
    """Encrypt plaintext letter indices.

    mode="random": each occurrence picks a homophone uniformly at random.
    mode="cycle":  each letter cycles through its homophones in order
                   (as the Zodiac killer largely did).
    """
    cipher = np.empty_like(plain)
    for letter, syms in enumerate(key.letter_to_syms):
        pos = np.flatnonzero(plain == letter)
        if mode == "random":
            choice = rng.integers(len(syms), size=len(pos))
        elif mode == "cycle":
            choice = np.arange(len(pos)) % len(syms)
        else:
            raise ValueError(f"unknown mode {mode!r}")
        cipher[pos] = syms[choice]
    return cipher


def decrypt(cipher: np.ndarray, sym_to_letter: np.ndarray) -> np.ndarray:
    """Apply a symbol -> letter mapping to a ciphertext."""
    return sym_to_letter[cipher]


def glyphs(cipher: np.ndarray, glyph_set: str = GLYPHS) -> str:
    """Render cipher symbol ids as printable glyphs."""
    if cipher.max(initial=0) >= len(glyph_set):
        raise ValueError("not enough glyphs for this many symbols")
    return "".join(glyph_set[s] for s in cipher)
