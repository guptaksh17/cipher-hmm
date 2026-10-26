"""Character-level English language models (bigram and trigram).

These models provide the FIXED transition matrix A and initial distribution pi
of the decipherment HMM. Smoothing is add-k; `weight_transitions` raises the
conditional probabilities to a power (language-model weight) and renormalizes.

The initial distribution is the unigram letter distribution, because test
passages start at arbitrary positions inside a book, not at word starts.

Run as a script to train and save the default models:
    python -m src.lm
"""
from __future__ import annotations

import string
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.corpus import DATA_DIR, load_train_corpus

ALPHABET = string.ascii_lowercase
ALPHABET_SPACE = ALPHABET + " "


def text_to_indices(text: str, alphabet: str = ALPHABET) -> np.ndarray:
    """Map a string over `alphabet` to an int array of letter indices."""
    lookup = np.full(256, -1, dtype=np.int64)
    for i, ch in enumerate(alphabet):
        lookup[ord(ch)] = i
    idx = lookup[np.frombuffer(text.encode("ascii"), dtype=np.uint8)]
    if (idx < 0).any():
        raise ValueError("text contains characters outside the alphabet")
    return idx


def indices_to_text(idx: np.ndarray, alphabet: str = ALPHABET) -> str:
    """Map an int array of letter indices back to a string."""
    return "".join(alphabet[i] for i in idx)


def _normalize_last(x: np.ndarray) -> np.ndarray:
    """Normalize an array so that it sums to one along the last axis."""
    return x / x.sum(axis=-1, keepdims=True)


@dataclass
class BigramLM:
    """First-order letter model: pi[i] = P(x_1=i), A[i, j] = P(x_t=j | x_{t-1}=i)."""

    alphabet: str
    pi: np.ndarray
    A: np.ndarray
    k: float

    @property
    def n_states(self) -> int:
        return len(self.alphabet)

    def log_prob(self, text: str) -> float:
        """Log-probability of a plaintext under the model."""
        x = text_to_indices(text, self.alphabet)
        return float(np.log(self.pi[x[0]]) + np.log(self.A[x[:-1], x[1:]]).sum())


@dataclass
class TrigramLM:
    """Second-order letter model.

    pi2[a, b] = P(x_1=a, x_2=b); T[a, b, c] = P(x_t=c | x_{t-2}=a, x_{t-1}=b).
    `unigram` is kept for one-letter inputs.
    """

    alphabet: str
    unigram: np.ndarray
    pi2: np.ndarray
    T: np.ndarray
    k: float

    @property
    def n_letters(self) -> int:
        return len(self.alphabet)

    def log_prob(self, text: str) -> float:
        """Log-probability of a plaintext under the model."""
        x = text_to_indices(text, self.alphabet)
        if len(x) == 1:
            return float(np.log(self.unigram[x[0]]))
        lp = np.log(self.pi2[x[0], x[1]])
        return float(lp + np.log(self.T[x[:-2], x[1:-1], x[2:]]).sum())


def ngram_counts(idx: np.ndarray, n: int, K: int) -> np.ndarray:
    """Count n-grams of letter indices; returns an array of shape (K,)*n."""
    code = np.zeros(len(idx) - n + 1, dtype=np.int64)
    for j in range(n):
        code = code * K + idx[j : len(idx) - n + 1 + j]
    return np.bincount(code, minlength=K**n).reshape((K,) * n).astype(np.float64)


def train_bigram(text: str, k: float = 0.5, alphabet: str = ALPHABET) -> BigramLM:
    """Train an add-k smoothed bigram model."""
    K = len(alphabet)
    idx = text_to_indices(text, alphabet)
    uni = ngram_counts(idx, 1, K) + k
    bi = ngram_counts(idx, 2, K) + k
    return BigramLM(alphabet, uni / uni.sum(), _normalize_last(bi), k)


def train_trigram(text: str, k: float = 0.5, alphabet: str = ALPHABET) -> TrigramLM:
    """Train an add-k smoothed trigram model."""
    K = len(alphabet)
    idx = text_to_indices(text, alphabet)
    uni = ngram_counts(idx, 1, K) + k
    bi = ngram_counts(idx, 2, K) + k
    tri = ngram_counts(idx, 3, K) + k
    return TrigramLM(alphabet, uni / uni.sum(), bi / bi.sum(), _normalize_last(tri), k)


def weight_transitions(P: np.ndarray, power: float) -> np.ndarray:
    """Raise conditional probabilities to `power` and renormalize (last axis).

    power > 1 sharpens the language model, power < 1 flattens it, 1 is identity.
    """
    if power == 1.0:
        return P
    logp = power * np.log(P)
    logp -= logp.max(axis=-1, keepdims=True)
    return _normalize_last(np.exp(logp))


def _model_path(order: int, k: float, spaces: bool) -> Path:
    tag = "_spaces" if spaces else ""
    return DATA_DIR / f"lm_order{order}_k{k:g}{tag}.npz"


def save_lm(lm: BigramLM | TrigramLM, path: Path) -> None:
    """Save a language model to an .npz file."""
    if isinstance(lm, BigramLM):
        np.savez_compressed(path, order=2, alphabet=lm.alphabet, k=lm.k, pi=lm.pi, A=lm.A)
    else:
        np.savez_compressed(path, order=3, alphabet=lm.alphabet, k=lm.k,
                            unigram=lm.unigram, pi2=lm.pi2, T=lm.T)


def load_lm(path: Path) -> BigramLM | TrigramLM:
    """Load a language model saved by `save_lm`."""
    d = np.load(path)
    alphabet, k = str(d["alphabet"]), float(d["k"])
    if int(d["order"]) == 2:
        return BigramLM(alphabet, d["pi"], d["A"], k)
    return TrigramLM(alphabet, d["unigram"], d["pi2"], d["T"], k)


def get_bigram(k: float = 0.5, spaces: bool = False) -> BigramLM:
    """Load the bigram model from data/, training and saving it if missing."""
    path = _model_path(2, k, spaces)
    if path.exists():
        return load_lm(path)  # type: ignore[return-value]
    lm = train_bigram(load_train_corpus(spaces), k, ALPHABET_SPACE if spaces else ALPHABET)
    save_lm(lm, path)
    return lm


def get_trigram(k: float = 0.5, spaces: bool = False) -> TrigramLM:
    """Load the trigram model from data/, training and saving it if missing."""
    path = _model_path(3, k, spaces)
    if path.exists():
        return load_lm(path)  # type: ignore[return-value]
    lm = train_trigram(load_train_corpus(spaces), k, ALPHABET_SPACE if spaces else ALPHABET)
    save_lm(lm, path)
    return lm


if __name__ == "__main__":
    from src.corpus import sample_passages

    bi, tri = get_bigram(), get_trigram()
    get_bigram(spaces=True)
    print("top unigrams:", "".join(ALPHABET[i] for i in np.argsort(-bi.pi)))
    p = sample_passages(1, 1000, seed=0)[0]
    print("held-out bits/letter  bigram: %.3f  trigram: %.3f"
          % (-bi.log_prob(p) / len(p) / np.log(2), -tri.log_prob(p) / len(p) / np.log(2)))
