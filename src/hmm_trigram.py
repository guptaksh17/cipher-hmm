"""Second-order (trigram) HMM via letter-pair states.

State (a, b) at time t means x_{t-1} = a, x_t = b; its index is a * L + b, so
there are L^2 = 676 states. Only transitions (a, b) -> (b, c) are possible,
with probability T[a, b, c] = P(c | a, b): each row of the 676 x 676 matrix has
just 26 non-zeros. Instead of a dense product (L^4 per step) every step is a
batched (L x L) @ (L x L) product over the shared middle letter b (L^3 per step).

Emissions are tied: state (a, b) emits with B[b, :], so the learned parameters
are still a 26 x S matrix and all algorithms in hmm.py apply unchanged.

The first state (a, x_0) has prior pi2[a, x_0]: `a` is the (unobserved) letter
before the ciphertext starts, which suits passages taken from mid-text.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.lm import TrigramLM, weight_transitions


@dataclass
class PairTransitions:
    """Sparse pair-state transitions built from a trigram model."""

    pi2: np.ndarray  # (L, L) joint prior of (x_{-1}, x_0)
    T: np.ndarray    # (L, L, L) T[a, b, c] = P(x_t = c | x_{t-2} = a, x_{t-1} = b)
    pi: np.ndarray = field(init=False, repr=False)
    _Tb: np.ndarray = field(init=False, repr=False)
    _log_T: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.pi = self.pi2.reshape(-1)
        self._Tb = np.ascontiguousarray(self.T.transpose(1, 0, 2))  # [b, a, c]
        with np.errstate(divide="ignore"):
            self._log_T = np.log(self.T)

    @property
    def n_letters(self) -> int:
        return self.T.shape[0]

    @property
    def n_states(self) -> int:
        return self.n_letters ** 2

    def forward_step(self, alpha: np.ndarray) -> np.ndarray:
        """new[r, (b, c)] = sum_a alpha[r, (a, b)] T[a, b, c]."""
        R, L = alpha.shape[0], self.n_letters
        x = alpha.reshape(R, L, L).transpose(2, 0, 1)            # [b, r, a]
        return np.matmul(x, self._Tb).transpose(1, 0, 2).reshape(R, L * L)

    def backward_step(self, v: np.ndarray) -> np.ndarray:
        """out[r, (a, b)] = sum_c T[a, b, c] v[r, (b, c)]."""
        R, L = v.shape[0], self.n_letters
        x = v.reshape(R, L, L).transpose(1, 2, 0)                # [b, c, r]
        return np.matmul(self._Tb, x).transpose(2, 1, 0).reshape(R, L * L)

    def max_step(self, delta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Best predecessor (a, b) of each state (b, c), in log space."""
        L = self.n_letters
        scores = delta.reshape(L, L)[:, :, None] + self._log_T   # [a, b, c]
        arg_a = scores.argmax(axis=0)                            # [b, c]
        best = np.take_along_axis(scores, arg_a[None], axis=0)[0]
        prev = arg_a * L + np.arange(L)[:, None]
        return best.reshape(-1), prev.reshape(-1)

    def expand(self, E: np.ndarray) -> np.ndarray:
        """State (a, b) emits like letter b."""
        return np.tile(E, self.n_letters)

    def collapse(self, gamma: np.ndarray) -> np.ndarray:
        """Sum out the previous letter a."""
        L = self.n_letters
        return gamma.reshape(*gamma.shape[:-1], L, L).sum(axis=-2)

    def state_letters(self, path: np.ndarray) -> np.ndarray:
        return path % self.n_letters


def trigram_transitions(lm: TrigramLM, power: float = 1.0) -> PairTransitions:
    """Pair-state HMM transitions from a trigram LM with weight `power`."""
    return PairTransitions(lm.pi2, weight_transitions(lm.T, power))
