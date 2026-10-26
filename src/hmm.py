"""Discrete HMM algorithms in NumPy: forward, backward, Baum-Welch, Viterbi.

Decipherment setting: transitions (pi, A) are a FIXED language model; only the
emission matrix B[k, s] = P(cipher symbol s | plaintext letter k) is learned.

Conventions
-----------
* K hidden states, S observation symbols, sequence length T.
* Many random restarts are run at once: B has shape (R, K, S) and per-position
  emission likelihoods E have shape (T, R, K) (time-major, so E[t] is contiguous).
* Forward and backward use per-step normalization (scaling); the log-likelihood
  is the sum of the log scaling factors. Viterbi works in log space.
* The transition step is pluggable (`Transitions`) so that the second-order
  model in hmm_trigram.py reuses all algorithms here unchanged. A Transitions
  object also says how hidden states relate to plaintext letters: emissions
  are always B[letter, symbol] (K_letters x S), `expand` turns per-letter
  emission likelihoods into per-state ones, and `collapse` sums state
  posteriors back to letter posteriors. For the first-order model states ARE
  letters and both are the identity.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol

import numpy as np


class Transitions(Protocol):
    """Transition structure of an HMM with fixed parameters."""

    n_states: int
    n_letters: int
    pi: np.ndarray  # (K,) initial distribution over states

    def forward_step(self, alpha: np.ndarray) -> np.ndarray:
        """(R, K) -> (R, K): sum_i alpha[r, i] A[i, j]."""

    def backward_step(self, v: np.ndarray) -> np.ndarray:
        """(R, K) -> (R, K): sum_j A[i, j] v[r, j]."""

    def max_step(self, delta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(K,) log scores -> (best (K,), argmax (K,)) over previous states."""

    def expand(self, E: np.ndarray) -> np.ndarray:
        """(T, R, n_letters) letter emission likelihoods -> (T, R, K) per state."""

    def collapse(self, gamma: np.ndarray) -> np.ndarray:
        """(..., K) state posteriors -> (..., n_letters) letter posteriors."""

    def state_letters(self, path: np.ndarray) -> np.ndarray:
        """Map a state path to the plaintext letter of each state."""


@dataclass
class DenseTransitions:
    """First-order transitions given by a dense (K, K) matrix A."""

    pi: np.ndarray
    A: np.ndarray
    log_A: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        with np.errstate(divide="ignore"):
            self.log_A = np.log(self.A)

    @property
    def n_states(self) -> int:
        return len(self.pi)

    @property
    def n_letters(self) -> int:
        return len(self.pi)

    def forward_step(self, alpha: np.ndarray) -> np.ndarray:
        return alpha @ self.A

    def backward_step(self, v: np.ndarray) -> np.ndarray:
        return v @ self.A.T

    def max_step(self, delta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        scores = delta[:, None] + self.log_A
        arg = scores.argmax(axis=0)
        return scores[arg, np.arange(len(arg))], arg

    def expand(self, E: np.ndarray) -> np.ndarray:
        return E

    def collapse(self, gamma: np.ndarray) -> np.ndarray:
        return gamma

    def state_letters(self, path: np.ndarray) -> np.ndarray:
        return path


def as_batch(B: np.ndarray) -> np.ndarray:
    """View a (K, S) emission matrix as a batch of one, (1, K, S)."""
    return B[None] if B.ndim == 2 else B


def emission_likelihoods(B: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """E[t, r, k] = B[r, k, obs[t]], time-major and contiguous."""
    return np.ascontiguousarray(as_batch(B)[:, :, obs].transpose(2, 0, 1))


def forward(trans: Transitions, E: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Scaled forward pass.

    Returns alpha_hat (T, R, K) with alpha_hat[t] = P(z_t | o_1..t) and the
    log scaling factors log_c (T, R); log P(o) = log_c.sum(axis=0).
    """
    T, R, K = E.shape
    alpha = np.empty_like(E)
    c = np.empty((T, R, 1))
    a = trans.pi * E[0]
    for t in range(T):
        if t > 0:
            a = trans.forward_step(alpha[t - 1])
            a *= E[t]
        c[t] = a.sum(axis=1, keepdims=True)
        np.divide(a, c[t], out=alpha[t])
    return alpha, np.log(c[:, :, 0])


def backward(trans: Transitions, E: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Scaled backward pass with its own normalization.

    Returns beta_hat (T, R, K), proportional to P(o_t+1..T | z_t), and the
    backward log-likelihood (R,), which must equal the forward one.
    """
    T, R, K = E.shape
    beta = np.empty_like(E)
    beta[T - 1] = 1.0
    d = np.ones((T, R, 1))
    for t in range(T - 2, -1, -1):
        b = trans.backward_step(beta[t + 1] * E[t + 1])
        d[t] = b.sum(axis=1, keepdims=True)
        np.divide(b, d[t], out=beta[t])
    loglik = np.log((trans.pi * E[0] * beta[0]).sum(axis=1)) + np.log(d[:, :, 0]).sum(axis=0)
    return beta, loglik


def posteriors(trans: Transitions, E: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """State posteriors gamma[t, r, k] = P(z_t = k | o) and log-likelihoods (R,)."""
    alpha, log_c = forward(trans, E)
    beta, _ = backward(trans, E)
    gamma = alpha * beta
    gamma /= gamma.sum(axis=2, keepdims=True)
    return gamma, log_c.sum(axis=0)


def letter_posteriors(trans: Transitions, E: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Letter posteriors P(x_t = k | o), (T, R, n_letters), and log-likelihoods (R,).

    `E` holds per-letter emission likelihoods. Memory-lean: only the forward
    variables are stored; the backward pass forms each posterior on the fly
    and keeps only its letter marginal (important for the 676 pair states).
    """
    Es = trans.expand(E)
    alpha, log_c = forward(trans, Es)
    T, R, K = Es.shape
    gamma = np.empty((T, R, trans.n_letters))
    beta = np.ones((R, K))
    for t in range(T - 1, -1, -1):
        if t < T - 1:
            beta = trans.backward_step(beta * Es[t + 1])
            beta /= beta.sum(axis=1, keepdims=True)
        g = alpha[t] * beta
        g /= g.sum(axis=1, keepdims=True)
        gamma[t] = trans.collapse(g)
    return gamma, log_c.sum(axis=0)


def log_likelihood(trans: Transitions, B: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """log P(obs) for each emission matrix in the batch, shape (R,)."""
    return forward(trans, trans.expand(emission_likelihoods(B, obs)))[1].sum(axis=0)


def expected_emission_counts(gamma: np.ndarray, onehot: np.ndarray) -> np.ndarray:
    """counts[r, k, s] = sum_t gamma[t, r, k] [obs_t = s], via one matrix product."""
    T, R, K = gamma.shape
    return (gamma.reshape(T, R * K).T @ onehot).reshape(R, K, -1)


def m_step(counts: np.ndarray, alpha: float = 0.0) -> np.ndarray:
    """Emission update with optional symmetric Dirichlet prior.

    alpha = 0 is maximum likelihood. alpha > 0 is the MAP estimate under a
    Dirichlet(alpha + 1) prior on each row of B: B = (n + alpha) / (N + S alpha).
    Rows with no expected counts (and alpha = 0) fall back to uniform.
    """
    num = counts + alpha
    tot = num.sum(axis=-1, keepdims=True)
    S = counts.shape[-1]
    return np.where(tot > 0, num / np.where(tot > 0, tot, 1.0), 1.0 / S)


def log_prior(B: np.ndarray, alpha: float) -> np.ndarray:
    """Unnormalized Dirichlet(alpha + 1) log-prior of each B in the batch, (R,)."""
    if alpha == 0:
        return np.zeros(len(as_batch(B)))
    return alpha * np.log(as_batch(B)).sum(axis=(1, 2))


def random_emissions(rng: np.random.Generator, n_restarts: int, K: int, S: int) -> np.ndarray:
    """Random initial emission matrices, each row drawn from a flat Dirichlet."""
    return rng.dirichlet(np.ones(S), size=(n_restarts, K))


@dataclass
class BaumWelchResult:
    """Outcome of batched Baum-Welch.

    B: final emissions (R, K, S); loglik: final log-likelihoods (R,);
    history[r]: objective (log-likelihood + log-prior) before each update and at
    the end; n_iter[r]: number of EM updates performed.
    """

    B: np.ndarray
    loglik: np.ndarray
    history: list[list[float]]
    loglik_history: list[list[float]]
    n_iter: np.ndarray


IterCallback = Callable[[int, np.ndarray, np.ndarray], None]


def baum_welch(trans: Transitions, obs: np.ndarray, B0: np.ndarray, alpha: float = 0.0,
               max_iter: int = 200, tol: float = 1e-6,
               callback: IterCallback | None = None) -> BaumWelchResult:
    """EM for the emissions only, with transitions fixed, for a batch of restarts.

    A restart stops when its objective improves by less than `tol` per
    observation, or after `max_iter` updates. `callback(it, B, loglik)` is
    called after every iteration with the full batch (for live displays).
    """
    B = as_batch(B0).astype(np.float64).copy()
    R, K, S = B.shape
    T = len(obs)
    onehot = np.zeros((T, S))
    onehot[np.arange(T), obs] = 1.0
    history: list[list[float]] = [[] for _ in range(R)]
    ll_history: list[list[float]] = [[] for _ in range(R)]
    n_iter = np.zeros(R, dtype=np.int64)
    last_ll = np.full(R, -np.inf)
    active = np.arange(R)
    for it in range(max_iter + 1):
        gamma, ll = letter_posteriors(trans, emission_likelihoods(B[active], obs))
        obj = ll + log_prior(B[active], alpha)
        last_ll[active] = ll
        done = np.zeros(len(active), dtype=bool)
        for j, r in enumerate(active):
            prev = history[r][-1] if history[r] else -np.inf
            history[r].append(float(obj[j]))
            ll_history[r].append(float(ll[j]))
            done[j] = it == max_iter or (obj[j] - prev) / T < tol
        update = ~done
        if update.any():
            idx = active[update]
            counts = expected_emission_counts(gamma[:, update], onehot)
            B[idx] = m_step(counts, alpha)
            n_iter[idx] += 1
        active = active[update]
        if callback is not None:
            callback(it, B, last_ll)
        if len(active) == 0:
            break
    return BaumWelchResult(B, last_ll, history, ll_history, n_iter)


def viterbi(trans: Transitions, B: np.ndarray, obs: np.ndarray) -> tuple[np.ndarray, float]:
    """Most probable path (log space), as letters, and its joint log-probability."""
    with np.errstate(divide="ignore"):
        log_E = np.log(trans.expand(emission_likelihoods(B, obs))[:, 0])
        delta = np.log(trans.pi) + log_E[0]
    T, K = log_E.shape
    back = np.empty((T, K), dtype=np.int64)
    for t in range(1, T):
        best, back[t] = trans.max_step(delta)
        delta = best + log_E[t]
    path = np.empty(T, dtype=np.int64)
    path[-1] = int(delta.argmax())
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return trans.state_letters(path), float(delta.max())


def posterior_decode(trans: Transitions, B: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """Maximum posterior marginal decoding: argmax_k P(z_t = k | o) at each t."""
    gamma, _ = letter_posteriors(trans, emission_likelihoods(B, obs))
    return gamma[:, 0].argmax(axis=1)
