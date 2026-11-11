"""Unsupervised decipherment: EM with random restarts, MAP option, key extraction.

Pipeline for a ciphertext (symbol ids 0..S-1):
  1. Fixed transitions from the language model, optionally sharpened (lm_power):
     a bigram LM gives a 26-state HMM, a trigram LM the 676 pair-state HMM.
  2. Baum-Welch on the emissions from `n_restarts` random initializations
     (batched); the restart with the highest log-likelihood is kept. Nothing
     about the true key or plaintext is used.
  3. Decodings of the best model:
       viterbi   - most probable state path,
       posterior - argmax of the per-position state posteriors,
       key       - a consistent symbol -> letter key applied to the ciphertext.
     The key comes from the expected letter/symbol counts N[k, s] =
     sum_t P(z_t = k | o) [o_t = s]. For simple substitution it is the
     one-to-one Hungarian assignment maximizing sum N (expected number of
     correctly decoded positions); for homophonic ciphers each symbol takes
     its argmax letter (many-to-one).
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.hmm import (BaumWelchResult, DenseTransitions, IterCallback, Transitions,
                     baum_welch, emission_likelihoods, expected_emission_counts,
                     letter_posteriors, random_emissions, viterbi)
from src.hmm_trigram import trigram_transitions
from src.lm import BigramLM, TrigramLM, weight_transitions
from src.refine import key_from_emissions, search_keys

log = logging.getLogger(__name__)


@dataclass
class DecipherConfig:
    """Settings of one decipherment run."""

    n_restarts: int = 50
    alpha: float = 0.0          # Dirichlet pseudo-count on rows of B (0 = ML)
    lm_power: float = 1.0       # exponent applied to transition probabilities
    max_iter: int = 500
    tol: float = 1e-5           # stop when objective gain per symbol < tol
    seed: int = 0
    one_to_one: bool | None = None  # None: one-to-one iff S == number of letters
    max_batch_bytes: float = 1e9    # restarts run in chunks to bound memory
    refine: bool = True             # polish the key by hill-climbing (src/refine.py)
    refine_order: int = 4           # n-gram order of the language model used to refine
    refine_candidates: int = 10     # top restarts whose keys are refined
    refine_kicks: int = 30          # iterated-local-search kicks after refinement


@dataclass
class DecipherResult:
    """Best model and its decodings (all as plaintext letter indices)."""

    key: np.ndarray            # (S,) symbol -> letter
    decoded: np.ndarray        # key[cipher]
    viterbi: np.ndarray
    posterior: np.ndarray
    B: np.ndarray              # (K, S) emissions of the best restart
    loglik: float
    restart_logliks: np.ndarray
    best_restart: int
    history: list[float]       # objective per EM iteration of the best restart
    n_iter: int
    runtime: float
    em_key: np.ndarray | None = None   # key straight from EM, before refinement
    key_score: float | None = None     # joint log-likelihood of the final key


def bigram_transitions(lm: BigramLM, power: float = 1.0) -> DenseTransitions:
    """First-order HMM transitions from a bigram LM with weight `power`."""
    return DenseTransitions(lm.pi, weight_transitions(lm.A, power))


def expected_counts(gamma: np.ndarray, cipher: np.ndarray, n_symbols: int) -> np.ndarray:
    """N[k, s]: expected number of positions with letter k emitting symbol s.

    `gamma` holds the letter posteriors of one model, shape (T, K).
    """
    onehot = np.zeros((len(cipher), n_symbols))
    onehot[np.arange(len(cipher)), cipher] = 1.0
    return expected_emission_counts(gamma[:, None], onehot)[0]


def one_to_one_key(N: np.ndarray) -> np.ndarray:
    """Hungarian assignment of symbols to letters maximizing total expected counts."""
    letters, symbols = linear_sum_assignment(-N)
    key = np.zeros(N.shape[1], dtype=np.int64)
    key[symbols] = letters
    return key


def many_to_one_key(N: np.ndarray) -> np.ndarray:
    """Each symbol goes to the letter that most often generated it."""
    return N.argmax(axis=0)


def make_transitions(lm: BigramLM | TrigramLM, power: float = 1.0) -> Transitions:
    """HMM transitions for a bigram (26 states) or trigram (676 states) LM."""
    if isinstance(lm, TrigramLM):
        return trigram_transitions(lm, power)
    return bigram_transitions(lm, power)


def restart_chunk_size(trans: Transitions, length: int, max_bytes: float) -> int:
    """How many restarts fit in one batch (about 3 float64 arrays of T x R x K)."""
    return max(1, int(max_bytes // (3 * 8 * length * trans.n_states)))


def run_em(trans: Transitions, cipher: np.ndarray, n_symbols: int,
           config: DecipherConfig, callback: IterCallback | None = None) -> BaumWelchResult:
    """Baum-Welch from `config.n_restarts` random initializations."""
    rng = np.random.default_rng(config.seed)
    B0 = random_emissions(rng, config.n_restarts, trans.n_letters, n_symbols)
    return run_em_from(trans, cipher, B0, config, callback)


def run_em_from(trans: Transitions, cipher: np.ndarray, B0: np.ndarray,
                config: DecipherConfig, callback: IterCallback | None = None) -> BaumWelchResult:
    """Baum-Welch from given initial emissions (R, K, S), in memory-bounded chunks."""
    size = restart_chunk_size(trans, len(cipher), config.max_batch_bytes)
    parts = [baum_welch(trans, cipher, B0[i : i + size], alpha=config.alpha,
                        max_iter=config.max_iter, tol=config.tol, callback=callback)
             for i in range(0, len(B0), size)]
    return BaumWelchResult(
        B=np.concatenate([p.B for p in parts]),
        loglik=np.concatenate([p.loglik for p in parts]),
        history=[h for p in parts for h in p.history],
        loglik_history=[h for p in parts for h in p.loglik_history],
        n_iter=np.concatenate([p.n_iter for p in parts]))


def decipher(cipher: np.ndarray, n_symbols: int, lm: BigramLM | TrigramLM,
             config: DecipherConfig | None = None,
             callback: IterCallback | None = None) -> DecipherResult:
    """Decipher a substitution ciphertext with a bigram or trigram HMM.

    `callback(it, B, loglik)` is called after every EM iteration of each
    restart chunk (one chunk unless memory forces more).
    """
    config = config or DecipherConfig()
    t0 = time.perf_counter()
    trans = make_transitions(lm, config.lm_power)
    res = run_em(trans, cipher, n_symbols, config, callback)
    return select_and_decode(trans, cipher, n_symbols, config, res, t0)


def select_and_decode(trans: Transitions, cipher: np.ndarray, n_symbols: int,
                      config: DecipherConfig, res: BaumWelchResult, t0: float,
                      extra_candidates: list[np.ndarray] | None = None) -> DecipherResult:
    """Pick the best restart, compute all decodings and, if enabled, refine the key.

    Refinement starts from the EM key, the keys implied by the top restarts and any
    `extra_candidates`, and keeps the key with the best joint log-likelihood.
    """
    for r, ll in enumerate(res.loglik):
        log.info("restart %3d: loglik %.2f after %d iterations", r, ll, res.n_iter[r])
    best = int(np.argmax(res.loglik))
    B = res.B[best]
    gamma = letter_posteriors(trans, emission_likelihoods(B, cipher))[0][:, 0]
    N = expected_counts(gamma, cipher, n_symbols)
    one_to_one = config.one_to_one if config.one_to_one is not None else n_symbols == N.shape[0]
    em_key = one_to_one_key(N) if one_to_one else many_to_one_key(N)
    key, score = em_key, None
    if config.refine:
        order = np.argsort(-res.loglik)[:config.refine_candidates]
        candidates = [em_key] + [key_from_emissions(res.B[r], one_to_one) for r in order]
        candidates += list(extra_candidates or [])
        key, score = search_keys(cipher, candidates, n_symbols, one_to_one,
                                 order=config.refine_order, n_kicks=config.refine_kicks,
                                 seed=config.seed)
    return DecipherResult(
        key=key, decoded=key[cipher], viterbi=viterbi(trans, B, cipher)[0],
        posterior=gamma.argmax(axis=1), B=B, loglik=float(res.loglik[best]),
        restart_logliks=res.loglik.copy(), best_restart=best, history=res.history[best],
        n_iter=int(res.n_iter[best]), runtime=time.perf_counter() - t0,
        em_key=em_key, key_score=score)


def decipher_staged(cipher: np.ndarray, n_symbols: int, bigram: BigramLM, trigram: TrigramLM,
                    config: DecipherConfig | None = None, n_pool: int = 1000,
                    n_keep: int = 50, init_smoothing: float = 0.01,
                    callback: IterCallback | None = None) -> DecipherResult:
    """Bigram-to-trigram staged decipherment.

    Stage 1: `n_pool` cheap random restarts with the bigram HMM (ML, alpha = 0).
    Stage 2: the `n_keep` highest-likelihood bigram solutions, lightly smoothed,
    initialize Baum-Welch on the trigram HMM with the settings in `config`
    (alpha, lm_power, tolerance). The best trigram restart is decoded.
    Random trigram restarts rarely reach the good optima of the 676-state model;
    starting from bigram solutions does, at a fraction of the cost.
    """
    config = config or DecipherConfig()
    t0 = time.perf_counter()
    pool_cfg = DecipherConfig(n_restarts=n_pool, max_iter=config.max_iter, tol=config.tol,
                              seed=config.seed, max_batch_bytes=config.max_batch_bytes)
    pool = run_em(bigram_transitions(bigram), cipher, n_symbols, pool_cfg)
    top = np.argsort(-pool.loglik)[:n_keep]
    B0 = pool.B[top] + init_smoothing
    B0 /= B0.sum(axis=-1, keepdims=True)
    trans = trigram_transitions(trigram, config.lm_power)
    res = run_em_from(trans, cipher, B0, config, callback)
    return select_and_decode(trans, cipher, n_symbols, config, res, t0,
                             extra_candidates=pool_candidates(pool, n_symbols, config))


def pool_candidates(pool: BaumWelchResult, n_symbols: int, config: DecipherConfig,
                    n: int = 20) -> list[np.ndarray]:
    """Keys implied by the best bigram-pool restarts (extra starting points for refinement)."""
    if not config.refine:
        return []
    n_letters = pool.B.shape[1]                       # B has shape (restarts, letters, symbols)
    one_to_one = config.one_to_one if config.one_to_one is not None else n_symbols == n_letters
    return [key_from_emissions(pool.B[r], one_to_one) for r in np.argsort(-pool.loglik)[:n]]
