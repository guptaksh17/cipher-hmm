"""Tests for the pair-state (trigram) HMM against brute force and a dense matrix."""
import itertools

import numpy as np
import pytest

from src.hmm import (DenseTransitions, baum_welch, emission_likelihoods, forward,
                     letter_posteriors, log_likelihood, random_emissions, viterbi)
from src.hmm_trigram import PairTransitions, trigram_transitions
from src.lm import get_trigram


def tiny_trigram(seed=0, L=3, S=4):
    rng = np.random.default_rng(seed)
    pi2 = rng.dirichlet(np.ones(L * L)).reshape(L, L)
    T = rng.dirichlet(np.ones(L), size=(L, L))
    B = rng.dirichlet(np.ones(S), size=L)
    return PairTransitions(pi2, T), B


def brute_force(trans, B, obs):
    """Sum/max over the phantom letter x_{-1} and all letter paths x_0..x_{T-1}."""
    L, n = trans.n_letters, len(obs)
    best, best_path, total = -np.inf, None, 0.0
    marg = np.zeros((n, L))
    for a, *x in itertools.product(range(L), repeat=n + 1):
        p = trans.pi2[a, x[0]] * B[x[0], obs[0]]
        prev2, prev1 = a, x[0]
        for t in range(1, n):
            p *= trans.T[prev2, prev1, x[t]] * B[x[t], obs[t]]
            prev2, prev1 = prev1, x[t]
        total += p
        marg[np.arange(n), x] += p
        if np.log(p) > best:
            best, best_path = np.log(p), np.array(x)
    return np.log(total), best_path, best, marg / total


@pytest.mark.parametrize("seed", range(3))
def test_trigram_forward_viterbi_posteriors_equal_brute_force(seed):
    trans, B = tiny_trigram(seed)
    obs = np.array([0, 3, 1, 1, 2, 0])
    ll_bf, path_bf, best_bf, marg_bf = brute_force(trans, B, obs)
    assert np.isclose(log_likelihood(trans, B, obs)[0], ll_bf)
    path, best = viterbi(trans, B, obs)
    assert np.array_equal(path, path_bf) and np.isclose(best, best_bf)
    gamma, ll = letter_posteriors(trans, emission_likelihoods(B, obs))
    assert np.allclose(gamma[:, 0], marg_bf) and np.isclose(ll[0], ll_bf)


def test_sparse_steps_equal_dense_pair_matrix():
    trans, _ = tiny_trigram(4, L=4)
    L = 4
    A = np.zeros((L * L, L * L))
    for a, b, c in itertools.product(range(L), repeat=3):
        A[a * L + b, b * L + c] = trans.T[a, b, c]
    dense = DenseTransitions(trans.pi, A)
    v = np.random.default_rng(0).random((5, L * L))
    assert np.allclose(trans.forward_step(v), dense.forward_step(v))
    assert np.allclose(trans.backward_step(v), dense.backward_step(v))
    d = np.log(np.random.default_rng(1).random(L * L))
    for got, ref in zip(trans.max_step(d), dense.max_step(d)):
        assert np.allclose(got, ref)


def test_trigram_baum_welch_non_decreasing_on_real_lm():
    trans = trigram_transitions(get_trigram())
    rng = np.random.default_rng(0)
    obs = rng.integers(30, size=200)
    res = baum_welch(trans, obs, random_emissions(rng, 3, 26, 30), max_iter=30, tol=0.0)
    assert res.B.shape == (3, 26, 30)
    for h in res.history:
        assert np.all(np.diff(h) >= -1e-6)


def test_trigram_long_sequence_no_underflow():
    trans = trigram_transitions(get_trigram())
    rng = np.random.default_rng(1)
    obs = rng.integers(54, size=3000)
    B = random_emissions(rng, 1, 26, 54)[0]
    assert np.isfinite(log_likelihood(trans, B, obs)[0])
    assert np.isfinite(viterbi(trans, B, obs)[1])
