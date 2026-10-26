"""Correctness tests for the HMM algorithms against brute force and hmmlearn."""
import itertools

import numpy as np
import pytest

from src.hmm import (DenseTransitions, as_batch, backward, baum_welch, emission_likelihoods,
                     forward, log_likelihood, m_step, posterior_decode, posteriors,
                     random_emissions, viterbi)


def tiny_hmm(seed=0, K=3, S=4):
    rng = np.random.default_rng(seed)
    pi = rng.dirichlet(np.ones(K))
    A = rng.dirichlet(np.ones(K), size=K)
    B = rng.dirichlet(np.ones(S), size=K)
    return DenseTransitions(pi, A), B


def path_logprob(trans, B, path, obs):
    lp = np.log(trans.pi[path[0]] * B[path[0], obs[0]])
    for t in range(1, len(obs)):
        lp += np.log(trans.A[path[t - 1], path[t]] * B[path[t], obs[t]])
    return lp


def brute_force(trans, B, obs):
    """All K^T state paths: total likelihood, best path, posterior marginals."""
    K, T = trans.n_states, len(obs)
    paths = list(itertools.product(range(K), repeat=T))
    lps = np.array([path_logprob(trans, B, p, obs) for p in paths])
    probs = np.exp(lps)
    marg = np.zeros((T, K))
    for p, pr in zip(paths, probs):
        marg[np.arange(T), list(p)] += pr
    return np.log(probs.sum()), np.array(paths[int(lps.argmax())]), lps.max(), marg / probs.sum()


OBS_CASES = [np.array([0, 1, 2, 3]), np.array([3, 3, 0, 1, 2, 1, 0])]


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("obs", OBS_CASES)
def test_forward_equals_brute_force(seed, obs):
    trans, B = tiny_hmm(seed)
    ll_bf, *_ = brute_force(trans, B, obs)
    assert np.isclose(log_likelihood(trans, B, obs)[0], ll_bf)


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("obs", OBS_CASES)
def test_viterbi_equals_brute_force(seed, obs):
    trans, B = tiny_hmm(seed)
    _, best_path, best_lp, _ = brute_force(trans, B, obs)
    path, lp = viterbi(trans, B, obs)
    assert np.array_equal(path, best_path)
    assert np.isclose(lp, best_lp)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_posteriors_equal_brute_force(seed):
    trans, B = tiny_hmm(seed)
    obs = OBS_CASES[1]
    *_, marg = brute_force(trans, B, obs)
    gamma, _ = posteriors(trans, emission_likelihoods(B, obs))
    assert np.allclose(gamma[:, 0], marg)
    assert np.array_equal(posterior_decode(trans, B, obs), marg.argmax(1))


@pytest.mark.parametrize("seed", range(5))
def test_forward_backward_same_likelihood(seed):
    rng = np.random.default_rng(seed)
    trans, _ = tiny_hmm(seed, K=5, S=7)
    B = random_emissions(rng, 4, 5, 7)            # batch of 4
    obs = rng.integers(7, size=300)
    E = emission_likelihoods(B, obs)
    _, log_c = forward(trans, E)
    _, ll_b = backward(trans, E)
    assert np.allclose(log_c.sum(0), ll_b)


def test_long_sequence_no_underflow():
    rng = np.random.default_rng(0)
    trans, B = tiny_hmm(0, K=26, S=26)
    obs = rng.integers(26, size=20000)
    ll = log_likelihood(trans, B, obs)[0]
    assert np.isfinite(ll) and ll < -20000
    path, lp = viterbi(trans, B, obs)
    assert np.isfinite(lp) and len(path) == 20000


def test_batch_matches_single():
    rng = np.random.default_rng(3)
    trans, _ = tiny_hmm(3, K=4, S=6)
    B = random_emissions(rng, 3, 4, 6)
    obs = rng.integers(6, size=50)
    batch = log_likelihood(trans, B, obs)
    single = [log_likelihood(trans, B[r], obs)[0] for r in range(3)]
    assert np.allclose(batch, single)


def test_m_step():
    counts = np.array([[[2.0, 0.0, 2.0], [0.0, 0.0, 0.0]]])
    assert np.allclose(m_step(counts), [[[0.5, 0, 0.5], [1 / 3] * 3]])
    assert np.allclose(m_step(counts, alpha=1.0)[0, 0], [3 / 7, 1 / 7, 3 / 7])


@pytest.mark.parametrize("alpha", [0.0, 0.1, 1.0])
def test_baum_welch_objective_non_decreasing(alpha):
    """EM never decreases the (penalized) log-likelihood with transitions fixed."""
    rng = np.random.default_rng(5)
    trans, B_true = tiny_hmm(5, K=6, S=8)
    obs = rng.integers(8, size=400)
    res = baum_welch(trans, obs, random_emissions(rng, 5, 6, 8), alpha=alpha,
                     max_iter=100, tol=0.0)
    for h in res.history:
        assert np.all(np.diff(h) >= -1e-8 * abs(h[0]))
        assert h[-1] > h[0]
    if alpha == 0:
        for h, hl in zip(res.history, res.loglik_history):
            assert np.allclose(h, hl)


def test_baum_welch_on_real_letters_non_decreasing():
    from src.lm import get_bigram
    lm = get_bigram()
    rng = np.random.default_rng(0)
    obs = rng.integers(26, size=300)
    res = baum_welch(DenseTransitions(lm.pi, lm.A), obs, random_emissions(rng, 3, 26, 26),
                     max_iter=50, tol=0.0)
    for h in res.history:
        assert np.all(np.diff(h) >= -1e-6)


def test_baum_welch_convergence_stops_early():
    rng = np.random.default_rng(6)
    trans, _ = tiny_hmm(6, K=3, S=4)
    obs = rng.integers(4, size=200)
    res = baum_welch(trans, obs, random_emissions(rng, 2, 3, 4), max_iter=1000, tol=1e-6)
    assert (res.n_iter < 1000).all()
    assert np.allclose(res.loglik, log_likelihood(trans, res.B, obs))


@pytest.mark.parametrize("seed", range(3))
def test_forward_matches_hmmlearn(seed):
    hmmlearn = pytest.importorskip("hmmlearn.hmm")
    rng = np.random.default_rng(seed)
    K, S = 5, 9
    trans, B = tiny_hmm(seed, K=K, S=S)
    obs = rng.integers(S, size=200)
    model = hmmlearn.CategoricalHMM(n_components=K, n_features=S, init_params="", params="")
    model.startprob_, model.transmat_, model.emissionprob_ = trans.pi, trans.A, B
    assert np.isclose(log_likelihood(trans, B, obs)[0], model.score(obs.reshape(-1, 1)))
    _, ref_path = model.decode(obs.reshape(-1, 1), algorithm="viterbi")
    assert np.array_equal(viterbi(trans, B, obs)[0], ref_path)
