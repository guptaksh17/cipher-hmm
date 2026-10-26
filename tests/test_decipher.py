"""Tests for restarts, MAP smoothing and key extraction in decipher.py."""
import numpy as np

from src.ciphers import encrypt, homophonic_key, simple_key
from src.corpus import sample_passages
from src.decipher import DecipherConfig, decipher, many_to_one_key, one_to_one_key
from src.lm import get_bigram, text_to_indices
from src.metrics import key_accuracy, symbol_accuracy


def make_cipher(seed, length, n_symbols=26):
    rng = np.random.default_rng(seed)
    plain = text_to_indices(sample_passages(1, length, seed=seed)[0])
    key = simple_key(rng) if n_symbols == 26 else homophonic_key(n_symbols, get_bigram().pi, rng)
    return plain, key, encrypt(plain, key, rng)


def test_one_to_one_key_is_permutation_and_optimal():
    rng = np.random.default_rng(0)
    perm = rng.permutation(26)
    N = rng.random((26, 26))
    N[perm, np.arange(26)] += 5.0  # true letter of symbol s is perm[s]
    key = one_to_one_key(N)
    assert sorted(key) == list(range(26))
    assert np.array_equal(key, perm)


def test_many_to_one_key():
    N = np.array([[3.0, 0.0, 1.0], [1.0, 2.0, 4.0]])
    assert list(many_to_one_key(N)) == [0, 1, 1]


def test_restarts_logged_and_best_selected():
    _, _, c = make_cipher(1, 300)
    res = decipher(c, 26, get_bigram(), DecipherConfig(n_restarts=8, max_iter=50))
    assert len(res.restart_logliks) == 8
    assert res.best_restart == int(np.argmax(res.restart_logliks))
    assert np.isclose(res.loglik, res.restart_logliks.max())
    assert np.all(np.diff(res.history) >= -1e-6)


def test_reproducible_with_seed():
    _, _, c = make_cipher(2, 300)
    cfg = DecipherConfig(n_restarts=5, max_iter=40, seed=3)
    r1, r2 = decipher(c, 26, get_bigram(), cfg), decipher(c, 26, get_bigram(), cfg)
    assert np.array_equal(r1.B, r2.B) and np.array_equal(r1.decoded, r2.decoded)


def test_map_smoothing_removes_zero_emissions():
    _, _, c = make_cipher(3, 100)
    lm = get_bigram()
    ml = decipher(c, 26, lm, DecipherConfig(n_restarts=4, max_iter=60, alpha=0.0))
    mp = decipher(c, 26, lm, DecipherConfig(n_restarts=4, max_iter=60, alpha=0.1))
    absent = np.setdiff1d(np.arange(26), c)
    if len(absent):  # symbols never seen get exactly zero probability under ML
        assert np.allclose(ml.B[:, absent], 0)
    assert (mp.B > 0).all()


def test_key_decoding_beats_frequency_level_on_simple():
    plain, key, c = make_cipher(4, 1000)
    res = decipher(c, 26, get_bigram(), DecipherConfig(n_restarts=20))
    assert sorted(res.key) == list(range(26))
    assert symbol_accuracy(res.decoded, plain) > 0.9
    assert key_accuracy(res.key, key.sym_to_letter, c) > 0.8


def test_homophonic_uses_many_to_one_key():
    plain, key, c = make_cipher(5, 400, n_symbols=40)
    res = decipher(c, 40, get_bigram(), DecipherConfig(n_restarts=4, max_iter=30))
    assert res.key.shape == (40,) and res.B.shape == (26, 40)
    assert np.array_equal(res.decoded, res.key[c])
