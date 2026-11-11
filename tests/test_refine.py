"""Tests for key refinement (src/refine.py) and its use in the decipher pipeline."""
import numpy as np

from src.ciphers import encrypt, homophonic_key, simple_key
from src.corpus import sample_passages
from src.decipher import DecipherConfig, decipher
from src.lm import get_bigram, text_to_indices
from src.metrics import symbol_accuracy
from src.refine import get_fourgram, key_score, refine_key, search_keys


def cipher_case(seed, length, n_symbols):
    rng = np.random.default_rng(seed)
    plain = text_to_indices(sample_passages(1, length, seed=seed)[0])
    key = simple_key(rng) if n_symbols == 26 else homophonic_key(n_symbols, get_bigram().pi, rng)
    return plain, key, encrypt(plain, key, rng)


def test_fourgram_is_normalised_log_probabilities():
    logp = get_fourgram()
    assert logp.shape == (26, 26, 26, 26)
    assert np.allclose(np.exp(logp.astype(float)).sum(axis=-1), 1, atol=1e-4)


def test_true_key_scores_higher_than_corrupted_keys():
    for n_symbols in (26, 54):
        plain, key, cipher = cipher_case(1, 400, n_symbols)
        true = key.sym_to_letter
        bad = true.copy()
        bad[[0, 1]] = bad[[1, 0]] if n_symbols == 26 else (bad[[0, 1]] + 3) % 26
        assert key_score(cipher, true, n_symbols, n_symbols == 26) > key_score(cipher, bad, n_symbols, n_symbols == 26)


def test_refine_repairs_a_partly_wrong_one_to_one_key():
    plain, key, cipher = cipher_case(2, 500, 26)
    rng = np.random.default_rng(0)
    start = key.sym_to_letter.copy()
    for _ in range(4):                                   # four random swaps
        a, b = rng.choice(26, size=2, replace=False)
        start[[a, b]] = start[[b, a]]
    assert symbol_accuracy(start[cipher], plain) < 0.9
    fixed = refine_key(cipher, start, 26, one_to_one=True)
    assert sorted(fixed) == list(range(26))              # still a permutation
    assert symbol_accuracy(fixed[cipher], plain) > 0.99


def test_refine_repairs_a_partly_wrong_homophonic_key():
    plain, key, cipher = cipher_case(3, 500, 54)
    start = key.sym_to_letter.copy()
    start[:8] = (start[:8] + 5) % 26                      # 8 symbols wrong
    fixed = refine_key(cipher, start, 54, one_to_one=False)
    assert symbol_accuracy(fixed[cipher], plain) > symbol_accuracy(start[cipher], plain) + 0.1
    assert symbol_accuracy(fixed[cipher], plain) > 0.95


def test_search_never_returns_a_worse_key_than_its_candidates():
    plain, key, cipher = cipher_case(4, 300, 26)
    rng = np.random.default_rng(1)
    cands = [rng.permutation(26) for _ in range(3)]
    best, score = search_keys(cipher, cands, 26, True, n_kicks=5)
    assert score >= max(key_score(cipher, refine_key(cipher, c, 26, True), 26, True) for c in cands) - 1e-6


def test_decipher_with_refinement_reaches_full_accuracy_on_300_letters():
    plain, key, cipher = cipher_case(5, 300, 26)
    r = decipher(cipher, 26, get_bigram(), DecipherConfig(n_restarts=20, seed=0))
    assert r.em_key is not None and r.key_score is not None
    assert symbol_accuracy(r.decoded, plain) >= symbol_accuracy(r.em_key[cipher], plain)
    assert symbol_accuracy(r.decoded, plain) > 0.98


def test_refinement_can_be_switched_off():
    plain, key, cipher = cipher_case(6, 200, 26)
    r = decipher(cipher, 26, get_bigram(), DecipherConfig(n_restarts=5, max_iter=50, refine=False))
    assert r.key_score is None and np.array_equal(r.key, r.em_key)
