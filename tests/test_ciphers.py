"""Tests for cipher generation, the frequency baseline and metrics."""
import numpy as np

from src.baseline import frequency_analysis
from src.ciphers import decrypt, encrypt, glyphs, homophone_counts, homophonic_key, simple_key
from src.corpus import sample_passages
from src.lm import get_bigram, text_to_indices
from src.metrics import key_accuracy, key_from_decoding, symbol_accuracy


def test_simple_key_is_permutation():
    key = simple_key(np.random.default_rng(0))
    assert sorted(key.sym_to_letter) == list(range(26))
    assert all(len(s) == 1 for s in key.letter_to_syms)


def test_simple_roundtrip():
    rng = np.random.default_rng(1)
    plain = text_to_indices(sample_passages(1, 500, seed=1)[0])
    key = simple_key(rng)
    c = encrypt(plain, key, rng)
    assert not np.array_equal(c, plain)
    assert np.array_equal(decrypt(c, key.sym_to_letter), plain)


def test_homophone_counts_proportional():
    freqs = get_bigram().pi
    for S in (26, 40, 54, 70):
        counts = homophone_counts(freqs, S)
        assert counts.sum() == S and counts.min() >= 1
    counts = homophone_counts(freqs, 70)
    assert counts[4] == counts.max()          # 'e' gets the most homophones
    assert counts[25] == 1                    # 'z' gets one


def test_homophonic_roundtrip_both_modes():
    rng = np.random.default_rng(2)
    plain = text_to_indices(sample_passages(1, 800, seed=2)[0])
    key = homophonic_key(54, get_bigram().pi, rng)
    assert sorted(np.concatenate(key.letter_to_syms)) == list(range(54))
    for mode in ("random", "cycle"):
        c = encrypt(plain, key, rng, mode=mode)
        assert np.array_equal(decrypt(c, key.sym_to_letter), plain)
    # homophones flatten the symbol distribution
    c = encrypt(plain, key, rng)
    assert np.bincount(c).max() < np.bincount(plain).max()


def test_cycle_mode_order():
    rng = np.random.default_rng(3)
    key = homophonic_key(30, np.ones(26), rng)
    letter = next(i for i, s in enumerate(key.letter_to_syms) if len(s) > 1)
    syms = key.letter_to_syms[letter]
    c = encrypt(np.full(2 * len(syms), letter), key, rng, mode="cycle")
    assert list(c) == list(syms) * 2


def test_glyphs():
    assert glyphs(np.array([0, 1, 25])) == "ABZ"
    assert len(set(glyphs(np.arange(70)))) == 70


def test_metrics():
    plain = np.array([0, 1, 2, 0])
    assert symbol_accuracy(np.array([0, 1, 0, 0]), plain) == 0.75
    true_key = np.array([0, 1, 2, 3])
    pred_key = np.array([0, 1, 0, 0])
    assert key_accuracy(pred_key, true_key, np.array([0, 1, 2])) == 2 / 3  # symbol 3 absent
    cipher = np.array([0, 0, 0, 1])
    assert list(key_from_decoding(cipher, np.array([5, 5, 4, 2]), 2)) == [5, 2]


def test_frequency_baseline_long_text():
    """Rank matching is far above chance on long simple-substitution text."""
    rng = np.random.default_rng(4)
    plain = text_to_indices(sample_passages(1, 20000, seed=4)[0])
    key = simple_key(rng)
    c = encrypt(plain, key, rng)
    pred = frequency_analysis(c, 26, get_bigram().pi)
    assert symbol_accuracy(decrypt(c, pred), plain) > 0.4
