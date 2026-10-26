"""Tests for the Zodiac 408 data and the trigram/staged decipherment entry points."""
import numpy as np

from src.ciphers import encrypt, homophonic_key
from src.corpus import sample_passages
from src.decipher import DecipherConfig, decipher, decipher_staged
from src.lm import get_bigram, get_trigram, indices_to_text, text_to_indices
from src.zodiac import N_COLS, N_ROWS, load_z408, true_key


def test_z408_data():
    z = load_z408()
    assert len(z.cipher) == N_ROWS * N_COLS == 408
    assert z.n_symbols == 54
    assert indices_to_text(z.plain).startswith("ilikekillingpeoplebecause")
    assert "".join(z.glyphs[s] for s in z.cipher) == z.text


def test_z408_solution_is_a_consistent_key():
    z = load_z408()
    assert np.array_equal(true_key(z)[z.cipher], z.plain)


def test_trigram_and_staged_run_on_homophonic():
    rng = np.random.default_rng(0)
    plain = text_to_indices(sample_passages(1, 150, seed=0)[0])
    c = encrypt(plain, homophonic_key(40, get_bigram().pi, rng), rng)
    cfg = DecipherConfig(n_restarts=3, max_iter=10)
    r = decipher(c, 40, get_trigram(), cfg)
    assert r.B.shape == (26, 40) and len(r.decoded) == 150
    s = decipher_staged(c, 40, get_bigram(), get_trigram(), cfg, n_pool=10, n_keep=3)
    assert len(s.restart_logliks) == 3 and np.all(np.diff(s.history) >= -1e-6)
