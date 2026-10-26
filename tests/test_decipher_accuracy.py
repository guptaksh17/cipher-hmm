"""End-to-end check: the bigram HMM breaks 1000-letter simple substitution ciphers.

Ten fixed seeds (key + held-out passage each). Measured when written: mean 0.950;
nine seeds exceed 0.90 and seed 0 reaches 0.858 (a real optimum of the bigram
model, not a convergence or restart problem). The test therefore asserts the
mean, the median and a per-seed floor rather than one hand-picked seed.
"""
import numpy as np
import pytest

from src.ciphers import encrypt, simple_key
from src.corpus import sample_passages
from src.hmm import DenseTransitions, baum_welch, random_emissions, viterbi
from src.lm import get_bigram, text_to_indices
from src.metrics import symbol_accuracy


def decipher_accuracy(seed: int, n_restarts: int = 50) -> float:
    lm = get_bigram()
    trans = DenseTransitions(lm.pi, lm.A)
    rng = np.random.default_rng(seed)
    plain = text_to_indices(sample_passages(1, 1000, seed=seed)[0])
    cipher = encrypt(plain, simple_key(rng), rng)
    res = baum_welch(trans, cipher, random_emissions(rng, n_restarts, 26, 26), max_iter=300)
    B = res.B[int(np.argmax(res.loglik))]  # unsupervised choice: highest likelihood
    return symbol_accuracy(viterbi(trans, B, cipher)[0], plain)


@pytest.mark.slow
def test_bigram_hmm_breaks_1000_letter_simple_substitution():
    accs = np.array([decipher_accuracy(seed) for seed in range(10)])
    print("per-seed symbol accuracy:", np.round(accs, 3))
    assert accs.mean() > 0.90
    assert np.median(accs) > 0.90
    assert accs.min() > 0.80
