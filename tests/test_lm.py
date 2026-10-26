"""Tests for corpus cleaning and the language models."""
import numpy as np

from src.corpus import TRAIN_BOOKS, TEST_BOOKS, clean_text, load_train_corpus, sample_passages, strip_gutenberg
from src.lm import (ALPHABET, get_bigram, get_trigram, indices_to_text, ngram_counts,
                    text_to_indices, train_bigram, train_trigram, weight_transitions)


def test_clean_text():
    assert clean_text("Hello, World! 123 it's") == "helloworldits"
    assert clean_text("Hello, World!  it's", keep_spaces=True) == "hello world it s"


def test_strip_gutenberg():
    raw = "licence\n*** START OF THE PROJECT GUTENBERG EBOOK X ***\nbody\n*** END OF THE PROJECT GUTENBERG EBOOK X ***\nlegal"
    assert strip_gutenberg(raw).strip() == "body"


def test_train_and_test_books_disjoint():
    assert not set(TRAIN_BOOKS) & set(TEST_BOOKS)


def test_corpus_size_and_alphabet():
    corpus = load_train_corpus()
    assert len(corpus) > 5_000_000
    assert set(corpus[:100_000]) <= set(ALPHABET)


def test_test_passages_not_in_training_corpus():
    corpus = load_train_corpus()
    for p in sample_passages(5, 60, seed=123):
        assert p not in corpus


def test_index_roundtrip():
    assert indices_to_text(text_to_indices("thequickbrownfox")) == "thequickbrownfox"


def test_ngram_counts():
    c = ngram_counts(np.array([0, 1, 0, 1]), 2, 3)
    assert c[0, 1] == 2 and c[1, 0] == 1 and c.sum() == 3


def test_models_are_normalized():
    bi, tri = get_bigram(), get_trigram()
    assert np.allclose(bi.pi.sum(), 1) and np.allclose(bi.A.sum(1), 1)
    assert np.allclose(tri.pi2.sum(), 1) and np.allclose(tri.T.sum(-1), 1)
    assert (bi.A > 0).all() and (tri.T > 0).all()  # add-k smoothing


def test_add_k_smoothing_value():
    lm = train_bigram("ab", k=1.0)
    assert np.isclose(lm.A[0, 1], 2 / 27) and np.isclose(lm.A[1, 0], 1 / 26)


def test_english_statistics():
    bi = get_bigram()
    assert ALPHABET[int(np.argmax(bi.pi))] == "e"
    t, h = ALPHABET.index("t"), ALPHABET.index("h")
    assert ALPHABET[int(np.argmax(bi.A[t]))] == "h"
    assert bi.log_prob("thequickbrownfox") > bi.log_prob("xqzjvkwpqzxjvkqz")


def test_trigram_beats_bigram_on_held_out_text():
    bi, tri = get_bigram(), get_trigram()
    p = sample_passages(1, 2000, seed=7)[0]
    assert tri.log_prob(p) > bi.log_prob(p)


def test_weight_transitions():
    A = get_bigram().A
    assert weight_transitions(A, 1.0) is A
    W = weight_transitions(A, 2.0)
    assert np.allclose(W.sum(1), 1)
    assert np.allclose(W, A**2 / (A**2).sum(1, keepdims=True))
    # sharpening lowers row entropy
    ent = lambda P: -(P * np.log(P)).sum(1)
    assert (ent(W) < ent(A)).all()
