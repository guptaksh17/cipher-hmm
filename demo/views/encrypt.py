"""Page 1: choose a plaintext, generate a random key, show the ciphertext."""
import numpy as np
import streamlit as st

from helpers import (CipherState, bigram, key_table, mono_html, set_cipher, test_texts)
from src.ciphers import GLYPHS, encrypt, homophone_counts, homophonic_key, simple_key
from src.corpus import TEST_BOOKS, clean_text
from src.lm import ALPHABET, indices_to_text, text_to_indices

DEFAULT_TEXT = ("It was the best of times, it was the worst of times, it was the age of "
                "wisdom, it was the age of foolishness, it was the epoch of belief, it was "
                "the epoch of incredulity, it was the season of Light, it was the season of "
                "Darkness, it was the spring of hope, it was the winter of despair.")


def _bump(key: str) -> None:
    st.session_state[key] = st.session_state.get(key, 1) + 1


def _plaintext() -> str:
    source = st.radio("Plaintext", ["Held-out book passage", "Type or paste text"],
                      horizontal=True)
    if source == "Held-out book passage":
        books = test_texts()
        book = st.selectbox("Book (never used to train the language model)", list(books),
                            format_func=lambda b: TEST_BOOKS[b])
        length = st.slider("Length (letters)", 50, 2000, 300, step=10)
        st.button("🔀 Another passage", on_click=_bump, args=("passage_seed",))
        rng = np.random.default_rng(st.session_state.get("passage_seed", 1))
        text = books[book]
        start = int(rng.integers(0, len(text) - length))
        return text[start:start + length]
    raw = st.text_area("Your text (letters are kept, everything else is removed)",
                       DEFAULT_TEXT, height=140)
    return clean_text(raw)


def _key(rng: np.random.Generator):
    kind = st.radio("Cipher", ["Simple substitution", "Homophonic substitution"],
                    horizontal=True)
    if kind == "Simple substitution":
        return simple_key(rng), "random"
    n = st.slider("Number of cipher symbols", 27, len(GLYPHS), 54,
                  help="Homophones per letter are roughly proportional to English letter "
                       "frequency; every letter gets at least one.")
    mode = st.selectbox("Homophone choice for each occurrence", ["random", "cycle"],
                        format_func=lambda m: {"random": "uniformly at random",
                                               "cycle": "cycle through them in order"}[m])
    counts = homophone_counts(bigram().pi, n)
    st.caption("Homophones per letter: " + " ".join(f"{ALPHABET[i]}{c}" for i, c in
                                                    enumerate(counts)))
    return homophonic_key(n, bigram().pi, rng), mode


def render() -> None:
    st.title("Encrypt")
    st.write("Pick a plaintext and a random key. The ciphertext is what the HMM gets on "
             "the next page; the plaintext and key are kept only to score the result.")
    left, right = st.columns(2, gap="large")
    with left:
        plain_str = _plaintext()
    with right:
        st.button("🔑 New random key", on_click=_bump, args=("key_seed",))
        rng = np.random.default_rng(st.session_state.get("key_seed", 1))
        key, mode = _key(rng)
    if len(plain_str) < 20:
        st.warning("Please enter at least 20 letters.")
        return
    plain = text_to_indices(plain_str)
    cipher = encrypt(plain, key, rng, mode=mode)
    state = CipherState(cipher=cipher, n_symbols=key.n_symbols, glyphs=GLYPHS[:key.n_symbols],
                        plain=plain, true_key=key.sym_to_letter,
                        label=f"{len(plain)} letters, {key.n_symbols} symbols")
    set_cipher(state)

    st.divider()
    c1, c2, c3 = st.columns(3)
    c1.metric("Length", f"{len(plain)} letters")
    c2.metric("Cipher symbols", key.n_symbols)
    c3.metric("Distinct symbols used", len(np.unique(cipher)))
    st.subheader("Plaintext")
    st.markdown(mono_html(indices_to_text(plain)), unsafe_allow_html=True)
    st.subheader("Ciphertext")
    st.markdown(mono_html(state.text()), unsafe_allow_html=True)
    with st.expander("Key (symbol → letter)"):
        st.dataframe(key_table(key.sym_to_letter, state).drop(columns=["decoded as", "correct"],
                                                              errors="ignore"),
                     hide_index=True, width="stretch")
    st.success("Ciphertext ready. Open **2. Decipher live** to break it.")
