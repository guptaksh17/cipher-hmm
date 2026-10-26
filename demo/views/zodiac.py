"""Page 4: the Zodiac 408 cipher, deciphered live with the staged trigram HMM."""
import time

import numpy as np
import pandas as pd
import streamlit as st

from helpers import (BATCH_BYTES, RESULTS, Throttle, bigram, grid_html, legend_html, transitions_for, z408,
                     zodiac_alpha)
from src.decipher import (DecipherConfig, bigram_transitions, run_em, run_em_from,
                          select_and_decode)
from src.hmm import viterbi
from src.lm import indices_to_text
from src.metrics import key_accuracy, symbol_accuracy
from src.zodiac import N_COLS, true_key



def _experiment_reference() -> str | None:
    path = RESULTS / "exp6_zodiac.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    row = df[(df.method == "staged") & df.selected].sort_values("alpha").iloc[-1]
    return (f"Experiment 6 (1000 bigram → 50 trigram restarts, 5 seeds, α = {row.alpha:g}) "
            f"reached **{row.acc_key:.1%}**.")


def render() -> None:
    z = z408()
    solution = indices_to_text(z.plain)
    st.title("Zodiac 408")
    st.markdown("Sent in three parts to Bay Area newspapers in 1969 and solved within a week "
                "by Donald and Bettye Harden. It is a homophonic substitution cipher: "
                f"**{z.n_symbols} symbols** for {len(np.unique(z.plain))} plaintext letters, "
                f"**{len(z.cipher)} symbols** long, "
                "with no word breaks. Below is the standard ASCII transcription in its "
                "17 × 24 layout.")
    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("Ciphertext")
        st.markdown(grid_html(z.text, N_COLS, upper=False), unsafe_allow_html=True)
    with right:
        st.subheader("Decipher")
        alpha = zodiac_alpha()
        n_pool = st.slider("Bigram restarts (pool)", 50, 1000, 200, step=50)
        n_keep = st.slider("Trigram restarts (best of pool)", 1, 50, 10)
        seed = st.number_input("Random seed", 0, 10_000, 2)
        st.caption(f"Pseudo-count α = {alpha:g}, chosen on synthetic 408-letter, 54-symbol "
                   "ciphers (experiment 3), not on the Zodiac. The best restart is picked by "
                   "log-likelihood; the solution is used only for colouring.")
        go = st.button("▶ Run staged decipherment", type="primary")
        ref = _experiment_reference()
        if ref:
            st.caption(ref)
    st.divider()
    status, grid_ph = st.empty(), st.empty()
    if go:
        cfg = DecipherConfig(n_restarts=n_keep, alpha=alpha, seed=int(seed),
                             max_batch_bytes=BATCH_BYTES)
        t0 = time.perf_counter()
        with st.spinner(f"Stage 1: {n_pool} bigram restarts…"):
            pool = run_em(bigram_transitions(bigram()), z.cipher, z.n_symbols,
                          DecipherConfig(n_restarts=n_pool, seed=int(seed),
                                                max_batch_bytes=BATCH_BYTES))
        B0 = pool.B[np.argsort(-pool.loglik)[:n_keep]] + 0.01
        B0 /= B0.sum(axis=-1, keepdims=True)
        trans = transitions_for("trigram")
        throttle = Throttle(0.5)

        def show(it, B):
            path = viterbi(trans, B, z.cipher)[0]
            status.markdown(f"Stage 2: trigram EM · iteration **{it}** · Viterbi accuracy "
                            f"**{symbol_accuracy(path, z.plain):.1%}**")
            grid_ph.markdown(grid_html(indices_to_text(path), N_COLS, solution),
                             unsafe_allow_html=True)

        def callback(it, B, ll):
            if throttle.ready(force=it == 0):
                throttle.draw(show, it, B[int(np.argmax(ll))])

        res = run_em_from(trans, z.cipher, B0, cfg, callback)
        st.session_state["zodiac_result"] = select_and_decode(trans, z.cipher, z.n_symbols,
                                                              cfg, res, t0)
    r = st.session_state.get("zodiac_result")
    if r is None:
        return
    status.empty()
    grid_ph.empty()
    c = st.columns(4)
    c[0].metric("Symbol accuracy", f"{symbol_accuracy(r.decoded, z.plain):.1%}")
    c[1].metric("Viterbi accuracy", f"{symbol_accuracy(r.viterbi, z.plain):.1%}")
    c[2].metric("Key accuracy (54 symbols)", f"{key_accuracy(r.key, true_key(z), z.cipher):.1%}")
    c[3].metric("Runtime", f"{r.runtime:.1f} s")
    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("Decoded (consistent key)")
        st.markdown(grid_html(indices_to_text(r.decoded), N_COLS, solution),
                    unsafe_allow_html=True)
        st.caption(legend_html(), unsafe_allow_html=True)
    with right:
        st.subheader("Known solution")
        st.markdown(grid_html(solution, N_COLS), unsafe_allow_html=True)
        st.caption("As published, with the author's misspellings (e.g. FORREST, PARADICE).")
