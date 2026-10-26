"""Page 2: run EM live and watch the decoding, the likelihood and the key sharpen."""
import time

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from helpers import (BATCH_BYTES, CipherState, Throttle, bigram, colored_html, emission_heatmap, get_cipher,
                     key_table, legend_html, loglik_figure, restarts_figure, set_cipher,
                     transitions_for)
from src.decipher import (DecipherConfig, bigram_transitions, run_em, run_em_from,
                          select_and_decode)
from src.hmm import viterbi
from src.lm import indices_to_text
from src.metrics import key_accuracy, symbol_accuracy

MODELS = {
    "bigram": "Bigram HMM (26 states, fast)",
    "staged": "Staged trigram HMM (bigram pool → 676 pair states)",
    "trigram": "Trigram HMM from random restarts (676 pair states)",
}
ALPHAS = [0.0, 0.001, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]


def _pasted_cipher() -> CipherState | None:
    raw = st.text_area("Ciphertext: every non-space character is one symbol; spaces and line "
                       "breaks are ignored.", height=120,
                       placeholder="e.g. paste a newspaper cryptogram or a symbol transcription")
    text = "".join(raw.split())
    if len(text) < 20:
        st.info("Paste at least 20 symbols.")
        return None
    glyphs = "".join(dict.fromkeys(text))
    ids = {g: i for i, g in enumerate(glyphs)}
    return CipherState(np.array([ids[g] for g in text]), len(glyphs), glyphs, None, None,
                       f"pasted, {len(text)} symbols")


def _controls(state: CipherState):
    c1, c2, c3 = st.columns(3)
    with c1:
        model = st.selectbox("Model", list(MODELS), format_func=MODELS.get)
        seed = st.number_input("Random seed", 0, 10_000, 0)
    with c2:
        if model == "staged":
            n_pool = st.slider("Bigram restarts (pool)", 20, 1000, 200, step=20)
            n_keep = st.slider("Trigram restarts (best of pool)", 1, 50, 10)
            restarts = n_keep
        else:
            restarts = st.slider("Random restarts", 1, 100, 20 if model == "bigram" else 10)
            n_pool = n_keep = 0
        max_iter = st.slider("Max EM iterations", 10, 500, 300, step=10)
    with c3:
        alpha = st.select_slider("Dirichlet pseudo-count α (0 = ML)", ALPHAS,
                                 value=0.02 if state.n_symbols > 26 else 0.0)
        power = st.slider("Language-model exponent", 1.0, 3.0, 1.0, step=0.25)
    cfg = DecipherConfig(n_restarts=restarts, alpha=alpha, lm_power=power, max_iter=max_iter,
                         seed=int(seed), max_batch_bytes=BATCH_BYTES)
    return model, cfg, n_pool, n_keep


def _draw(ph, state, truth, trans, it, B, history, stage):
    path = viterbi(trans, B, state.cipher)[0]
    decoded = indices_to_text(path)
    acc = f" · Viterbi accuracy **{symbol_accuracy(path, state.plain):.1%}**" \
        if state.plain is not None else ""
    ph["status"].markdown(f"{stage} · iteration **{it}** · best log-likelihood "
                          f"**{history[-1]:.1f}**{acc}")
    ph["text"].markdown(colored_html(decoded, truth), unsafe_allow_html=True)
    fig = loglik_figure(history)
    ph["ll"].pyplot(fig)
    plt.close(fig)
    fig = emission_heatmap(B, state, "Emission matrix B of the current best restart")
    ph["heat"].pyplot(fig)
    plt.close(fig)


def _run(state: CipherState, model: str, cfg: DecipherConfig, n_pool: int, n_keep: int):
    truth = indices_to_text(state.plain) if state.plain is not None else None
    ph = {"status": st.empty(), "text": st.empty()}
    left, right = st.columns([1, 1.3])
    with left:
        ph["ll"] = st.empty()
    with right:
        ph["heat"] = st.empty()
    trans = transitions_for("bigram" if model == "bigram" else "trigram", cfg.lm_power)
    history: list[float] = []
    throttle = Throttle()
    stage = ["EM"]

    def callback(it, B, ll):
        history.append(float(np.max(ll)))
        if throttle.ready(force=it == 0):
            throttle.draw(_draw, ph, state, truth, trans, it, B[int(np.argmax(ll))], history,
                          stage[0])

    t0 = time.perf_counter()
    if model == "staged":
        with st.spinner(f"Stage 1: {n_pool} bigram restarts…"):
            pool_cfg = DecipherConfig(n_restarts=n_pool, seed=cfg.seed, max_iter=cfg.max_iter,
                                      max_batch_bytes=BATCH_BYTES)
            pool = run_em(bigram_transitions(bigram()), state.cipher, state.n_symbols, pool_cfg)
        B0 = pool.B[np.argsort(-pool.loglik)[:n_keep]] + 0.01
        B0 /= B0.sum(axis=-1, keepdims=True)
        stage[0] = f"Stage 2: trigram EM from the {n_keep} best bigram solutions"
        res = run_em_from(trans, state.cipher, B0, cfg, callback)
    else:
        res = run_em(trans, state.cipher, state.n_symbols, cfg, callback)
    result = select_and_decode(trans, state.cipher, state.n_symbols, cfg, res, t0)
    _draw(ph, state, truth, trans, result.n_iter, result.B, history,
          f"Finished (best restart {result.best_restart})")
    st.session_state["result"] = dict(result=result, model=model, power=cfg.lm_power,
                                      cipher_id=state.fingerprint(), history=history)


def _show_result(state: CipherState) -> None:
    saved = st.session_state.get("result")
    if not saved or saved["cipher_id"] != state.fingerprint():
        return
    r = saved["result"]
    st.subheader("Result of the best restart")
    cols = st.columns(5)
    if state.plain is not None:
        cols[0].metric("Consistent-key accuracy", f"{symbol_accuracy(r.decoded, state.plain):.1%}")
        cols[1].metric("Viterbi accuracy", f"{symbol_accuracy(r.viterbi, state.plain):.1%}")
        cols[2].metric("Posterior accuracy", f"{symbol_accuracy(r.posterior, state.plain):.1%}")
        cols[3].metric("Key accuracy", f"{key_accuracy(r.key, state.true_key, state.cipher):.1%}")
    cols[4].metric("Runtime", f"{r.runtime:.1f} s")
    st.markdown("**Decoded with the consistent key** (every occurrence of a symbol decoded "
                "the same way)")
    truth = indices_to_text(state.plain) if state.plain is not None else None
    st.markdown(colored_html(indices_to_text(r.decoded), truth), unsafe_allow_html=True)
    if truth is not None:
        st.caption(legend_html(), unsafe_allow_html=True)
    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("**Learned key**")
        st.dataframe(key_table(r.key, state), hide_index=True, width="stretch",
                     height=300)
    with c2:
        st.markdown("**Final log-likelihood of each restart** (local optima of EM)")
        fig = restarts_figure(r.restart_logliks, r.best_restart)
        st.pyplot(fig)
        plt.close(fig)


def render() -> None:
    st.title("Decipher live")
    source = st.radio("Ciphertext", ["From the Encrypt page", "Paste your own"],
                      horizontal=True)
    state = get_cipher() if source == "From the Encrypt page" else _pasted_cipher()
    if state is None:
        if source == "From the Encrypt page":
            st.info("Create a ciphertext on **1. Encrypt** first, or paste your own.")
        return
    if source == "Paste your own":
        set_cipher(state)
    st.caption(f"Ciphertext: {state.label}. The model sees only the symbols; the true "
               "plaintext is used only to colour letters green (right) or red (wrong).")
    model, cfg, n_pool, n_keep = _controls(state)
    if st.button("▶ Run decipherment", type="primary"):
        _run(state, model, cfg, n_pool, n_keep)
    _show_result(state)
