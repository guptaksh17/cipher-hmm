"""Page 3: the Viterbi trellis on a short window of the ciphertext."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from helpers import (GRID, MUTED, ORANGE, SEQ_CMAP, SURFACE, TEXT, TEXT_2, get_cipher,
                     transitions_for)
from src.hmm import Transitions
from src.lm import ALPHABET


def viterbi_trellis(trans: Transitions, B: np.ndarray, obs: np.ndarray):
    """Log-space Viterbi keeping every column: delta (W, K), backpointers (W, K), path."""
    with np.errstate(divide="ignore"):
        log_E = np.log(B[:, obs]).T
        delta = np.empty_like(log_E)
        delta[0] = np.log(trans.pi) + log_E[0]
    back = np.zeros(log_E.shape, dtype=np.int64)
    for t in range(1, len(obs)):
        best, back[t] = trans.max_step(delta[t - 1])
        delta[t] = best + log_E[t]
    path = np.empty(len(obs), dtype=np.int64)
    path[-1] = int(delta[-1].argmax())
    for t in range(len(obs) - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return delta, back, path


def draw_trellis(delta, back, path, glyphs: str, step: int, show_path: bool, top_k: int):
    """Columns 0..step-1 of the trellis; best path in orange once backtracked."""
    W, K = delta.shape
    fig, ax = plt.subplots(figsize=(max(7.0, 1.0 * W + 1.5), 6.0), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for t in range(1, step):  # backpointers of the top-k states of each column
        for j in np.argsort(-delta[t])[:top_k]:
            ax.plot([t - 1, t], [back[t, j], j], color=MUTED, alpha=0.45, linewidth=0.9,
                    zorder=1)
    for t in range(step):
        rel = np.exp(delta[t] - delta[t].max())  # score relative to the column's best
        ax.scatter(np.full(K, t), np.arange(K), c=rel, cmap=SEQ_CMAP, vmin=0, vmax=1, s=70,
                   edgecolors=GRID, linewidths=0.6, zorder=2)
    if show_path:
        ax.plot(np.arange(W), path, color=ORANGE, linewidth=3, zorder=3)
        ax.scatter(np.arange(W), path, s=120, facecolors="none", edgecolors=ORANGE,
                   linewidths=2.2, zorder=4)
        for t in range(W):
            ax.text(t, K + 0.4, ALPHABET[path[t]].upper(), ha="center", va="top",
                    fontsize=13, color=ORANGE, fontweight="bold")
    for t in range(W):
        ax.text(t, -1.3, glyphs[t], ha="center", va="bottom", fontsize=14, color=TEXT,
                family="monospace")
    ax.set_xlim(-0.6, W - 0.4)
    ax.set_ylim(K + 1.6, -2.2)
    ax.set_yticks(range(K), list(ALPHABET), fontsize=8, color=TEXT_2)
    ax.set_xticks(range(W), [f"t={t + 1}" for t in range(W)], fontsize=8, color=TEXT_2)
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_ylabel("hidden state (plaintext letter)", color=TEXT_2)
    fig.tight_layout()
    return fig


def _emissions(state):
    """Emission matrix to use: learned (page 2) or the true key's, as the user picks."""
    saved = st.session_state.get("result")
    options = []
    if saved and saved["cipher_id"] == state.fingerprint():
        options.append("Learned on page 2")
    if state.plain is not None:
        options.append("True key (for reference)")
    if not options:
        st.info("Run **2. Decipher live** on this ciphertext first, so there is a learned "
                "emission matrix to use.")
        return None
    choice = st.radio("Emission matrix B", options, horizontal=True)
    if choice == "Learned on page 2":
        return saved["result"].B
    N = np.full((26, state.n_symbols), 1e-3)
    np.add.at(N, (state.plain, state.cipher), 1.0)
    return N / N.sum(axis=1, keepdims=True)


def render() -> None:
    st.title("Inside the HMM: the Viterbi trellis")
    st.markdown(
        "Each column is one cipher symbol; each dot is a hidden state (a plaintext letter). "
        "Moving left to right, Viterbi keeps for every state the score of the best path "
        "ending there,  \n"
        r"$\delta_t(j) = \max_i \big[\delta_{t-1}(i) + \log a_{ij}\big] + \log b_j(o_t)$, "
        "and remembers which predecessor $i$ achieved it (grey lines). After the last "
        "column it backtracks from the best final state along those pointers (orange).")
    state = get_cipher()
    if state is None:
        st.info("Create or paste a ciphertext on pages 1 or 2 first.")
        return
    B = _emissions(state)
    if B is None:
        return
    c1, c2, c3 = st.columns(3)
    W = c1.slider("Window length", 6, 14, 10)
    start = c2.slider("Window start", 0, max(0, len(state.cipher) - W), 0)
    top_k = c3.slider("Back-pointers drawn per column", 1, 26, 6)
    obs = state.cipher[start:start + W]
    W = len(obs)
    trans = transitions_for("bigram")
    delta, back, path = viterbi_trellis(trans, B, obs)
    step = st.slider("Forward pass: columns computed", 1, W, W)
    show_path = st.toggle("Backtrack and show the best path", value=True, disabled=step < W)
    glyphs = "".join(state.glyphs[s] for s in obs)
    fig = draw_trellis(delta, back, path, glyphs, step, show_path and step == W, top_k)
    st.pyplot(fig)
    plt.close(fig)
    st.caption("Dot colour: δ relative to the best state in the same column (dark = close to "
               "the best). The trellis uses the bigram transitions, since the 676 pair states of "
               "the trigram model would not fit on screen.")
    if state.plain is not None:
        truth = "".join(ALPHABET[x] for x in state.plain[start:start + W])
        st.markdown(f"Best path: **{''.join(ALPHABET[p] for p in path)}** · true plaintext: "
                    f"**{truth}**")
    col = st.slider("Inspect column", 1, step, step)
    t = col - 1
    top = np.argsort(-delta[t])[:5]
    st.dataframe(pd.DataFrame({
        "state": [ALPHABET[j] for j in top],
        "δ (log score)": np.round(delta[t, top], 2),
        "best predecessor": ["start" if t == 0 else ALPHABET[back[t, j]] for j in top],
    }), hide_index=True)
