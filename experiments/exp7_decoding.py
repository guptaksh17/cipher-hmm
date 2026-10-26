"""Experiment 7: Viterbi vs posterior (forward-backward) decoding.

Every HMM run in experiments 1 and 5 records three decodings of the same best
model: Viterbi (most probable path), posterior / maximum posterior marginal
(argmax_k P(x_t = k | o) per position), and the consistent symbol -> letter key
derived from expected counts. This script compares them; it reuses those runs
instead of repeating them, so run experiments 1 and 5 first.
Writes results/exp7_decoding_summary.csv and a plot.
"""
import sys

import pandas as pd

from common import (DECODINGS, RESULTS, finish_figure, line_with_band, output_name,
                    parse_args, plt, save_csv, setup_style, summarize)

COLS = list(DECODINGS)


def load(name: str, quick: bool) -> pd.DataFrame:
    path = RESULTS / f"{output_name(name, quick)}.csv"
    if not path.exists():
        sys.exit(f"{path.name} not found: run {name.split('_')[0]} first")
    return pd.read_csv(path)


def main() -> None:
    args = parse_args(__doc__)
    e1 = load("exp1_length", args.quick)
    e5 = load("exp5_homophonic", args.quick)
    e1 = e1[e1.method.isin(["bigram", "staged"])].assign(n_symbols=26)
    e5 = e5[e5.method.isin(["bigram", "staged"])]
    s1 = summarize(e1, ["method", "length"], COLS).assign(source="exp1")
    s5 = summarize(e5, ["method", "length", "n_symbols"], COLS).assign(source="exp5")
    name = output_name("exp7_decoding", args.quick)
    save_csv(pd.concat([s1, s5], ignore_index=True), name + "_summary")

    setup_style()
    L5 = min(e5.length)
    panels = [("bigram", s1, "length", "A. Bigram HMM, simple substitution"),
              ("staged", s1, "length", "B. Staged trigram HMM, simple substitution"),
              ("bigram", s5[s5.length == L5], "n_symbols", f"C. Bigram HMM, {L5} letters"),
              ("staged", s5[s5.length == L5], "n_symbols", f"D. Staged trigram HMM, {L5} letters")]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.6), sharey=True,
                             gridspec_kw=dict(hspace=0.42))
    for ax, (m, s, x, title) in zip(axes.flat, panels):
        d = s[s.method == m].sort_values(x)
        for col in COLS:
            line_with_band(ax, d[x], d[f"{col}_mean"], d[f"{col}_std"], DECODINGS[col],
                           label_end=False)
        if x == "length":
            ax.set_xscale("log")
            ax.set_xticks(d[x], [str(v) for v in d[x]])
            ax.set_xlabel("ciphertext length (letters)")
        else:
            ax.set_xticks(d[x], [str(v) for v in d[x]])
            ax.set_xlabel("number of cipher symbols")
        ax.set_ylim(0, 1.02)
        ax.set_title(title, fontsize=10)
    for ax in axes[:, 0]:
        ax.set_ylabel("symbol accuracy")
    axes[0, 0].legend(fontsize=8.5, loc="lower right")
    fig.suptitle("Decoding the same learned model three ways", x=0.01, ha="left", fontsize=12,
                 fontweight="bold")
    finish_figure(fig, name, "Mean ± 1 std; runs from experiments 1 and 5. Viterbi and "
                  "posterior decode each position separately; the consistent key decodes "
                  "every occurrence of a symbol the same way.")
    out = pd.concat([s1, s5])
    print(out[["source", "method", "length", "n_symbols"] + [f"{c}_mean" for c in COLS]]
          .round(3).to_string(index=False))


if __name__ == "__main__":
    main()
