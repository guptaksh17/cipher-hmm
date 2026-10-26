"""Experiment 5: simple vs homophonic substitution, accuracy vs number of cipher symbols.

S = 26 is simple substitution; 40, 54 and 70 are homophonic keys with homophone
counts proportional to English letter frequency. Ciphertexts of 408 letters (the
Zodiac length) and 1000 letters, 10 ciphers per setting.
Methods: frequency analysis, bigram HMM (50 restarts), staged HMM (alpha 0.02).
Writes results/exp5_homophonic.csv, its summary and a plot.
"""
from common import (METHODS, finish_figure, load_csv, top_legend, line_with_band, make_instance, output_name,
                    parse_args, plt, run_method, run_tasks, save_csv, setup_style, summarize)

SYMBOLS = [26, 40, 54, 70]
LENGTHS = [408, 1000]
N_INSTANCES = 10
METHOD_ORDER = ["freq", "bigram", "staged"]
ALPHA_STAGED = 0.02


def task(t: dict) -> list[dict]:
    inst = make_instance(t["length"], t["n_symbols"], t["index"])
    r = run_method(t["method"], inst.cipher, t["n_symbols"], inst.plain, inst.key.sym_to_letter,
                   seed=t["index"], alpha=ALPHA_STAGED if t["method"] == "staged" else 0.0,
                   n_restarts=t["n_restarts"], n_pool=t["n_pool"], n_keep=t["n_keep"])
    return [dict(length=t["length"], n_symbols=t["n_symbols"], index=t["index"],
                 method=t["method"], **r)]


def plot(summary, name: str) -> None:
    setup_style()
    lengths = sorted(summary.length.unique())
    fig, axes = plt.subplots(1, len(lengths), figsize=(5.5 * len(lengths), 4.2), sharey=True,
                             squeeze=False)
    for ax, L in zip(axes[0], lengths):
        for m in METHOD_ORDER:
            d = summary[(summary.method == m) & (summary.length == L)]
            line_with_band(ax, d.n_symbols, d.acc_key_mean, d.acc_key_std, METHODS[m],
                           label_end=False)
        ax.set_xticks(SYMBOLS, ["26\n(simple)"] + [str(s) for s in SYMBOLS[1:]])
        ax.set_xlabel("number of cipher symbols")
        ax.set_title(f"{L}-letter ciphertexts", fontsize=10)
        ax.set_ylim(0, 1.02)
    axes[0][0].set_ylabel("symbol accuracy")
    top_legend(fig, axes[0][0], "Simple vs homophonic substitution")
    finish_figure(fig, name, "Mean ± 1 std over 10 ciphers. Homophones per letter ∝ English "
                  "letter frequency; each occurrence picks a homophone uniformly at random.")


def main() -> None:
    args = parse_args(__doc__)
    name = output_name("exp5_homophonic", args.quick)
    if args.plot_only:
        return plot(load_csv(name + "_summary"), name)
    if args.quick:
        symbols, lengths, n = [26, 54], [408], 2
        budget = dict(n_restarts=10, n_pool=40, n_keep=5)
    else:
        symbols, lengths, n = SYMBOLS, LENGTHS, N_INSTANCES
        budget = dict(n_restarts=50, n_pool=1000, n_keep=50)
    tasks = [dict(length=L, n_symbols=S, index=i, method=m, **budget)
             for L in sorted(lengths, reverse=True) for S in symbols for i in range(n)
             for m in METHOD_ORDER]
    df = run_tasks(task, tasks, args.workers).sort_values(["method", "length", "n_symbols", "index"])
    save_csv(df, name)
    cols = ["acc_key", "acc_viterbi", "acc_posterior", "key_acc", "runtime"]
    summary = summarize(df, ["method", "length", "n_symbols"], cols)
    save_csv(summary, name + "_summary")
    plot(summary, name)
    print(summary.pivot_table(index=["length", "n_symbols"], columns="method",
                              values="acc_key_mean").round(3))


if __name__ == "__main__":
    main()
