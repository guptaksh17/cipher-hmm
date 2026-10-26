"""Experiment 1: accuracy vs ciphertext length on simple substitution ciphers.

Methods: frequency analysis, bigram HMM, trigram HMM from random restarts, and
the staged bigram-to-trigram HMM. 10 random keys and held-out passages per length.
Writes results/exp1_length.csv, exp1_length_summary.csv and two plots.
"""
from common import (METHODS, finish_figure, load_csv, top_legend, line_with_band, make_instance, output_name,
                    parse_args, plt, run_method, run_tasks, save_csv, setup_style, summarize)

LENGTHS = [50, 100, 200, 300, 500, 1000, 2000]
N_INSTANCES = 10
METHOD_ORDER = ["freq", "bigram", "trigram", "staged"]


def task(t: dict) -> list[dict]:
    inst = make_instance(t["length"], 26, t["index"])
    r = run_method(t["method"], inst.cipher, 26, inst.plain, inst.key.sym_to_letter,
                   seed=t["index"], n_restarts=t["n_restarts"], n_pool=t["n_pool"],
                   n_keep=t["n_keep"])
    return [dict(length=t["length"], index=t["index"], method=t["method"], **r)]


def plot(summary, name: str) -> None:
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, col, title in ((axes[0], "acc_key", "Symbol accuracy"),
                           (axes[1], "key_acc", "Key accuracy")):
        for m in METHOD_ORDER:
            d = summary[summary.method == m]
            line_with_band(ax, d.length, d[f"{col}_mean"], d[f"{col}_std"], METHODS[m],
                           label_end=(ax is axes[1]))
        ax.set_xscale("log")
        ax.set_xticks(LENGTHS, [str(x) for x in LENGTHS])
        ax.set_xlabel("ciphertext length (letters)")
        ax.set_ylim(0, 1.02)
        ax.set_title(title)
    axes[0].set_ylabel("fraction correct")
    top_legend(fig, axes[0], "Simple substitution: accuracy vs ciphertext length")
    finish_figure(fig, name, "Mean ± 1 std over 10 random keys and held-out passages per "
                  "length. HMMs: 50 restarts, best log-likelihood kept; staged: 1000 bigram "
                  "restarts → 50 trigram.")

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for m in METHOD_ORDER[1:]:
        d = summary[summary.method == m]
        line_with_band(ax, d.length, d.runtime_mean, d.runtime_std, METHODS[m])
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks(LENGTHS, [str(x) for x in LENGTHS])
    ax.set_xlabel("ciphertext length (letters)"); ax.set_ylabel("seconds per decipherment")
    ax.set_title("Runtime per decipherment (one CPU core)")
    finish_figure(fig, name + "_runtime")


def main() -> None:
    args = parse_args(__doc__)
    name = output_name("exp1_length", args.quick)
    if args.plot_only:
        return plot(load_csv(name + "_summary"), name)
    lengths, n = ([100, 300], 2) if args.quick else (LENGTHS, N_INSTANCES)
    budget = dict(n_restarts=10, n_pool=40, n_keep=5) if args.quick else \
        dict(n_restarts=50, n_pool=1000, n_keep=50)
    tasks = [dict(length=L, index=i, method=m, **budget)
             for L in sorted(lengths, reverse=True) for i in range(n) for m in METHOD_ORDER]
    df = run_tasks(task, tasks, args.workers).sort_values(["method", "length", "index"])
    save_csv(df, name)
    cols = ["acc_key", "acc_viterbi", "acc_posterior", "key_acc", "emission_tv", "runtime"]
    summary = summarize(df, ["method", "length"], cols)
    save_csv(summary, name + "_summary")
    plot(summary, name)
    print(summary.pivot(index="length", columns="method", values="acc_key_mean").round(3))


if __name__ == "__main__":
    main()
