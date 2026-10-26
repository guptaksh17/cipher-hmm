"""Experiment 3: ML vs MAP emission estimates (Dirichlet pseudo-count alpha).

A: bigram HMM on short simple substitution ciphers (100, 200, 300 letters),
   20 ciphers each, where the data are too few for a reliable ML estimate.
B: staged HMM on 408-letter, 54-symbol homophonic ciphers (Zodiac-like), 20 ciphers,
   the bigram pool shared across alpha values. This part selects the alpha that
   is later used for the Zodiac 408 (experiment 6), so that alpha is never tuned
   on the Zodiac itself.
Also records `emission_tv`, the distance between learned and true emissions,
to show the estimator's error directly.
Writes results/exp3_map.csv, its summary and plots.
"""
from common import (DecipherConfig, MAX_BATCH_BYTES, METHODS, finish_figure, load_csv, ordinal_colors,
                    line_with_band, make_instance, output_name, parse_args, plt, run_method,
                    run_tasks, save_csv, score, setup_style, staged_from_pool, staged_pool,
                    summarize)

ALPHAS_A = [0.0, 0.001, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
ALPHAS_B = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
LENGTHS_A = [100, 200, 300]
N_INSTANCES = 20


def task(t: dict) -> list[dict]:
    rows = []
    if t["part"] == "A":
        inst = make_instance(t["length"], 26, t["index"])
        for a in t["alphas"]:
            r = run_method("bigram", inst.cipher, 26, inst.plain, inst.key.sym_to_letter,
                           seed=t["index"], alpha=a)
            rows.append(dict(part="A", method="bigram", length=t["length"], alpha=a,
                             index=t["index"], **r))
        return rows
    inst = make_instance(408, 54, t["index"])
    pool, pool_time = staged_pool(inst.cipher, 54, t["n_pool"], t["index"])
    for a in t["alphas"]:
        cfg = DecipherConfig(alpha=a, seed=t["index"], max_batch_bytes=MAX_BATCH_BYTES)
        res = staged_from_pool(inst.cipher, 54, pool, pool_time, t["n_keep"], cfg)
        rows.append(dict(part="B", method="staged", length=408, alpha=a, index=t["index"],
                         **score(res, inst.plain, inst.key.sym_to_letter, inst.cipher)))
    return rows


def _alpha_axis(ax, alphas) -> None:
    ax.set_xscale("symlog", linthresh=0.001, linscale=0.4)
    ax.set_xticks(alphas, ["0 (ML)" if a == 0 else f"{a:g}" for a in alphas], rotation=45)
    ax.set_xlabel("Dirichlet pseudo-count α")
    ax.minorticks_off()


def plot(summary, name: str) -> None:
    setup_style()
    for col, ylabel, suffix, ylim in (("acc_key", "symbol accuracy", "", (0, 1.02)),
                                      ("emission_tv", "TV distance to true emissions",
                                       "_emission_error", None)):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
        a = summary[summary.part == "A"]
        lengths = sorted(a.length.unique())
        for L, color in zip(lengths, ordinal_colors(len(lengths))):
            d = a[a.length == L]
            line_with_band(axes[0], d.alpha, d[f"{col}_mean"], d[f"{col}_std"],
                           (f"{L} letters", color, "o", "-"))
        axes[0].set_title("A. Bigram HMM, short simple substitution ciphers", fontsize=10)
        b = summary[summary.part == "B"]
        line_with_band(axes[1], b.alpha, b[f"{col}_mean"], b[f"{col}_std"], METHODS["staged"],
                       label_end=False)
        best = b.loc[b.acc_key_mean.idxmax()]
        if col == "acc_key":
            axes[1].annotate(f"best α = {best.alpha:g}", (best.alpha, best.acc_key_mean),
                             xytext=(0, -16), textcoords="offset points", ha="center",
                             fontsize=8.5, color="#52514e")
        axes[1].set_title("B. Staged trigram HMM, homophonic 408 × 54", fontsize=10)
        for ax, alphas in ((axes[0], sorted(a.alpha.unique())), (axes[1], sorted(b.alpha.unique()))):
            _alpha_axis(ax, alphas)
            if ylim:
                ax.set_ylim(*ylim)
        axes[0].set_ylabel(ylabel)
        finish_figure(fig, name + suffix, "Mean ± 1 std over 20 ciphers per setting. α = 0 is "
                      "maximum likelihood; α > 0 is the MAP estimate under a Dirichlet(α + 1) "
                      "prior on each row of B.")


def main() -> None:
    args = parse_args(__doc__)
    name = output_name("exp3_map", args.quick)
    if args.plot_only:
        return plot(load_csv(name + "_summary"), name)
    if args.quick:
        alphas_a, alphas_b, lengths, n, n_pool, n_keep = [0.0, 0.1], [0.0, 0.1], [100], 2, 40, 5
    else:
        alphas_a, alphas_b, lengths, n, n_pool, n_keep = (ALPHAS_A, ALPHAS_B, LENGTHS_A,
                                                         N_INSTANCES, 1000, 50)
    tasks = [dict(part="B", index=i, alphas=alphas_b, n_pool=n_pool, n_keep=n_keep)
             for i in range(n)]
    tasks += [dict(part="A", length=L, index=i, alphas=alphas_a) for L in lengths
              for i in range(n)]
    df = run_tasks(task, tasks, args.workers).sort_values(["part", "length", "alpha", "index"])
    save_csv(df, name)
    summary = summarize(df, ["part", "method", "length", "alpha"],
                        ["acc_key", "acc_viterbi", "key_acc", "emission_tv"])
    save_csv(summary, name + "_summary")
    plot(summary, name)
    print(summary[["part", "length", "alpha", "acc_key_mean", "acc_key_std",
                   "emission_tv_mean"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
