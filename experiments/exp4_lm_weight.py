"""Experiment 4: language-model weight (transition probabilities raised to a power).

A: bigram HMM on simple substitution ciphers of 200 and 500 letters.
B: staged HMM on 408-letter, 54-symbol homophonic ciphers; the power is applied to
   the trigram stage (the bigram pool stays at power 1), alpha = 0.02.
10 ciphers per setting. Writes results/exp4_lm_weight.csv, its summary and a plot.
"""
from common import (DecipherConfig, MAX_BATCH_BYTES, METHODS, finish_figure, load_csv, ordinal_colors,
                    line_with_band, make_instance, output_name, parse_args, plt, run_method,
                    run_tasks, save_csv, score, setup_style, staged_from_pool, staged_pool,
                    summarize)

POWERS = [1.0, 1.25, 1.5, 2.0, 2.5, 3.0]
LENGTHS_A = [200, 500]
N_INSTANCES = 10
ALPHA_STAGED = 0.02


def task(t: dict) -> list[dict]:
    rows = []
    if t["part"] == "A":
        inst = make_instance(t["length"], 26, t["index"])
        for p in t["powers"]:
            r = run_method("bigram", inst.cipher, 26, inst.plain, inst.key.sym_to_letter,
                           seed=t["index"], lm_power=p)
            rows.append(dict(part="A", method="bigram", length=t["length"], power=p,
                             index=t["index"], **r))
        return rows
    inst = make_instance(408, 54, t["index"])
    pool, pool_time = staged_pool(inst.cipher, 54, t["n_pool"], t["index"])
    for p in t["powers"]:
        cfg = DecipherConfig(refine=False, alpha=ALPHA_STAGED, lm_power=p, seed=t["index"],
                             max_batch_bytes=MAX_BATCH_BYTES)
        res = staged_from_pool(inst.cipher, 54, pool, pool_time, t["n_keep"], cfg)
        rows.append(dict(part="B", method="staged", length=408, power=p, index=t["index"],
                         **score(res, inst.plain, inst.key.sym_to_letter, inst.cipher)))
    return rows


def plot(summary, name: str, n: int, n_pool: int, n_keep: int) -> None:
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    a = summary[summary.part == "A"]
    lengths = sorted(a.length.unique())
    for L, color in zip(lengths, ordinal_colors(len(lengths))):
        d = a[a.length == L]
        line_with_band(axes[0], d.power, d.acc_key_mean, d.acc_key_std,
                       (f"{L} letters", color, "o", "-"))
    axes[0].set_title("A. Bigram HMM, simple substitution", fontsize=10)
    b = summary[summary.part == "B"]
    line_with_band(axes[1], b.power, b.acc_key_mean, b.acc_key_std, METHODS["staged"],
                   label_end=False)
    axes[1].set_title("B. Staged trigram HMM, homophonic 408 × 54", fontsize=10)
    for ax in axes:
        ax.set_xlabel("language-model exponent")
        ax.set_xticks(POWERS, [f"{p:g}" for p in POWERS])
        ax.set_ylim(0, 1.02)
    axes[0].set_ylabel("symbol accuracy")
    finish_figure(fig, name, f"Mean ± 1 std over {n} ciphers. Transition probabilities raised "
                  "to the exponent and renormalized before EM; exponent 1 is the plain model. "
                  f"Staged: {n_pool} bigram restarts → {n_keep} trigram.")


def main() -> None:
    args = parse_args(__doc__)
    name = output_name("exp4_lm_weight", args.quick)
    if args.plot_only:
        df = load_csv(name)
        b = df[df.part == "B"]
        return plot(load_csv(name + "_summary"), name, int(b["index"].nunique()),
                    int(b.n_pool.iloc[0]), int(b.n_keep.iloc[0]))
    powers, lengths, n, n_pool, n_keep = (([1.0, 2.0], [200], 2, 40, 5) if args.quick
                                          else (POWERS, LENGTHS_A, N_INSTANCES, 1000, 50))
    n, n_pool, n_keep = args.instances or n, args.pool or n_pool, args.keep or n_keep
    tasks = [dict(part="B", index=i, powers=powers, n_pool=n_pool, n_keep=n_keep)
             for i in range(n)]
    tasks += [dict(part="A", length=L, index=i, powers=powers) for L in lengths for i in range(n)]
    df = run_tasks(task, tasks, args.workers).sort_values(["part", "length", "power", "index"])
    df["n_pool"], df["n_keep"] = df.part.map({"A": 0, "B": n_pool}), df.part.map({"A": 0, "B": n_keep})
    save_csv(df, name)
    summary = summarize(df, ["part", "method", "length", "power"],
                        ["acc_key", "acc_viterbi", "key_acc", "loglik"])
    save_csv(summary, name + "_summary")
    plot(summary, name, n, n_pool, n_keep)
    print(summary[["part", "length", "power", "acc_key_mean", "acc_key_std"]].round(3)
          .to_string(index=False))


if __name__ == "__main__":
    main()
