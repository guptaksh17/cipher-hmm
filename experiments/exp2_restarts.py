"""Experiment 2: number of random restarts vs accuracy and runtime.

A: bigram HMM on 300-letter simple substitution ciphers, R random restarts.
B: staged HMM on 408-letter, 54-symbol homophonic ciphers (Zodiac-like); here R is
   the number of best bigram solutions (from a fixed pool of 1000) that seed the
   trigram stage.
Writes results/exp2_restarts.csv, its summary and two plots.
"""
import time

from common import (METHODS, DecipherConfig, MAX_BATCH_BYTES, finish_figure, load_csv, line_with_band,
                    make_instance, output_name, parse_args, plt, run_method, run_tasks,
                    save_csv, score, setup_style, staged_from_pool, staged_pool, summarize)

RESTARTS = [1, 5, 10, 25, 50, 100]
N_INSTANCES = 10
ALPHA_STAGED = 0.02  # chosen on synthetic ciphers (experiment 3)


def task(t: dict) -> list[dict]:
    if t["part"] == "A":
        inst = make_instance(300, 26, t["index"])
        r = run_method("bigram", inst.cipher, 26, inst.plain, inst.key.sym_to_letter,
                       seed=t["index"], n_restarts=t["restarts"])
        return [dict(part="A", method="bigram", restarts=t["restarts"], index=t["index"], **r)]
    inst = make_instance(408, 54, t["index"])
    pool, pool_time = staged_pool(inst.cipher, 54, t["n_pool"], t["index"])
    rows = []
    for R in t["restarts"]:
        t0 = time.perf_counter()
        cfg = DecipherConfig(refine=False, alpha=ALPHA_STAGED, seed=t["index"], max_batch_bytes=MAX_BATCH_BYTES)
        res = staged_from_pool(inst.cipher, 54, pool, pool_time, R, cfg)
        rows.append(dict(part="B", method="staged", restarts=R, index=t["index"],
                         stage2_runtime=time.perf_counter() - t0, pool_runtime=pool_time,
                         **score(res, inst.plain, inst.key.sym_to_letter, inst.cipher)))
    return rows


def plot(summary, name: str, n: int, n_pool: int) -> None:
    setup_style()
    titles = {"A": "A. Bigram HMM, simple substitution, 300 letters",
              "B": "B. Staged trigram HMM, homophonic, 408 letters × 54 symbols"}
    for col, ylabel, suffix in (("acc_key", "symbol accuracy", ""),
                                ("runtime", "seconds per decipherment", "_runtime")):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=(col == "acc_key"))
        for ax, part in zip(axes, "AB"):
            d = summary[summary.part == part]
            m = d.method.iloc[0]
            line_with_band(ax, d.restarts, d[f"{col}_mean"], d[f"{col}_std"], METHODS[m],
                           label_end=False)
            if col == "runtime" and part == "B":
                line_with_band(ax, d.restarts, d.stage2_runtime_mean, d.stage2_runtime_std,
                               ("trigram stage only", METHODS["trigram"][1], "^", "--"),
                               label_end=False)
                ax.legend(fontsize=8.5, loc="upper left")
            ax.set_xscale("log")
            ax.set_xticks(RESTARTS, [str(r) for r in RESTARTS])
            ax.set_xlabel("restarts" if part == "A" else f"trigram restarts (from {n_pool} bigram)")
            ax.set_title(titles[part], fontsize=10)
            if col == "acc_key":
                ax.set_ylim(0, 1.02)
        axes[0].set_ylabel(ylabel)
        finish_figure(fig, name + suffix, f"Mean ± 1 std over {n} ciphers. Best restart chosen "
                      "by log-likelihood. Restarts run as one vectorized batch, so runtime "
                      "grows sub-linearly.")


def main() -> None:
    args = parse_args(__doc__)
    name = output_name("exp2_restarts", args.quick)
    if args.plot_only:
        df = load_csv(name)
        b = df[df.part == "B"]
        return plot(load_csv(name + "_summary"), name, int(b["index"].nunique()),
                    int(b.n_pool.iloc[0]))
    restarts, n, n_pool = ([1, 5], 2, 40) if args.quick else (RESTARTS, N_INSTANCES, 1000)
    n, n_pool = args.instances or n, args.pool or n_pool
    restarts = [R for R in restarts if R <= n_pool]
    tasks = [dict(part="B", index=i, restarts=restarts, n_pool=n_pool) for i in range(n)]
    tasks += [dict(part="A", index=i, restarts=R) for R in restarts for i in range(n)]
    df = run_tasks(task, tasks, args.workers).sort_values(["part", "restarts", "index"])
    df["n_pool"] = df.part.map({"A": 0, "B": n_pool})
    save_csv(df, name)
    cols = ["acc_key", "key_acc", "runtime", "stage2_runtime"]
    summary = summarize(df.fillna({"stage2_runtime": 0.0}), ["part", "method", "restarts"], cols)
    save_csv(summary, name + "_summary")
    plot(summary, name, n, n_pool)
    print(summary[["part", "restarts", "acc_key_mean", "acc_key_std", "runtime_mean"]].round(3))


if __name__ == "__main__":
    main()
