"""Experiment 6: the Zodiac 408 cipher.

Methods: frequency analysis, bigram HMM, trigram HMM from random restarts, and the
staged HMM (with alpha = 0 and with the alpha chosen on synthetic ciphers in
experiment 3). Each HMM method runs with 5 seeds; per method the run with the
highest log-likelihood is selected, which uses no knowledge of the solution.
Writes results/exp6_zodiac.csv, a method comparison plot, the decoded grid next to
the known solution with errors highlighted, and exp6_zodiac_decoded.txt.
"""
import numpy as np

from common import (CRITICAL, MAX_BATCH_BYTES, METHODS, RESULTS, TEXT, DecipherConfig,
                    finish_figure, load_csv, output_name, parse_args, plt, run_freq, run_tasks,
                    save_csv, score, selected_alpha, setup_style)
from src.baseline import frequency_analysis
from src.decipher import decipher, decipher_staged
from src.lm import get_bigram, get_trigram, indices_to_text
from src.zodiac import N_COLS, N_ROWS, load_z408, true_key

SEEDS = [0, 1, 2, 3, 4]


def task(t: dict) -> list[dict]:
    z = load_z408()
    tk = true_key(z)
    m = t["method"]
    if m == "freq":
        r = run_freq(z.cipher, z.n_symbols, z.plain, tk)
        decoded = frequency_analysis(z.cipher, z.n_symbols, get_bigram().pi)[z.cipher]
        return [dict(method=m, alpha=0.0, seed=0, decoded=indices_to_text(decoded), **r)]
    cfg = DecipherConfig(refine=False, n_restarts=t["n_restarts"], alpha=t["alpha"], seed=t["seed"],
                         max_batch_bytes=MAX_BATCH_BYTES)
    if m == "staged":
        res = decipher_staged(z.cipher, z.n_symbols, get_bigram(), get_trigram(), cfg,
                              n_pool=t["n_pool"], n_keep=t["n_keep"])
    else:
        res = decipher(z.cipher, z.n_symbols, get_bigram() if m == "bigram" else get_trigram(), cfg)
    return [dict(method=m, alpha=t["alpha"], seed=t["seed"],
                 decoded=indices_to_text(res.decoded), **score(res, z.plain, tk, z.cipher))]


def label(row) -> str:
    base = METHODS[row.method][0]
    return f"{base}, α = {row.alpha:g}" if row.method == "staged" else base


def plot_methods(df, name: str) -> None:
    setup_style()
    groups = list(df.groupby(["method", "alpha"], sort=False))
    fig, ax = plt.subplots(figsize=(8, 0.6 * len(groups) + 1.4))
    for y, ((m, a), d) in enumerate(groups):
        color = METHODS[m][1]
        ax.scatter(d.acc_key, [y] * len(d), s=26, color=color, alpha=0.45, linewidths=0)
        best = d[d.selected].iloc[0]
        ax.scatter([best.acc_key], [y], s=90, color=color, edgecolors=TEXT, linewidths=1.0,
                   zorder=3)
        ax.annotate(f"{best.acc_key:.1%}", (best.acc_key, y), xytext=(0, 10),
                    textcoords="offset points", ha="center", va="bottom", fontsize=9, color=TEXT)
    ax.set_yticks(range(len(groups)), [label(d.iloc[0]) for _, d in groups])
    ax.margins(y=0.14)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("symbol accuracy on the Zodiac 408")
    ax.set_title("Zodiac 408: accuracy by method")
    ax.grid(axis="y", visible=False)
    finish_figure(fig, name, "Large dot: run with the highest log-likelihood (selected without "
                  "the solution). Small dots: the other seeds.")


def plot_grid(z, decoded: str, title: str, name: str) -> None:
    setup_style()
    solution = indices_to_text(z.plain)
    fig, axes = plt.subplots(1, 3, figsize=(15, 8.2))
    panels = [("Ciphertext (transcription)", z.text, None),
              ("Decoded", decoded, solution),
              ("Known solution (as published)", solution, None)]
    for ax, (ptitle, text, ref) in zip(axes, panels):
        ax.set_xlim(-0.5, N_COLS - 0.5)
        ax.set_ylim(N_ROWS - 0.5, -0.5)
        ax.axis("off")
        ax.set_title(ptitle, fontsize=10, loc="center")
        for i, ch in enumerate(text):
            r, c = divmod(i, N_COLS)
            wrong = ref is not None and ch != ref[i]
            ax.text(c, r, ch.upper(), ha="center", va="center", family="monospace",
                    fontsize=10.5, color=CRITICAL if wrong else TEXT,
                    fontweight="bold" if wrong else "normal")
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="bold", color=TEXT)
    finish_figure(fig, name, "Red bold letters in the middle panel differ from the known "
                  "solution. Misspellings in the solution are the author's own.")


def main() -> None:
    args = parse_args(__doc__)
    if args.plot_only:  # the decoded grid needs the decoded text, so only this chart
        return plot_methods(load_csv(output_name("exp6_zodiac", args.quick)),
                            output_name("exp6_zodiac", args.quick))
    alpha = selected_alpha()
    seeds = [0] if args.quick else SEEDS
    budget = dict(n_restarts=5, n_pool=30, n_keep=5) if args.quick else \
        dict(n_restarts=50, n_pool=1000, n_keep=50)
    tasks = [dict(method="freq")]
    for s in seeds:
        tasks += [dict(method="staged", alpha=alpha, seed=s, **budget),
                  dict(method="staged", alpha=0.0, seed=s, **budget),
                  dict(method="trigram", alpha=0.0, seed=s, **budget),
                  dict(method="bigram", alpha=0.0, seed=s, **budget)]
    df = run_tasks(task, tasks, args.workers)
    rank = {"freq": 0, "bigram": 1, "trigram": 2, "staged": 3}
    df = (df.assign(rank=df.method.map(rank)).sort_values(["rank", "alpha", "seed"])
          .drop(columns="rank").reset_index(drop=True))
    best_idx = df.fillna({"loglik": 0}).groupby(["method", "alpha"]).loglik.idxmax()
    df["selected"] = df.index.isin(best_idx)
    name = output_name("exp6_zodiac", args.quick)
    save_csv(df.drop(columns="decoded"), name)
    plot_methods(df, name)

    z = load_z408()
    final = df[(df.method == "staged") & (df.alpha == alpha) & df.selected].iloc[0]
    plot_grid(z, final.decoded, f"Zodiac 408 decoded by the staged trigram HMM (α = {alpha:g}): "
              f"{final.acc_key:.1%} of symbols correct", name + "_grid")
    solution = indices_to_text(z.plain)
    lines = [f"Zodiac 408, staged trigram HMM, alpha={alpha:g}, seed={final.seed}, "
             f"symbol accuracy {final.acc_key:.4f}, key accuracy {final.key_acc:.4f}",
             "row  decoded            solution           errors"]
    for r in range(N_ROWS):
        d, s = final.decoded[r * N_COLS:(r + 1) * N_COLS], solution[r * N_COLS:(r + 1) * N_COLS]
        marks = "".join("^" if a != b else " " for a, b in zip(d, s))
        lines.append(f"{r + 1:3d}  {d}  {s}  {marks}")
    path = RESULTS / f"{name}_decoded.txt"
    path.write_text("\n".join(lines) + "\n")
    print(f"wrote results/{path.name}")
    sel = df[df.selected][["method", "alpha", "seed", "acc_key", "acc_viterbi", "key_acc",
                           "loglik", "runtime"]]
    print(sel.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
