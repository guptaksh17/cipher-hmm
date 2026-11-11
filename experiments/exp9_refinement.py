"""Experiment 9: key refinement after EM (hill-climbing with a 4-gram model).

The settings of the interactive demo, on 8 ciphers per scenario and the Zodiac 408
with 5 seeds. Each run records the key straight from EM and the refined key
(src/refine.py: refine the EM key and the keys of the top restarts, keep the best
by joint log-likelihood, then iterated local search), so both are measured on the
same ciphertexts. Writes results/exp9_refinement.csv, its summary and a plot.
"""
import numpy as np

from common import (SLOTS, DecipherConfig, finish_figure, load_csv, make_instance, output_name,
                    parse_args, plt, run_tasks, save_csv, setup_style, summarize)
from src.decipher import decipher, decipher_staged
from src.lm import get_bigram, get_trigram
from src.metrics import key_accuracy, symbol_accuracy
from src.zodiac import load_z408, true_key

SCENARIOS = {  # name: (length, symbols, method, label)
    "simple300_bigram": (300, 26, "bigram", "Simple, 300 letters\n(bigram)"),
    "simple150_staged": (150, 26, "staged", "Simple, 150 letters\n(staged)"),
    "simple300_staged": (300, 26, "staged", "Simple, 300 letters\n(staged)"),
    "homo54_300_staged": (300, 54, "staged", "Homophonic 54,\n300 letters"),
    "homo54_408_staged": (408, 54, "staged", "Homophonic 54,\n408 letters"),
    "zodiac": (408, 54, "staged", "Zodiac 408\n(5 seeds)"),
}
N_INSTANCES = 8
N_ZODIAC_SEEDS = 5


def task(t: dict) -> list[dict]:
    name, i = t["scenario"], t["index"]
    if name == "zodiac":
        z = load_z408()
        cipher, plain, S, tk = z.cipher, z.plain, z.n_symbols, true_key(z)
    else:
        L, S, _, _ = SCENARIOS[name]
        inst = make_instance(L, S, 9000 + i)
        cipher, plain, tk = inst.cipher, inst.plain, inst.key.sym_to_letter
    method = SCENARIOS[name][2]
    cfg = DecipherConfig(n_restarts=20 if method == "bigram" else t["n_keep"], alpha=0.02 if S > 26 else 0.0,
                         seed=i, refine=True, max_batch_bytes=3e8)
    if method == "bigram":
        r = decipher(cipher, S, get_bigram(), cfg)
    else:
        r = decipher_staged(cipher, S, get_bigram(), get_trigram(), cfg, n_pool=t["n_pool"], n_keep=t["n_keep"])
    return [dict(scenario=name, index=i, em_acc=symbol_accuracy(r.em_key[cipher], plain),
                 refined_acc=symbol_accuracy(r.decoded, plain), viterbi_acc=symbol_accuracy(r.viterbi, plain),
                 em_key_acc=key_accuracy(r.em_key, tk, cipher), refined_key_acc=key_accuracy(r.key, tk, cipher),
                 runtime=r.runtime)]


def plot(summary, name: str) -> None:
    setup_style()
    order = [s for s in SCENARIOS if s in set(summary.scenario)]
    d = summary.set_index("scenario").loc[order]
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(10.5, 4.4))
    for off, col, label, color in ((-0.19, "em_acc", "Key from EM", SLOTS[0]),
                                   (0.19, "refined_acc", "After refinement", SLOTS[1])):
        bars = ax.bar(x + off, d[f"{col}_mean"], width=0.36, color=color, label=label, zorder=2)
        ax.errorbar(x + off, d[f"{col}_mean"], yerr=d[f"{col}_std"], fmt="none", ecolor="#52514e",
                    elinewidth=1, capsize=3, zorder=3)
        for b, v in zip(bars, d[f"{col}_mean"]):
            ax.annotate(f"{v:.0%}", (b.get_x() + b.get_width() / 2, v), xytext=(0, 3),
                        textcoords="offset points", ha="center", fontsize=8.5, color="#0b0b0b")
    ax.set_xticks(x, [SCENARIOS[s][3] for s in order], fontsize=9)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("symbol accuracy")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", ncol=2, fontsize=9)
    ax.set_title("Key refinement after EM: same ciphers, demo settings")
    finish_figure(fig, name, "Mean ± 1 std over 8 ciphers per scenario (Zodiac: 5 seeds). Refinement: "
                  "hill-climbing on 4-gram log-likelihood + emission term, from the top restarts' keys, "
                  "then iterated local search; the best key is chosen by score alone.")


def main() -> None:
    args = parse_args(__doc__)
    name = output_name("exp9_refinement", args.quick)
    if args.plot_only:
        return plot(load_csv(name + "_summary"), name)
    n, nz, n_pool, n_keep = (2, 1, 40, 5) if args.quick else (N_INSTANCES, N_ZODIAC_SEEDS, 200, 10)
    tasks = [dict(scenario=s, index=i, n_pool=n_pool, n_keep=n_keep)
             for s in SCENARIOS for i in range(nz if s == "zodiac" else n)]
    df = run_tasks(task, tasks, args.workers).sort_values(["scenario", "index"])
    save_csv(df, name)
    summary = summarize(df, ["scenario"], ["em_acc", "refined_acc", "viterbi_acc", "em_key_acc",
                                           "refined_key_acc", "runtime"])
    save_csv(summary, name + "_summary")
    plot(summary, name)
    print(summary[["scenario", "em_acc_mean", "refined_acc_mean", "refined_acc_std", "runtime_mean"]]
          .round(3).to_string(index=False))


if __name__ == "__main__":
    main()
