"""Experiment 8: EM progress.

For the best restart (highest final log-likelihood) the run is replayed from its
initial emissions, recording the log-likelihood and the Viterbi decoding after
every EM iteration. Two cases:
  simple - bigram HMM, 500-letter simple substitution cipher, 50 random restarts;
  zodiac - trigram stage of the staged HMM on the Zodiac 408.
The log-likelihood curves of all restarts are drawn too, showing the local optima.
Writes results/exp8_em_progress_<case>.csv (one row per iteration), a plot and a
text file of decoded snapshots at selected iterations for each case.
"""
import numpy as np
import pandas as pd

from common import (MAX_BATCH_BYTES, MUTED, RESULTS, SLOTS, TEXT, DecipherConfig,
                    finish_figure, make_instance, output_name, parse_args, plt, save_csv,
                    selected_alpha, setup_style)
from src.decipher import bigram_transitions, run_em
from src.hmm import baum_welch, random_emissions, viterbi
from src.hmm_trigram import trigram_transitions
from src.lm import get_bigram, get_trigram, indices_to_text
from src.metrics import symbol_accuracy
from src.zodiac import load_z408

SNAPSHOTS = [0, 1, 2, 3, 5, 10, 20, 50, 100, 200]


def replay(trans, cipher, B0_all, alpha, plain, max_iter=500, tol=1e-5):
    """Run all restarts, then replay the best one recording each iteration."""
    res = baum_welch(trans, cipher, B0_all, alpha=alpha, max_iter=max_iter, tol=tol)
    best = int(np.argmax(res.loglik))
    Bs = [B0_all[best].copy()]
    one = baum_welch(trans, cipher, B0_all[best:best + 1], alpha=alpha, max_iter=max_iter,
                     tol=tol, callback=lambda it, B, ll: Bs.append(B[0].copy()))
    lls = one.loglik_history[0]
    rows = []
    for it, ll in enumerate(lls):
        path = viterbi(trans, Bs[it], cipher)[0]
        rows.append(dict(iteration=it, loglik=ll, acc_viterbi=symbol_accuracy(path, plain),
                         decoded=indices_to_text(path)))
    return pd.DataFrame(rows), res.loglik_history, best


def plot(df, all_curves, best, title, name, n_chars=68):
    setup_style()
    fig = plt.figure(figsize=(12, 8.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.9], hspace=0.35)
    ax_ll, ax_acc = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    for r, h in enumerate(all_curves):
        if r != best:
            ax_ll.plot(range(len(h)), h, color=MUTED, alpha=0.35, linewidth=0.8)
    ax_ll.plot(df.iteration, df.loglik, color=SLOTS[0], label="best restart")
    ax_ll.plot([], [], color=MUTED, linewidth=0.8, label="other restarts")
    lo = np.percentile([h[-1] for h in all_curves], 5)
    span = abs(df.loglik.max() - lo)
    ax_ll.set_ylim(lo - 0.15 * span, df.loglik.max() + 0.05 * span + 1)
    ax_ll.set_xlabel("EM iteration"); ax_ll.set_ylabel("log-likelihood")
    ax_ll.set_title("Log-likelihood per iteration", fontsize=10)
    ax_ll.legend(fontsize=8.5, loc="lower right")
    ax_acc.plot(df.iteration, df.acc_viterbi, color=SLOTS[0])
    ax_acc.set_ylim(0, 1.02)
    ax_acc.set_xlabel("EM iteration"); ax_acc.set_ylabel("Viterbi symbol accuracy")
    ax_acc.set_title("Accuracy of the best restart (uses the true plaintext)", fontsize=10)
    ax_txt = fig.add_subplot(gs[1, :]); ax_txt.axis("off")
    snaps = [i for i in SNAPSHOTS if i < len(df)] + [len(df) - 1]
    y = 1.0
    ax_txt.text(0, y, "Viterbi decoding of the first letters at selected iterations",
                fontsize=10, fontweight="bold", color=TEXT, transform=ax_txt.transAxes)
    for i in dict.fromkeys(snaps):
        y -= 0.085
        row = df.iloc[i]
        ax_txt.text(0, y, f"iter {i:>3}  {row.acc_viterbi:5.1%}  {row.decoded[:n_chars]}",
                    family="monospace", fontsize=8.8, color=TEXT, transform=ax_txt.transAxes)
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="bold")
    finish_figure(fig, name)


def write_snapshots(df, path, header):
    snaps = sorted(set(i for i in SNAPSHOTS if i < len(df)) | {len(df) - 1})
    lines = [header] + [f"iteration {i:4d}  loglik {df.loglik[i]:10.2f}  accuracy "
                        f"{df.acc_viterbi[i]:.3f}\n  {df.decoded[i]}" for i in snaps]
    path.write_text("\n".join(lines) + "\n")
    print(f"wrote results/{path.name}")


def main() -> None:
    args = parse_args(__doc__)
    n_restarts, n_pool, n_keep = (5, 30, 5) if args.quick else (50, 1000, 50)

    inst = make_instance(500, 26, 0)
    B0 = random_emissions(np.random.default_rng(0), n_restarts, 26, 26)
    df, curves, best = replay(bigram_transitions(get_bigram()), inst.cipher, B0, 0.0, inst.plain)
    name = output_name("exp8_em_progress_simple", args.quick)
    save_csv(df, name)
    plot(df, curves, best, "EM progress: bigram HMM, 500-letter simple substitution", name)
    write_snapshots(df, RESULTS / f"{name}_snapshots.txt",
                    f"Bigram HMM, simple substitution, 500 letters, best of {n_restarts} restarts")

    z = load_z408()
    alpha = selected_alpha()
    cfg = DecipherConfig(n_restarts=n_pool, seed=0, max_batch_bytes=MAX_BATCH_BYTES)
    pool = run_em(bigram_transitions(get_bigram()), z.cipher, z.n_symbols, cfg)
    B0 = pool.B[np.argsort(-pool.loglik)[:n_keep]] + 0.01
    B0 /= B0.sum(axis=-1, keepdims=True)
    df, curves, best = replay(trigram_transitions(get_trigram()), z.cipher, B0, alpha, z.plain)
    name = output_name("exp8_em_progress_zodiac", args.quick)
    save_csv(df, name)
    plot(df, curves, best, f"EM progress: Zodiac 408, trigram stage of the staged HMM "
         f"(α = {alpha:g}, initialized from bigram solutions)", name)
    write_snapshots(df, RESULTS / f"{name}_snapshots.txt",
                    f"Zodiac 408, trigram stage (alpha={alpha:g}), best of {n_keep} restarts "
                    f"seeded from {n_pool} bigram restarts")


if __name__ == "__main__":
    main()
