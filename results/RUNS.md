# How the results in this folder were produced

Run on 2026-10-26, an 8-core Mac (4 performance + 4 efficiency cores), 7 worker
processes. All seeds are fixed; rerunning a script with the same options gives the
same numbers (runtimes aside).

| Experiment | Command | Ciphers per setting | Staged budget (bigram pool → trigram restarts) | Wall time |
|---|---|---|---|---|
| 1 Length | `python exp1_length.py` | 10 | 1000 → 50 | 67 min |
| 2 Restarts | `python exp2_restarts.py --instances 5 --pool 300` | 5 | 300 → 1…100 | 3 min |
| 3 ML vs MAP | `python exp3_map.py` | 20 | 1000 → 50 | 35 min |
| 4 LM weight | `python exp4_lm_weight.py --instances 5 --pool 300 --keep 30` | 5 | 300 → 30 | 2 min |
| 5 Homophonic | `python exp5_homophonic.py` | 10 | 1000 → 50 | 107 min |
| 6 Zodiac 408 | `python exp6_zodiac.py` | 5 seeds | 1000 → 50 | 6 min |
| 7 Decoding | `python exp7_decoding.py` | reuses 1 and 5 | n/a | <1 min |
| 8 EM progress | `python exp8_em_progress.py` | 1 | 1000 → 50 | 1 min |

Experiments 2 and 4 used a reduced budget (5 ciphers, smaller staged pool) to save
time; `run_all.sh` runs every experiment at the full default budget.

Runtimes in the CSVs were measured with 7 runs sharing the CPU, so a single run
on an idle machine is faster (for example about 50 s instead of about 220 s for a
staged Zodiac run).

Alpha for the Zodiac (experiments 6 and 8) is read from experiment 3, part B:
the best alpha on synthetic 408-letter, 54-symbol ciphers (0.02). It is never
tuned on the Zodiac itself.
