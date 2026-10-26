"""Page 5: the main plots and tables from results/."""
import pandas as pd
import streamlit as st

from helpers import RESULTS

FIGURES = [
    ("exp1_length.png", "1. Accuracy vs ciphertext length",
     "Simple substitution, mean ± 1 std over 10 ciphers per length. Exact numbers are in "
     "the data table below."),
    ("exp2_restarts.png", "2. Number of random restarts",
     "EM has many local optima, so the best of several random restarts is kept. In the "
     "staged method the bigram pool does most of the search."),
    ("exp3_map.png", "3. ML vs MAP emission estimates",
     "A small Dirichlet pseudo-count helps on short ciphertexts; large values over-smooth."),
    ("exp3_map_emission_error.png", "3b. Error of the emission estimate",
     "The prior pulls B towards uniform (more bias), yet decoding improves: less variance."),
    ("exp4_lm_weight.png", "4. Language-model weight",
     "Transition probabilities raised to a power and renormalized before EM."),
    ("exp5_homophonic.png", "5. Simple vs homophonic substitution",
     "More homophones flatten symbol statistics; longer ciphertexts compensate."),
    ("exp6_zodiac.png", "6. Zodiac 408 by method", None),
    ("exp6_zodiac_grid.png", "6b. Zodiac 408 decoded vs the known solution", None),
    ("exp7_decoding.png", "7. Viterbi vs posterior vs consistent-key decoding",
     "A consistent symbol → letter key beats both position-wise decodings."),
    ("exp8_em_progress_simple.png", "8. EM progress, simple substitution", None),
    ("exp8_em_progress_zodiac.png", "8b. EM progress, Zodiac 408 (trigram stage)", None),
    ("exp1_length_runtime.png", "Runtime vs ciphertext length", None),
]


def render() -> None:
    st.title("Results")
    available = [f for f in FIGURES if (RESULTS / f[0]).exists()]
    if not available:
        st.warning("No results yet: run `experiments/run_all.sh`.")
        return
    titles = [t for _, t, _ in available]
    choice = st.radio("Figure", titles, label_visibility="collapsed")
    name, title, note = available[titles.index(choice)]
    st.subheader(title)
    if note:
        st.write(note)
    st.image(str(RESULTS / name), width="stretch")
    csv = RESULTS / (name.split(".")[0].replace("_grid", "").replace("_emission_error", "")
                     .replace("_runtime", "") + "_summary.csv")
    if csv.exists():
        with st.expander("Data (summary table)"):
            st.dataframe(pd.read_csv(csv), hide_index=True, width="stretch")
    runs = RESULTS / "RUNS.md"
    if runs.exists():
        with st.expander("How these results were produced"):
            st.markdown(runs.read_text())
