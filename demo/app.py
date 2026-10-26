"""Streamlit demo: unsupervised decipherment of substitution ciphers with HMMs.

Run from the project root:
    streamlit run demo/app.py
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from views import decipher_live, encrypt, results, trellis, zodiac  # noqa: E402

st.set_page_config(page_title="Cipher HMM", page_icon="🔐", layout="wide")

pages = [
    st.Page(encrypt.render, title="1. Encrypt", icon="🔒", url_path="encrypt", default=True),
    st.Page(decipher_live.render, title="2. Decipher live", icon="📈", url_path="decipher"),
    st.Page(trellis.render, title="3. Inside the HMM", icon="🧭", url_path="trellis"),
    st.Page(zodiac.render, title="4. Zodiac 408", icon="♈", url_path="zodiac"),
    st.Page(results.render, title="5. Results", icon="📊", url_path="results"),
]
with st.sidebar:
    st.markdown("**Unsupervised decipherment with Hidden Markov Models**")
    st.caption("Hidden states: plaintext letters. Observations: cipher symbols. "
               "Transitions: a fixed English letter model. Emissions (the key): learned "
               "with Baum-Welch from the ciphertext alone.")
st.navigation(pages).run()
