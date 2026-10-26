"""Load the Zodiac 408 ciphertext (24 x 17 ASCII transcription) and its known solution."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.corpus import DATA_DIR
from src.lm import text_to_indices

ZODIAC_DIR = DATA_DIR / "zodiac"
N_COLS, N_ROWS = 17, 24


@dataclass
class Zodiac408:
    """Ciphertext as symbol ids plus the glyph of each id and the known plaintext."""

    text: str             # 408 transcription characters, row by row
    glyphs: str           # glyphs[s] = transcription character of symbol id s
    cipher: np.ndarray    # (408,) symbol ids
    plain: np.ndarray     # (408,) letter indices of the known solution

    @property
    def n_symbols(self) -> int:
        return len(self.glyphs)


def load_z408() -> Zodiac408:
    """Read data/zodiac/*.txt; symbol ids are assigned in order of first appearance."""
    text = "".join((ZODIAC_DIR / "z408_cipher.txt").read_text().split())
    solution = "".join((ZODIAC_DIR / "z408_solution.txt").read_text().split())
    assert len(text) == len(solution) == N_COLS * N_ROWS
    glyphs = "".join(dict.fromkeys(text))
    ids = {g: i for i, g in enumerate(glyphs)}
    cipher = np.array([ids[g] for g in text], dtype=np.int64)
    return Zodiac408(text, glyphs, cipher, text_to_indices(solution))


def true_key(z: Zodiac408) -> np.ndarray:
    """Symbol -> letter key implied by the known solution (consistent by construction)."""
    key = np.zeros(z.n_symbols, dtype=np.int64)
    key[z.cipher] = z.plain
    return key
