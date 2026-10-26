"""Download and clean Project Gutenberg texts.

Training books (language model) and test books (held-out plaintexts) are
disjoint, so test passages never appear in the language-model corpus.

Run as a script to build data/corpus_train.txt and data/test/*.txt:
    python -m src.corpus
"""
from __future__ import annotations

import re
import string
import time
from pathlib import Path

import numpy as np
import requests

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
TEST_DIR = DATA_DIR / "test"
TRAIN_CORPUS = DATA_DIR / "corpus_train.txt"
TRAIN_CORPUS_SPACES = DATA_DIR / "corpus_train_spaces.txt"

ALPHABET = string.ascii_lowercase

# Gutenberg ebook id -> title. Language-model training books.
TRAIN_BOOKS: dict[int, str] = {
    2600: "War and Peace",
    135: "Les Miserables",
    2701: "Moby Dick",
    1342: "Pride and Prejudice",
    1400: "Great Expectations",
    98: "A Tale of Two Cities",
    1661: "The Adventures of Sherlock Holmes",
    345: "Dracula",
    84: "Frankenstein",
    766: "David Copperfield",
    145: "Middlemarch",
    1023: "Bleak House",
}

# Held-out books used only for test plaintexts.
TEST_BOOKS: dict[int, str] = {
    11: "Alice's Adventures in Wonderland",
    120: "Treasure Island",
    43: "The Strange Case of Dr Jekyll and Mr Hyde",
    768: "Wuthering Heights",
    35: "The Time Machine",
    76: "Adventures of Huckleberry Finn",
}

_URL = "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt"
_START_RE = re.compile(r"\*\*\*\s*START OF (THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", re.I | re.S)
_END_RE = re.compile(r"\*\*\*\s*END OF (THE|THIS) PROJECT GUTENBERG EBOOK", re.I)


def download_book(book_id: int, raw_dir: Path = RAW_DIR) -> str:
    """Download one Gutenberg book (cached on disk) and return its raw text."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / f"pg{book_id}.txt"
    if not path.exists():
        resp = requests.get(_URL.format(id=book_id), timeout=60)
        resp.raise_for_status()
        resp.encoding = "utf-8"
        path.write_text(resp.text, encoding="utf-8")
        time.sleep(1.0)  # be polite to the Gutenberg servers
    return path.read_text(encoding="utf-8")


def strip_gutenberg(text: str) -> str:
    """Remove the Project Gutenberg header and licence footer."""
    start = _START_RE.search(text)
    if start:
        text = text[start.end():]
    end = _END_RE.search(text)
    if end:
        text = text[: end.start()]
    return text


def clean_text(text: str, keep_spaces: bool = False) -> str:
    """Lowercase and keep only a-z (optionally single spaces between words)."""
    text = text.lower()
    if keep_spaces:
        text = re.sub(r"[^a-z]+", " ", text)
        return text.strip()
    return re.sub(r"[^a-z]", "", text)


def load_clean_book(book_id: int, keep_spaces: bool = False) -> str:
    """Download (if needed), strip and clean a book."""
    return clean_text(strip_gutenberg(download_book(book_id)), keep_spaces)


def build_corpora() -> None:
    """Build the training corpus and the held-out test texts on disk."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    train = [load_clean_book(b) for b in TRAIN_BOOKS]
    TRAIN_CORPUS.write_text("".join(train))
    train_sp = [load_clean_book(b, keep_spaces=True) for b in TRAIN_BOOKS]
    TRAIN_CORPUS_SPACES.write_text(" ".join(train_sp))
    for b in TEST_BOOKS:
        (TEST_DIR / f"{b}.txt").write_text(load_clean_book(b))
        (TEST_DIR / f"{b}_spaces.txt").write_text(load_clean_book(b, keep_spaces=True))


def load_train_corpus(keep_spaces: bool = False) -> str:
    """Return the cleaned training corpus, building it if necessary."""
    path = TRAIN_CORPUS_SPACES if keep_spaces else TRAIN_CORPUS
    if not path.exists():
        build_corpora()
    return path.read_text()


def load_test_texts() -> dict[int, str]:
    """Return the cleaned held-out test books (letters only), keyed by Gutenberg id."""
    if not all((TEST_DIR / f"{b}.txt").exists() for b in TEST_BOOKS):
        build_corpora()
    return {b: (TEST_DIR / f"{b}.txt").read_text() for b in TEST_BOOKS}


def sample_passages(n: int, length: int, seed: int = 0) -> list[str]:
    """Draw n random contiguous passages of `length` letters from the test books."""
    rng = np.random.default_rng(seed)
    texts = list(load_test_texts().values())
    out = []
    for _ in range(n):
        t = texts[rng.integers(len(texts))]
        start = rng.integers(0, len(t) - length)
        out.append(t[start : start + length])
    return out


if __name__ == "__main__":
    build_corpora()
    print(f"train corpus: {TRAIN_CORPUS.stat().st_size / 1e6:.2f} MB letters")
    for b, title in TEST_BOOKS.items():
        print(f"test {b:5d} {title}: {(TEST_DIR / f'{b}.txt').stat().st_size} letters")
