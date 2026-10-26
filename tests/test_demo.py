"""Smoke tests for the Streamlit demo (no browser needed)."""
import sys
from pathlib import Path

import numpy as np
import pytest

DEMO = Path(__file__).resolve().parent.parent / "demo"
sys.path.insert(0, str(DEMO))
AppTest = pytest.importorskip("streamlit.testing.v1").AppTest


def test_app_starts_on_encrypt_page_without_errors():
    at = AppTest.from_file(str(DEMO / "app.py"), default_timeout=60).run()
    assert not at.exception
    assert at.title[0].value == "Encrypt"
    assert "cipher" in at.session_state


def test_render_helpers():
    import helpers
    html = helpers.colored_html("abc", "abd")
    assert html.count(helpers.GOOD) == 2 and html.count(helpers.BAD) == 1
    grid = helpers.grid_html("abcd", cols=2, ref="abce")
    assert grid.count("<tr>") == 2 and grid.count(helpers.BAD) == 1
    state = helpers.CipherState(np.array([1, 0, 1]), 2, "XY", np.array([0, 1, 0]),
                                np.array([1, 0]), "test")
    assert state.text() == "YXY"
    assert list(helpers.symbol_order(state)) == [1, 0]  # grouped by true letter


def test_throttle_limits_redraws():
    import helpers
    t = helpers.Throttle(min_interval=10.0)
    assert t.ready()
    t.draw(lambda: None)
    assert not t.ready() and t.ready(force=True)
