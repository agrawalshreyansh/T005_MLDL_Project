"""Every dashboard page must render without an exception (needs the real data and at least one prediction file)."""
import pytest

from deep_learning.dl import results
from src import config

PAGES = ["Network map", "Replay", "Sensor explorer", "Leaderboard", "Graph lens", "Training", "About"]
ready = config.TRAFFIC_CLEAN_NPZ.is_file() and bool(results.available())


@pytest.mark.skipif(not ready, reason="processed data or predictions not present")
@pytest.mark.parametrize("page", PAGES)
def test_page_renders(page):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(config.PROJECT_ROOT / "deep_learning" / "dashboard" / "app.py"), default_timeout=180)
    at.run()
    at.sidebar.radio[0].set_value(page).run()
    assert not at.exception, at.exception[0].value
