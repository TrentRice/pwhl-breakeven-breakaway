"""Checks on Step 10 outputs (skipped if not built)."""

from pathlib import Path

import pandas as pd
import pytest

TABLES = Path(__file__).resolve().parents[1] / "outputs" / "tables"


@pytest.fixture(scope="module")
def pred():
    path = TABLES / "backtest_nhl_predictions.csv"
    if not path.exists():
        pytest.skip("run `python -m src.backtest` first")
    return pd.read_csv(path)


def test_every_nhl_ot_power_play_predicted_once(pred):
    assert len(pred) == 580   # 581 OT PP minors; one pair of same-second minors is a single power play
    assert not pred.duplicated(["season", "game_id", "ot_remaining_s"]).any()
    assert pred["p_model"].between(0, 1).all()


def test_baseline_never_uses_its_own_season(pred):
    # the constant baseline is fit on the other seasons, so it differs from the test season's own rate
    own = pred.groupby("season")["pp_goal"].mean()
    base = pred.groupby("season")["p_baseline"].first()
    assert (own - base).abs().max() > 0


def test_case_studies_cover_all_nine_ot_penalties():
    path = TABLES / "case_studies.csv"
    if not path.exists():
        pytest.skip("run `python -m src.backtest` first")
    cs = pd.read_csv(path)
    assert len(cs) == 9
    assert cs["P_shot_better"].between(0, 1).all()
