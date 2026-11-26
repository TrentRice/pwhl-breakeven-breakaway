"""Sanity checks on the modeling tables in data/processed (skipped if not built)."""

from pathlib import Path

import pandas as pd
import pytest

PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"
INTERIM = Path(__file__).resolve().parents[1] / "data" / "interim"
NAMES = ("shootout_attempts", "pp_intervals", "ot_intervals", "ot_penalties")


@pytest.fixture(scope="module")
def t():
    if not all((PROCESSED / f"{n}.parquet").exists() for n in NAMES):
        pytest.skip("run `python -m src.build_tables` first")
    return {n: pd.read_parquet(PROCESSED / f"{n}.parquet") for n in NAMES}


def test_shootout_attempts_complete(t):
    so = t["shootout_attempts"]
    assert so["shooter_id"].notna().all()
    assert so["goalie_id"].notna().mean() > 0.999
    assert set(so["goal"].unique()) <= {0, 1}
    pwhl_reg = so[(so["league"] == "PWHL") & (so["season_type"] == "regular") & (so["is_penalty_shot"] == 0)]
    assert len(pwhl_reg) == 287


def test_pp_segments_valid(t):
    pp = t["pp_intervals"]
    assert (pp["duration_s"] > 0).all()
    assert (pp["spell_offset_s"] >= 0).all()
    assert pp["strength"].isin(["5v4", "5v3", "4v3", "6v5", "6v4", "4v4"]).sum() / len(pp) > 0.99


def test_pwhl_pp_goals_match_feed(t):
    pp = t["pp_intervals"]
    ev = pd.read_parquet(INTERIM / "pwhl_events.parquet", columns=["event_type", "is_pp"])
    feed_ppg = int(ev.loc[ev["event_type"] == "goal", "is_pp"].sum())
    ours = int(pp.loc[pp["league"] == "PWHL", "goals_for"].sum())
    assert abs(ours - feed_ppg) <= 5


def test_ot_intervals_balance(t):
    ot = t["ot_intervals"]
    assert (ot.groupby(["league", "game_id"]).size() == 2).all()
    home, away = ot[ot["is_home"] == 1].set_index(["league", "game_id"]), ot[ot["is_home"] == 0].set_index(["league", "game_id"])
    assert (home["gf_3v3"] == away.loc[home.index, "ga_3v3"]).all()
    assert (ot["toi_3v3_s"] <= 300).all()


def test_pwhl_regular_season_ot_penalties(t):
    otp = t["ot_penalties"]
    pwhl = otp[(otp["league"] == "PWHL") & (otp["season_type"] == "regular")]
    assert len(pwhl) == 9                       # matches Rachman (2026)
    assert pwhl["ps_awarded"].sum() == 1        # game 12 (2024): breakaway penalty shot
    assert pwhl["pp_created"].sum() == 8
    assert (pwhl["ot_remaining_s"].between(0, 300)).all()
