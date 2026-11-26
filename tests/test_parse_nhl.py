"""Tests for the NHL parser: helpers (offline) and sanity checks on parsed data."""

from pathlib import Path

import pandas as pd
import pytest

from src.parse_nhl import decode_situation, elapsed, mmss, snap_boundary

INTERIM = Path(__file__).resolve().parents[1] / "data" / "interim"


# ---------------------------------------------------------------- helpers

def test_decode_situation_digit_order():
    # [away goalie][away skaters][home skaters][home goalie]
    assert decode_situation("1451") == (1, 4, 5, 1)   # away team shorthanded
    assert decode_situation("0651") == (0, 6, 5, 1)   # away goalie pulled
    assert decode_situation("1331") == (1, 3, 3, 1)   # 3-on-3 OT
    assert decode_situation(None) == (None, None, None, None)
    assert decode_situation("12") == (None, None, None, None)


def test_time_helpers():
    assert mmss("02:38") == 158
    assert elapsed(1, 0) == 0
    assert elapsed(3, 1200) == 3600
    assert elapsed(4, 300) == 3900


def test_snap_boundary_uses_penalty_expiry():
    # code still 5v4 at t=200, back to 5v5 at t=230; the minor expired at 220
    assert snap_boundary(200, 230, [220]) == 220
    # no expiry in between: change happened at the play where it was seen
    assert snap_boundary(200, 230, [300]) == 230


# ---------------------------------------------------------------- parsed data (skipped if not built)

@pytest.fixture(scope="module")
def tables():
    if not (INTERIM / "nhl_events.parquet").exists():
        pytest.skip("run `python -m src.parse_nhl` first")
    return {n: pd.read_parquet(INTERIM / f"nhl_{n}.parquet") for n in ("games", "events", "strength")}


def test_no_negative_times(tables):
    ev = tables["events"]
    timed = ev[ev["period_type"] != "SO"]
    assert (timed["elapsed_s"] >= 0).all()
    assert (timed["period_s"].between(0, 1200)).all()


def test_goals_complete_in_almost_all_games(tables):
    assert tables["games"]["goals_complete"].mean() > 0.98


def test_strength_intervals_cover_each_game(tables):
    s = tables["strength"].sort_values(["game_id", "start_s"])
    games = tables["games"].set_index("game_id")
    assert (s.groupby("game_id")["start_s"].first() == 0).all()
    same_game = s["game_id"].values[1:] == s["game_id"].values[:-1]
    assert (s["start_s"].values[1:][same_game] == s["end_s"].values[:-1][same_game]).all()
    last = s.groupby("game_id")["end_s"].last()
    assert (last == games.loc[last.index, "game_end_s"]).all()


def test_overtime_is_mostly_three_on_three(tables):
    s = tables["strength"]
    ot = s[s["start_s"] >= 3600]
    share_3v3 = ot.loc[(ot["home_skaters"] == 3) & (ot["away_skaters"] == 3), "duration_s"].sum() / ot["duration_s"].sum()
    assert share_3v3 > 0.8


def test_shootout_attempts_have_shooter_and_goalie(tables):
    so = tables["events"].query("event_type == 'shootout'")
    assert len(so) > 5000
    assert so["player_id"].notna().all()
    assert so["goalie_id"].notna().mean() > 0.999


def test_empty_net_goals_have_conceding_goalie_digit_zero(tables):
    ev = tables["events"]
    en = ev[(ev["event_type"] == "goal") & (ev["is_en"] == 1) & ev["home_goalie"].notna()]
    conceding_goalie = en["away_goalie"].where(en["is_home"] == 1, en["home_goalie"])
    assert (conceding_goalie == 0).mean() > 0.95
