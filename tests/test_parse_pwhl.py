"""Tests for the PWHL parser: strength-engine rules (offline) and sanity checks on parsed data."""

from pathlib import Path

import pandas as pd
import pytest

from src.parse_pwhl import Penalty, StrengthEngine, elapsed

H, A = "home", "away"
INTERIM = Path(__file__).resolve().parents[1] / "data" / "interim"


def engine(season_type="regular"):
    return StrengthEngine(home=H, away=A, season_type=season_type)


def minor(team, t, player="p1", pid="x"):
    return Penalty(pid=pid, team=team, player=player, called_s=t, duration_s=120, is_minor=True)


def states(eng):
    return [(a, b, hs, as_) for a, b, hs, as_, _, _ in eng.intervals]


# ---------------------------------------------------------------- engine rules

def test_elapsed_periods():
    assert elapsed(1, 0, "regular") == 0
    assert elapsed(3, 1200, "regular") == 3600
    assert elapsed(4, 300, "regular") == 3900
    assert elapsed(5, 100, "playoff") == 3600 + 1200 + 100


def test_minor_expires_after_two_minutes():
    eng = engine()
    eng.add_penalty(minor(A, 100), 100, None)
    eng.close(1000)
    assert states(eng) == [(0, 100, 5, 5), (100, 220, 5, 4), (220, 1000, 5, 5)]


def test_pp_goal_ends_minor():
    eng = engine()
    eng.add_penalty(minor(A, 100), 100, None)
    eng.advance(150, inclusive=False)
    eng.goal(H, 150)
    eng.close(1000)
    assert states(eng)[1] == (100, 150, 5, 4)
    assert states(eng)[2] == (150, 1000, 5, 5)


def test_major_not_ended_by_pp_goal():
    eng = engine()
    major = Penalty(pid="m", team=A, player="p1", called_s=100, duration_s=300, is_minor=False)
    eng.add_penalty(major, 100, None)
    eng.goal(H, 150)
    eng.close(1000)
    assert states(eng)[1] == (100, 400, 5, 4)


def test_double_minor_served_back_to_back():
    eng = engine()
    first, second = minor(A, 100, pid="a"), minor(A, 100, pid="b")
    eng.add_penalty(first, 100, None)
    eng.add_penalty(second, 100, first)
    eng.close(1000)
    assert states(eng) == [(0, 100, 5, 5), (100, 340, 5, 4), (340, 1000, 5, 5)]


def test_third_penalty_waits():
    eng = engine()
    for i, t in enumerate((100, 110, 120)):
        eng.add_penalty(minor(A, t, player=f"p{i}", pid=str(i)), t, None)
    eng.close(1000)
    # 5-on-3 until the first expires at 220, then the third starts: still 5-on-3 until 230
    assert states(eng)[:3] == [(0, 100, 5, 5), (100, 110, 5, 4), (110, 230, 5, 3)]
    assert states(eng)[3] == (230, 340, 5, 4)


def test_overtime_minor_gives_four_on_three():
    eng = engine()
    eng.boundary(3600)
    eng.add_penalty(minor(A, 3700), 3700, None)
    eng.close(3900)
    assert states(eng)[-3:] == [(3600, 3700, 3, 3), (3700, 3820, 4, 3), (3820, 3900, 3, 3)]


def test_regulation_penalty_carries_into_ot_as_four_on_three():
    eng = engine()
    eng.add_penalty(minor(A, 3550), 3550, None)
    eng.boundary(3600)
    eng.close(3900)
    assert states(eng)[-3:] == [(3550, 3600, 5, 4), (3600, 3670, 4, 3), (3670, 3900, 3, 3)]


def test_playoff_overtime_is_five_on_five():
    eng = engine("playoff")
    eng.add_penalty(minor(A, 3700), 3700, None)
    eng.close(4000)
    assert (3700, 3820, 5, 4) in states(eng)


def test_goalie_swap_does_not_empty_net():
    eng = engine()
    eng.goalie(H, "g1", None, 0)
    eng.goalie(H, "g2", None, 500)   # new goalie in
    eng.goalie(H, None, "g1", 500)   # old goalie out
    assert eng.goalie_in[H] and eng.net[H] == "g2"


def test_pulled_goalie_adds_skater():
    eng = engine()
    eng.goalie(H, "g1", None, 0)
    eng.goalie(H, None, "g1", 3500)
    eng.close(3600)
    assert states(eng)[-1] == (3500, 3600, 6, 5)


# ---------------------------------------------------------------- parsed data (skipped if not built)

@pytest.fixture(scope="module")
def tables():
    if not (INTERIM / "pwhl_events.parquet").exists():
        pytest.skip("run `python -m src.parse_pwhl` first")
    return {n: pd.read_parquet(INTERIM / f"pwhl_{n}.parquet") for n in ("games", "events", "penalties", "strength")}


def test_no_negative_or_out_of_range_times(tables):
    ev = tables["events"]
    timed = ev[ev["period_type"] != "SO"]
    assert (timed["elapsed_s"] >= 0).all()
    assert (timed["period_s"] >= 0).all()
    reg = timed[timed["period"] <= 3]
    assert (reg["period_s"] <= 1200).all()


def test_goals_match_final_score(tables):
    ev, games = tables["events"], tables["games"]
    goals = ev[ev["event_type"] == "goal"].groupby(["game_id", "is_home"]).size().unstack(fill_value=0)
    g = games.set_index("game_id")
    so_winner_home = g["shootout"].eq(1) & (g["home_goals"] > g["away_goals"])
    so_winner_away = g["shootout"].eq(1) & (g["away_goals"] > g["home_goals"])
    home = goals.reindex(g.index, fill_value=0)[1] + so_winner_home.astype(int)
    away = goals.reindex(g.index, fill_value=0)[0] + so_winner_away.astype(int)
    assert (home == g["home_goals"]).all()
    assert (away == g["away_goals"]).all()


def test_strength_intervals_cover_each_game(tables):
    s, games = tables["strength"], tables["games"].set_index("game_id")
    for gid, grp in s.groupby("game_id"):
        grp = grp.sort_values("start_s")
        assert grp["start_s"].iloc[0] == 0
        assert (grp["start_s"].iloc[1:].values == grp["end_s"].iloc[:-1].values).all()
        assert grp["end_s"].iloc[-1] == games.loc[gid, "game_end_s"]
    assert s[["home_skaters", "away_skaters"]].isin([3, 4, 5, 6]).all().all()


def test_strength_matches_goal_on_ice_counts(tables):
    ev = tables["events"]
    g = ev[(ev["event_type"] == "goal") & (ev["n_plus"] >= 3) & (ev["n_minus"] >= 3)
           & (ev["season_type"] == "regular")]
    own = g["home_skaters"].where(g["is_home"] == 1, g["away_skaters"])
    opp = g["away_skaters"].where(g["is_home"] == 1, g["home_skaters"])
    match = ((own == g["n_plus"]) & (opp == g["n_minus"])).mean()
    assert match > 0.95


def test_every_penalty_is_in_penalty_table(tables):
    assert len(tables["penalties"]) == (tables["events"]["event_type"] == "penalty").sum()
