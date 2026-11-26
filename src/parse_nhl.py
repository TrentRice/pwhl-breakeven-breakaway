"""Parse cached NHL play-by-play into tidy tables.

Usage (from the repo root, with the venv active):
    python -m src.parse_nhl

Reads data/raw/nhl/ and writes, under data/interim/:
    nhl_games.parquet       one row per regular-season game
    nhl_events.parquet      one row per play, with skater counts decoded from situationCode
    nhl_penalties.parquet   every penalty with type, duration and the situation when called
    nhl_strength.parquet    strength-state intervals per game

situationCode is 4 digits: [away goalie][away skaters][home skaters][home goalie]. Verified on
the raw data: on every sampled empty-net goal the conceding team's goalie digit is 0, and after a
minor at 5-on-5 the penalized side drops to 4 skaters (1451 = away penalized, 1541 = home).

Time conventions (CONVENTIONS.md): elapsed_s is seconds elapsed in the game. Periods 1-3 are
1200 s; regular-season OT (period 4) is 300 s of 3-on-3. Shootout events have elapsed_s NaN.

Strength intervals come from situationCode changes between consecutive plays. The code changes
at the first play after the change, so a boundary is placed at a penalty's scheduled expiry when
one falls between the two plays, and otherwise at the play where the new code first appears.
Codes on period-end / game-end markers are not used (they are unreliable).

Feed gaps handled here: shootout misses have no goalie in the feed, so the goalie the shooter's
team faced in that shootout is filled in; about 1% of games are missing a goal in the play-by-play,
flagged as goals_complete = 0 in nhl_games so modeling tables can exclude them.

Event types follow the PWHL tables where they overlap: shootout attempts are 'shootout' and
in-game penalty-shot attempts are 'penaltyshot' (one shooter vs one goalie: code 1010 or 0101).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "nhl"
OUT_DIR = ROOT / "data" / "interim"

REG_PERIOD_S = 1200
REG_END_S = 3600
OT_S = 300

SHOT_TYPES = {"goal", "shot-on-goal", "missed-shot", "failed-shot-attempt"}
PENALTY_SHOT_CODES = {"1010", "0101"}
MARKERS = {"period-end", "game-end", "shootout-complete"}  # codes on these are not trusted
TYPE_NAMES = {
    "shot-on-goal": "shot", "missed-shot": "missed_shot", "blocked-shot": "blocked_shot",
    "failed-shot-attempt": "failed_shot", "delayed-penalty": "delayed_penalty",
    "period-start": "period_start", "period-end": "period_end", "game-end": "game_end",
    "shootout-complete": "shootout_complete",
}


# ---------------------------------------------------------------- helpers

def mmss(t: str) -> int:
    m, s = t.split(":")
    return int(m) * 60 + int(s)


def elapsed(period: int, period_s: int) -> int:
    if period <= 3:
        return (period - 1) * REG_PERIOD_S + period_s
    return REG_END_S + (period - 4) * OT_S + period_s


def decode_situation(code: str | None) -> tuple:
    """'1451' -> (away_goalie=1, away_skaters=4, home_skaters=5, home_goalie=1)."""
    if not code or len(code) != 4 or not code.isdigit():
        return (None, None, None, None)
    return tuple(int(c) for c in code)


def snap_boundary(t_prev: float, t_cur: float, expiries: list[float]) -> float:
    """Where a strength change seen at t_cur (last seen unchanged at t_prev) most likely happened."""
    inside = [e for e in expiries if t_prev < e <= t_cur]
    return max(inside) if inside else t_cur


# ---------------------------------------------------------------- per game

def parse_game(d: dict) -> tuple[dict, list[dict], list[dict], list[tuple]]:
    gid = d["id"]
    home, away = d["homeTeam"], d["awayTeam"]
    team_code = {home["id"]: home["abbrev"], away["id"]: away["abbrev"]}
    plays = sorted(d["plays"], key=lambda p: p.get("sortOrder", 0))

    rows, pens = [], []
    so_counts: dict[int, int] = {}
    for p in plays:
        pd_ = p["periodDescriptor"]
        ptype = pd_["periodType"]
        det = p.get("details") or {}
        key = p["typeDescKey"]
        code = p.get("situationCode")
        ag, as_, hs, hg = decode_situation(code)
        team = det.get("eventOwnerTeamId")
        period_s = mmss(p["timeInPeriod"]) if ptype != "SO" else None
        ev_type = TYPE_NAMES.get(key, key)
        if ptype == "SO" and key in SHOT_TYPES:
            ev_type = "shootout"
        elif key in SHOT_TYPES and code in PENALTY_SHOT_CODES:
            ev_type = "penaltyshot"

        row = {
            "game_id": gid, "season": d["season"], "event_idx": p.get("sortOrder"),
            "period": pd_["number"], "period_type": ptype, "period_s": period_s,
            "elapsed_s": elapsed(pd_["number"], period_s) if period_s is not None else np.nan,
            "event_type": ev_type, "nhl_type": key,
            "team_id": team, "team_code": team_code.get(team),
            "is_home": None if team is None else int(team == home["id"]),
            "player_id": det.get("shootingPlayerId") or det.get("scoringPlayerId") or det.get("committedByPlayerId")
                         or det.get("hittingPlayerId") or det.get("winningPlayerId") or det.get("playerId"),
            "goalie_id": det.get("goalieInNetId"),
            "shot_type": det.get("shotType"), "x": det.get("xCoord"), "y": det.get("yCoord"),
            "zone": det.get("zoneCode"),
            "is_goal": int(key == "goal") if key in SHOT_TYPES else None,
            "is_en": int(key == "goal" and ptype != "SO" and "goalieInNetId" not in det) if key == "goal" else None,
            "so_round": None,
            "situation_code": code,
            "away_goalie": ag, "away_skaters": as_, "home_skaters": hs, "home_goalie": hg,
            "penalty_type": det.get("typeCode") if key == "penalty" else None,
            "penalty_minutes": det.get("duration") if key == "penalty" else None,
            "penalty_desc": det.get("descKey") if key == "penalty" else None,
            "served_by": det.get("servedByPlayerId"), "drawn_by": det.get("drawnByPlayerId"),
        }
        if ev_type == "shootout" and team is not None:
            so_counts[team] = so_counts.get(team, 0) + 1
            row["so_round"] = so_counts[team]
        rows.append(row)
        if key == "penalty":
            pens.append({k: row[k] for k in (
                "game_id", "season", "event_idx", "period", "period_type", "period_s", "elapsed_s",
                "team_id", "team_code", "is_home", "player_id", "served_by", "drawn_by",
                "penalty_type", "penalty_minutes", "penalty_desc", "situation_code",
                "away_skaters", "home_skaters")})

    # ---- shootout misses carry no goalie in the feed: use the goalie this team faced in the shootout
    so_rows = [r for r in rows if r["event_type"] == "shootout"]
    faced = {}
    for r in rows:  # fallback: last goalie this team shot on before the shootout
        if r["event_type"] in ("goal", "shot", "missed_shot") and r["goalie_id"] is not None:
            faced[r["team_id"]] = r["goalie_id"]
    for r in so_rows:  # preferred: the goalie this team faced in the shootout itself
        if r["goalie_id"] is not None:
            faced[r["team_id"]] = r["goalie_id"]
    for r in so_rows:
        if r["goalie_id"] is None and r["team_id"] in faced:
            r["goalie_id"] = faced[r["team_id"]]

    # ---- does the play-by-play have every goal? (~1% of NHL games are missing one)
    goals = {home["id"]: 0, away["id"]: 0}
    for r in rows:
        if r["event_type"] == "goal" and r["team_id"] in goals:
            goals[r["team_id"]] += 1
    so_win = (d.get("gameOutcome") or {}).get("lastPeriodType") == "SO"
    if so_win and home.get("score") is not None:
        goals[home["id"] if home["score"] > away["score"] else away["id"]] += 1
    goals_complete = int(goals[home["id"]] == home.get("score") and goals[away["id"]] == away.get("score"))

    # ---- game end (before any shootout)
    timed = [r for r in rows if not np.isnan(r["elapsed_s"])]
    outcome = (d.get("gameOutcome") or {}).get("lastPeriodType")
    if outcome == "SO" or any(r["event_type"] == "shootout" for r in rows):
        end = REG_END_S + OT_S
    else:
        end = max((r["elapsed_s"] for r in timed), default=REG_END_S)
        end = max(end, REG_END_S)

    # ---- strength intervals from situationCode changes
    expiries = [r["elapsed_s"] + 60 * (r["penalty_minutes"] or 0) for r in rows
                if r["nhl_type"] == "penalty" and (r["penalty_minutes"] or 0) > 0 and not np.isnan(r["elapsed_s"])]
    expiries += [r["elapsed_s"] + 120 for r in rows if r["nhl_type"] == "penalty" and r["penalty_minutes"] == 4
                 and not np.isnan(r["elapsed_s"])]  # first half of a double minor
    usable = [r for r in timed if r["situation_code"] and r["nhl_type"] not in MARKERS
              and r["away_skaters"] is not None and r["elapsed_s"] <= end]
    intervals = []
    if usable:
        cur = usable[0]["situation_code"]
        start, t_prev = 0.0, usable[0]["elapsed_s"]
        for r in usable[1:]:
            t = r["elapsed_s"]
            if r["situation_code"] != cur:
                b = snap_boundary(t_prev, t, expiries)
                # a new period starts in its own state
                if r["nhl_type"] == "period_start" or r["period_s"] == 0:
                    b = t
                if b > start:
                    intervals.append((start, b, cur))
                    start = b
                cur = r["situation_code"]
            t_prev = t
        if end > start:
            intervals.append((start, end, cur))

    game = {
        "game_id": gid, "season": d["season"], "date": d.get("gameDate"),
        "home_team_id": home["id"], "home_team": home["abbrev"], "away_team_id": away["id"], "away_team": away["abbrev"],
        "home_goals": home.get("score"), "away_goals": away.get("score"),
        "last_period_type": outcome, "game_end_s": end, "goals_complete": goals_complete,
    }
    return game, rows, pens, intervals


# ---------------------------------------------------------------- driver

def parse_all() -> dict[str, pd.DataFrame]:
    index = pd.read_csv(RAW_DIR / "games.csv")
    index = index[(index["has_pbp"] == 1) & index["game_state"].isin(["OFF", "FINAL"])]
    games, rows, pens, strength = [], [], [], []
    for i, gid in enumerate(index["game_id"], 1):
        d = json.loads((RAW_DIR / "games" / f"{gid}_pbp.json").read_text())
        g, r, p, iv = parse_game(d)
        games.append(g)
        rows += r
        pens += p
        for a, b, code in iv:
            ag, as_, hs, hg = decode_situation(code)
            strength.append((gid, d["season"], a, b, code, as_, hs, ag, hg))
        if i % 2000 == 0:
            print(f"  parsed {i:,}/{len(index):,} games")

    events = pd.DataFrame(rows)
    for col in ("player_id", "goalie_id", "served_by", "drawn_by", "team_id"):
        events[col] = events[col].astype("Int64")
    strength = pd.DataFrame(strength, columns=["game_id", "season", "start_s", "end_s", "situation_code",
                                               "away_skaters", "home_skaters", "away_goalie", "home_goalie"])
    strength["duration_s"] = strength["end_s"] - strength["start_s"]
    return {"games": pd.DataFrame(games), "events": events, "penalties": pd.DataFrame(pens), "strength": strength}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tables = parse_all()
    for name, df in tables.items():
        df.to_parquet(OUT_DIR / f"nhl_{name}.parquet", index=False)
        print(f"nhl_{name}.parquet: {len(df):,} rows")


if __name__ == "__main__":
    main()
