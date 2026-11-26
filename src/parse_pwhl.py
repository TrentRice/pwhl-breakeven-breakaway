"""Parse cached PWHL play-by-play into tidy tables and reconstruct strength states.

Usage (from the repo root, with the venv active):
    python -m src.parse_pwhl

Reads data/raw/pwhl/ and writes, under data/interim/:
    pwhl_games.parquet       one row per finished game
    pwhl_events.parquet      one row per play-by-play event, with the strength state just before it
    pwhl_penalties.parquet   every penalty with reconstructed start, end and end reason
    pwhl_strength.parquet    strength-state intervals per game (skaters and goalie for each team)

Time conventions (CONVENTIONS.md): elapsed_s is seconds elapsed in the game. Periods 1-3 are
1200 s. Overtime is period 4+: 300 s of 3-on-3 in regular season and preseason, 1200 s of 5-on-5
per period in the playoffs. Shootout events have no game time (elapsed_s is NaN).

Strength reconstruction (the feed has no on-ice data except on goals):
- Only penalties the feed flags as creating a power play (pp >= 1) change strength; penalties
  that were turned into penalty shots are not served; misconducts never change strength.
- At most two penalties per team run at once; a third waits until one ends.
- Several minors on the same player at the same time are served back to back (double minors).
- A power-play goal ends the earliest-starting minor of the shorthanded team. Majors run in full.
- Regulation and playoff OT: skaters = 5 - penalties running (minimum 3).
  Regular-season/preseason OT (3-on-3): skaters = 3 + max(0, opponent's penalties - own penalties),
  so a minor gives the other team a 4th skater (4-on-3) and two give a 5th (5-on-3).
- A pulled goalie adds one skater.
The result is checked against the on-ice skater counts recorded on goals (see validate()).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "pwhl"
OUT_DIR = ROOT / "data" / "interim"

REG_PERIOD_S = 1200
REG_END_S = 3 * REG_PERIOD_S  # 3600
OT_3V3_S = 300
PLAYOFF_OT_S = 1200
STRENGTH_CLASSES = {"Minor", "Major", "Match"}


# ---------------------------------------------------------------- time helpers

def ot_length(season_type: str) -> int:
    return PLAYOFF_OT_S if season_type == "playoff" else OT_3V3_S


def elapsed(period: int, period_s: int, season_type: str) -> int:
    """Seconds elapsed in the game at `period_s` seconds into `period`."""
    if period <= 3:
        return (period - 1) * REG_PERIOD_S + period_s
    return REG_END_S + (period - 4) * ot_length(season_type) + period_s


def is_3v3(t: float, season_type: str) -> bool:
    """True when time t (elapsed) is in 3-on-3 overtime."""
    return season_type != "playoff" and t >= REG_END_S


# ---------------------------------------------------------------- strength engine

@dataclass
class Penalty:
    pid: str
    team: str            # penalized team id
    player: str | None   # player serving
    called_s: int
    duration_s: int
    is_minor: bool
    start_s: float | None = None
    end_s: float | None = None
    end_reason: str | None = None
    chain_next: "Penalty | None" = None


@dataclass
class StrengthEngine:
    """Tracks penalties and goalies for both teams and records strength intervals."""
    home: str
    away: str
    season_type: str
    t0: float = 0.0
    active: dict = field(default_factory=dict)
    queue: dict = field(default_factory=dict)
    goalie_in: dict = field(default_factory=dict)
    net: dict = field(default_factory=dict)  # goalie id currently in net, per team
    intervals: list = field(default_factory=list)
    _state: tuple | None = None
    _state_start: float = 0.0

    def __post_init__(self):
        for team in (self.home, self.away):
            self.active[team] = []
            self.queue[team] = []
            self.goalie_in[team] = True
        self._state = self.state(0.0)

    # ---- state
    def n_box(self, team: str) -> int:
        return len(self.active[team])

    def skaters(self, team: str, t: float) -> int:
        opp = self.away if team == self.home else self.home
        if is_3v3(t, self.season_type):
            n = 3 + max(0, self.n_box(opp) - self.n_box(team))
            n = min(n, 5)
        else:
            n = max(3, 5 - self.n_box(team))
        return n + (0 if self.goalie_in[team] else 1)

    def state(self, t: float) -> tuple:
        return (self.skaters(self.home, t), self.skaters(self.away, t),
                int(self.goalie_in[self.home]), int(self.goalie_in[self.away]))

    def _mark(self, t: float) -> None:
        """Record a new interval if the state changed at time t."""
        new = self.state(t)
        if new != self._state:
            if t > self._state_start:
                self.intervals.append((self._state_start, t, *self._state))
            self._state, self._state_start = new, t

    # ---- penalties
    def _start(self, p: Penalty, t: float) -> None:
        p.start_s, p.end_s = t, t + p.duration_s
        self.active[p.team].append(p)

    def _finish(self, p: Penalty, t: float, reason: str) -> None:
        self.active[p.team].remove(p)
        p.end_s, p.end_reason = t, reason
        if p.chain_next is not None:  # second half of a double minor
            self._start(p.chain_next, t)
        elif self.queue[p.team] and len(self.active[p.team]) < 2:
            self._start(self.queue[p.team].pop(0), t)

    def add_penalty(self, p: Penalty, t: float, chained_to: Penalty | None) -> None:
        if chained_to is not None:
            chained_to.chain_next = p
        elif len(self.active[p.team]) < 2:
            self._start(p, t)
        else:
            self.queue[p.team].append(p)
        self._mark(t)

    def advance(self, t: float, inclusive: bool) -> None:
        """Expire penalties ending before t (or at t when inclusive), in time order."""
        while True:
            running = [p for team in self.active for p in self.active[team]]
            due = [p for p in running if (p.end_s <= t if inclusive else p.end_s < t)]
            if not due:
                return
            p = min(due, key=lambda q: q.end_s)
            end = p.end_s
            self._finish(p, end, "expired")
            self._mark(end)

    def boundary(self, t: float) -> None:
        """Mark a rule change in strength (start of 3-on-3 OT)."""
        self.advance(t, inclusive=True)
        self._mark(t)

    def goal(self, scorer: str, t: float) -> None:
        """A goal by `scorer` ends the earliest minor of a shorthanded opponent."""
        opp = self.away if scorer == self.home else self.home
        if self.n_box(opp) > self.n_box(scorer):
            minors = [p for p in self.active[opp] if p.is_minor]
            if minors:
                self._finish(min(minors, key=lambda q: q.start_s), t, "pp_goal")
                self._mark(t)

    def goalie(self, team: str, goalie_in: str | None, goalie_out: str | None, t: float) -> None:
        """Apply a goalie_change event. A swap arrives as 'new in' then 'old out', so an
        'out' only empties the net when the goalie leaving is the one currently in net."""
        if goalie_in is not None:
            self.net[team] = goalie_in
        elif goalie_out is not None and self.net.get(team) in (goalie_out, None):
            self.net[team] = None
        self.goalie_in[team] = self.net.get(team) is not None
        self._mark(t)

    def close(self, t_end: float) -> None:
        self.advance(t_end, inclusive=True)
        for team in self.active:
            for p in list(self.active[team]):
                p.end_s, p.end_reason = t_end, "game_end"
            for p in self.queue[team]:
                p.end_reason = "never_started"
        if t_end > self._state_start:
            self.intervals.append((self._state_start, t_end, *self._state))


# ---------------------------------------------------------------- event extraction

def _int(x):
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


def event_period(e: dict) -> int | None:
    return _int(e.get("period_id")) or _int(e.get("period"))


def event_team(e: dict) -> str | None:
    ev = e["event"]
    if ev in ("shot", "blocked_shot"):
        return e.get("player_team_id") or e.get("team_id")
    if ev == "faceoff":
        return e.get("win_team_id")
    return e.get("team_id")


def extract_event(e: dict, idx: int, season_type: str) -> dict:
    ev = e["event"]
    period = event_period(e) if ev != "shootout" else None
    period_s = _int(e.get("s")) if ev != "shootout" else None
    row = {
        "event_idx": idx,
        "event_type": ev,
        "period": period,
        "period_type": "SO" if ev == "shootout" else ("REG" if (period or 0) <= 3 else "OT"),
        "period_s": period_s,
        "elapsed_s": elapsed(period, period_s, season_type) if period is not None else np.nan,
        "team_id": event_team(e),
        "player_id": None, "goalie_id": None,
        "shot_type": e.get("shot_type_description"),
        "x": _int(e.get("x_location")), "y": _int(e.get("y_location")),
        "is_goal": None, "is_pp": None, "is_sh": None, "is_en": None, "is_ps": None,
        "n_plus": None, "n_minus": None,
        "penalty_class": None, "penalty_minutes": None, "penalty_desc": None,
        "pp_flag": None, "ps_awarded": None, "bench": None, "served_by": None,
        "so_round": None, "goalie_in": None, "goalie_out": None,
    }
    if ev in ("shot", "blocked_shot"):
        row.update(player_id=e.get("player_id"), goalie_id=e.get("goalie_id"),
                   is_goal=int(bool(e.get("game_goal_id"))) if ev == "shot" else 0)
    elif ev == "goal":
        row.update(player_id=e.get("goal_player_id"), is_goal=1,
                   is_pp=int(e.get("power_play") == "1"), is_sh=int(e.get("short_handed") == "1"),
                   is_en=int(e.get("empty_net") == "1"), is_ps=int(e.get("penalty_shot") == "1"),
                   n_plus=len(e.get("plus") or []), n_minus=len(e.get("minus") or []))
    elif ev == "penalty":
        row.update(player_id=e.get("player_id"), served_by=e.get("player_served"),
                   penalty_class=e.get("penalty_class"), penalty_minutes=float(e.get("minutes") or 0),
                   penalty_desc=e.get("lang_penalty_description"), pp_flag=_int(e.get("pp")),
                   ps_awarded=int(e.get("penalty_shot") == "1"), bench=int(e.get("bench") == "1"))
    elif ev == "shootout":
        row.update(player_id=e.get("player_id"), goalie_id=e.get("goalie_id"),
                   so_round=_int(e.get("shot_order")), is_goal=int(e.get("goal") == "1"))
    elif ev == "penaltyshot":
        row.update(player_id=e.get("player_id"), goalie_id=e.get("goalie_id"),
                   is_goal=int(e.get("result") == "goal"), is_ps=1)
    elif ev == "goalie_change":
        row.update(goalie_in=e.get("goalie_in_id"), goalie_out=e.get("goalie_out_id"))
    elif ev == "hit":
        row.update(player_id=e.get("player_id"))
    elif ev == "faceoff":
        row.update(player_id=e.get("home_player_id") if e.get("home_win") == "1" else e.get("visitor_player_id"))
    return row


# ---------------------------------------------------------------- per game

def game_end_s(rows: list[dict], season_type: str, status: str) -> float:
    """Elapsed time at which play ended (before any shootout)."""
    timed = [r for r in rows if not np.isnan(r["elapsed_s"])]
    last_event = max((r["elapsed_s"] for r in timed), default=REG_END_S)
    ot_goals = [r["elapsed_s"] for r in timed if r["event_type"] == "goal" and r["elapsed_s"] > REG_END_S]
    if any(r["event_type"] == "shootout" for r in rows):
        rule_end = REG_END_S + ot_length(season_type)
    elif "OT" in status and ot_goals:          # "Final OT", "Final 2OT", ...
        rule_end = max(ot_goals)
    elif "OT" in status:                       # OT played without a deciding goal (preseason)
        rule_end = REG_END_S + ot_length(season_type)
    else:
        rule_end = REG_END_S
    # Never cut off recorded play (e.g. preseason OT played regardless of score).
    return float(max(rule_end, last_event))


# Order of simultaneous events: goals before penalties (a penalty at the same second
# cannot have been on the ice for the goal), goalie changes last.
_TIE_ORDER = {"goal": 0, "shot": 1, "blocked_shot": 1, "hit": 1, "penaltyshot": 1,
              "penalty": 2, "faceoff": 3, "goalie_change": 4, "shootout": 5}


def parse_game(game: dict, events: list[dict]) -> tuple[list[dict], list[dict], list[tuple]]:
    st = game["season_type"]
    home, away = str(game["home_team_id"]), str(game["away_team_id"])
    rows = [extract_event(e, i, st) for i, e in enumerate(events)]
    end = game_end_s(rows, st, game["game_status"])

    eng = StrengthEngine(home=home, away=away, season_type=st)
    penalties: list[Penalty] = []
    pen_rows: list[dict] = []
    timed = sorted((r for r in rows if not np.isnan(r["elapsed_s"])),
                   key=lambda r: (r["elapsed_s"], _TIE_ORDER.get(r["event_type"], 9), r["event_idx"]))
    ot_marked = False

    for r in timed:
        t = r["elapsed_s"]
        if t > end:
            r.update(home_skaters=None, away_skaters=None, home_goalie=None, away_goalie=None)
            continue
        if not ot_marked and is_3v3(t, st):
            eng.boundary(REG_END_S)
            ot_marked = True
        # goals happen with the state just before them; a penalty ending at the same second
        # is treated as still running (clock shows whole seconds).
        eng.advance(t, inclusive=r["event_type"] != "goal")
        hs, as_, hg, ag = eng.state(t)
        r.update(home_skaters=hs, away_skaters=as_, home_goalie=hg, away_goalie=ag)

        team = str(r["team_id"]) if r["team_id"] is not None else None
        if r["event_type"] == "goal" and team in (home, away):
            eng.goal(team, t)
        elif r["event_type"] == "penalty" and team in (home, away):
            minutes = r["penalty_minutes"] or 0
            affects = (r["penalty_class"] in STRENGTH_CLASSES and (r["pp_flag"] or 0) >= 1
                       and not r["ps_awarded"] and minutes in (2.0, 5.0))
            pen_row = {"event_idx": r["event_idx"], "team_id": team, "called_s": t,
                       "affects_strength": int(affects), "start_s": None, "end_s": None,
                       "end_reason": None if affects else
                       ("penalty_shot" if r["ps_awarded"] else "no_strength_effect")}
            pen_rows.append(pen_row)
            if affects:
                p = Penalty(pid=str(r["event_idx"]), team=team, player=r["served_by"], called_s=t,
                            duration_s=int(minutes * 60), is_minor=minutes == 2.0)
                chained_to = None
                if p.is_minor and p.player:
                    same = [q for q in penalties if q.team == team and q.player == p.player
                            and q.called_s == t and q.is_minor and q.chain_next is None]
                    chained_to = same[-1] if same else None
                penalties.append(p)
                pen_row["_obj"] = p
                eng.add_penalty(p, t, chained_to)
        elif r["event_type"] == "goalie_change" and team in (home, away):
            eng.goalie(team, r["goalie_in"], r["goalie_out"], t)

        # goalie being scored on / shot at, from the tracked net (goal events lack it)
        if r["event_type"] == "goal" and team in (home, away):
            opp = away if team == home else home
            r["goalie_id"] = eng.net.get(opp)

    if not ot_marked and end > REG_END_S and is_3v3(end, st):
        eng.boundary(REG_END_S)
    eng.close(end)

    for pr in pen_rows:
        p = pr.pop("_obj", None)
        if p is not None:
            pr.update(start_s=p.start_s, end_s=p.end_s, end_reason=p.end_reason)
    for r in rows:
        r.setdefault("home_skaters", None)
        r.setdefault("away_skaters", None)
        r.setdefault("home_goalie", None)
        r.setdefault("away_goalie", None)
    game["game_end_s"] = end
    return rows, pen_rows, eng.intervals


# ---------------------------------------------------------------- driver

def load_games() -> pd.DataFrame:
    g = pd.read_csv(RAW_DIR / "games.csv", dtype={"game_id": str, "season_id": str})
    g = g[(g["final"] == 1) & (g["has_pxp"] == 1)].copy()
    ids = {}
    for path in (RAW_DIR / "schedules").glob("season_*.json"):
        for s in json.loads(path.read_text())["SiteKit"]["Schedule"]:
            ids[str(s["game_id"])] = (str(s["home_team"]), str(s["visiting_team"]))
    g["home_team_id"] = g["game_id"].map(lambda x: ids[x][0])
    g["away_team_id"] = g["game_id"].map(lambda x: ids[x][1])
    return g.rename(columns={"away_goals": "away_goals", "home_goals": "home_goals"})


def parse_all() -> dict[str, pd.DataFrame]:
    games = load_games()
    ev_rows, pen_rows, str_rows, game_rows = [], [], [], []
    for game in games.to_dict("records"):
        events = json.loads((RAW_DIR / "games" / f"{game['game_id']}_pxp.json").read_text())["GC"]["Pxpverbose"]
        rows, pens, intervals = parse_game(game, events)
        meta = {"game_id": game["game_id"], "season_id": game["season_id"], "season_type": game["season_type"]}
        team_code = {game["home_team_id"]: game["home_team"], game["away_team_id"]: game["away_team"]}
        for r in rows:
            r.update(meta)
            r["team_code"] = team_code.get(str(r["team_id"]))
            r["is_home"] = None if r["team_id"] is None else int(str(r["team_id"]) == game["home_team_id"])
        for p in pens:
            p.update(meta)
        ev_rows += rows
        pen_rows += pens
        str_rows += [{**meta, "start_s": a, "end_s": b, "home_skaters": hs, "away_skaters": as_,
                      "home_goalie": hg, "away_goalie": ag} for a, b, hs, as_, hg, ag in intervals]
        game_rows.append(game)

    events = pd.DataFrame(ev_rows)
    penalties = pd.DataFrame(pen_rows).merge(
        events[["game_id", "event_idx", "period", "period_s", "player_id", "served_by",
                "penalty_class", "penalty_minutes", "penalty_desc", "pp_flag", "ps_awarded", "bench"]],
        on=["game_id", "event_idx"], how="left")
    strength = pd.DataFrame(str_rows)
    strength["duration_s"] = strength["end_s"] - strength["start_s"]
    games_df = pd.DataFrame(game_rows)

    lead = ["game_id", "season_id", "season_type", "event_idx", "period", "period_type", "period_s", "elapsed_s",
            "event_type", "team_id", "team_code", "is_home"]
    events = events[lead + [c for c in events.columns if c not in lead]]
    for col in ("team_id", "player_id", "goalie_id", "served_by", "goalie_in", "goalie_out"):
        events[col] = events[col].astype("string")
    return {"games": games_df, "events": events, "penalties": penalties, "strength": strength}


def validate(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Compare reconstructed skaters with the on-ice counts recorded on each goal."""
    ev = tables["events"]
    goals = ev[(ev["event_type"] == "goal") & ev["home_skaters"].notna()].copy()
    goals = goals[(goals["n_plus"] >= 3) & (goals["n_minus"] >= 3)]  # complete on-ice lists only
    goals["for_skaters"] = np.where(goals["is_home"] == 1, goals["home_skaters"], goals["away_skaters"])
    goals["against_skaters"] = np.where(goals["is_home"] == 1, goals["away_skaters"], goals["home_skaters"])
    goals["match"] = (goals["for_skaters"] == goals["n_plus"]) & (goals["against_skaters"] == goals["n_minus"])
    return goals


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tables = parse_all()
    for name, df in tables.items():
        df.to_parquet(OUT_DIR / f"pwhl_{name}.parquet", index=False)
        print(f"pwhl_{name}.parquet: {len(df):,} rows")
    goals = validate(tables)
    print(f"\nStrength check on {len(goals):,} goals with complete on-ice lists: "
          f"{goals['match'].mean():.1%} match")
    print(goals.groupby("season_type")["match"].agg(["mean", "size"]).round(3).to_string())


if __name__ == "__main__":
    main()
