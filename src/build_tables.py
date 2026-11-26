"""Build the modeling tables (PROJECT_PLAN Step 5) from the parsed PWHL and NHL tables.

Usage (from the repo root, with the venv active):
    python -m src.build_tables

Reads data/interim/{pwhl,nhl}_*.parquet and writes, under data/processed/:
    shootout_attempts.parquet  every shootout attempt and in-game penalty shot (shooter, goalie, goal)
    pp_intervals.parquet       man-advantage time: one row per spell segment at a constant strength
    ot_intervals.parquet       3-on-3 overtime time and goals, one row per team per game
    ot_penalties.parquet       every penalty called in 3-on-3 overtime, with what followed
and outputs/tables/data_summary.{csv,md} with counts by league and season.

Definitions
- Man advantage compares skaters from penalties only: a pulled goalie's extra attacker does not
  count, so empty-net time is never a power play.
- A spell is an unbroken run of man-advantage time for one team. It is split into segments when
  the strength changes (e.g. 5v3 -> 5v4). spell_offset_s is how far into the spell a segment
  starts; first_stoppage_s is when the first faceoff after the spell's opening faceoff happened
  (the No Escape window ends no earlier than this).
- A goal belongs to the segment with start < goal time <= end (the state just before the goal).
- Overtime tables cover 3-on-3 overtime only (regular season and preseason); playoff OT is 5-on-5.
- League-specific flags: no_escape = 1 for PWHL seasons from 2024-25 on (PWHL season_id >= 4);
  tied_after_reg = 0 marks PWHL preseason games that played OT without being tied;
  goals_complete = 0 marks NHL games whose play-by-play is missing a goal (exclude from rates).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "outputs" / "tables"

REG_END_S = 3600
OT_S = 300
NO_ESCAPE_FIRST_PWHL_SEASON_ID = 4  # 2024-25 preseason onward


# ---------------------------------------------------------------- load and harmonize

def _pwhl_season_label(name: str) -> str:
    return name.split()[0]  # "2024-25 Regular Season" -> "2024-25"


def load_pwhl() -> dict[str, pd.DataFrame]:
    g = pd.read_parquet(INTERIM / "pwhl_games.parquet")
    games = pd.DataFrame({
        "league": "PWHL",
        "game_id": g["game_id"].astype(str),
        "season": g["season_name"].map(_pwhl_season_label),
        "season_type": g["season_type"],
        "home_team": g["home_team"], "away_team": g["away_team"],
        "home_goals": g["home_goals"].astype(int), "away_goals": g["away_goals"].astype(int),
        "outcome": np.select([g["game_status"].str.contains("SO"), g["game_status"].str.contains("OT")],
                             ["SO", "OT"], "REG"),
        "game_end_s": g["game_end_s"].astype(float),
        "goals_complete": 1,
        "no_escape": (g["season_id"].astype(int) >= NO_ESCAPE_FIRST_PWHL_SEASON_ID).astype(int),
    })
    ev = pd.read_parquet(INTERIM / "pwhl_events.parquet", columns=[
        "game_id", "event_idx", "event_type", "period_type", "elapsed_s", "is_home",
        "player_id", "goalie_id", "is_goal", "so_round"])
    pen = pd.read_parquet(INTERIM / "pwhl_penalties.parquet")
    pen = pd.DataFrame({
        "game_id": pen["game_id"].astype(str), "event_idx": pen["event_idx"],
        "called_s": pen["called_s"].astype(float),
        "is_home": (pen["team_id"] == pen["game_id"].map(
            g.set_index("game_id")["home_team_id"].astype(str))).astype(int),
        "penalty_class": pen["penalty_class"], "penalty_minutes": pen["penalty_minutes"],
        "penalty_desc": pen["penalty_desc"],
        "ps_awarded": pen["ps_awarded"].astype(int),
    })
    st = pd.read_parquet(INTERIM / "pwhl_strength.parquet")
    return {"games": games, "events": _norm_events(ev), "penalties": pen, "strength": _norm_strength(st)}


def load_nhl() -> dict[str, pd.DataFrame]:
    g = pd.read_parquet(INTERIM / "nhl_games.parquet")
    s = g["season"].astype(str)
    games = pd.DataFrame({
        "league": "NHL",
        "game_id": g["game_id"].astype(str),
        "season": s.str[:4] + "-" + s.str[6:],
        "season_type": "regular",
        "home_team": g["home_team"], "away_team": g["away_team"],
        "home_goals": g["home_goals"].astype(int), "away_goals": g["away_goals"].astype(int),
        "outcome": g["last_period_type"].fillna("REG"),
        "game_end_s": g["game_end_s"].astype(float),
        "goals_complete": g["goals_complete"].astype(int),
        "no_escape": 0,
    })
    ev = pd.read_parquet(INTERIM / "nhl_events.parquet", columns=[
        "game_id", "event_idx", "event_type", "period_type", "elapsed_s", "is_home",
        "player_id", "goalie_id", "is_goal", "so_round"])
    pen = pd.read_parquet(INTERIM / "nhl_penalties.parquet")
    pen = pd.DataFrame({
        "game_id": pen["game_id"].astype(str), "event_idx": pen["event_idx"],
        "called_s": pen["elapsed_s"].astype(float), "is_home": pen["is_home"].astype("Int64"),
        "penalty_class": pen["penalty_type"].map({"MIN": "Minor", "BEN": "Minor", "MAJ": "Major",
                                                  "MIS": "Misconduct", "GAM": "Misconduct",
                                                  "MAT": "Match", "PS": "Penalty shot"}),
        "penalty_minutes": pen["penalty_minutes"], "penalty_desc": pen["penalty_desc"],
        "ps_awarded": (pen["penalty_type"] == "PS").astype(int),
    })
    st = pd.read_parquet(INTERIM / "nhl_strength.parquet")
    return {"games": games, "events": _norm_events(ev), "penalties": pen, "strength": _norm_strength(st)}


def _norm_events(ev: pd.DataFrame) -> pd.DataFrame:
    ev = ev.copy()
    ev["game_id"] = ev["game_id"].astype(str)
    for c in ("player_id", "goalie_id"):
        ev[c] = ev[c].astype("string")
    return ev


def _norm_strength(st: pd.DataFrame) -> pd.DataFrame:
    st = st[["game_id", "start_s", "end_s", "home_skaters", "away_skaters", "home_goalie", "away_goalie"]].copy()
    st["game_id"] = st["game_id"].astype(str)
    # skaters from penalties only: remove a pulled goalie's extra attacker
    st["home_pen"] = st["home_skaters"] - (1 - st["home_goalie"])
    st["away_pen"] = st["away_skaters"] - (1 - st["away_goalie"])
    return st.sort_values(["game_id", "start_s"]).reset_index(drop=True)


# ---------------------------------------------------------------- shootout attempts

def build_shootout_attempts(league: dict) -> pd.DataFrame:
    g = league["games"].set_index("game_id")
    ev = league["events"]
    so = ev[ev["event_type"].isin(["shootout", "penaltyshot"])].copy()
    side = np.where(so["is_home"] == 1, "home", "away")
    gi = g.loc[so["game_id"]]
    return pd.DataFrame({
        "league": gi["league"].values, "season": gi["season"].values, "season_type": gi["season_type"].values,
        "game_id": so["game_id"].values, "event_idx": so["event_idx"].values,
        "is_penalty_shot": (so["event_type"] == "penaltyshot").astype(int).values,
        "so_round": so["so_round"].values,
        "elapsed_s": so["elapsed_s"].values,
        "shooter_id": so["player_id"].values, "goalie_id": so["goalie_id"].values,
        "shooter_team": np.where(side == "home", gi["home_team"].values, gi["away_team"].values),
        "goalie_team": np.where(side == "home", gi["away_team"].values, gi["home_team"].values),
        "is_home": so["is_home"].values,
        "goal": so["is_goal"].astype(int).values,
    })


# ---------------------------------------------------------------- power-play intervals

def build_pp_intervals(league: dict) -> pd.DataFrame:
    games = league["games"].set_index("game_id")
    st = league["strength"]
    ev = league["events"]
    goals = ev[(ev["event_type"] == "goal") & (ev["period_type"] != "SO")][["game_id", "elapsed_s", "is_home"]]
    faceoffs = ev[ev["event_type"] == "faceoff"][["game_id", "elapsed_s"]]
    goals_by = {k: v for k, v in goals.groupby("game_id")}
    fo_by = {k: np.sort(v["elapsed_s"].values) for k, v in faceoffs.groupby("game_id")}

    rows = []
    for gid, grp in st.groupby("game_id", sort=False):
        game = games.loc[gid]
        recs = grp.to_dict("records")
        # advantaged side and strength label per interval
        segs = []
        for r in recs:
            hp, ap = r["home_pen"], r["away_pen"]
            if hp == ap:
                segs.append(None)
                continue
            side = "home" if hp > ap else "away"
            segs.append((side, f"{int(max(hp, ap))}v{int(min(hp, ap))}"))
        # merge consecutive identical segments, then group into spells
        merged = []
        for r, s in zip(recs, segs):
            if s is not None and merged and merged[-1]["key"] == s and merged[-1]["end_s"] == r["start_s"]:
                merged[-1]["end_s"] = r["end_s"]
            else:
                merged.append({"key": s, "start_s": r["start_s"], "end_s": r["end_s"]})
        g_goals = goals_by.get(gid)
        fo = fo_by.get(gid, np.array([]))
        spell_id, spell_start, prev = -1, None, None
        for i, m in enumerate(merged):
            if m["key"] is None:
                prev = None
                continue
            side, strength = m["key"]
            if prev is None or prev["key"][0] != side or prev["end_s"] != m["start_s"]:
                spell_id += 1
                spell_start = m["start_s"]
            prev = m
            a, b = m["start_s"], m["end_s"]
            gf = ga = 0
            first_gf = np.nan
            gf_at_end = ga_at_end = False
            if g_goals is not None:
                inside = g_goals[(g_goals["elapsed_s"] > a) & (g_goals["elapsed_s"] <= b)]
                for_side = inside["is_home"] == (1 if side == "home" else 0)
                gf, ga = int(for_side.sum()), int((~for_side).sum())
                if gf:
                    first_gf = inside.loc[for_side, "elapsed_s"].min() - spell_start
                gf_at_end = bool((inside.loc[for_side, "elapsed_s"] == b).any())
                ga_at_end = bool((inside.loc[~for_side, "elapsed_s"] == b).any())
            nxt = merged[i + 1]["key"] if i + 1 < len(merged) else "END"
            if gf_at_end:
                reason = "goal_for"
            elif ga_at_end:
                reason = "goal_against"
            elif nxt == "END" or b >= game["game_end_s"]:
                reason = "game_end"
            elif nxt is None or nxt[0] != side:
                reason = "advantage_over"
            else:
                reason = "strength_change"
            later_fo = fo[fo > spell_start]
            rows.append({
                "league": game["league"], "season": game["season"], "season_type": game["season_type"],
                "game_id": gid, "spell": spell_id, "team": game[f"{side}_team"],
                "opponent": game["away_team" if side == "home" else "home_team"], "is_home": int(side == "home"),
                "strength": strength, "start_s": a, "end_s": b, "duration_s": b - a,
                "spell_start_s": spell_start, "spell_offset_s": a - spell_start,
                "first_stoppage_s": (later_fo[0] - spell_start) if len(later_fo) else np.nan,
                "in_ot": int(a >= REG_END_S), "goals_for": gf, "goals_against": ga,
                "first_goal_for_s": first_gf, "end_reason": reason,
                "no_escape": game["no_escape"], "goals_complete": game["goals_complete"],
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- overtime intervals

def _score_at(goals: pd.DataFrame, t: float) -> tuple[int, int]:
    before = goals[goals["elapsed_s"] <= t]
    return int((before["is_home"] == 1).sum()), int((before["is_home"] == 0).sum())


def build_ot_intervals(league: dict) -> pd.DataFrame:
    games = league["games"]
    ot_games = games[(games["season_type"] != "playoff") & (games["game_end_s"] > REG_END_S)].set_index("game_id")
    st = league["strength"]
    st = st[st["game_id"].isin(ot_games.index) & (st["start_s"] >= REG_END_S)]
    ev = league["events"]
    goals = ev[(ev["event_type"] == "goal") & (ev["period_type"] != "SO") & ev["game_id"].isin(ot_games.index)]
    goals_by = {k: v for k, v in goals.groupby("game_id")}

    rows = []
    for gid, grp in st.groupby("game_id"):
        game = ot_games.loc[gid]
        g_goals = goals_by.get(gid, goals.iloc[0:0])
        hs, as_ = _score_at(g_goals, REG_END_S)
        pure = grp[(grp["home_pen"] == 3) & (grp["away_pen"] == 3) & (grp["home_goalie"] == 1) & (grp["away_goalie"] == 1)]
        ot_goals = g_goals[g_goals["elapsed_s"] > REG_END_S]
        in_pure = ot_goals[[((pure["start_s"] < t) & (t <= pure["end_s"])).any() for t in ot_goals["elapsed_s"]]] \
            if len(ot_goals) else ot_goals
        for side, opp in (("home", "away"), ("away", "home")):
            is_h = 1 if side == "home" else 0
            won = game[f"{side}_goals"] > game[f"{opp}_goals"]
            rows.append({
                "league": game["league"], "season": game["season"], "season_type": game["season_type"],
                "game_id": gid, "team": game[f"{side}_team"], "opponent": game[f"{opp}_team"], "is_home": is_h,
                "tied_after_reg": int(hs == as_),
                "ot_toi_s": grp["end_s"].max() - REG_END_S,
                "toi_3v3_s": float((pure["end_s"] - pure["start_s"]).sum()),
                "gf_3v3": int((in_pure["is_home"] == is_h).sum()), "ga_3v3": int((in_pure["is_home"] != is_h).sum()),
                "gf_ot": int((ot_goals["is_home"] == is_h).sum()), "ga_ot": int((ot_goals["is_home"] != is_h).sum()),
                "result": ("W" if won else "L") + ("_SO" if game["outcome"] == "SO" else "_OT" if game["outcome"] == "OT" else "_REG"),
                "goals_complete": game["goals_complete"],
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- overtime penalties

def build_ot_penalties(league: dict, pp: pd.DataFrame) -> pd.DataFrame:
    games = league["games"]
    games = games[games["season_type"] != "playoff"].set_index("game_id")
    pen = league["penalties"]
    pen = pen[pen["game_id"].isin(games.index) & (pen["called_s"] > REG_END_S)
              & (pen["penalty_class"].isin(["Minor", "Penalty shot"]) | (pen["ps_awarded"] == 1))].copy()
    ev = league["events"]
    goals = ev[(ev["event_type"] == "goal") & (ev["period_type"] != "SO") & ev["game_id"].isin(pen["game_id"])]
    goals_by = {k: v for k, v in goals.groupby("game_id")}
    st = league["strength"]
    st_by = {k: v for k, v in st[st["game_id"].isin(pen["game_id"])].groupby("game_id")}
    pp_by = {k: v for k, v in pp[pp["game_id"].isin(pen["game_id"])].groupby("game_id")}
    all_pen_by = {k: v for k, v in league["penalties"][league["penalties"]["game_id"].isin(pen["game_id"])].groupby("game_id")}

    rows = []
    for r in pen.itertuples(index=False):
        game = games.loc[r.game_id]
        side = "home" if r.is_home == 1 else "away"
        fouled = "away" if side == "home" else "home"
        t = r.called_s
        g_goals = goals_by.get(r.game_id, goals.iloc[0:0])
        hs, as_ = _score_at(g_goals[g_goals["elapsed_s"] < t], t)
        s = st_by[r.game_id]
        before = s[(s["start_s"] < t) & (t <= s["end_s"])]
        state = f"{int(before['home_pen'].iloc[0])}v{int(before['away_pen'].iloc[0])}" if len(before) else None
        others = all_pen_by[r.game_id]
        coincidental = int(((others["called_s"] == t) & (others["is_home"] != r.is_home)).any())
        spells = pp_by.get(r.game_id)
        spell = None
        if spells is not None:
            cand = spells[(spells["team"] == game[f"{fouled}_team"]) & (spells["spell_start_s"] >= t)
                          & (spells["spell_start_s"] <= t + 5)]
            if len(cand):
                spell = spells[(spells["spell"] == cand["spell"].iloc[0])]
        fouled_won = game[f"{fouled}_goals"] > game[f"{side}_goals"]
        rows.append({
            "league": game["league"], "season": game["season"], "season_type": game["season_type"],
            "game_id": r.game_id, "event_idx": r.event_idx,
            "called_s": t, "ot_elapsed_s": t - REG_END_S, "ot_remaining_s": REG_END_S + OT_S - t,
            "penalized_team": game[f"{side}_team"], "fouled_team": game[f"{fouled}_team"],
            "fouled_is_home": int(fouled == "home"),
            "penalty_class": r.penalty_class, "penalty_desc": r.penalty_desc,
            "ps_awarded": int(r.ps_awarded), "coincidental": coincidental,
            "state_at_call": state, "tied_at_call": int(hs == as_),
            "pp_created": int(spell is not None),
            "pp_duration_s": float(spell["duration_s"].sum()) if spell is not None else np.nan,
            "pp_goal": int(spell["goals_for"].sum() > 0) if spell is not None else 0,
            "sh_goal": int(spell["goals_against"].sum() > 0) if spell is not None else 0,
            "fouled_result": ("W" if fouled_won else "L") + "_" + game["outcome"],
            "goals_complete": game["goals_complete"],
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- summary

def data_summary(games, so, pp, ot, otp) -> pd.DataFrame:
    keys = ["league", "season", "season_type"]
    s = games.groupby(keys).agg(games=("game_id", "size"),
                                ot_games=("outcome", lambda x: (x == "OT").sum()),
                                so_games=("outcome", lambda x: (x == "SO").sum()))
    s["shootout_attempts"] = so[so["is_penalty_shot"] == 0].groupby(keys).size()
    s["penalty_shots"] = so[so["is_penalty_shot"] == 1].groupby(keys).size()
    reg = pp[(pp["in_ot"] == 0) & (pp["strength"] == "5v4")]
    s["pp_5v4_min"] = reg.groupby(keys)["duration_s"].sum() / 60
    s["pp_5v4_goals"] = reg.groupby(keys)["goals_for"].sum()
    ot4 = pp[(pp["in_ot"] == 1) & (pp["strength"] == "4v3")]
    s["pp_4v3_min"] = ot4.groupby(keys)["duration_s"].sum() / 60
    s["pp_4v3_goals"] = ot4.groupby(keys)["goals_for"].sum()
    home = ot[ot["is_home"] == 1]
    s["ot_3v3_min"] = home.groupby(keys)["toi_3v3_s"].sum() / 60
    s["ot_3v3_goals"] = ot.groupby(keys)["gf_3v3"].sum()
    s["ot_penalties"] = otp.groupby(keys).size()
    s["ot_minors_creating_pp"] = otp.groupby(keys)["pp_created"].sum()
    return s.fillna(0).round(1).reset_index()


def markdown_table(df: pd.DataFrame) -> str:
    def fmt(v):
        return f"{v:g}" if isinstance(v, float) else str(v)
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def main() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    out = {"shootout_attempts": [], "pp_intervals": [], "ot_intervals": [], "ot_penalties": []}
    all_games = []
    for name, loader in (("PWHL", load_pwhl), ("NHL", load_nhl)):
        league = loader()
        so = build_shootout_attempts(league)
        pp = build_pp_intervals(league)
        ot = build_ot_intervals(league)
        otp = build_ot_penalties(league, pp)
        for k, v in zip(out, (so, pp, ot, otp)):
            out[k].append(v)
        all_games.append(league["games"])
        print(f"{name}: {len(so):,} attempts, {len(pp):,} PP segments, {len(ot):,} OT team-rows, {len(otp):,} OT penalties")
    tables = {k: pd.concat(v, ignore_index=True) for k, v in out.items()}
    for k, v in tables.items():
        v.to_parquet(PROCESSED / f"{k}.parquet", index=False)
    summary = data_summary(pd.concat(all_games, ignore_index=True), tables["shootout_attempts"],
                           tables["pp_intervals"], tables["ot_intervals"], tables["ot_penalties"])
    summary.to_csv(TABLES / "data_summary.csv", index=False)
    (TABLES / "data_summary.md").write_text(
        "# Data summary\n\nCounts by league and season, computed by `src/build_tables.py`.\n\n"
        + markdown_table(summary) + "\n")
    print(f"\nWrote {', '.join(tables)} to data/processed and data_summary to outputs/tables")


if __name__ == "__main__":
    main()
