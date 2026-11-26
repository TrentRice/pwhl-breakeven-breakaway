"""Decision engine: penalty shot vs power play (RQ3; PROJECT_PLAN.md §5.4, Step 9).

Usage (from the repo root, with the venv active):
    python -m src.decision        # writes outputs/tables/decision_*.csv and the p*(t) draws

Combines posterior draws of the three fitted models (paired by draw index; the posteriors are
independent):
- penalty-shot model  -> p, the shooter's success probability on an in-game penalty shot (incl. delta)
- power-play model    -> 4-on-3 scoring rate (early = before the first stoppage, late = after) and the
                          shorthanded rate nu
- overtime model      -> W0(t), P(win | tied 3-on-3 with t seconds left), incl. the shootout

For a minor called with t seconds left in tied 3-on-3 OT (fouled team = A):
    P(win | PS) = p + (1 - p) * W0(t)                     (a miss returns to 3-on-3, same clock)
    P(win | PP) = P(A scores during the PP) + P(no goal during the PP) * W0(t - d),   d = min(120, t)
    p*(t)       = [P(win | PP) - W0(t)] / [1 - W0(t)]      (break-even penalty-shot success rate)
The power play is computed second by second (exact within each second), averaging over the observed
PWHL distribution of the first-stoppage time. A shorthanded goal ends the game as a loss.

Season-to-season variation (Step 8 decision): the league power-play rate for a new season gets an extra
random factor exp(eps), eps ~ Normal(0, sigma_season), with sigma_season estimated from the spread of
NHL season 5-on-4 rates after removing Poisson noise.

All quantities are arrays of posterior draws: shape (S,) or (S, 301) over t = 0..300 seconds left.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
from scipy.special import expit

from src.model_ot_winprob import cumulative_G, level_rates, shape_grid, shootout_win_prob
from src.model_penalty_shot import SEED, draws_of

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
MODELS = PROCESSED / "models"
TABLES = ROOT / "outputs" / "tables"

OT_LEN, PP_LEN = 300, 120
N_STOPPAGE_QUANTILES = 25
REPORT_TIMES = [270, 210, 150, 120, 90, 60, 30, 10, 5]   # seconds left


# ---------------------------------------------------------------- inputs

def season_sigma() -> float:
    """Spread of league 5-on-4 rates across NHL seasons (log scale), Poisson noise removed."""
    pp = pd.read_parquet(PROCESSED / "pp_intervals.parquet")
    pp = pp[(pp["league"] == "NHL") & (pp["goals_complete"] == 1) & (pp["strength"] == "5v4") & (pp["in_ot"] == 0)]
    s = pp.groupby("season").agg(goals=("goals_for", "sum"), secs=("duration_s", "sum"))
    log_rate = np.log(s["goals"] / s["secs"])
    var = log_rate.var(ddof=1) - (1 / s["goals"]).mean()
    return float(np.sqrt(max(var, 0.0)))


def stoppage_quantiles() -> np.ndarray:
    """First-stoppage times (seconds into the PP) for PWHL power plays under No Escape, as quantiles."""
    pp = pd.read_parquet(PROCESSED / "pp_intervals.parquet")
    x = pp[(pp["league"] == "PWHL") & (pp["season_type"] != "preseason") & (pp["no_escape"] == 1)
           & (pp["spell_offset_s"] == 0)]["first_stoppage_s"].fillna(np.inf).to_numpy()
    q = (np.arange(N_STOPPAGE_QUANTILES) + 0.5) / N_STOPPAGE_QUANTILES
    return np.quantile(x, q)


@dataclass
class Posterior:
    n: int
    alpha: np.ndarray
    delta: np.ndarray          # PWHL-updated in-game penalty-shot term (context only)
    delta_nhl: np.ndarray      # NHL in-game penalty-shot term (cautious bound)
    shooter_effect: np.ndarray
    sigma_s: np.ndarray
    goalie_effect: np.ndarray
    sigma_g: np.ndarray
    shooter_ids: list
    goalie_ids: list
    pp_rate: np.ndarray        # late 4-on-3 rate per second (league, incl. season variation)
    rho: np.ndarray
    nu: np.ndarray             # shorthanded rate per second
    lam: np.ndarray            # 3-on-3 rate per second per team (OT average)
    G: np.ndarray              # (S, 301) cumulative OT shape for t seconds left
    stoppages: np.ndarray
    sigma_season: float


def load_posterior(season_variation: bool = True, ps=None, pp=None, ot=None) -> Posterior:
    """Combine the fitted PWHL models. ps / pp / ot override the saved fits (used by sensitivity refits)."""
    ps = ps if ps is not None else az.from_netcdf(MODELS / "penalty_shot_pwhl.nc")
    pp = pp if pp is not None else az.from_netcdf(MODELS / "powerplay_pwhl.nc")
    ot = ot if ot is not None else az.from_netcdf(MODELS / "ot_pwhl.nc")
    ps_nhl = az.from_netcdf(MODELS / "penalty_shot_nhl.nc")
    n = min(draws_of(ps, "alpha").shape[0], draws_of(pp, "b0").shape[0], draws_of(ot, "b0").shape[0],
            draws_of(ps_nhl, "delta").shape[0])
    rng = np.random.default_rng(SEED)
    sig = season_sigma()
    eps = rng.normal(0, sig, n) if season_variation else np.zeros(n)
    lam_h, lam_a = level_rates(ot)
    assert np.allclose(lam_h, lam_a), "decision engine assumes no home term in the OT model"
    return Posterior(
        n=n,
        alpha=draws_of(ps, "alpha")[:n], delta=draws_of(ps, "delta")[:n], delta_nhl=draws_of(ps_nhl, "delta")[:n],
        shooter_effect=draws_of(ps, "shooter_effect")[:n], sigma_s=draws_of(ps, "sigma_s")[:n],
        goalie_effect=draws_of(ps, "goalie_effect")[:n], sigma_g=draws_of(ps, "sigma_g")[:n],
        shooter_ids=[str(x) for x in ps["posterior"]["shooter"].values],
        goalie_ids=[str(x) for x in ps["posterior"]["goalie"].values],
        pp_rate=(draws_of(pp, "rate5v4_per60")[:n] / 3600) * draws_of(pp, "r")[:n] * np.exp(eps),
        rho=draws_of(pp, "rho")[:n], nu=draws_of(pp, "nu_per60")[:n] / 3600,
        lam=lam_h[:n], G=cumulative_G(shape_grid(ot)[:n]),
        stoppages=stoppage_quantiles(), sigma_season=sig,
    )


# ---------------------------------------------------------------- building blocks

def _effect(post: Posterior, kind: str, pid: str | None) -> np.ndarray:
    """Posterior draws of a shooter's or goalie's effect; None = league average; unknown id = new player."""
    if pid is None:
        return np.zeros(post.n)
    ids = post.shooter_ids if kind == "shooter" else post.goalie_ids
    eff = post.shooter_effect if kind == "shooter" else post.goalie_effect
    if str(pid) in ids:
        return eff[:, ids.index(str(pid))]
    sigma = post.sigma_s if kind == "shooter" else post.sigma_g
    return np.random.default_rng(SEED).normal(0, sigma)


DELTA_CASES = {
    "zero": "Main: chosen shooter, shootout-like (delta = 0)",
    "nhl": "Cautious: full historical penalty-shot gap (NHL delta)",
    "estimated": "Context: PWHL-updated delta (9 PWHL penalty shots)",
}


def delta_draws(post: Posterior, case: str) -> np.ndarray:
    """In-game penalty-shot adjustment for a pilot-rule shot. See DELTA_CASES; 'zero' is the main case
    because under the pilot rule the team picks its shooter (decided before Step 10)."""
    if case == "zero":
        return np.zeros(post.n)
    if case == "nhl":
        return post.delta_nhl
    if case == "estimated":
        return post.delta
    raise ValueError(f"unknown delta case {case!r}")


def p_shot(post: Posterior, shooter_id: str | None = None, goalie_id: str | None = None,
           delta: str = "zero") -> np.ndarray:
    """Pilot-rule penalty-shot success probability for a shooter vs a goalie (None = league average)."""
    return expit(post.alpha + delta_draws(post, delta) + _effect(post, "shooter", shooter_id) + _effect(post, "goalie", goalie_id))


def p_shootout_attempt(post: Posterior, shooter_id: str | None, goalie_id: str | None) -> np.ndarray:
    return expit(post.alpha + _effect(post, "shooter", shooter_id) + _effect(post, "goalie", goalie_id))


def shootout_S(post: Posterior, shooters_a=(None,) * 5, goalie_a=None, shooters_b=(None,) * 5, goalie_b=None) -> np.ndarray:
    """P(team A wins the shootout): A's five shooters vs B's goalie and vice versa."""
    pa = np.column_stack([p_shootout_attempt(post, s, goalie_b) for s in shooters_a])
    pb = np.column_stack([p_shootout_attempt(post, s, goalie_a) for s in shooters_b])
    return shootout_win_prob(pa, pb)


def w0(post: Posterior, S_a: np.ndarray) -> np.ndarray:
    """W0(t) for t = 0..300 seconds left, equal 3-on-3 rates for both teams. Shape (S, 301)."""
    survive = np.exp(-2 * post.lam[:, None] * post.G)
    return 0.5 * (1 - survive) + survive * S_a[:, None]


def pp_paths(post: Posterior) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For PP lengths d = 0..120 s: P(fouled team scores by d), P(shorthanded goal by d), P(no goal by d).
    Exact within each second; averaged over the first-stoppage distribution. Each shape (S, 121)."""
    S, F = post.n, len(post.stoppages)
    mu_late = post.pp_rate[:, None] * np.ones((1, F))
    mu_early = mu_late * post.rho[:, None]
    nu = post.nu[:, None] * np.ones((1, F))
    surv = np.ones((S, F))
    pa, pb = np.zeros((S, F, PP_LEN + 1)), np.zeros((S, F, PP_LEN + 1))
    for tau in range(PP_LEN):
        mu = np.where(tau < post.stoppages[None, :], mu_early, mu_late)
        tot = mu + nu
        p_any = 1 - np.exp(-tot)
        pa[:, :, tau + 1] = pa[:, :, tau] + surv * p_any * mu / tot
        pb[:, :, tau + 1] = pb[:, :, tau] + surv * p_any * nu / tot
        surv = surv * np.exp(-tot)
    pa, pb = pa.mean(1), pb.mean(1)
    return pa, pb, 1 - pa - pb


def win_prob_pp(post: Posterior, S_a: np.ndarray, paths=None) -> np.ndarray:
    """P(win | take the power play) for t = 0..300 seconds left. Shape (S, 301)."""
    pa, _, none = paths if paths is not None else pp_paths(post)
    W = w0(post, S_a)
    t = np.arange(OT_LEN + 1)
    d = np.minimum(PP_LEN, t)
    return pa[:, d] + none[:, d] * W[:, t - d]


def win_prob_ps(post: Posterior, p: np.ndarray, S_a: np.ndarray) -> np.ndarray:
    """P(win | take the penalty shot) for t = 0..300 seconds left. Shape (S, 301)."""
    W = w0(post, S_a)
    return p[:, None] + (1 - p[:, None]) * W


def breakeven(post: Posterior, S_a: np.ndarray, paths=None) -> np.ndarray:
    """p*(t): penalty-shot success rate at which both options are equally good. Shape (S, 301)."""
    W = w0(post, S_a)
    return (win_prob_pp(post, S_a, paths) - W) / (1 - W)


def p_shot_better(post: Posterior, p: np.ndarray, S_a: np.ndarray, paths=None) -> np.ndarray:
    """P(penalty shot is the better choice) at each t: share of draws with p > p*(t). Shape (301,)."""
    return (p[:, None] > breakeven(post, S_a, paths)).mean(0)


def crossover_seconds(prob_better: np.ndarray) -> int | None:
    """Most seconds left at which the penalty shot is more likely than not the better choice
    (None if never). Scans from 5:00 down."""
    ok = np.where(prob_better[1:] >= 0.5)[0]
    return int(ok.max() + 1) if len(ok) else None


# ---------------------------------------------------------------- 2026-27 rosters (provided by hand)

ROSTERS = ROOT / "data" / "raw" / "pwhl" / "rosters_2026_27.csv"


def _norm(name: str) -> str:
    import re
    import unicodedata
    name = re.sub(r'["“”][^"“”]*["“”]', " ", str(name))   # drop quoted nicknames: Caroline "KK" Harvey
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    s = "".join(ch if ch.isalnum() or ch == " " else "" for ch in s.replace("-", " "))  # drop accents/punctuation
    return " ".join(s.split())


def load_rosters() -> pd.DataFrame:
    """2026-27 rosters (team, name, position, status) matched to PWHL player ids by name.
    Players with no PWHL record get player_id NaN (treated as league average)."""
    from src.model_penalty_shot import pwhl_names
    r = pd.read_csv(ROSTERS)
    by_name, by_initial = {}, {}
    for pid, name in pwhl_names().items():
        n = _norm(name)
        by_name.setdefault(n, set()).add(pid)
        parts = n.split()
        if len(parts) >= 2:
            by_initial.setdefault((parts[0][0], parts[-1]), set()).add(pid)

    def match(n):
        ids = by_name.get(_norm(n), set())
        if not ids:  # fallback for nicknames (Abigail / Abby): first initial + last name, only if unique
            parts = _norm(n).split()
            ids = by_initial.get((parts[0][0], parts[-1]), set()) if len(parts) >= 2 else set()
        return next(iter(ids)) if len(ids) == 1 else (None if not ids else "AMBIGUOUS")
    r["player_id"] = r["name"].map(match)
    return r


def team_lineup(post: Posterior, roster: pd.DataFrame, team: str, n_shooters: int = 3) -> tuple[pd.DataFrame, str | None]:
    """A team's top shooters and its main goalie (most shootout attempts faced; None = league average).

    Shooters: players with PWHL shootout history first, ranked by posterior effect (coaches pick from
    players they have used; Step 7 found rarely used shooters convert less, so an untested player is not
    assumed to beat a tested one). If fewer than n have history, untested forwards (then defenders) fill
    in at league average."""
    t = roster[roster["team"] == team]
    skaters = t[t["position"] != "G"].copy()
    eff = {pid: post.shooter_effect[:, post.shooter_ids.index(pid)].mean() for pid in skaters["player_id"].dropna()
           if pid in post.shooter_ids}
    skaters["effect"] = skaters["player_id"].map(eff)
    skaters["history"] = skaters["effect"].notna()
    skaters["effect"] = skaters["effect"].fillna(0.0)
    skaters["is_forward"] = skaters["position"].eq("F")
    top = skaters.sort_values(["history", "effect", "is_forward"], ascending=False).head(n_shooters)
    goalies = t[t["position"] == "G"]
    gk = pd.read_csv(TABLES / "penalty_shot_pwhl_goalies.csv", dtype={"goalie_id": str}).set_index("goalie_id")["attempts"]
    known = goalies[goalies["player_id"].isin(gk.index)]
    main_goalie = known.assign(n=known["player_id"].map(gk)).sort_values("n", ascending=False)["player_id"].iloc[0] if len(known) else None
    return top, main_goalie


# ---------------------------------------------------------------- outputs

def q(x: np.ndarray, axis=0) -> tuple:
    return x.mean(axis), np.quantile(x, 0.05, axis), np.quantile(x, 0.95, axis)


HEATMAP_TIMES = (150, 105)   # 2:30 left (close-call zone) and 1:45 left (around the crossover)


def lineups(post: Posterior, roster: pd.DataFrame) -> dict:
    """Per team: top 5 shooters (ids or None for untested players) and main goalie id (or None)."""
    out = {}
    for team in sorted(roster["team"].unique()):
        top5, goalie = team_lineup(post, roster, team, n_shooters=5)
        ids = [pid if h else None for pid, h in zip(top5["player_id"], top5["history"])]
        out[team] = {"shooters": ids, "names": list(top5["name"]), "goalie": goalie}
    return out


def ot_penalty_inputs(games_per_team: float) -> dict:
    """How often a team is fouled in tied 3-on-3 OT, and when (seconds left), from the data:
    frequency from PWHL regular seasons, timing from NHL OT power plays (PWHL has too few)."""
    otp = pd.read_parquet(PROCESSED / "ot_penalties.parquet")
    ot = pd.read_parquet(PROCESSED / "ot_intervals.parquet")
    g = pd.read_parquet(ROOT / "data" / "interim" / "pwhl_games.parquet")
    reg_games = int((g["season_type"] == "regular").sum())
    pw_ot = ot[(ot["league"] == "PWHL") & (ot["season_type"] == "regular") & (ot["is_home"] == 1) & (ot["tied_after_reg"] == 1)]
    pw_pens = otp[(otp["league"] == "PWHL") & (otp["season_type"] == "regular") & (otp["pp_created"] == 1)]
    nhl = otp[(otp["league"] == "NHL") & (otp["pp_created"] == 1) & (otp["goals_complete"] == 1)]
    t = np.clip(np.round(nhl["ot_remaining_s"]).astype(int), 1, OT_LEN)
    weights = np.bincount(t, minlength=OT_LEN + 1).astype(float)
    p_ot = len(pw_ot) / reg_games
    pens_per_ot_game = len(pw_pens) / len(pw_ot)
    return {"p_game_reaches_ot": p_ot, "ot_pp_minors_per_ot_game": pens_per_ot_game,
            "fouled_ot_minors_per_team_season": games_per_team * p_ot * pens_per_ot_game / 2,
            "time_weights": weights / weights.sum(), "n_pwhl_ot_games": len(pw_ot), "n_pwhl_ot_minors": len(pw_pens),
            "n_nhl_timing_penalties": len(nhl)}


def main() -> None:
    post = load_posterior()
    paths = pp_paths(post)
    S_avg = np.full(post.n, 0.5)
    t = np.arange(OT_LEN + 1)

    # 1. League-average teams: the break-even curve (does not depend on delta)
    W = w0(post, S_avg)
    wpp = win_prob_pp(post, S_avg, paths)
    pstar = breakeven(post, S_avg, paths)
    curve = pd.DataFrame({"seconds_left": t})
    for name, arr in (("pstar", pstar), ("win_pp", wpp), ("w0", W)):
        m, lo, hi = q(arr)
        curve[f"{name}_mean"], curve[f"{name}_lo90"], curve[f"{name}_hi90"] = m, lo, hi
    for case in DELTA_CASES:
        curve[f"p_shot_better_avg_shooter_{case}"] = p_shot_better(post, p_shot(post, delta=case), S_avg, paths)
    curve.to_csv(TABLES / "decision_breakeven_average.csv", index=False)
    np.savez_compressed(MODELS / "decision_pstar_average.npz", pstar=pstar.astype(np.float32))

    # 2. 2026-27 rosters: each team's best shooter under each delta case (vs league-average goalie and opponent)
    roster = load_rosters()
    lu = lineups(post, roster)
    team_S = {tm: shootout_S(post, tuple(v["shooters"]), v["goalie"]) for tm, v in lu.items()}
    rows = []
    for case, label in DELTA_CASES.items():
        entries = [("league average", "average shooter", None, S_avg)]
        entries += [(tm, v["names"][0], v["shooters"][0], team_S[tm]) for tm, v in lu.items()]
        for team, name, sid, S in entries:
            p = p_shot(post, sid, delta=case)
            better = p_shot_better(post, p, S, paths)
            m, lo, hi = q(p)
            row = {"delta_case": case, "case_label": label, "team": team, "shooter": name, "p_shot_mean": m,
                   "p_shot_lo90": lo, "p_shot_hi90": hi, "shootout_S_mean": float(S.mean()),
                   "crossover_seconds_left": crossover_seconds(better)}
            row.update({f"P_shot_better_{s}s": better[s] for s in REPORT_TIMES})
            rows.append(row)
    cases = pd.DataFrame(rows)
    cases.to_csv(TABLES / "decision_delta_cases.csv", index=False)
    cases[cases["delta_case"] == "zero"].drop(columns=["delta_case", "case_label"]) \
        .to_csv(TABLES / "decision_team_best_shooters.csv", index=False)

    # 3. Heatmap: each team's top 3 shooters vs each team's main goalie (evenly matched teams, main case)
    pstar_h = {s: pstar[:, s] for s in HEATMAP_TIMES}
    names = dict(zip(roster["player_id"], roster["name"]))
    mrows = []
    for tm, v in lu.items():
        for rank, (sid, sname) in enumerate(zip(v["shooters"][:3], v["names"][:3]), 1):
            for gt, gv in lu.items():
                p = p_shot(post, sid, gv["goalie"])
                row = {"team": tm, "rank": rank, "shooter": sname, "goalie_team": gt,
                       "goalie": names.get(gv["goalie"], "league average") if gv["goalie"] else "league average",
                       "own_team": int(gt == tm), "p_shot_mean": float(p.mean())}
                row.update({f"P_shot_better_{s}s": float((p > pstar_h[s]).mean()) for s in HEATMAP_TIMES})
                mrows.append(row)
    pd.DataFrame(mrows).to_csv(TABLES / "decision_matchups.csv", index=False)

    # 4. Team cards: best shooter vs every opponent (real matchup: shooter vs opponent's goalie,
    #    shootout between the two teams' lineups), plus vs a league-average goalie and opponent
    card_rows, curves = [], []
    for tm, v in lu.items():
        opps = {"league average": (None, team_S[tm])}   # league-average goalie and opponent lineup
        for ot_, ov in lu.items():
            if ot_ != tm:
                opps[ot_] = (ov["goalie"], shootout_S(post, tuple(v["shooters"]), v["goalie"], tuple(ov["shooters"]), ov["goalie"]))
        cross, at150 = {}, {}
        for opp, (goalie, S) in opps.items():
            better = p_shot_better(post, p_shot(post, v["shooters"][0], goalie), S, paths)
            cross[opp] = crossover_seconds(better)
            at150[opp] = float(better[150])
            curves.append(pd.DataFrame({"team": tm, "opponent": opp, "seconds_left": t, "P_shot_better": better}))
        opp_only = {k: c for k, c in cross.items() if k != "league average"}
        lo_team = min(opp_only, key=lambda k: opp_only[k] or 0)
        hi_team = max(opp_only, key=lambda k: opp_only[k] or 0)
        card_rows.append({"team": tm, "best_shooter": v["names"][0], "top3": ", ".join(v["names"][:3]),
                          "main_goalie": names.get(v["goalie"], "league average") if v["goalie"] else "league average",
                          "shootout_S_vs_average": float(team_S[tm].mean()),
                          "crossover_vs_average": cross["league average"],
                          "P_shot_better_2m30_vs_average": at150["league average"],
                          "P_shot_better_2m30_min": min(v_ for k, v_ in at150.items() if k != "league average"),
                          "P_shot_better_2m30_max": max(v_ for k, v_ in at150.items() if k != "league average"),
                          "crossover_min": opp_only[lo_team], "crossover_min_opponent": lo_team,
                          "crossover_max": opp_only[hi_team], "crossover_max_opponent": hi_team})
    pd.DataFrame(card_rows).to_csv(TABLES / "decision_team_cards.csv", index=False)
    pd.concat(curves).to_csv(TABLES / "decision_team_card_curves.csv", index=False)

    # 5. Cost of choosing wrong, by time left (league-average teams; average shooter and best 2026-27 shooter)
    best_team = max(lu, key=lambda k: p_shot(post, lu[k]["shooters"][0]).mean())
    cost_rows = []
    for who, sid in (("average shooter", None), (f"best shooter ({lu[best_team]['names'][0]})", lu[best_team]["shooters"][0])):
        diff = win_prob_ps(post, p_shot(post, sid), S_avg) - wpp      # > 0: penalty shot better
        m, lo, hi = q(diff)
        cost_rows.append(pd.DataFrame({"shooter": who, "seconds_left": t, "ps_minus_pp_mean": m, "lo90": lo, "hi90": hi,
                                       "cost_of_wrong_choice": np.abs(m)}))
    cost = pd.concat(cost_rows)
    cost.to_csv(TABLES / "decision_cost_by_time.csv", index=False)

    # 6. League impact (RQ4): always PP vs always PS vs optimal, per team and league-wide
    sched = pd.read_json(ROOT / "data" / "raw" / "pwhl" / "schedules" / "season_11.json")["SiteKit"]["Schedule"]
    teams_sched = pd.Series([g["home_team_code"] for g in sched] + [g["visiting_team_code"] for g in sched])
    games_per_team = float(teams_sched.value_counts().mean())
    inp = ot_penalty_inputs(games_per_team)
    wts = inp["time_weights"]
    imp = []
    for tm, v in list(lu.items()) + [("league average", {"shooters": [None], "names": ["average shooter"]})]:
        S = team_S.get(tm, S_avg)
        wps_t = win_prob_ps(post, p_shot(post, v["shooters"][0]), S)
        wpp_t = win_prob_pp(post, S, paths)
        choose_ps = wps_t.mean(0) > wpp_t.mean(0)              # decide on posterior-mean win probability
        opt = np.where(choose_ps[None, :], wps_t, wpp_t)
        g_pp = (opt - wpp_t) @ wts                              # per penalty, per draw
        g_ps = (opt - wps_t) @ wts
        per_season = inp["fouled_ot_minors_per_team_season"]
        imp.append({"team": tm, "shooter": v["names"][0], "share_of_penalties_where_PS_is_optimal": float(choose_ps @ wts),
                    "gain_vs_always_pp_per_penalty_mean": g_pp.mean(), "gain_vs_always_pp_lo90": np.quantile(g_pp, 0.05),
                    "gain_vs_always_pp_hi90": np.quantile(g_pp, 0.95),
                    "gain_vs_always_ps_per_penalty_mean": g_ps.mean(),
                    "fouled_ot_minors_per_season": per_season,
                    "standings_points_per_season_vs_always_pp": g_pp.mean() * per_season,
                    "standings_points_per_season_vs_always_ps": g_ps.mean() * per_season})
    impact = pd.DataFrame(imp)
    league_pens = inp["fouled_ot_minors_per_team_season"] * len(lu)
    impact.to_csv(TABLES / "decision_league_impact.csv", index=False)
    pd.DataFrame([{k: v for k, v in inp.items() if k != "time_weights"} | {
        "games_per_team_2026_27": games_per_team, "teams": len(lu), "league_ot_minors_per_season": league_pens,
        "expected_results_changed_per_season_vs_always_pp": impact.loc[impact["team"] != "league average", "gain_vs_always_pp_per_penalty_mean"].mean() * league_pens,
    }]).to_csv(TABLES / "decision_league_impact_inputs.csv", index=False)

    # 7. Phases, components, checks
    phases = []
    for label, lo, hi in (("early (3:00-5:00 left)", 180, 300), ("middle (2:00-3:00 left)", 120, 180), ("late (under 2:00)", 1, 119)):
        sl = slice(lo, hi + 1)
        m, l, h = q(pstar[:, sl].mean(1))
        row = {"phase": label, "pstar_mean": m, "pstar_lo90": l, "pstar_hi90": h}
        row.update({f"P_shot_better_avg_shooter_{c}": float(curve[f"p_shot_better_avg_shooter_{c}"].iloc[sl].mean())
                    for c in DELTA_CASES})
        phases.append(row)
    pa, pb, none = paths
    comp = pd.DataFrame({"pp_length_s": [30, 60, 90, 120],
                         "P_pp_goal_mean": [pa[:, d].mean() for d in (30, 60, 90, 120)],
                         "P_sh_goal_mean": [pb[:, d].mean() for d in (30, 60, 90, 120)]})
    pd.DataFrame(phases).to_csv(TABLES / "decision_phases.csv", index=False)
    comp.to_csv(TABLES / "decision_pp_components.csv", index=False)
    chk = [
        {"check": "p*(t) at 5 seconds left is small (PS should almost always win)", "value": float(pstar[:, 5].mean())},
        {"check": "p*(t) is 0 at 0 seconds left", "value": float(np.abs(pstar[:, 0]).max())},
        {"check": "P(PP goal) for a full 2:00 PP (compare NHL 38% with 2:00+ left; PWHL 3 of 6)", "value": float(pa[:, 120].mean())},
        {"check": "sigma_season used for PP rate (log scale)", "value": post.sigma_season},
    ]
    for case in DELTA_CASES:
        chk.append({"check": f"average shooter pilot-rule penalty-shot p ({case})", "value": float(p_shot(post, delta=case).mean())})
    checks = pd.DataFrame(chk)
    checks.to_csv(TABLES / "decision_checks.csv", index=False)

    pd.set_option("display.width", 200)
    print("Team cards:\n", pd.DataFrame(card_rows).round(3).to_string(index=False))
    print("\nLeague impact:\n", impact.round(4).to_string(index=False))
    print("\nImpact inputs:\n", pd.read_csv(TABLES / "decision_league_impact_inputs.csv").round(3).T.to_string())
    print("\nCost of wrong choice (average shooter):\n",
          cost[cost["shooter"] == "average shooter"].iloc[[300, 180, 120, 105, 90, 60, 30, 10]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
