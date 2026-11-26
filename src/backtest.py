"""Validation and robustness (PROJECT_PLAN.md §5.6, Step 10).

Usage (from the repo root, with the venv active):
    python -m src.backtest            # all three parts (about 10-15 minutes)
    python -m src.backtest --quick    # fewer draws, for checking the pipeline

1. NHL backtest, leave one season out. For each NHL season, refit the NHL power-play model on the
   other seasons (and the first-stoppage distribution from those seasons) and predict P(power-play goal)
   for that season's OT power plays, given the time left (PP cut to min(2:00, time left)). Every prediction
   is out-of-sample. Compared with a constant-rate baseline (training-season OT PP conversion).
   The 2:00+ vs under-2:00 split is a check, not a target.
2. PWHL case studies: the 9 regular-season OT penalties. Lineups are the players with recorded events for
   each team in that game; goalies are those in net when the penalty was called. Uses the final PWHL fits,
   so these are in-sample illustrations, not tests.
3. Sensitivity analysis on the decision: r +/-25%, No Escape early bump off, flat priors (all three PWHL
   models refit with the stage-1 outside-knowledge priors instead of NHL-informed ones), the three delta
   cases, the chosen-shooter uplift (regular shootout shooters' observed rate), and no season variation.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist

from src import decision as D
from src import model_ot_winprob as OT
from src import model_penalty_shot as PS
from src import model_powerplay as PP
from src.model_penalty_shot import SEED, draws_of, fit

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
INTERIM = ROOT / "data" / "interim"
TABLES = ROOT / "outputs" / "tables"
TIME_BINS = [(0, 60, "under 1:00"), (60, 120, "1:00-2:00"), (120, 301, "2:00 or more")]


# ---------------------------------------------------------------- 1. NHL leave-one-season-out backtest

def _stoppages(seasons: list[str]) -> np.ndarray:
    pp = pd.read_parquet(PROCESSED / "pp_intervals.parquet")
    x = pp[(pp["league"] == "NHL") & pp["season"].isin(seasons) & (pp["spell_offset_s"] == 0)]["first_stoppage_s"]
    q = (np.arange(D.N_STOPPAGE_QUANTILES) + 0.5) / D.N_STOPPAGE_QUANTILES
    return np.quantile(x.fillna(np.inf).to_numpy(), q)


def nhl_loso(draws: int, tune: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pieces = PP.load_pieces()
    nhl = pieces[pieces["league"] == "NHL"]
    otp = pd.read_parquet(PROCESSED / "ot_penalties.parquet")
    otp = otp[(otp["league"] == "NHL") & (otp["pp_created"] == 1) & (otp["goals_complete"] == 1)].copy()
    # two minors to the same team at the same second create one power play: count it once
    otp = otp.drop_duplicates(["game_id", "called_s", "penalized_team"])
    seasons = sorted(nhl["season"].unique())
    preds = []
    for s in seasons:
        train = [x for x in seasons if x != s]
        idata = fit(PP.nhl_model(PP.aggregate(nhl[nhl["season"].isin(train)], ["ot4v3", "early"]), PP.NHL_PRIORS), draws, tune)
        b0, r, rho, nu = (draws_of(idata, n) for n in ("b0", "r", "rho", "nu_per60"))
        post = SimpleNamespace(n=len(b0), stoppages=_stoppages(train), pp_rate=np.exp(b0) * r, rho=rho, nu=nu / 3600)
        pa, _, _ = D.pp_paths(post)
        test = otp[otp["season"] == s].copy()
        d = np.minimum(120, np.round(test["ot_remaining_s"]).astype(int).clip(1, 300))
        test["p_model"] = pa[:, d].mean(0)
        tr = otp[otp["season"].isin(train)]
        test["p_baseline"] = tr["pp_goal"].mean()
        preds.append(test)
        print(f"  {s}: {len(test)} OT power plays")
    pred = pd.concat(preds)
    y = pred["pp_goal"].to_numpy()

    def scores(p):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return float(np.mean((y - p) ** 2)), float(np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    summary = pd.DataFrame([{"method": m, "brier": scores(pred[c])[0], "log_score": scores(pred[c])[1], "n": len(pred)}
                            for m, c in (("model (leave-one-season-out)", "p_model"), ("constant rate", "p_baseline"))])
    cal = []
    for lo, hi, label in TIME_BINS:
        g = pred[(pred["ot_remaining_s"] >= lo) & (pred["ot_remaining_s"] < hi)]
        k, n = int(g["pp_goal"].sum()), len(g)
        cal.append({"time_left": label, "n": n, "observed": k / n, "observed_lo90": beta_dist.ppf(0.05, k, n - k + 1) if k else 0.0,
                    "observed_hi90": beta_dist.ppf(0.95, k + 1, n - k), "predicted": g["p_model"].mean(),
                    "constant_baseline": g["p_baseline"].mean()})
    return pred, summary, pd.DataFrame(cal)


# ---------------------------------------------------------------- 2. PWHL case studies

def case_studies(post: D.Posterior) -> pd.DataFrame:
    otp = pd.read_parquet(PROCESSED / "ot_penalties.parquet")
    otp = otp[(otp["league"] == "PWHL") & (otp["season_type"] == "regular")]
    ev = pd.read_parquet(INTERIM / "pwhl_events.parquet",
                         columns=["game_id", "event_type", "elapsed_s", "team_code", "player_id", "goalie_id"])
    paths = D.pp_paths(post)
    names = PS.pwhl_names()
    rows = []
    for r in otp.itertuples(index=False):
        g = ev[ev["game_id"] == str(r.game_id)]
        t = int(round(r.ot_remaining_s))

        def lineup(team):
            ids = g.loc[(g["team_code"] == team) & g["player_id"].notna(), "player_id"].astype(str).unique()
            tested = sorted([i for i in ids if i in post.shooter_ids],
                            key=lambda i: -post.shooter_effect[:, post.shooter_ids.index(i)].mean())
            return (tested + [None] * 5)[:5]

        def goalie_facing(team):  # goalie in net for the other team: last shot by `team` before the call
            sh = g[(g["team_code"] == team) & (g["event_type"] == "shot") & (g["elapsed_s"] <= r.called_s) & g["goalie_id"].notna()]
            return str(sh["goalie_id"].iloc[-1]) if len(sh) else None

        fouled, pen = r.fouled_team, r.penalized_team
        sh_f, sh_p = lineup(fouled), lineup(pen)
        g_p, g_f = goalie_facing(fouled), goalie_facing(pen)       # g_p: penalized team's goalie
        S = D.shootout_S(post, tuple(sh_f), g_f, tuple(sh_p), g_p)
        p = D.p_shot(post, sh_f[0], g_p)
        pstar = D.breakeven(post, S, paths)[:, t]
        diff = D.win_prob_ps(post, p, S)[:, t] - D.win_prob_pp(post, S, paths)[:, t]
        rec = "penalty shot" if diff.mean() > 0 else "power play"
        rows.append({
            "season": r.season, "game_id": r.game_id, "time_left": f"{t // 60}:{t % 60:02d}", "seconds_left": t,
            "fouled_team": fouled, "penalized_team": pen, "penalty": r.penalty_desc,
            "best_shooter": names.get(sh_f[0], "league average") if sh_f[0] else "league average",
            "goalie_faced": names.get(g_p, "league average") if g_p else "league average",
            "p_shot": float(p.mean()), "breakeven": float(pstar.mean()), "P_shot_better": float((p > pstar).mean()),
            "win_prob_gain_shot_minus_pp": float(diff.mean()), "model_recommends": rec,
            "what_happened": ("penalty shot awarded (breakaway foul)" if r.ps_awarded else
                              ("PP goal" if r.pp_goal else ("shorthanded goal" if r.sh_goal else "no goal on PP"))),
            "fouled_team_result": r.fouled_result,
        })
    return pd.DataFrame(rows).sort_values(["season", "seconds_left"], ascending=[True, False])


# ---------------------------------------------------------------- 3. Sensitivity analysis

def summarize(post: D.Posterior, case: str, p: np.ndarray, note: str = "") -> dict:
    paths = D.pp_paths(post)
    S = np.full(post.n, 0.5)
    pstar = D.breakeven(post, S, paths)
    better = (p[:, None] > pstar).mean(0)
    return {"case": case, "note": note, "shooter_p": float(p.mean()),
            "breakeven_2m30": float(pstar[:, 150].mean()), "breakeven_1m30": float(pstar[:, 90].mean()),
            "breakeven_1m00": float(pstar[:, 60].mean()),
            "P_shot_better_2m30": float(better[150]), "P_shot_better_1m30": float(better[90]),
            "P_shot_better_1m00": float(better[60]), "crossover_seconds": D.crossover_seconds(better)}


def regulars_rate(n: int, min_attempts: int = 7) -> tuple[np.ndarray, str]:
    so = pd.read_parquet(PROCESSED / "shootout_attempts.parquet")
    so = so[(so["league"] == "PWHL") & (so["season_type"] == "regular") & (so["is_penalty_shot"] == 0)]
    att = so.groupby("shooter_id")["goal"].transform("size")
    reg = so[att >= min_attempts]
    k, m = int(reg["goal"].sum()), len(reg)
    draws = np.random.default_rng(SEED).beta(k + 1, m - k + 1, n)
    return draws, f"regular shooters ({min_attempts}+ attempts): {k}/{m} = {k / m:.1%}"


def sensitivity(draws: int, tune: int) -> pd.DataFrame:
    base = D.load_posterior()
    rows = [summarize(base, "main (as fitted)", D.p_shot(base), "delta = 0, average shooter")]
    for case in ("nhl", "estimated"):
        rows.append(summarize(base, f"delta: {case}", D.p_shot(base, delta=case), D.DELTA_CASES[case]))
    for mult in (0.75, 1.25):
        post = D.load_posterior()
        post.pp_rate = post.pp_rate * mult
        rows.append(summarize(post, f"r x {mult}", D.p_shot(post), "4-on-3 multiplier scaled"))
    post = D.load_posterior()
    post.rho = np.ones_like(post.rho)
    rows.append(summarize(post, "No Escape bump off (rho = 1)", D.p_shot(post), "no early-PP scoring burst"))
    post = D.load_posterior(season_variation=False)
    rows.append(summarize(post, "no season variation", D.p_shot(post), "sigma_season = 0"))
    p_reg, note = regulars_rate(base.n)
    rows.append(summarize(base, "chosen-shooter uplift", p_reg, note))

    # Flat priors: refit all three PWHL models with the stage-1 outside-knowledge priors
    print("  flat-prior refits ...")
    so = PS.load_attempts()
    ps_flat = fit(PS.build_model(so[so["league"] == "PWHL"], PS.NHL_PRIORS), draws, tune)
    pieces = PP.load_pieces()
    pp_flat = fit(PP.pwhl_model(pieces[pieces["league"] == "PWHL"], PP.NHL_PRIORS), draws, tune)
    bins = OT.load_bins()
    ot_flat = fit(OT.ot_model(bins[bins["league"] == "PWHL"], OT.NHL_PRIORS, "linear"), draws, tune)
    post = D.load_posterior(ps=ps_flat, pp=pp_flat, ot=ot_flat)
    rows.append(summarize(post, "flat priors (no NHL borrowing)", D.p_shot(post),
                          "all three PWHL models refit with outside-knowledge priors"))
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    draws, tune = (300, 300) if args.quick else (1000, 1000)
    pd.set_option("display.width", 220)

    print("1. NHL leave-one-season-out backtest ...")
    pred, summary, cal = nhl_loso(draws, tune)
    pred[["season", "game_id", "ot_remaining_s", "pp_goal", "p_model", "p_baseline"]].to_csv(TABLES / "backtest_nhl_predictions.csv", index=False)
    summary.to_csv(TABLES / "backtest_nhl_summary.csv", index=False)
    cal.to_csv(TABLES / "backtest_nhl_calibration.csv", index=False)
    print(summary.round(4).to_string(index=False)); print(cal.round(3).to_string(index=False))

    print("\n2. PWHL case studies ...")
    cs = case_studies(D.load_posterior())
    cs.to_csv(TABLES / "case_studies.csv", index=False)
    print(cs.drop(columns=["game_id", "seconds_left", "penalty"]).round(3).to_string(index=False))

    print("\n3. Sensitivity analysis ...")
    sens = sensitivity(draws, tune)
    sens.to_csv(TABLES / "sensitivity.csv", index=False)
    print(sens.drop(columns=["note"]).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
