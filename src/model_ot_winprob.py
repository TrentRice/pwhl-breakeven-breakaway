"""Overtime win-probability model (PROJECT_PLAN.md §5.3 and §5.6, Step 8).

Usage (from the repo root, with the venv active):
    python -m src.model_ot_winprob            # full run
    python -m src.model_ot_winprob --quick    # fewer draws, for checking the pipeline

Pieces
1. Time-varying 3-on-3 scoring rate (piecewise-exponential / Poisson survival model). For each game
   tied after regulation, side (home/away) and OT minute bin k (0-4), using pure 3-on-3 time only
   (both goalies in, no penalties):
       goals ~ Poisson(exposure * exp(eta)),  eta = b0 + shape(k) [+ h * side]
   side = +1/2 for home, -1/2 for away (h = home-ice log rate ratio). The home term is a candidate
   like every other term: without it, home and away score at the same rate (50% home share).
   Two candidate shapes:
     - log-linear:  shape(k) = beta * x_k, x_k = (minute midpoint / 300 - 1/2)
     - per-minute:  shape(k) = c_k, c_0 = 0
   Stage 1 (NHL, large sample) fixes the shape; the per-minute version is kept only if it passes the
   rule fixed in advance (LOO gain >= 4 and > 2 SE on training data AND better holdout log score).
   Stage 2 (PWHL) uses NHL posteriors as widened priors for the shape and h; the level b0 has a wide
   prior around the NHL value. Shrunk team effects are a candidate under the same rule, with the
   PWHL holdout done leave-one-season-out (too few OT games for a single holdout season).
2. Shootout win probability S_A, computed exactly per posterior draw of the penalty-shot model
   (Poisson-binomial over 5 rounds, then sudden death), not by Monte Carlo.
3. W0(t): P(team A wins | tied 3-on-3, t seconds left), on the 1-second grid:
       W0(t) = pi_A (1 - exp(-(lam_A + lam_B) G(t))) + exp(-(lam_A + lam_B) G(t)) S_A
   with lam the average rates, g(u) the normalized shape (mean 1 over 0-300 s) and
   G(t) = integral of g from 300 - t to 300.

Outputs: data/processed/models/ot_{nhl,pwhl}.nc and outputs/tables/ot_*.csv.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
from scipy.special import expit, logsumexp
from scipy.stats import poisson

from src.model_penalty_shot import SEED, draws_of, fit, loo_gain

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"
MODELS = PROCESSED / "models"
TABLES = ROOT / "outputs" / "tables"

REG_END = 3600
OT_LEN = 300
BINS = 5
X_BIN = (np.arange(BINS) * 60 + 30) / OT_LEN - 0.5   # minute midpoints, centered

# Stage-1 (NHL) priors from outside knowledge: a 3-on-3 team scores very roughly 6 per 60.
NHL_PRIORS = {"b0_mu": float(np.log(6 / 3600)), "b0_sd": 1.0, "beta_mu": 0.0, "beta_sd": 1.0,
              "c_mu": np.zeros(BINS - 1), "c_sd": np.ones(BINS - 1), "h_mu": 0.0, "h_sd": 0.5}
WIDEN = {"b0": 0.5, "beta": 0.2, "c": 0.2, "h": 0.3}
TEAM_SIGMA_SD = 0.2

NHL_TRAIN = [f"{y}-{str(y + 1)[2:]}" for y in range(2015, 2023)]
NHL_TEST = ["2023-24", "2024-25", "2025-26"]


# ---------------------------------------------------------------- data

def tied_games() -> pd.DataFrame:
    ot = pd.read_parquet(PROCESSED / "ot_intervals.parquet")
    g = ot[(ot["is_home"] == 1) & (ot["season_type"] == "regular") & (ot["tied_after_reg"] == 1)
           & (ot["goals_complete"] == 1)]
    return g[["league", "season", "game_id", "team", "opponent", "result"]].rename(
        columns={"team": "home_team", "opponent": "away_team"})


def _strength(league: str) -> pd.DataFrame:
    s = pd.read_parquet(INTERIM / f"{league.lower()}_strength.parquet")
    s = s[s["start_s"] < REG_END + OT_LEN]
    s = s[s["end_s"] > REG_END]
    pure = (s["home_skaters"] == 3) & (s["away_skaters"] == 3) & (s["home_goalie"] == 1) & (s["away_goalie"] == 1)
    s = s[pure][["game_id", "start_s", "end_s"]].copy()
    s["game_id"] = s["game_id"].astype(str)
    return s


def _goals(league: str) -> pd.DataFrame:
    cols = ["game_id", "event_type", "elapsed_s", "is_home", "home_skaters", "away_skaters", "home_goalie", "away_goalie"]
    ev = pd.read_parquet(INTERIM / f"{league.lower()}_events.parquet", columns=cols)
    ev = ev[(ev["event_type"] == "goal") & (ev["elapsed_s"] > REG_END)]
    pure = (ev["home_skaters"] == 3) & (ev["away_skaters"] == 3) & (ev["home_goalie"] == 1) & (ev["away_goalie"] == 1)
    ev = ev[pure].copy()
    ev["game_id"] = ev["game_id"].astype(str)
    u = ev["elapsed_s"] - REG_END
    ev["bin"] = np.clip(np.ceil(u / 60).astype(int) - 1, 0, BINS - 1)
    return ev[["game_id", "bin", "is_home"]]


def load_bins() -> pd.DataFrame:
    """One row per (game, side, OT minute bin): pure 3-on-3 exposure and goals scored by that side."""
    games = tied_games()
    out = []
    for league, g in games.groupby("league"):
        s = _strength(league)
        s = s[s["game_id"].isin(g["game_id"])]
        rows = []
        for k in range(BINS):
            lo, hi = REG_END + 60 * k, REG_END + 60 * (k + 1)
            e = np.clip(np.minimum(s["end_s"], hi) - np.maximum(s["start_s"], lo), 0, None)
            rows.append(pd.DataFrame({"game_id": s["game_id"], "bin": k, "exposure_s": e}))
        exp_ = pd.concat(rows).groupby(["game_id", "bin"], as_index=False)["exposure_s"].sum()
        exp_ = exp_[exp_["exposure_s"] > 0]
        goals = _goals(league)
        goals = goals[goals["game_id"].isin(g["game_id"])]
        for side, is_h in (("home", 1), ("away", 0)):
            gs = goals[goals["is_home"] == is_h].groupby(["game_id", "bin"]).size().rename("goals")
            d = exp_.join(gs, on=["game_id", "bin"]).fillna({"goals": 0})
            d["side"] = 0.5 if side == "home" else -0.5
            d["team"] = d["game_id"].map(g.set_index("game_id")[f"{side}_team"])
            d["opponent"] = d["game_id"].map(g.set_index("game_id")["away_team" if side == "home" else "home_team"])
            out.append(d.assign(league=league))
    df = pd.concat(out, ignore_index=True)
    df["season"] = df["game_id"].map(games.set_index("game_id")["season"])
    df["goals"] = df["goals"].astype(int)
    return df


# ---------------------------------------------------------------- models

def ot_model(df: pd.DataFrame, priors: dict, shape: str = "linear", teams: bool = False, home: bool = False):
    """Poisson model for 3-on-3 goals by OT minute bin. shape = 'linear' (log-linear in OT time) or 'minute'.
    The home term is a candidate like any other (PROJECT_PLAN §5.6): without it, home and away score at the
    same rate (home share of OT goals = 50%)."""
    team_idx = pd.Index(sorted(set(df["team"]) | set(df["opponent"])))
    with pm.Model(coords={"team": team_idx}) as m:
        b0 = pm.Normal("b0", priors["b0_mu"], priors["b0_sd"])
        k = df["bin"].to_numpy()
        if shape == "linear":
            beta = pm.Normal("beta", priors["beta_mu"], priors["beta_sd"])
            shp = beta * X_BIN[k]
            pm.Deterministic("log_shape", beta * X_BIN)
        else:
            c = pm.Normal("c", priors["c_mu"], priors["c_sd"], shape=BINS - 1)
            cfull = pm.math.concatenate([[0.0], c])
            shp = cfull[k]
            pm.Deterministic("log_shape", cfull)
        eta = b0 + shp
        if home:
            h = pm.Normal("h", priors["h_mu"], priors["h_sd"])
            eta = eta + h * df["side"].to_numpy()
        if teams:
            s_a = pm.HalfNormal("sigma_att", TEAM_SIGMA_SD)
            s_d = pm.HalfNormal("sigma_def", TEAM_SIGMA_SD)
            a = pm.Deterministic("att", s_a * pm.Normal("z_att", 0, 1, dims="team"), dims="team")
            d = pm.Deterministic("dfn", s_d * pm.Normal("z_def", 0, 1, dims="team"), dims="team")
            eta = eta + a[team_idx.get_indexer(df["team"])] + d[team_idx.get_indexer(df["opponent"])]
        pm.Poisson("goals", mu=df["exposure_s"].to_numpy() * pm.math.exp(eta), observed=df["goals"].to_numpy())
    return m


def home_draws(idata) -> np.ndarray:
    """Home log rate ratio per draw; zeros when the model has no home term."""
    b0 = draws_of(idata, "b0")
    return draws_of(idata, "h") if "h" in idata["posterior"] else np.zeros_like(b0)


def stage2_priors(nhl_idata, shape: str, nhl_home_idata) -> dict:
    """PWHL priors from the NHL posterior (home-term prior from the NHL home-candidate fit)."""
    widen = lambda sd, extra: np.sqrt(sd ** 2 + extra ** 2)
    b0, h = draws_of(nhl_idata, "b0"), draws_of(nhl_home_idata, "h")
    p = {"b0_mu": float(b0.mean()), "b0_sd": float(widen(b0.std(), WIDEN["b0"])),
         "h_mu": float(h.mean()), "h_sd": float(widen(h.std(), WIDEN["h"])),
         "beta_mu": 0.0, "beta_sd": 1.0, "c_mu": np.zeros(BINS - 1), "c_sd": np.ones(BINS - 1)}
    if shape == "linear":
        b = draws_of(nhl_idata, "beta")
        p.update(beta_mu=float(b.mean()), beta_sd=float(widen(b.std(), WIDEN["beta"])))
    else:
        c = draws_of(nhl_idata, "c")
        p.update(c_mu=c.mean(0), c_sd=widen(c.std(0), WIDEN["c"]))
    return p


def diagnostics(idata, label: str) -> dict:
    names = [v for v in ("b0", "h", "beta", "c", "sigma_att", "sigma_def", "z_att", "z_def") if v in idata["posterior"]]
    s = az.summary(idata, var_names=names, ci_prob=0.9)
    div = int(idata["sample_stats"]["diverging"].values.sum())
    out = {"fit": label, "max_r_hat": float(s["r_hat"].max()), "min_ess_bulk": float(s["ess_bulk"].min()), "divergences": div}
    out["ok"] = out["max_r_hat"] < 1.01 and out["min_ess_bulk"] > 400 and div == 0
    return out


def predict_mu(idata, df: pd.DataFrame, train_teams: pd.Index | None = None) -> np.ndarray:
    rng = np.random.default_rng(SEED)
    b0, h, ls = draws_of(idata, "b0"), home_draws(idata), draws_of(idata, "log_shape")
    k = df["bin"].to_numpy()
    eta = b0[:, None] + ls[:, k] + h[:, None] * df["side"].to_numpy()
    if "att" in idata["posterior"]:
        for eff, sig, col in (("att", "sigma_att", "team"), ("dfn", "sigma_def", "opponent")):
            e, s = draws_of(idata, eff), draws_of(idata, sig)
            idx = train_teams.get_indexer(df[col])
            eta = eta + np.where(idx >= 0, e[:, np.clip(idx, 0, None)], rng.normal(0, 1, (len(s), len(df))) * s[:, None])
    return df["exposure_s"].to_numpy() * np.exp(eta)


def log_score_draws(mu: np.ndarray, y: np.ndarray) -> float:
    lp = poisson.logpmf(y[None, :], mu)
    return float(np.sum(logsumexp(lp, axis=0) - np.log(mu.shape[0])))


def constant_rate_score(train: pd.DataFrame, test: pd.DataFrame) -> float:
    """Baseline: one constant 3-on-3 rate per team-side (no time shape, no home term)."""
    rate = train["goals"].sum() / train["exposure_s"].sum()
    return float(np.sum(poisson.logpmf(test["goals"].to_numpy(), test["exposure_s"].to_numpy() * rate)))


# ---------------------------------------------------------------- win probability pieces

def shape_grid(idata) -> np.ndarray:
    """Normalized g(u) on the 1-second grid u = 0..299 for every draw: shape (S, 300), mean 1 per draw."""
    ls = draws_of(idata, "log_shape")
    k = np.minimum(np.arange(OT_LEN) // 60, BINS - 1)
    if "beta" in idata["posterior"]:
        u = (np.arange(OT_LEN) + 0.5) / OT_LEN - 0.5
        g = np.exp(draws_of(idata, "beta")[:, None] * u[None, :])
    else:
        g = np.exp(ls[:, k])
    return g / g.mean(1, keepdims=True)


def level_rates(idata) -> tuple[np.ndarray, np.ndarray]:
    """Average per-second 3-on-3 scoring rate for the home and away side (mean over OT), per draw."""
    b0, h, ls = draws_of(idata, "b0"), home_draws(idata), draws_of(idata, "log_shape")
    if "beta" in idata["posterior"]:
        u = (np.arange(OT_LEN) + 0.5) / OT_LEN - 0.5
        mean_shape = np.exp(draws_of(idata, "beta")[:, None] * u[None, :]).mean(1)
    else:
        mean_shape = np.exp(ls).mean(1)  # equal-width bins
    base = np.exp(b0) * mean_shape
    return base * np.exp(h / 2), base * np.exp(-h / 2)


def cumulative_G(g: np.ndarray) -> np.ndarray:
    """G(t) for t = 0..300 seconds left: integral of g over the last t seconds of OT. Shape (S, 301)."""
    tail = np.cumsum(g[:, ::-1], axis=1)              # sum of g over the last t seconds
    return np.concatenate([np.zeros((g.shape[0], 1)), tail], axis=1)


def w0_curve(lam_a: np.ndarray, lam_b: np.ndarray, G: np.ndarray, S_a: np.ndarray) -> np.ndarray:
    """W0(t) for t = 0..300 per draw. lam per second (averaged over OT), G from cumulative_G, S_a per draw."""
    tot = (lam_a + lam_b)[:, None]
    survive = np.exp(-tot * G)
    return (lam_a / (lam_a + lam_b))[:, None] * (1 - survive) + survive * S_a[:, None]


def poisson_binomial(p: np.ndarray) -> np.ndarray:
    """Distribution of the number of goals in independent attempts. p: (S, n) -> (S, n+1)."""
    dist = np.zeros((p.shape[0], p.shape[1] + 1))
    dist[:, 0] = 1.0
    for j in range(p.shape[1]):
        q = p[:, j:j + 1]
        dist[:, 1:] = dist[:, 1:] * (1 - q) + dist[:, :-1] * q
        dist[:, 0] = dist[:, 0] * (1 - q[:, 0])
    return dist


def shootout_win_prob(p_a: np.ndarray, p_b: np.ndarray) -> np.ndarray:
    """P(team A wins the shootout) per draw, exactly. p_a, p_b: (S, 5) success probabilities of each
    team's five shooters against the other team's goalie. Five rounds (stopping early never changes the
    winner), then sudden death using each team's average shooter (shooters may repeat)."""
    da, db = poisson_binomial(p_a), poisson_binomial(p_b)
    win = np.zeros(p_a.shape[0])
    for a in range(6):
        win += da[:, a] * db[:, :a].sum(1)
    tie = (da * db).sum(1)
    pa, pb = p_a.mean(1), p_b.mean(1)
    sd = pa * (1 - pb) / (pa * (1 - pb) + pb * (1 - pa))
    return win + tie * sd


# ---------------------------------------------------------------- driver

def rule(gain: float, se: float, holdout_better: bool) -> bool:
    """Decision rule fixed in advance (PROJECT_PLAN §5.6)."""
    return gain >= 4 and gain > 2 * se and holdout_better


def team_shootout_table(ps_idata, n_draws: int) -> pd.DataFrame:
    """Each team's P(win shootout) vs a league-average opponent, using its five highest-rated shooters
    and its most-used goalie (2025-26 rosters) from the penalty-shot posterior."""
    shooters = pd.read_csv(TABLES / "penalty_shot_pwhl_shooters.csv", dtype={"shooter_id": str})
    goalies = pd.read_csv(TABLES / "penalty_shot_pwhl_goalies.csv", dtype={"goalie_id": str})
    a = draws_of(ps_idata, "alpha")[:n_draws]
    s_eff = draws_of(ps_idata, "shooter_effect")[:n_draws]
    g_eff = draws_of(ps_idata, "goalie_effect")[:n_draws]
    s_ids = [str(x) for x in ps_idata["posterior"]["shooter"].values]
    g_ids = [str(x) for x in ps_idata["posterior"]["goalie"].values]
    avg = np.tile(expit(a)[:, None], (1, 5))  # league-average shooters vs league-average goalie
    rows = [{"team": "league average", "shooters": "", "goalie": "", "S_mean": 0.5, "S_lo90": 0.5, "S_hi90": 0.5}]
    for team, grp in shooters.groupby("team"):
        top = grp.sort_values("p_mean", ascending=False).head(5)
        gk = goalies[goalies["team"] == team].sort_values("attempts", ascending=False).head(1)
        si = [s_ids.index(x) for x in top["shooter_id"]]
        g = g_eff[:, g_ids.index(gk["goalie_id"].iloc[0])] if len(gk) else np.zeros(len(a))
        p_a = expit(a[:, None] + s_eff[:, si])                      # vs average goalie
        p_a = np.pad(p_a, ((0, 0), (0, 5 - p_a.shape[1])), mode="edge") if p_a.shape[1] < 5 else p_a
        p_b = expit(a[:, None] + g[:, None]) * np.ones((1, 5))      # average shooters vs this goalie
        S = shootout_win_prob(p_a, p_b)
        rows.append({"team": team, "shooters": ", ".join(top["name"].astype(str)),
                     "goalie": gk["name"].iloc[0] if len(gk) else "", "S_mean": S.mean(),
                     "S_lo90": np.quantile(S, 0.05), "S_hi90": np.quantile(S, 0.95)})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    draws, tune = (300, 300) if args.quick else (1000, 1000)
    MODELS.mkdir(parents=True, exist_ok=True)

    bins = load_bins()
    nhl, pwhl = bins[bins["league"] == "NHL"], bins[bins["league"] == "PWHL"]
    diags, holdout, choice, checks = [], [], [], []

    # 1. NHL shape: linear vs per-minute, chosen on training seasons, scored on the last three
    print("NHL shape holdout ...")
    tr, te = nhl[nhl["season"].isin(NHL_TRAIN)], nhl[nhl["season"].isin(NHL_TEST)]
    lin = fit(ot_model(tr, NHL_PRIORS, "linear"), draws, tune)
    mnt = fit(ot_model(tr, NHL_PRIORS, "minute"), draws, tune)
    diags += [diagnostics(lin, "NHL holdout linear"), diagnostics(mnt, "NHL holdout per-minute")]
    y = te["goals"].to_numpy()
    s_lin, s_mnt = log_score_draws(predict_mu(lin, te), y), log_score_draws(predict_mu(mnt, te), y)
    s_const = constant_rate_score(tr, te)
    holdout += [{"league": "NHL", "split": "train 2015-23, test 2023-26", "method": m, "log_score_total": v, "test_goals": int(y.sum())}
                for m, v in (("linear shape", s_lin), ("per-minute shape", s_mnt), ("constant rate", s_const))]
    gain, se = loo_gain(mnt, lin)
    use_minute = rule(gain, se, s_mnt > s_lin)
    choice.append({"candidate": "per-minute shape (vs linear)", "loo_gain": gain, "se": se,
                   "holdout_candidate": s_mnt, "holdout_base": s_lin, "keep": use_minute})
    shape = "minute" if use_minute else "linear"

    # 1b. NHL home term: candidate vs no home term (same rule)
    print("NHL home-term holdout ...")
    base_tr = mnt if use_minute else lin
    hm_tr = fit(ot_model(tr, NHL_PRIORS, shape, home=True), draws, tune)
    diags.append(diagnostics(hm_tr, "NHL holdout home"))
    s_base = s_mnt if use_minute else s_lin
    s_home = log_score_draws(predict_mu(hm_tr, te), y)
    holdout.append({"league": "NHL", "split": "train 2015-23, test 2023-26", "method": f"{shape} shape + home",
                    "log_score_total": s_home, "test_goals": int(y.sum())})
    gain, se = loo_gain(hm_tr, base_tr)
    nhl_home = rule(gain, se, s_home > s_base)
    choice.append({"candidate": "NHL home term (vs none)", "loo_gain": gain, "se": se,
                   "holdout_candidate": s_home, "holdout_base": s_base, "keep": nhl_home})

    # 2. NHL full fit with the chosen structure -> PWHL priors
    print(f"NHL full fit ({shape}, home: {nhl_home}) ...")
    nhl_full = fit(ot_model(nhl, NHL_PRIORS, shape, home=nhl_home), draws, tune)
    nhl_home_fit = nhl_full if nhl_home else fit(ot_model(nhl, NHL_PRIORS, shape, home=True), draws, tune)
    diags.append(diagnostics(nhl_full, "NHL full"))
    nhl_full.to_netcdf(MODELS / "ot_nhl.nc")
    priors2 = stage2_priors(nhl_full, shape, nhl_home_fit)

    # 3. PWHL: leave-one-season-out for the level and the candidates
    print("PWHL leave-one-season-out ...")
    tot = {"base": 0.0, "base + home": 0.0, "base + team effects": 0.0, "constant rate": 0.0}
    for season in sorted(pwhl["season"].unique()):
        tr, te = pwhl[pwhl["season"] != season], pwhl[pwhl["season"] == season]
        teams_tr = pd.Index(sorted(set(tr["team"]) | set(tr["opponent"])))
        base = fit(ot_model(tr, priors2, shape), draws, tune)
        hm = fit(ot_model(tr, priors2, shape, home=True), draws, tune)
        tm = fit(ot_model(tr, priors2, shape, teams=True), draws, tune)
        diags += [diagnostics(f, f"PWHL leave out {season} {n}") for n, f in (("base", base), ("home", hm), ("teams", tm))]
        y = te["goals"].to_numpy()
        sc = {"base": log_score_draws(predict_mu(base, te), y),
              "base + home": log_score_draws(predict_mu(hm, te), y),
              "base + team effects": log_score_draws(predict_mu(tm, te, teams_tr), y),
              "constant rate": constant_rate_score(tr, te)}
        for m, v in sc.items():
            tot[m] += v
            holdout.append({"league": "PWHL", "split": f"leave out {season}", "method": m, "log_score_total": v,
                            "test_goals": int(y.sum())})
    for m, v in tot.items():
        holdout.append({"league": "PWHL", "split": "leave-one-season-out total", "method": m, "log_score_total": v,
                        "test_goals": int(pwhl["goals"].sum())})

    # 4. PWHL final fit (all seasons); candidates judged with LOO on all PWHL data + leave-one-season-out
    print("PWHL final fit ...")
    final = fit(ot_model(pwhl, priors2, shape), draws, tune)
    hm_all = fit(ot_model(pwhl, priors2, shape, home=True), draws, tune)
    tm_all = fit(ot_model(pwhl, priors2, shape, teams=True), draws, tune)
    diags += [diagnostics(final, "PWHL final"), diagnostics(hm_all, "PWHL home (all data)"),
              diagnostics(tm_all, "PWHL team effects (all data)")]
    keep = {}
    for name, cand, key in (("PWHL home term (vs none)", hm_all, "base + home"),
                            ("PWHL team effects (vs base)", tm_all, "base + team effects")):
        gain, se = loo_gain(cand, final)
        keep[key] = rule(gain, se, tot[key] > tot["base"])
        choice.append({"candidate": name, "loo_gain": gain, "se": se, "holdout_candidate": tot[key],
                       "holdout_base": tot["base"], "keep": keep[key]})
    if keep["base + home"] or keep["base + team effects"]:
        final = fit(ot_model(pwhl, priors2, shape, home=keep["base + home"], teams=keep["base + team effects"]), draws, tune)
        diags.append(diagnostics(final, "PWHL final (with kept candidates)"))
    final.to_netcdf(MODELS / "ot_pwhl.nc")

    # 5. Summaries: level, home term, shape by minute
    rows = []
    for label, idata in (("NHL", nhl_full), ("PWHL", final)):
        lam_h, lam_a = level_rates(idata)
        g = shape_grid(idata)
        quants = {"both teams per60": (lam_h + lam_a) * 3600, "home share of goals": lam_h / (lam_h + lam_a),
                  "home log rate ratio h": home_draws(idata)}
        if "beta" in idata["posterior"]:
            quants["beta (log-rate change start to end of OT)"] = draws_of(idata, "beta")
        for k in range(BINS):
            quants[f"g(u) minute {k}-{k + 1}"] = g[:, 60 * k:60 * (k + 1)].mean(1)
        for q, d in quants.items():
            rows.append({"league": label, "quantity": q, "mean": d.mean(), "lo90": np.quantile(d, 0.05), "hi90": np.quantile(d, 0.95)})
    params = pd.DataFrame(rows)

    # 6. Shootout and W0(t) for league-average teams (PWHL)
    ps = az.from_netcdf(MODELS / "penalty_shot_pwhl.nc") if hasattr(az, "from_netcdf") else None
    if ps is None:
        import xarray as xr
        ps = xr.open_datatree(MODELS / "penalty_shot_pwhl.nc")
    n = min(draws_of(final, "b0").shape[0], draws_of(ps, "alpha").shape[0])
    so_tab = team_shootout_table(ps, n)
    lam_h, lam_a = (x[:n] for x in level_rates(final))
    G = cumulative_G(shape_grid(final)[:n])
    w_home = w0_curve(lam_h, lam_a, G, np.full(n, 0.5))
    t = np.arange(OT_LEN + 1)
    w0 = pd.DataFrame({"seconds_left": t, "home_W0_mean": w_home.mean(0), "home_W0_lo90": np.quantile(w_home, 0.05, 0),
                       "home_W0_hi90": np.quantile(w_home, 0.95, 0),
                       "P_goal_before_shootout_mean": (1 - np.exp(-(lam_h + lam_a)[:, None] * G)).mean(0)})

    # 7. Checks (not targets)
    games = tied_games()
    for label, idata, lg in (("NHL", nhl_full, "NHL"), ("PWHL", final, "PWHL")):
        lh, la = level_rates(idata)
        Gf = cumulative_G(shape_grid(idata))[:, -1]
        p_no_goal = np.exp(-(lh + la) * Gf)
        g_l = games[games["league"] == lg]
        checks.append({"check": f"{label}: P(tied game reaches shootout)", "observed": float((g_l["result"].str.endswith("SO")).mean()),
                       "predicted_mean": float(p_no_goal.mean()), "predicted_lo90": float(np.quantile(p_no_goal, 0.05)),
                       "predicted_hi90": float(np.quantile(p_no_goal, 0.95)),
                       "note": "model covers pure 3-on-3 only; OT power-play goals also prevent shootouts"})
        d = bins[bins["league"] == lg]
        mu = predict_mu(idata, d, pd.Index(sorted(set(d["team"]) | set(d["opponent"]))))
        for k in range(BINS):
            m = (d["bin"] == k).to_numpy()
            checks.append({"check": f"{label}: goals in OT minute {k}-{k + 1}", "observed": float(d.loc[m, "goals"].sum()),
                           "predicted_mean": float(mu[:, m].sum(1).mean()), "predicted_lo90": float(np.quantile(mu[:, m].sum(1), 0.05)),
                           "predicted_hi90": float(np.quantile(mu[:, m].sum(1), 0.95)), "note": "in-sample"})

    params.to_csv(TABLES / "ot_params.csv", index=False)
    pd.DataFrame(holdout).to_csv(TABLES / "ot_holdout.csv", index=False)
    pd.DataFrame(choice).to_csv(TABLES / "ot_model_choice.csv", index=False)
    pd.DataFrame(checks).to_csv(TABLES / "ot_checks.csv", index=False)
    pd.DataFrame(diags).to_csv(TABLES / "ot_diagnostics.csv", index=False)
    so_tab.to_csv(TABLES / "ot_shootout_by_team.csv", index=False)
    w0.to_csv(TABLES / "ot_w0_average_teams.csv", index=False)

    print("\nDiagnostics:\n", pd.DataFrame(diags).to_string(index=False))
    print("\nModel choice:\n", pd.DataFrame(choice).round(3).to_string(index=False))
    print("\nHoldout:\n", pd.DataFrame(holdout).round(2).to_string(index=False))
    print("\nParameters:\n", params.round(3).to_string(index=False))
    print("\nChecks:\n", pd.DataFrame(checks).round(3).to_string(index=False))
    print("\nShootout by team:\n", so_tab.round(3).to_string(index=False))
    print("\nW0 (home, average teams):\n", w0.iloc[[0, 30, 60, 120, 180, 240, 300]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
