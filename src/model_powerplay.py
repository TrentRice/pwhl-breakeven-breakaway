"""Power-play scoring model (RQ2; PROJECT_PLAN.md §5.2 and §5.6, Step 8).

Usage (from the repo root, with the venv active):
    python -m src.model_powerplay            # full run
    python -m src.model_powerplay --quick    # fewer draws, for checking the pipeline

Technique: Poisson rate regression with an exposure offset (a piecewise-exponential survival model).
Every power-play segment is split into an "early" piece (from the start of the power play up to the
first stoppage, when a tired penalty-killing unit is guaranteed relief) and a "late" piece. For a piece
with `exposure` seconds:
    goals ~ Poisson(exposure * rate)
    log(rate) = b0 + log(r) * [4-on-3 in OT] + log(rho) * early [+ gamma * early * no_escape]
                [+ u_team (power play) + v_opp (penalty kill)]
Shorthanded goals during OT 4-on-3 time:  goals_against ~ Poisson(exposure * nu).

Two stages:
1. NHL (regulation 5-on-4 + OT 4-on-3, games with complete goals) with wide priors from outside
   knowledge: 5-on-4 scoring of roughly 6.5 goals per 60, no assumed direction for r or rho.
   Cells are aggregated (exact for a Poisson model with no random effects).
2. PWHL (regular season + playoffs) with the NHL posterior as prior for r, rho and nu, widened.
   b0 gets a wide prior around the NHL value (the leagues score at different rates).

Candidate terms (each tested once, against the base model, with the rule fixed in advance:
LOO gain >= 4 and > 2 SE on training data AND better holdout log score):
- team effects (power-play team and penalty-killing team, shrunk);
- No Escape change (gamma: how much the early/late ratio changed from 2024 to 2024-25 on).

Validation (time-based):
- PWHL: train 2024 regular + playoffs and 2024-25 regular + 2025 playoffs; test 2025-26 + 2026 playoffs.
- NHL r: train 2015-16 to 2022-23; test OT 4-on-3 goals in 2023-24 to 2025-26.
Metrics: Poisson log score per piece vs (a) league rate per strength, (b) raw team rates.

Outputs: data/processed/models/powerplay_{nhl,pwhl}.nc and outputs/tables/powerplay_*.csv.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
from scipy.special import logsumexp
from scipy.stats import poisson

from src.model_penalty_shot import SEED, draws_of, fit, loo_gain

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
MODELS = PROCESSED / "models"
TABLES = ROOT / "outputs" / "tables"

PER60 = 3600.0
# Stage-1 (NHL) priors, from outside knowledge only (log scale, rates per second).
NHL_PRIORS = {"b0_mu": float(np.log(6.5 / PER60)), "b0_sd": 1.0,
              "log_r_mu": 0.0, "log_r_sd": 1.0,
              "log_rho_mu": 0.0, "log_rho_sd": 0.5,
              "log_nu_mu": float(np.log(1.0 / PER60)), "log_nu_sd": 1.5}
# Allowances added (in quadrature) to NHL posterior sds for the PWHL priors.
WIDEN = {"b0": 0.5, "log_r": 0.2, "log_rho": 0.2, "log_nu": 0.5}
GAMMA_SD = 0.5        # prior sd for the No Escape change in log(rho)
TEAM_SIGMA_SD = 0.2   # half-normal scale for team effects (log scale: about +/-20%)

NHL_TRAIN = [f"{y}-{str(y + 1)[2:]}" for y in range(2015, 2023)]
NHL_TEST = ["2023-24", "2024-25", "2025-26"]
PWHL_TRAIN = ["2024", "2024-25", "2025"]
PWHL_TEST = ["2025-26", "2026"]


# ---------------------------------------------------------------- data

def load_pieces() -> pd.DataFrame:
    """Split each regulation 5-on-4 and OT 4-on-3 segment into early / late pieces."""
    pp = pd.read_parquet(PROCESSED / "pp_intervals.parquet")
    pp = pp[(pp["season_type"] != "preseason") & (pp["goals_complete"] == 1)]
    pp = pp[((pp["strength"] == "5v4") & (pp["in_ot"] == 0)) | ((pp["strength"] == "4v3") & (pp["in_ot"] == 1))].copy()
    a = pp["spell_offset_s"].to_numpy()
    b = a + pp["duration_s"].to_numpy()
    fs = pp["first_stoppage_s"].fillna(np.inf).to_numpy()
    g1 = pp["first_goal_for_s"].to_numpy()
    gf = pp["goals_for"].to_numpy()
    first_early = (gf > 0) & (g1 <= fs)
    rows = []
    for early, lo, hi in ((1, a, np.minimum(b, fs)), (0, np.maximum(a, fs), b)):
        exp_s = np.clip(hi - lo, 0, None)
        # the first goal goes to the piece it happened in; any further goals (majors, rare) to late
        goals = np.where(early == 1, first_early.astype(int), gf - first_early.astype(int))
        frac = np.divide(exp_s, b - a, out=np.zeros_like(exp_s), where=(b - a) > 0)
        piece = pp.assign(early=early, exposure_s=exp_s, goals=goals,
                          goals_against=pp["goals_against"] * frac)  # SH goals split by time (only totals used)
        rows.append(piece[exp_s > 0])
    out = pd.concat(rows, ignore_index=True)
    out["ot4v3"] = (out["strength"] == "4v3").astype(int)
    return out[["league", "season", "season_type", "game_id", "spell", "team", "opponent", "no_escape",
                "ot4v3", "early", "exposure_s", "goals", "goals_against"]]


def aggregate(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    return df.groupby(keys, as_index=False).agg(exposure_s=("exposure_s", "sum"), goals=("goals", "sum"),
                                                 goals_against=("goals_against", "sum"))


# ---------------------------------------------------------------- models

def nhl_model(cells: pd.DataFrame, priors: dict):
    """Stage 1: aggregated NHL cells (ot4v3 x early), no random effects."""
    with pm.Model() as m:
        b0 = pm.Normal("b0", priors["b0_mu"], priors["b0_sd"])
        log_r = pm.Normal("log_r", priors["log_r_mu"], priors["log_r_sd"])
        log_rho = pm.Normal("log_rho", priors["log_rho_mu"], priors["log_rho_sd"])
        log_nu = pm.Normal("log_nu", priors["log_nu_mu"], priors["log_nu_sd"])
        eta = b0 + log_r * cells["ot4v3"].to_numpy() + log_rho * cells["early"].to_numpy()
        pm.Poisson("goals", mu=cells["exposure_s"].to_numpy() * pm.math.exp(eta), observed=cells["goals"].to_numpy())
        ot = cells[cells["ot4v3"] == 1]
        pm.Poisson("goals_against", mu=ot["exposure_s"].sum() * pm.math.exp(log_nu),
                   observed=int(round(ot["goals_against"].sum())))
        for name, v in (("r", log_r), ("rho", log_rho)):
            pm.Deterministic(name, pm.math.exp(v))
        pm.Deterministic("rate5v4_per60", pm.math.exp(b0) * PER60)
        pm.Deterministic("nu_per60", pm.math.exp(log_nu) * PER60)
    return m


def pwhl_model(df: pd.DataFrame, priors: dict, teams: bool = False, no_escape: bool = False):
    """Stage 2: PWHL pieces. Optional shrunk team effects and a No Escape change in log(rho)."""
    team_idx = pd.Index(sorted(set(df["team"]) | set(df["opponent"])))
    with pm.Model(coords={"team": team_idx}) as m:
        b0 = pm.Normal("b0", priors["b0_mu"], priors["b0_sd"])
        log_r = pm.Normal("log_r", priors["log_r_mu"], priors["log_r_sd"])
        log_rho = pm.Normal("log_rho", priors["log_rho_mu"], priors["log_rho_sd"])
        log_nu = pm.Normal("log_nu", priors["log_nu_mu"], priors["log_nu_sd"])
        early = df["early"].to_numpy()
        eta = b0 + log_r * df["ot4v3"].to_numpy() + log_rho * early
        if no_escape:
            gamma = pm.Normal("gamma_no_escape", 0, GAMMA_SD)
            eta = eta + gamma * early * df["no_escape"].to_numpy()
        if teams:
            s_u = pm.HalfNormal("sigma_pp_team", TEAM_SIGMA_SD)
            s_v = pm.HalfNormal("sigma_pk_team", TEAM_SIGMA_SD)
            u = pm.Deterministic("pp_team", s_u * pm.Normal("z_pp", 0, 1, dims="team"), dims="team")
            v = pm.Deterministic("pk_team", s_v * pm.Normal("z_pk", 0, 1, dims="team"), dims="team")
            eta = eta + u[team_idx.get_indexer(df["team"])] + v[team_idx.get_indexer(df["opponent"])]
        pm.Poisson("goals", mu=df["exposure_s"].to_numpy() * pm.math.exp(eta), observed=df["goals"].to_numpy())
        ot = df[df["ot4v3"] == 1]
        pm.Poisson("goals_against", mu=ot["exposure_s"].sum() * pm.math.exp(log_nu),
                   observed=int(round(ot["goals_against"].sum())))
        pm.Deterministic("r", pm.math.exp(log_r))
        pm.Deterministic("rho", pm.math.exp(log_rho))
        pm.Deterministic("rate5v4_per60", pm.math.exp(b0) * PER60)
        pm.Deterministic("nu_per60", pm.math.exp(log_nu) * PER60)
    return m


def stage2_priors(nhl_idata) -> dict:
    widen = lambda sd, extra: float(np.sqrt(sd ** 2 + extra ** 2))
    out = {}
    for name, key in (("b0", "b0"), ("log_r", "log_r"), ("log_rho", "log_rho"), ("log_nu", "log_nu")):
        d = draws_of(nhl_idata, name)
        out[f"{key}_mu"], out[f"{key}_sd"] = float(d.mean()), widen(d.std(), WIDEN[key])
    return out


def diagnostics(idata, label: str) -> dict:
    names = [v for v in ("b0", "log_r", "log_rho", "log_nu", "gamma_no_escape", "sigma_pp_team", "sigma_pk_team",
                         "z_pp", "z_pk") if v in idata["posterior"]]
    s = az.summary(idata, var_names=names, ci_prob=0.9)
    div = int(idata["sample_stats"]["diverging"].values.sum())
    out = {"fit": label, "max_r_hat": float(s["r_hat"].max()), "min_ess_bulk": float(s["ess_bulk"].min()), "divergences": div}
    out["ok"] = out["max_r_hat"] < 1.01 and out["min_ess_bulk"] > 400 and div == 0
    return out


# ---------------------------------------------------------------- prediction and scoring

def predict_mu(idata, df: pd.DataFrame, train_teams: pd.Index | None = None) -> np.ndarray:
    """Expected goals per piece for every posterior draw: shape (S, n). Teams unseen in training
    get a fresh effect drawn from Normal(0, sigma)."""
    rng = np.random.default_rng(SEED)
    b0, log_r, log_rho = (draws_of(idata, n) for n in ("b0", "log_r", "log_rho"))
    eta = b0[:, None] + log_r[:, None] * df["ot4v3"].to_numpy() + log_rho[:, None] * df["early"].to_numpy()
    if "gamma_no_escape" in idata["posterior"]:
        eta = eta + draws_of(idata, "gamma_no_escape")[:, None] * (df["early"] * df["no_escape"]).to_numpy()
    if "pp_team" in idata["posterior"]:
        for eff, sig, col in (("pp_team", "sigma_pp_team", "team"), ("pk_team", "sigma_pk_team", "opponent")):
            e, s = draws_of(idata, eff), draws_of(idata, sig)
            idx = train_teams.get_indexer(df[col])
            vals = np.where(idx >= 0, e[:, np.clip(idx, 0, None)], rng.normal(0, 1, (len(s), len(df))) * s[:, None])
            eta = eta + vals
    return df["exposure_s"].to_numpy() * np.exp(eta)


def log_score_draws(mu: np.ndarray, y: np.ndarray) -> float:
    """Mean per-piece log predictive density, averaging the Poisson likelihood over draws."""
    lp = poisson.logpmf(y[None, :], mu)
    return float(np.mean(logsumexp(lp, axis=0) - np.log(mu.shape[0])))


def baselines(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, np.ndarray]:
    league = train.groupby("ot4v3").apply(lambda g: g["goals"].sum() / g["exposure_s"].sum(), include_groups=False)
    raw = train.groupby(["team", "ot4v3"]).apply(lambda g: g["goals"].sum() / g["exposure_s"].sum(), include_groups=False)
    lr = test["ot4v3"].map(league).to_numpy()
    tr = np.array([raw.get((t, o), np.nan) for t, o in zip(test["team"], test["ot4v3"])])
    tr = np.where(np.isnan(tr) | (tr == 0), lr, tr)
    e = test["exposure_s"].to_numpy()
    return {"league rate": lr * e, "raw team rate": tr * e}


def score_fixed(mu: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(poisson.logpmf(y, np.clip(mu, 1e-12, None))))


# ---------------------------------------------------------------- driver

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    draws, tune = (300, 300) if args.quick else (1000, 1000)
    MODELS.mkdir(parents=True, exist_ok=True)

    pieces = load_pieces()
    nhl, pwhl = pieces[pieces["league"] == "NHL"], pieces[pieces["league"] == "PWHL"]
    diags, holdout_rows, checks = [], [], []

    # 1. NHL holdout for r: train early seasons, predict OT 4-on-3 goals in the last three
    print("NHL holdout ...")
    tr_cells = aggregate(nhl[nhl["season"].isin(NHL_TRAIN)], ["ot4v3", "early"])
    nhl_tr = fit(nhl_model(tr_cells, NHL_PRIORS), draws, tune)
    diags.append(diagnostics(nhl_tr, "NHL holdout"))
    te = aggregate(nhl[nhl["season"].isin(NHL_TEST)], ["season", "ot4v3", "early"])
    mu = predict_mu(nhl_tr, te)
    te_ot = te["ot4v3"] == 1
    rate5 = tr_cells.loc[tr_cells["ot4v3"] == 0, "goals"].sum() / tr_cells.loc[tr_cells["ot4v3"] == 0, "exposure_s"].sum()
    checks.append({"check": "NHL test OT 4v3 goals", "observed": int(te.loc[te_ot, "goals"].sum()),
                   "predicted_mean": float(mu[:, te_ot.to_numpy()].sum(1).mean()),
                   "predicted_lo90": float(np.quantile(mu[:, te_ot.to_numpy()].sum(1), 0.05)),
                   "predicted_hi90": float(np.quantile(mu[:, te_ot.to_numpy()].sum(1), 0.95)),
                   "baseline_no_multiplier": float(rate5 * te.loc[te_ot, "exposure_s"].sum())})

    # 2. NHL full fit -> PWHL priors
    print("NHL full fit ...")
    nhl_full = fit(nhl_model(aggregate(nhl, ["ot4v3", "early"]), NHL_PRIORS), draws, tune)
    diags.append(diagnostics(nhl_full, "NHL full"))
    nhl_full.to_netcdf(MODELS / "powerplay_nhl.nc")
    priors2 = stage2_priors(nhl_full)

    # 3. PWHL holdout: base and candidates, fit on training seasons only
    print("PWHL holdout + candidates ...")
    tr, te = pwhl[pwhl["season"].isin(PWHL_TRAIN)], pwhl[pwhl["season"].isin(PWHL_TEST)]
    train_teams = pd.Index(sorted(set(tr["team"]) | set(tr["opponent"])))
    fits = {"base": fit(pwhl_model(tr, priors2), draws, tune),
            "base + team effects": fit(pwhl_model(tr, priors2, teams=True), draws, tune),
            "base + No Escape change": fit(pwhl_model(tr, priors2, no_escape=True), draws, tune)}
    y = te["goals"].to_numpy()
    for name, idata in fits.items():
        diags.append(diagnostics(idata, f"PWHL holdout {name}"))
        holdout_rows.append({"method": name, "log_score": log_score_draws(predict_mu(idata, te, train_teams), y)})
    for name, mu_b in baselines(tr, te).items():
        holdout_rows.append({"method": name, "log_score": score_fixed(mu_b, y)})
    for r in holdout_rows:
        r.update(n_train_pieces=len(tr), n_test_pieces=len(te), test_goals=int(y.sum()))
    scores = {r["method"]: r["log_score"] for r in holdout_rows}

    # Decision rule (fixed in advance)
    choice = []
    for name in ("base + team effects", "base + No Escape change"):
        gain, se = loo_gain(fits[name], fits["base"], var_name="goals")
        keep = gain >= 4 and gain > 2 * se and scores[name] > scores["base"]
        choice.append({"candidate": name, "loo_gain": gain, "se": se, "holdout_log_score": scores[name],
                       "holdout_log_score_base": scores["base"], "keep": keep})
    choice = pd.DataFrame(choice)
    use_teams = bool(choice.loc[choice["candidate"] == "base + team effects", "keep"].iloc[0])
    use_ne = bool(choice.loc[choice["candidate"] == "base + No Escape change", "keep"].iloc[0])

    # Calibration check on the test season: predicted vs observed goals, early vs late, by team
    best = fits["base"]
    mu_te = predict_mu(best, te, train_teams)
    for label, mask in (("early pieces", te["early"] == 1), ("late pieces", te["early"] == 0),
                        ("all regulation 5v4", te["ot4v3"] == 0), ("OT 4v3", te["ot4v3"] == 1)):
        m = mask.to_numpy()
        tot = mu_te[:, m].sum(1)
        checks.append({"check": f"PWHL test {label}", "observed": int(y[m].sum()), "predicted_mean": float(tot.mean()),
                       "predicted_lo90": float(np.quantile(tot, 0.05)), "predicted_hi90": float(np.quantile(tot, 0.95))})

    # 4. Final PWHL fit on all seasons with the chosen structure
    print(f"PWHL final fit (teams: {use_teams}, No Escape change: {use_ne}) ...")
    final = fit(pwhl_model(pwhl, priors2, teams=use_teams, no_escape=use_ne), draws, tune)
    diags.append(diagnostics(final, "PWHL final"))
    final.to_netcdf(MODELS / "powerplay_pwhl.nc")

    # Reference fit with the No Escape term, reported for interest whatever the decision
    ne_ref = final if use_ne else fit(pwhl_model(pwhl, priors2, no_escape=True), draws, tune)
    if not use_ne:
        diags.append(diagnostics(ne_ref, "PWHL No Escape reference"))

    # Outputs
    rows = []
    for label, idata in (("NHL full", nhl_full), ("PWHL final", final), ("PWHL No Escape reference", ne_ref)):
        for v in ("rate5v4_per60", "r", "rho", "nu_per60", "gamma_no_escape", "sigma_pp_team", "sigma_pk_team"):
            if v in idata["posterior"]:
                d = draws_of(idata, v)
                rows.append({"fit": label, "param": v, "mean": d.mean(), "lo90": np.quantile(d, 0.05),
                             "hi90": np.quantile(d, 0.95)})
    params = pd.DataFrame(rows)
    params.to_csv(TABLES / "powerplay_params.csv", index=False)
    pd.DataFrame([{"param": k, "value": v} for k, v in priors2.items()]).to_csv(TABLES / "powerplay_pwhl_priors.csv", index=False)
    pd.DataFrame(holdout_rows).to_csv(TABLES / "powerplay_holdout.csv", index=False)
    choice.to_csv(TABLES / "powerplay_model_choice.csv", index=False)
    pd.DataFrame(checks).to_csv(TABLES / "powerplay_checks.csv", index=False)
    pd.DataFrame(diags).to_csv(TABLES / "powerplay_diagnostics.csv", index=False)

    print("\nDiagnostics:\n", pd.DataFrame(diags).to_string(index=False))
    print("\nParameters:\n", params.round(3).to_string(index=False))
    print("\nHoldout (PWHL test, mean log score per piece):\n", pd.DataFrame(holdout_rows).round(4).to_string(index=False))
    print("\nModel choice:\n", choice.round(3).to_string(index=False))
    print("\nChecks:\n", pd.DataFrame(checks).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
