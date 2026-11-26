"""Penalty-shot success model (RQ1; PROJECT_PLAN.md §5.1 and §5.6, Step 7).

Usage (from the repo root, with the venv active):
    python -m src.model_penalty_shot            # full run (about 10-15 minutes)
    python -m src.model_penalty_shot --quick    # fewer draws, for checking the pipeline

Model: Bayesian hierarchical logistic regression for each shootout attempt / penalty shot k,
shooter i, goalie j:
    goal_k ~ Bernoulli(p_k)
    logit(p_k) = alpha + s_i + g_j + delta * penalty_shot_k  [+ beta_home * home_k + beta_sd * sudden_death_k]
    s_i = sigma_s * z_i,  g_j = sigma_g * w_j,  z, w ~ Normal(0, 1)   (non-centered random effects)

Two stages:
1. NHL fit with wide priors from outside knowledge (not from our own exploration):
   alpha ~ Normal(logit(1/3), 1): long-run shootout conversion is roughly a third;
   sigma_s, sigma_g ~ HalfNormal(0.5); delta ~ Normal(0, 0.5).
2. PWHL fit with the NHL posterior as prior, widened by a league-difference allowance
   (added in quadrature to the posterior sd) because the leagues differ.

Validation (time-based holdout, structure chosen on training data only):
- NHL: train 2015-16 to 2022-23, test 2023-24 to 2025-26.
- PWHL: train 2024 + 2024-25, test 2025-26 (NHL prior from the full NHL fit; separate league).
- Metrics: mean log score and Brier score vs (a) league average and (b) raw shooter rates.
- Extra terms (home, sudden-death round) are compared with PSIS-LOO on the PWHL training data.

Outputs:
- data/processed/models/penalty_shot_{nhl,pwhl}.nc         posterior draws (final fits)
- outputs/tables/penalty_shot_*.csv                         summaries, holdout metrics, checks
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
from scipy.special import expit, logit

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
MODELS = PROCESSED / "models"
TABLES = ROOT / "outputs" / "tables"
RAW_PWHL = ROOT / "data" / "raw" / "pwhl" / "games"

SEED = 2027
REG_ROUNDS = {"PWHL": 5, "NHL": 3}

# Stage-1 (NHL) priors: outside knowledge only.
NHL_PRIORS = {"alpha_mu": float(logit(1 / 3)), "alpha_sd": 1.0, "sigma_s_sd": 0.5, "sigma_g_sd": 0.5,
              "delta_mu": 0.0, "delta_sd": 0.5}
# League-difference allowance added (in quadrature) to NHL posterior sds for the PWHL prior.
WIDEN = {"alpha": 0.25, "log_sigma": 0.35, "delta": 0.25, "beta": 0.15}
EXTRA_SD = 0.3  # prior sd for optional home / sudden-death terms (log-odds)

NHL_TRAIN = [f"{y}-{str(y + 1)[2:]}" for y in range(2015, 2023)]
NHL_TEST = ["2023-24", "2024-25", "2025-26"]
PWHL_TRAIN = ["2024", "2024-25"]
PWHL_TEST = ["2025-26"]


# ---------------------------------------------------------------- data

def load_attempts() -> pd.DataFrame:
    so = pd.read_parquet(PROCESSED / "shootout_attempts.parquet")
    so = so[(so["season_type"] != "preseason") & so["goalie_id"].notna()].copy()
    so["home"] = so["is_home"].astype(int)
    so["sudden_death"] = [int(r > REG_ROUNDS[lg]) if not ps else 0
                          for r, lg, ps in zip(so["so_round"].fillna(0), so["league"], so["is_penalty_shot"])]
    return so.reset_index(drop=True)


def pwhl_names() -> dict[str, str]:
    """Player id -> 'First Last' from every player record in the cached PWHL play-by-play."""
    names = {}
    for path in RAW_PWHL.glob("*_pxp.json"):
        for e in json.loads(path.read_text())["GC"]["Pxpverbose"]:
            for info in e.values():
                infos = info if isinstance(info, list) else [info]   # goal events list on-ice players
                for i in infos:
                    if isinstance(i, dict) and i.get("player_id") and "first_name" in i:
                        names[str(i["player_id"])] = f"{i.get('first_name', '')} {i.get('last_name', '')}".strip()
    return names


def _season_label(league: str, raw) -> str:
    if league == "PWHL":
        return str(raw).split()[0]
    s = str(raw)
    return f"{s[:4]}-{s[6:]}"


def shooter_skill(league: str, seasons: list[str] | None = None) -> pd.Series:
    """Regulation shooting skill per player: empirical-Bayes shrunk shooting percentage
    (goals / shots on goal), on the log-odds scale, standardized within the league.

    Fixed in advance (PROJECT_PLAN §5.6): one definition, no alternatives tried. Uses regulation and
    overtime play only (never shootouts or penalty shots), and only `seasons` when given, so holdout
    fits never see test-season data. Players with no shots get the league prior mean.
    """
    I = ROOT / "data" / "interim"
    if league == "PWHL":
        ev = pd.read_parquet(I / "pwhl_events.parquet", columns=["game_id", "event_type", "period_type", "player_id", "goalie_id", "is_goal"])
        games = pd.read_parquet(I / "pwhl_games.parquet", columns=["game_id", "season_name", "season_type"])
        games = games[games["season_type"] != "preseason"]
        season = games.set_index("game_id")["season_name"].map(lambda x: _season_label("PWHL", x))
        sog = ev[(ev["event_type"] == "shot") & (ev["period_type"] != "SO") & ev["goalie_id"].notna()].copy()
        sog["goal"] = sog["is_goal"].fillna(0)
    else:
        ev = pd.read_parquet(I / "nhl_events.parquet", columns=["game_id", "event_type", "period_type", "player_id", "is_en"])
        games = pd.read_parquet(I / "nhl_games.parquet", columns=["game_id", "season"])
        season = games.set_index("game_id")["season"].map(lambda x: _season_label("NHL", x))
        sog = ev[ev["event_type"].isin(["shot", "goal"]) & (ev["period_type"] != "SO") & (ev["is_en"] != 1)].copy()
        sog["goal"] = (sog["event_type"] == "goal").astype(int)
    sog["season"] = sog["game_id"].map(season)
    sog = sog[sog["season"].notna()]
    if seasons is not None:
        sog = sog[sog["season"].isin(seasons)]
    sog["player_id"] = sog["player_id"].astype(str)
    per = sog.groupby("player_id")["goal"].agg(goals="sum", shots="size")
    # beta prior by method of moments on players with 20+ shots
    est = per[per["shots"] >= 20]
    rate = est["goals"] / est["shots"]
    m, v = rate.mean(), rate.var()
    v = max(v - (m * (1 - m) / est["shots"]).mean(), 1e-5)  # remove binomial noise
    k = m * (1 - m) / v - 1
    a, b = m * k, (1 - m) * k
    shrunk = logit((per["goals"] + a) / (per["shots"] + a + b))
    prior_mean = float(logit(m))
    mu, sd = shrunk.mean(), shrunk.std()
    z = (shrunk - mu) / sd
    z.attrs["unseen_value"] = (prior_mean - mu) / sd  # standardized value for players with no shots
    return z


def skill_for(ids: pd.Index, skill: pd.Series) -> np.ndarray:
    return skill.reindex(ids.astype(str)).fillna(skill.attrs["unseen_value"]).to_numpy()


def team_of(df: pd.DataFrame, id_col: str, team_col: str) -> pd.Series:
    """Most recent team for each player id."""
    return df.sort_values(["season", "game_id"]).groupby(id_col)[team_col].last()


# ---------------------------------------------------------------- model

def build_model(df: pd.DataFrame, priors: dict, extras: bool = False, skill: pd.Series | None = None):
    """Hierarchical logistic model. With `skill`, each shooter's effect is centered on
    beta_skill * (standardized regulation shooting skill) instead of 0."""
    shooters = pd.Index(df["shooter_id"].unique())
    goalies = pd.Index(df["goalie_id"].unique())
    coords = {"shooter": shooters, "goalie": goalies}
    with pm.Model(coords=coords) as model:
        alpha = pm.Normal("alpha", priors["alpha_mu"], priors["alpha_sd"])
        if "log_sigma_s_mu" in priors:  # stage 2: lognormal from NHL posterior
            sigma_s = pm.LogNormal("sigma_s", priors["log_sigma_s_mu"], priors["log_sigma_s_sd"])
            sigma_g = pm.LogNormal("sigma_g", priors["log_sigma_g_mu"], priors["log_sigma_g_sd"])
        else:
            sigma_s = pm.HalfNormal("sigma_s", priors["sigma_s_sd"])
            sigma_g = pm.HalfNormal("sigma_g", priors["sigma_g_sd"])
        delta = pm.Normal("delta", priors["delta_mu"], priors["delta_sd"])
        z = pm.Normal("z_shooter", 0, 1, dims="shooter")
        w = pm.Normal("z_goalie", 0, 1, dims="goalie")
        if skill is not None:
            beta = pm.Normal("beta_skill", priors.get("beta_mu", 0.0), priors.get("beta_sd", 0.5))
            x = pm.Data("skill", skill_for(shooters, skill), dims="shooter")
            s = pm.Deterministic("shooter_effect", beta * x + sigma_s * z, dims="shooter")
        else:
            s = pm.Deterministic("shooter_effect", sigma_s * z, dims="shooter")
        g = pm.Deterministic("goalie_effect", sigma_g * w, dims="goalie")
        eta = (alpha + s[shooters.get_indexer(df["shooter_id"])] + g[goalies.get_indexer(df["goalie_id"])]
               + delta * df["is_penalty_shot"].to_numpy())
        if extras:
            b_home = pm.Normal("beta_home", 0, EXTRA_SD)
            b_sd = pm.Normal("beta_sudden_death", 0, EXTRA_SD)
            eta = eta + b_home * df["home"].to_numpy() + b_sd * df["sudden_death"].to_numpy()
        pm.Bernoulli("goal", logit_p=eta, observed=df["goal"].to_numpy())
    return model


def fit(model, draws: int, tune: int):
    with model:
        idata = pm.sample(draws=draws, tune=tune, chains=4, cores=4, target_accept=0.9, random_seed=SEED,
                          progressbar=False, idata_kwargs={"log_likelihood": True})
    return idata


def diagnostics(idata, label: str) -> dict:
    names = ["alpha", "sigma_s", "sigma_g", "delta", "z_shooter", "z_goalie"]
    names += [v for v in ("beta_skill", "beta_home", "beta_sudden_death") if v in idata["posterior"]]
    s = az.summary(idata, var_names=names, ci_prob=0.9)
    div = int(idata["sample_stats"]["diverging"].values.sum())
    out = {"fit": label, "max_r_hat": float(s["r_hat"].max()), "min_ess_bulk": float(s["ess_bulk"].min()),
           "divergences": div}
    out["ok"] = out["max_r_hat"] < 1.01 and out["min_ess_bulk"] > 400 and div == 0
    return out


def draws_of(idata, name: str) -> np.ndarray:
    """Posterior draws flattened over chains: shape (S,) or (S, k)."""
    v = idata["posterior"][name].values
    return v.reshape(-1, *v.shape[2:])


def stage2_priors(nhl_idata) -> dict:
    """PWHL priors from the NHL posterior, widened by a league-difference allowance."""
    a, ss, sg, d = (draws_of(nhl_idata, n) for n in ("alpha", "sigma_s", "sigma_g", "delta"))
    widen = lambda sd, extra: float(np.sqrt(sd ** 2 + extra ** 2))
    priors = {
        "alpha_mu": float(a.mean()), "alpha_sd": widen(a.std(), WIDEN["alpha"]),
        "log_sigma_s_mu": float(np.log(ss).mean()), "log_sigma_s_sd": widen(np.log(ss).std(), WIDEN["log_sigma"]),
        "log_sigma_g_mu": float(np.log(sg).mean()), "log_sigma_g_sd": widen(np.log(sg).std(), WIDEN["log_sigma"]),
        "delta_mu": float(d.mean()), "delta_sd": widen(d.std(), WIDEN["delta"]),
    }
    if "beta_skill" in nhl_idata["posterior"]:
        b = draws_of(nhl_idata, "beta_skill")
        priors.update(beta_mu=float(b.mean()), beta_sd=widen(b.std(), WIDEN["beta"]))
    return priors


# ---------------------------------------------------------------- prediction and scoring

def predict(idata, train: pd.DataFrame, test: pd.DataFrame, extras: bool = False,
            skill: pd.Series | None = None) -> np.ndarray:
    """Posterior-mean P(goal) for test attempts. Players unseen in training get a fresh
    random effect drawn from Normal(beta * skill, sigma) in every posterior draw (skill model)
    or Normal(0, sigma) (core model)."""
    rng = np.random.default_rng(SEED)
    a, ss, sg, d = (draws_of(idata, n) for n in ("alpha", "sigma_s", "sigma_g", "delta"))
    s_eff, g_eff = draws_of(idata, "shooter_effect"), draws_of(idata, "goalie_effect")
    beta = draws_of(idata, "beta_skill") if skill is not None else None
    shooters = pd.Index(train["shooter_id"].unique())
    goalies = pd.Index(train["goalie_id"].unique())
    si, gi = shooters.get_indexer(test["shooter_id"]), goalies.get_indexer(test["goalie_id"])
    x_test = skill_for(pd.Index(test["shooter_id"]), skill) if skill is not None else np.zeros(len(test))
    S = a.shape[0]
    out = np.empty(len(test))
    for k, (i, j, ps) in enumerate(zip(si, gi, test["is_penalty_shot"].to_numpy())):
        if i >= 0:
            s = s_eff[:, i]
        else:
            s = rng.normal(0, ss) + (beta * x_test[k] if beta is not None else 0)
        g = g_eff[:, j] if j >= 0 else rng.normal(0, sg)
        eta = a + s + g + d * ps
        if extras:
            eta = eta + draws_of(idata, "beta_home") * test["home"].iloc[k] \
                + draws_of(idata, "beta_sudden_death") * test["sudden_death"].iloc[k]
        out[k] = expit(eta).mean()
        assert eta.shape == (S,)
    return out


def loo_gain(candidate, base, var_name: str | None = None) -> tuple[float, float]:
    """PSIS-LOO elpd gain of `candidate` over `base` (same data) and its standard error,
    from the pointwise differences. `var_name` picks the observed variable when a model has several."""
    a = np.asarray(az.loo(candidate, pointwise=True, var_name=var_name).elpd_i).ravel()
    b = np.asarray(az.loo(base, pointwise=True, var_name=var_name).elpd_i).ravel()
    diff = a - b
    return float(diff.sum()), float(np.sqrt(len(diff) * diff.var()))


def score(y: np.ndarray, p: np.ndarray) -> dict:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {"log_score": float(np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
            "brier": float(np.mean((y - p) ** 2))}


def baselines(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, np.ndarray]:
    league = np.full(len(test), train["goal"].mean())
    raw = train.groupby("shooter_id")["goal"].mean()
    # raw shooter rate (no shrinkage); unseen shooters get the league average; clipped so 0/1 records
    # don't produce infinite log loss
    raw_p = test["shooter_id"].map(raw).fillna(train["goal"].mean()).clip(0.01, 0.99).to_numpy()
    return {"league average": league, "raw shooter rate": raw_p}


def holdout(league: str, train: pd.DataFrame, test: pd.DataFrame, priors: dict, priors_skill: dict,
            draws: int, tune: int, train_seasons: list[str]):
    """Fit core and skill models on training seasons; score both on the test seasons.
    Skill is computed from training seasons only. Returns rows, calibration, diagnostics, fits."""
    skill_tr = shooter_skill(league, train_seasons)
    core = fit(build_model(train, priors), draws, tune)
    sk = fit(build_model(train, priors_skill, skill=skill_tr), draws, tune)
    diag = [diagnostics(core, f"{league} holdout core"), diagnostics(sk, f"{league} holdout skill")]
    y = test["goal"].to_numpy()
    p_core = predict(core, train, test)
    p_skill = predict(sk, train, test, skill=skill_tr)
    rows = [{"league": league, "method": "hierarchical model", **score(y, p_core)},
            {"league": league, "method": "hierarchical + scoring skill", **score(y, p_skill)}]
    rows += [{"league": league, "method": m, **score(y, p)} for m, p in baselines(train, test).items()]
    p_model = p_core
    calib = pd.DataFrame({"p": p_model, "y": y})
    calib["bin"] = pd.qcut(calib["p"], q=min(5, max(2, len(calib) // 40)), duplicates="drop")
    calib = calib.groupby("bin", observed=True).agg(n=("y", "size"), predicted=("p", "mean"), observed=("y", "mean")).reset_index()
    calib.insert(0, "league", league)
    calib["bin"] = calib["bin"].astype(str)
    for r in rows:
        r.update(n_train=len(train), n_test=len(test),
                 test_new_shooters=int((~test["shooter_id"].isin(train["shooter_id"])).sum()))
    calib_skill = pd.DataFrame({"p": p_skill, "y": y})
    calib_skill["bin"] = pd.qcut(calib_skill["p"], q=min(5, max(2, len(y) // 40)), duplicates="drop")
    calib_skill = calib_skill.groupby("bin", observed=True).agg(n=("y", "size"), predicted=("p", "mean"),
                                                                observed=("y", "mean")).reset_index()
    calib_skill.insert(0, "league", f"{league} (skill)")
    calib_skill["bin"] = calib_skill["bin"].astype(str)
    return rows, pd.concat([calib, calib_skill]), diag, core, sk


# ---------------------------------------------------------------- final-fit summaries

def player_table(idata, df: pd.DataFrame, kind: str, names: dict | None) -> pd.DataFrame:
    """P(goal) for each shooter vs an average goalie (or each goalie vs an average shooter)."""
    a = draws_of(idata, "alpha")
    eff = draws_of(idata, f"{kind}_effect")
    ids = list(idata["posterior"][kind].values)
    p = expit(a[:, None] + eff)
    id_col, team_col = (("shooter_id", "shooter_team") if kind == "shooter" else ("goalie_id", "goalie_team"))
    counts = df.groupby(id_col)["goal"].agg(attempts="size", goals="sum")
    t = pd.DataFrame({
        id_col: ids, "p_mean": p.mean(0), "p_lo90": np.quantile(p, 0.05, 0), "p_hi90": np.quantile(p, 0.95, 0),
        "effect_mean": eff.mean(0),
    }).join(counts, on=id_col).join(team_of(df, id_col, team_col).rename("team"), on=id_col)
    t["raw_rate"] = t["goals"] / t["attempts"]
    if names is not None:
        t.insert(1, "name", t[id_col].map(names))
    if kind == "goalie":  # for goalies, report save probability framing too
        t["p_goal_against_mean"] = t.pop("p_mean")
    return t.sort_values("effect_mean", ascending=(kind == "goalie")).reset_index(drop=True)


def ppc_by_bucket(model, idata, df: pd.DataFrame) -> pd.DataFrame:
    """Posterior predictive check: success rate for shooters grouped by number of attempts."""
    with model:
        pp = pm.sample_posterior_predictive(idata, random_seed=SEED, progressbar=False)
    yrep = pp["posterior_predictive"]["goal"].values.reshape(-1, len(df))
    n = df.groupby("shooter_id")["goal"].transform("size").to_numpy()
    bucket = pd.cut(n, [0, 1, 3, 6, np.inf], labels=["1", "2-3", "4-6", "7+"])
    rows = []
    for b in bucket.categories:
        m = np.asarray(bucket == b)
        if m.sum() == 0:
            continue
        sim = yrep[:, m].mean(1)
        rows.append({"shooter_attempts": b, "attempts": int(m.sum()), "observed": df["goal"].to_numpy()[m].mean(),
                     "sim_mean": sim.mean(), "sim_lo90": np.quantile(sim, 0.05), "sim_hi90": np.quantile(sim, 0.95)})
    return pd.DataFrame(rows)


def prior_predictive(df: pd.DataFrame, priors: dict) -> pd.DataFrame:
    """What the NHL priors imply before seeing data: league rate and spread of shooter rates."""
    with build_model(df, priors):
        pr = pm.sample_prior_predictive(draws=1000, random_seed=SEED)
    a = pr["prior"]["alpha"].values.ravel()
    ss = pr["prior"]["sigma_s"].values.ravel()
    league = expit(a)
    good_shooter = expit(a + ss)  # a shooter one sd above average
    q = [0.05, 0.5, 0.95]
    return pd.DataFrame({"quantity": ["league success rate", "shooter +1 sd success rate"],
                         **{f"q{int(x * 100)}": [np.quantile(league, x), np.quantile(good_shooter, x)] for x in q}})


def param_table(idata, label: str) -> pd.DataFrame:
    s = az.summary(idata, var_names=[v for v in ("alpha", "sigma_s", "sigma_g", "delta", "beta_skill", "beta_home", "beta_sudden_death")
                                     if v in idata["posterior"]], ci_prob=0.9)
    s.insert(0, "fit", label)
    a = draws_of(idata, "alpha")
    extra = pd.DataFrame([{"fit": label, "mean": float(expit(a).mean()),
                           "eti90_lb": float(np.quantile(expit(a), 0.05)), "eti90_ub": float(np.quantile(expit(a), 0.95))}],
                         index=["league_rate (expit alpha)"])
    return pd.concat([s, extra])


# ---------------------------------------------------------------- driver

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="fewer draws, to check the pipeline")
    args = ap.parse_args()
    draws, tune = (300, 300) if args.quick else (1000, 1000)
    MODELS.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)

    so = load_attempts()
    nhl, pwhl = so[so["league"] == "NHL"], so[so["league"] == "PWHL"]
    diags, holdout_rows, calib_rows = [], [], []

    # Prior predictive check for the stage-1 priors
    prior_predictive(nhl, NHL_PRIORS).to_csv(TABLES / "penalty_shot_prior_predictive.csv", index=False)

    # 1. NHL holdout: core vs skill (independent priors; skill from training seasons only)
    print("NHL holdout ...")
    r, c, d, _, _ = holdout("NHL", nhl[nhl["season"].isin(NHL_TRAIN)], nhl[nhl["season"].isin(NHL_TEST)],
                            NHL_PRIORS, NHL_PRIORS, draws, tune, NHL_TRAIN)
    holdout_rows += r; calib_rows.append(c); diags += d

    # 2. NHL full fits (core and skill) -> PWHL priors for each variant
    print("NHL full fits ...")
    nhl_skill = shooter_skill("NHL")
    nhl_core = fit(build_model(nhl, NHL_PRIORS), draws, tune)
    nhl_sk = fit(build_model(nhl, NHL_PRIORS, skill=nhl_skill), draws, tune)
    diags += [diagnostics(nhl_core, "NHL full core"), diagnostics(nhl_sk, "NHL full skill")]
    priors_core, priors_skill = stage2_priors(nhl_core), stage2_priors(nhl_sk)

    # 3. PWHL holdout with NHL-informed priors
    print("PWHL holdout ...")
    p_train, p_test = pwhl[pwhl["season"].isin(PWHL_TRAIN)], pwhl[pwhl["season"].isin(PWHL_TEST)]
    r, c, d, core, sk = holdout("PWHL", p_train, p_test, priors_core, priors_skill, draws, tune, PWHL_TRAIN)
    holdout_rows += r; calib_rows.append(c); diags += d

    # 4. Candidate terms vs core: PSIS-LOO on PWHL training data + holdout (structure chosen on training only)
    print("LOO: candidates vs core ...")
    ext = fit(build_model(p_train, priors_core, extras=True), draws, tune)
    diags.append(diagnostics(ext, "PWHL extended (train)"))
    loo = az.compare({"core": core, "core + scoring skill": sk, "core + home + sudden death": ext})
    loo.to_csv(TABLES / "penalty_shot_loo.csv")
    ext_score = score(p_test["goal"].to_numpy(), predict(ext, p_train, p_test, extras=True))
    holdout_rows.append({"league": "PWHL", "method": "hierarchical + home + sudden death", **ext_score,
                         "n_train": len(p_train), "n_test": len(p_test), "test_new_shooters": r[0]["test_new_shooters"]})
    pwhl_scores = {row["method"]: row["log_score"] for row in holdout_rows if row["league"] == "PWHL"}

    # Decision rule (fixed in advance): keep a candidate only if it clearly improves LOO on the training
    # data (elpd gain >= 4 and > 2 standard errors) AND improves the holdout log score.
    choice = []
    for name, idata_c, method in (("core + scoring skill", sk, "hierarchical + scoring skill"),
                                  ("core + home + sudden death", ext, "hierarchical + home + sudden death")):
        gain, dse = loo_gain(idata_c, core)
        keep = gain >= 4 and gain > 2 * dse and pwhl_scores[method] > pwhl_scores["hierarchical model"]
        choice.append({"candidate": name, "loo_gain": gain, "dse": dse, "holdout_log_score": pwhl_scores[method],
                       "holdout_log_score_core": pwhl_scores["hierarchical model"], "keep": keep})
    choice = pd.DataFrame(choice)
    choice.to_csv(TABLES / "penalty_shot_model_choice.csv", index=False)
    use_skill = bool(choice.loc[choice["candidate"] == "core + scoring skill", "keep"].iloc[0])
    use_extras = bool(choice.loc[choice["candidate"] == "core + home + sudden death", "keep"].iloc[0])

    # 5. Final PWHL fit on all seasons, with the chosen structure
    print(f"PWHL final fit (skill: {use_skill}, extras: {use_extras}) ...")
    pwhl_skill = shooter_skill("PWHL") if use_skill else None
    pwhl_model = build_model(pwhl, priors_skill if use_skill else priors_core, extras=use_extras, skill=pwhl_skill)
    pwhl_idata = fit(pwhl_model, draws, tune)
    diags.append(diagnostics(pwhl_idata, "PWHL final"))
    pwhl_idata.to_netcdf(MODELS / "penalty_shot_pwhl.nc")
    nhl_final = nhl_sk if use_skill else nhl_core
    nhl_final.to_netcdf(MODELS / "penalty_shot_nhl.nc")

    names = pwhl_names()
    pd.concat([param_table(nhl_core, "NHL full core"), param_table(nhl_sk, "NHL full skill"),
               param_table(pwhl_idata, "PWHL final")]).to_csv(TABLES / "penalty_shot_params.csv")
    used_priors = priors_skill if use_skill else priors_core
    pd.DataFrame([{"param": k, "value": v} for k, v in used_priors.items()]).to_csv(TABLES / "penalty_shot_pwhl_priors.csv", index=False)
    shooters = player_table(pwhl_idata, pwhl, "shooter", names)
    if use_skill:
        shooters["skill_z"] = skill_for(pd.Index(shooters["shooter_id"]), pwhl_skill)
    shooters.to_csv(TABLES / "penalty_shot_pwhl_shooters.csv", index=False)
    player_table(pwhl_idata, pwhl, "goalie", names).to_csv(TABLES / "penalty_shot_pwhl_goalies.csv", index=False)
    ppc_by_bucket(pwhl_model, pwhl_idata, pwhl).to_csv(TABLES / "penalty_shot_ppc.csv", index=False)
    pd.DataFrame(holdout_rows).to_csv(TABLES / "penalty_shot_holdout.csv", index=False)
    pd.concat(calib_rows).to_csv(TABLES / "penalty_shot_calibration.csv", index=False)
    pd.DataFrame(diags).to_csv(TABLES / "penalty_shot_diagnostics.csv", index=False)

    print("\nDiagnostics:\n", pd.DataFrame(diags).to_string(index=False))
    print("\nHoldout:\n", pd.DataFrame(holdout_rows).round(4).to_string(index=False))
    print("\nLOO (PWHL training):\n", loo.round(2).to_string())
    print("\nModel choice:\n", choice.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
