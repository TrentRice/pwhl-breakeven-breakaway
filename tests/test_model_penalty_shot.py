"""Offline tests for the penalty-shot model helpers (no sampling)."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from src.model_penalty_shot import NHL_PRIORS, WIDEN, baselines, score, stage2_priors


def test_score_perfect_and_coinflip():
    y = np.array([1, 0, 1, 0])
    assert score(y, np.array([0.5] * 4)) == pytest.approx({"log_score": np.log(0.5), "brier": 0.25})
    perfect = score(y, y.astype(float))
    assert perfect["brier"] < 1e-9 and perfect["log_score"] > -1e-5  # predictions are clipped off 0/1


def test_baselines_unseen_shooter_gets_league_average():
    train = pd.DataFrame({"shooter_id": ["a", "a", "b"], "goal": [1, 1, 0]})
    test = pd.DataFrame({"shooter_id": ["a", "c"], "goal": [1, 0]})
    b = baselines(train, test)
    assert b["league average"] == pytest.approx([2 / 3, 2 / 3])
    assert b["raw shooter rate"][0] == pytest.approx(0.99)  # 2-for-2, clipped
    assert b["raw shooter rate"][1] == pytest.approx(2 / 3)


def test_stage2_priors_are_widened():
    rng = np.random.default_rng(0)
    post = xr.Dataset({
        "alpha": (("chain", "draw"), rng.normal(-0.8, 0.03, (2, 500))),
        "sigma_s": (("chain", "draw"), np.abs(rng.normal(0.2, 0.02, (2, 500)))),
        "sigma_g": (("chain", "draw"), np.abs(rng.normal(0.3, 0.02, (2, 500)))),
        "delta": (("chain", "draw"), rng.normal(-0.1, 0.1, (2, 500))),
    })
    p = stage2_priors({"posterior": post})
    assert p["alpha_mu"] == pytest.approx(-0.8, abs=0.01)
    assert p["alpha_sd"] >= WIDEN["alpha"]           # never tighter than the league allowance
    assert p["log_sigma_s_sd"] >= WIDEN["log_sigma"]
    assert p["delta_sd"] >= WIDEN["delta"]


def test_nhl_priors_are_independent_and_wide():
    # centered near one-third (outside knowledge), sd 1 on the log-odds scale
    from scipy.special import expit
    assert expit(NHL_PRIORS["alpha_mu"]) == pytest.approx(1 / 3)
    assert NHL_PRIORS["alpha_sd"] >= 1.0
