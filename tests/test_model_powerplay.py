"""Tests for the power-play model helpers (no sampling)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import poisson

from src.model_powerplay import load_pieces, log_score_draws

PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"


def test_log_score_matches_poisson_for_identical_draws():
    mu = np.full((10, 3), 0.5)
    y = np.array([0, 1, 2])
    assert log_score_draws(mu, y) == pytest.approx(poisson.logpmf(y, 0.5).mean())


@pytest.fixture(scope="module")
def pieces():
    if not (PROCESSED / "pp_intervals.parquet").exists():
        pytest.skip("run `python -m src.build_tables` first")
    return load_pieces()


def test_pieces_preserve_exposure_and_goals(pieces):
    pp = pd.read_parquet(PROCESSED / "pp_intervals.parquet")
    pp = pp[(pp["season_type"] != "preseason") & (pp["goals_complete"] == 1)]
    pp = pp[((pp["strength"] == "5v4") & (pp["in_ot"] == 0)) | ((pp["strength"] == "4v3") & (pp["in_ot"] == 1))]
    assert pieces["exposure_s"].sum() == pytest.approx(pp["duration_s"].sum())
    assert pieces["goals"].sum() == pp["goals_for"].sum()
    assert (pieces["exposure_s"] > 0).all()
    assert set(pieces["early"].unique()) <= {0, 1}


def test_goals_land_in_a_piece_with_exposure(pieces):
    assert (pieces.loc[pieces["goals"] > 0, "exposure_s"] > 0).all()
