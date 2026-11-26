"""Tests for the decision engine math on a small synthetic posterior (no fitted models needed)."""

import numpy as np
import pytest

from src.decision import Posterior, breakeven, crossover_seconds, pp_paths, w0, win_prob_pp, win_prob_ps
from src.model_ot_winprob import cumulative_G


def synthetic(n=50, pp_rate=11 / 3600, nu=0.4 / 3600, lam=5.8 / 3600, rho=1.15):
    return Posterior(
        n=n, alpha=np.full(n, -0.85), delta=np.full(n, -0.2), delta_nhl=np.full(n, -0.13), shooter_effect=np.zeros((n, 1)), sigma_s=np.full(n, 0.2),
        goalie_effect=np.zeros((n, 1)), sigma_g=np.full(n, 0.2), shooter_ids=["s"], goalie_ids=["g"],
        pp_rate=np.full(n, pp_rate), rho=np.full(n, rho), nu=np.full(n, nu), lam=np.full(n, lam),
        G=cumulative_G(np.ones((n, 300))), stoppages=np.array([10.0, 40.0, np.inf]), sigma_season=0.0)


def test_pp_paths_are_probabilities():
    pa, pb, none = pp_paths(synthetic())
    assert np.allclose(pa + pb + none, 1)
    assert (np.diff(pa, axis=1) >= 0).all() and pa[:, 0].max() == 0


def test_breakeven_equals_pp_minus_sh_for_even_teams():
    post = synthetic()
    S = np.full(post.n, 0.5)
    pa, pb, _ = pp_paths(post)
    pstar = breakeven(post, S)
    assert pstar[:, 300] == pytest.approx(pa[:, 120] - pb[:, 120])   # full PP available
    assert pstar[:, 60] == pytest.approx(pa[:, 60] - pb[:, 60])      # PP cut to 60 s
    assert np.abs(pstar[:, 0]).max() < 1e-12


def test_options_equal_at_breakeven():
    post = synthetic()
    S = np.full(post.n, 0.6)
    pstar = breakeven(post, S)
    for t in (30, 150, 300):
        p = pstar[:, t]
        assert win_prob_ps(post, p, S)[:, t] == pytest.approx(win_prob_pp(post, S)[:, t])


def test_w0_goes_to_shootout_at_zero_and_is_half_for_even_teams():
    post = synthetic()
    assert w0(post, np.full(post.n, 0.7))[:, 0] == pytest.approx(0.7)
    assert w0(post, np.full(post.n, 0.5)) == pytest.approx(0.5)


def test_crossover_seconds():
    better = np.zeros(301)
    better[1:96] = 0.8
    assert crossover_seconds(better) == 95
    assert crossover_seconds(np.zeros(301)) is None


def test_delta_cases():
    from src.decision import delta_draws, p_shot
    from scipy.special import expit
    post = synthetic()
    assert (delta_draws(post, "zero") == 0).all()
    assert p_shot(post, delta="zero") == pytest.approx(expit(-0.85))
    assert p_shot(post, delta="nhl") == pytest.approx(expit(-0.85 - 0.13))
    assert p_shot(post, delta="estimated") == pytest.approx(expit(-0.85 - 0.2))
    with pytest.raises(ValueError):
        delta_draws(post, "other")


def test_team_lineup_prefers_players_with_shootout_history():
    import pandas as pd
    from src.decision import team_lineup
    post = synthetic()
    post.shooter_ids = ["a", "b"]
    post.shooter_effect = np.column_stack([np.full(post.n, -0.3), np.full(post.n, 0.2)])  # both tested
    roster = pd.DataFrame({"team": "X", "name": ["Tested weak", "Tested strong", "Untested F", "Untested D"],
                           "position": ["F", "F", "F", "D"], "player_id": ["a", "b", None, None]})
    top, _ = team_lineup(post, roster, "X", n_shooters=3)
    # tested players first (even the below-average one), then an untested forward before a defender
    assert list(top["name"]) == ["Tested strong", "Tested weak", "Untested F"]
