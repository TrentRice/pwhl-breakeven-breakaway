"""Tests for the overtime win-probability math (no sampling)."""

import numpy as np
import pytest

from src.model_ot_winprob import cumulative_G, poisson_binomial, shootout_win_prob, w0_curve


def test_poisson_binomial_matches_binomial_for_equal_p():
    from scipy.stats import binom
    d = poisson_binomial(np.full((1, 5), 0.3))[0]
    assert d == pytest.approx(binom.pmf(np.arange(6), 5, 0.3))


def test_shootout_symmetry_and_direction():
    p = np.full((2, 5), 0.3)
    assert shootout_win_prob(p, p) == pytest.approx([0.5, 0.5])
    better = shootout_win_prob(np.full((1, 5), 0.4), np.full((1, 5), 0.3))[0]
    worse = shootout_win_prob(np.full((1, 5), 0.3), np.full((1, 5), 0.4))[0]
    assert better > 0.5 and better + worse == pytest.approx(1.0)


def test_shootout_extremes():
    assert shootout_win_prob(np.ones((1, 5)) * 0.999, np.full((1, 5), 0.001))[0] > 0.99


def test_cumulative_G_flat_shape_is_time_left():
    G = cumulative_G(np.ones((1, 300)))
    assert G[0, 0] == 0 and G[0, 60] == 60 and G[0, 300] == 300


def test_cumulative_G_counts_the_end_of_overtime():
    # a shape that is zero except in the last minute: G only grows over the last 60 seconds left
    g = np.zeros((1, 300))
    g[0, 240:] = 5.0
    G = cumulative_G(g)
    assert G[0, 60] == pytest.approx(300) and G[0, 120] == pytest.approx(300)


def test_w0_limits():
    G = cumulative_G(np.ones((1, 300)))
    w = w0_curve(np.array([1 / 300]), np.array([1 / 600]), G, np.array([0.3]))
    assert w[0, 0] == pytest.approx(0.3)                       # no time left: shootout decides
    assert 0.3 < w[0, 300] < 2 / 3                              # heads toward the scoring share
    long_G = cumulative_G(np.full((1, 300), 100.0))
    assert w0_curve(np.array([1 / 300]), np.array([1 / 600]), long_G, np.array([0.3]))[0, 300] == pytest.approx(2 / 3, abs=1e-3)
