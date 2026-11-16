"""Offline tests for scraper helpers (no network)."""

import pytest

from src.scrape_pwhl import parse_payload, season_type


def test_parse_plain_json():
    assert parse_payload('{"a": 1}') == {"a": 1}


def test_parse_jsonp_wrapper():
    assert parse_payload('angular.callbacks._0({"a": [1, 2]});') == {"a": [1, 2]}


def test_parse_garbage_raises():
    with pytest.raises(ValueError):
        parse_payload("<html>error</html>")


@pytest.mark.parametrize(
    "season, expected",
    [
        ({"season_name": "2025-26 Regular Season", "playoff": "0"}, "regular"),
        ({"season_name": "2026 Playoffs", "playoff": "1"}, "playoff"),
        ({"season_name": "2026-27 Pre-Season", "playoff": "0"}, "preseason"),
        ({"season_name": "2025-26 Preseason", "playoff": "0"}, "preseason"),
    ],
)
def test_season_type(season, expected):
    assert season_type(season) == expected
