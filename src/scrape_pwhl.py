"""Download PWHL schedules, play-by-play and game summaries from the HockeyTech feed.

Usage (from the repo root, with the venv active):
    python -m src.scrape_pwhl --probe          # quick check: seasons, one schedule, one game
    python -m src.scrape_pwhl --all            # every season
    python -m src.scrape_pwhl --season 8 --season 10

Output, under data/raw/pwhl/:
    seasons.json                     season list (refetched every run)
    schedules/season_{id}.json       schedule per season (refetched every run; games get added and finished)
    games/{game_id}_pxp.json         play-by-play, cached, written once per finished game
    games/{game_id}_summary.json     game summary, cached, written once per finished game
    games.csv                        one row per game (index)
    failures.log                     games that failed after retries

Only finished games (schedule field final == "1") are downloaded. Re-running skips
games already on disk, so an interrupted run resumes where it stopped.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from collections import Counter
from pathlib import Path

import requests

FEED_URL = "https://lscluster.hockeytech.com/feed/index.php"
# Public client key used by thepwhl.com's own pages.
BASE_PARAMS = {"key": "446521baf8c38984", "client_code": "pwhl", "fmt": "json"}

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "pwhl"
MIN_INTERVAL_S = 0.6  # ~1.7 requests/second, under the 2 req/s project limit
MAX_RETRIES = 4
TIMEOUT_S = 30

log = logging.getLogger("scrape_pwhl")
_session = requests.Session()
_session.headers["User-Agent"] = "Mozilla/5.0 (research scraper)"
_last_request = 0.0


# ---------------------------------------------------------------- helpers

def parse_payload(text: str) -> dict:
    """Parse a feed response. Strips a JSONP wrapper like `cb({...})` if present."""
    text = text.strip()
    if not text.startswith("{"):
        start, end = text.find("("), text.rfind(")")
        if start == -1 or end == -1:
            raise ValueError(f"Unrecognized response: {text[:80]!r}")
        text = text[start + 1:end]
    return json.loads(text)


def season_type(season: dict) -> str:
    """Classify a season record as 'regular', 'playoff' or 'preseason'."""
    if season.get("playoff") == "1":
        return "playoff"
    if "pre" in season.get("season_name", "").lower():
        return "preseason"
    return "regular"


def fetch(**params) -> dict:
    """GET one feed view with rate limiting and retries (exponential backoff)."""
    global _last_request
    for attempt in range(1, MAX_RETRIES + 1):
        wait = MIN_INTERVAL_S - (time.monotonic() - _last_request)
        if wait > 0:
            time.sleep(wait)
        _last_request = time.monotonic()
        try:
            r = _session.get(FEED_URL, params={**params, **BASE_PARAMS}, timeout=TIMEOUT_S)
            r.raise_for_status()
            return parse_payload(r.text)
        except (requests.RequestException, ValueError) as exc:
            if attempt == MAX_RETRIES:
                raise
            backoff = 2 ** attempt
            log.warning("Request %s failed (%s); retry %d in %ds", params, exc, attempt, backoff)
            time.sleep(backoff)
    raise RuntimeError("unreachable")


def write_json(path: Path, data: dict) -> None:
    """Write atomically so an interrupted run never leaves a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data), encoding="utf-8")
    tmp.replace(path)


# ---------------------------------------------------------------- feed views

def get_seasons() -> list[dict]:
    data = fetch(feed="modulekit", view="seasons")
    write_json(RAW_DIR / "seasons.json", data)
    return data["SiteKit"]["Seasons"]


def get_schedule(season_id: str) -> list[dict]:
    data = fetch(feed="modulekit", view="schedule", season_id=season_id)
    write_json(RAW_DIR / "schedules" / f"season_{season_id}.json", data)
    return data["SiteKit"]["Schedule"]


def game_paths(game_id: str) -> tuple[Path, Path]:
    d = RAW_DIR / "games"
    return d / f"{game_id}_pxp.json", d / f"{game_id}_summary.json"


def download_game(game_id: str) -> None:
    """Download play-by-play and summary for one game, skipping files already cached."""
    pxp_path, summary_path = game_paths(game_id)
    if not pxp_path.exists():
        data = fetch(feed="gc", tab="pxpverbose", game_id=game_id)
        data["GC"]["Pxpverbose"]  # KeyError if the response is not play-by-play
        write_json(pxp_path, data)
    if not summary_path.exists():
        data = fetch(feed="gc", tab="gamesummary", game_id=game_id)
        data["GC"]["Gamesummary"]
        write_json(summary_path, data)


# ---------------------------------------------------------------- driver

INDEX_FIELDS = [
    "season_id", "season_name", "season_type", "game_id", "date", "home_team", "away_team",
    "home_goals", "away_goals", "game_status", "final", "overtime", "shootout",
    "has_pxp", "has_summary",
]


def index_row(season: dict, g: dict) -> dict:
    pxp_path, summary_path = game_paths(g["game_id"])
    return {
        "season_id": season["season_id"],
        "season_name": season["season_name"],
        "season_type": season_type(season),
        "game_id": g["game_id"],
        "date": g["date_played"],
        "home_team": g["home_team_code"],
        "away_team": g["visiting_team_code"],
        "home_goals": g["home_goal_count"],
        "away_goals": g["visiting_goal_count"],
        "game_status": g["game_status"],
        "final": g["final"],
        "overtime": int(g["game_status"] in ("Final OT", "Final SO")),
        "shootout": int(g["game_status"] == "Final SO"),
        "has_pxp": int(pxp_path.exists()),
        "has_summary": int(summary_path.exists()),
    }


def scrape(season_ids: list[str] | None) -> list[dict]:
    seasons = get_seasons()
    if season_ids:
        wanted = set(season_ids)
        seasons = [s for s in seasons if s["season_id"] in wanted]
        missing = wanted - {s["season_id"] for s in seasons}
        if missing:
            raise SystemExit(f"Unknown season id(s): {sorted(missing)}")

    rows, failures = [], []
    for season in sorted(seasons, key=lambda s: int(s["season_id"])):
        schedule = get_schedule(season["season_id"])
        finished = [g for g in schedule if g.get("final") == "1"]
        log.info("Season %s (%s): %d games, %d final",
                 season["season_id"], season["season_name"], len(schedule), len(finished))
        for i, g in enumerate(finished, 1):
            try:
                download_game(g["game_id"])
            except Exception as exc:  # log and keep going; rerun to retry
                failures.append((g["game_id"], repr(exc)))
                log.error("Game %s failed: %s", g["game_id"], exc)
            if i % 25 == 0:
                log.info("  %d/%d", i, len(finished))
        rows.extend(index_row(season, g) for g in schedule)

    with open(RAW_DIR / "games.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=INDEX_FIELDS)
        w.writeheader()
        w.writerows(rows)
    if failures:
        with open(RAW_DIR / "failures.log", "a", encoding="utf-8") as f:
            for gid, err in failures:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}\t{gid}\t{err}\n")
        log.warning("%d games failed; see failures.log and rerun", len(failures))
    return rows


def probe() -> None:
    """Print what the feed returns, for checking field names before a full run."""
    seasons = get_seasons()
    print("Seasons:")
    for s in sorted(seasons, key=lambda s: int(s["season_id"])):
        print(f"  {s['season_id']:>3}  {season_type(s):<9}  {s['season_name']}")

    today = time.strftime("%Y-%m-%d")
    started = [s for s in seasons if season_type(s) == "regular" and s["start_date"] <= today]
    latest_done = max(started, key=lambda s: int(s["season_id"]))
    schedule = get_schedule(latest_done["season_id"])
    print(f"\nSchedule for {latest_done['season_name']}: {len(schedule)} games")
    print("  status counts:", dict(Counter(g["game_status"] for g in schedule if g.get("final") == "1")))

    so_games = [g for g in schedule if g["game_status"] == "Final SO"]
    if not so_games:
        print("  no shootout games to probe")
        return
    gid = so_games[0]["game_id"]
    download_game(gid)
    pxp_path, summary_path = game_paths(gid)
    events = json.loads(pxp_path.read_text())["GC"]["Pxpverbose"]
    summary = json.loads(summary_path.read_text())["GC"]["Gamesummary"]
    print(f"\nGame {gid}: {len(events)} play-by-play events")
    print("  event types:", dict(Counter(e["event"] for e in events)))
    print("  shootout attempts:", sum(e["event"] == "shootout" for e in events))
    print("  summary penaltyshots:", summary.get("penaltyshots"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--probe", action="store_true", help="check the feed on one shootout game")
    mode.add_argument("--all", action="store_true", help="scrape every season")
    mode.add_argument("--season", action="append", metavar="ID", help="scrape one season (repeatable)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if args.probe:
        probe()
    else:
        scrape(None if args.all else args.season)


if __name__ == "__main__":
    main()
