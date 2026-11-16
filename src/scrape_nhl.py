"""Download NHL regular-season play-by-play (2015-16 through 2025-26) via nhl-api-py.

Usage (from the repo root, with the venv active):
    python -m src.scrape_nhl --probe                 # one week of schedule + one game
    python -m src.scrape_nhl --all                   # every season in SEASONS
    python -m src.scrape_nhl --season 20232024       # one season (repeatable)

Output, under data/raw/nhl/:
    schedules/week_{YYYY-MM-DD}.json   weekly schedule pages (cached; past weeks don't change)
    games/{game_id}_pbp.json           play-by-play, cached, written once per finished game
    games.csv                          one row per regular-season game (index)
    failures.log                       games that failed after retries

Only finished regular-season games (gameType 2, gameState OFF/FINAL) are downloaded.
Re-running skips files already on disk, so an interrupted run resumes where it stopped.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

import httpx
from nhlpy import NHLClient
from nhlpy.http_client import RateLimitExceededException, ResourceNotFoundException, ServerErrorException

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "nhl"
MIN_INTERVAL_S = 0.6  # ~1.7 requests/second, under the 2 req/s project limit
MAX_RETRIES = 5

# First season with 3-on-3 OT through the last complete season.
SEASONS = [f"{y}{y + 1}" for y in range(2015, 2026)]
REGULAR_SEASON = 2
FINISHED_STATES = {"OFF", "FINAL"}

log = logging.getLogger("scrape_nhl")
_client = NHLClient()
_last_request = 0.0


# ---------------------------------------------------------------- helpers

def season_window(season: str) -> tuple[date, date]:
    """Generous date window that contains a season's regular season (incl. the 2020-21 shift)."""
    start_year = int(season[:4])
    return date(start_year, 9, 1), date(start_year + 1, 7, 31)


def call(fn, *args, **kwargs):
    """Call an nhl-api-py method with rate limiting and retries on transient errors."""
    global _last_request
    for attempt in range(1, MAX_RETRIES + 1):
        wait = MIN_INTERVAL_S - (time.monotonic() - _last_request)
        if wait > 0:
            time.sleep(wait)
        _last_request = time.monotonic()
        try:
            return fn(*args, **kwargs)
        except ResourceNotFoundException:
            raise
        except (RateLimitExceededException, ServerErrorException, httpx.HTTPError) as exc:
            if attempt == MAX_RETRIES:
                raise
            backoff = 2 ** attempt * (5 if isinstance(exc, RateLimitExceededException) else 1)
            log.warning("%s failed (%s); retry %d in %ds", fn.__name__, exc, attempt, backoff)
            time.sleep(backoff)
    raise RuntimeError("unreachable")


def write_json(path: Path, data: dict) -> None:
    """Write atomically so an interrupted run never leaves a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data), encoding="utf-8")
    tmp.replace(path)


# ---------------------------------------------------------------- schedule

def get_week(start: date) -> dict:
    """Weekly schedule page starting at `start`. Cached once the whole week is in the past."""
    path = RAW_DIR / "schedules" / f"week_{start.isoformat()}.json"
    if path.exists():
        return json.loads(path.read_text())
    data = call(_client.schedule.weekly_schedule, date=start.isoformat())
    if start + timedelta(days=7) < date.today():
        write_json(path, data)
    return data


def season_games(season: str) -> list[dict]:
    """All regular-season games of one season, walking the weekly schedule pages."""
    first, last = season_window(season)
    games, seen, cur = [], set(), first
    while cur <= last:
        week = get_week(cur)
        for day in week.get("gameWeek", []):
            for g in day.get("games", []):
                if str(g.get("season")) == season and g.get("gameType") == REGULAR_SEASON and g["id"] not in seen:
                    seen.add(g["id"])
                    games.append(g)
        nxt = week.get("nextStartDate")
        cur = date.fromisoformat(nxt) if nxt else cur + timedelta(days=7)
    return games


# ---------------------------------------------------------------- games

def pbp_path(game_id: int) -> Path:
    return RAW_DIR / "games" / f"{game_id}_pbp.json"


def download_game(game_id: int) -> None:
    path = pbp_path(game_id)
    if path.exists():
        return
    data = call(_client.game_center.play_by_play, game_id=str(game_id))
    data["plays"]  # KeyError if the response is not play-by-play
    write_json(path, data)


INDEX_FIELDS = [
    "season", "game_id", "date", "home_team", "away_team", "home_goals", "away_goals",
    "game_state", "last_period_type", "has_pbp",
]


def index_row(g: dict) -> dict:
    return {
        "season": g["season"],
        "game_id": g["id"],
        "date": g.get("gameDate") or g.get("startTimeUTC", "")[:10],
        "home_team": g["homeTeam"]["abbrev"],
        "away_team": g["awayTeam"]["abbrev"],
        "home_goals": g["homeTeam"].get("score"),
        "away_goals": g["awayTeam"].get("score"),
        "game_state": g.get("gameState"),
        "last_period_type": (g.get("gameOutcome") or {}).get("lastPeriodType"),
        "has_pbp": int(pbp_path(g["id"]).exists()),
    }


# ---------------------------------------------------------------- driver

def scrape(seasons: list[str]) -> None:
    rows, failures = [], []
    for season in seasons:
        games = season_games(season)
        finished = [g for g in games if g.get("gameState") in FINISHED_STATES]
        todo = [g for g in finished if not pbp_path(g["id"]).exists()]
        log.info("Season %s: %d games, %d finished, %d to download", season, len(games), len(finished), len(todo))
        for i, g in enumerate(todo, 1):
            try:
                download_game(g["id"])
            except Exception as exc:  # log and keep going; rerun to retry
                failures.append((g["id"], repr(exc)))
                log.error("Game %s failed: %s", g["id"], exc)
            if i % 100 == 0:
                log.info("  %s: %d/%d", season, i, len(todo))
        rows.extend(index_row(g) for g in games)

    # Merge with any existing index so single-season runs don't drop other seasons.
    index_path = RAW_DIR / "games.csv"
    merged = {}
    if index_path.exists():
        with open(index_path, newline="", encoding="utf-8") as f:
            merged = {r["game_id"]: r for r in csv.DictReader(f)}
    merged.update({str(r["game_id"]): r for r in rows})
    with open(index_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=INDEX_FIELDS)
        w.writeheader()
        w.writerows(sorted(merged.values(), key=lambda r: int(r["game_id"])))

    if failures:
        with open(RAW_DIR / "failures.log", "a", encoding="utf-8") as f:
            for gid, err in failures:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}\t{gid}\t{err}\n")
        log.warning("%d games failed; see failures.log and rerun", len(failures))
    log.info("Done. %d games indexed.", len(merged))


def probe() -> None:
    """Check one week of schedule and one shootout game before a full run."""
    week = get_week(date(2023, 11, 1))
    games = [g for d in week["gameWeek"] for g in d["games"] if g.get("gameType") == REGULAR_SEASON]
    print(f"Week of 2023-11-01: {len(games)} regular-season games")
    print("  outcomes:", dict(Counter((g.get("gameOutcome") or {}).get("lastPeriodType") for g in games)))
    so = [g for g in games if (g.get("gameOutcome") or {}).get("lastPeriodType") == "SO"]
    if not so:
        return
    gid = so[0]["id"]
    download_game(gid)
    plays = json.loads(pbp_path(gid).read_text())["plays"]
    print(f"\nGame {gid}: {len(plays)} plays")
    print("  period types:", dict(Counter(p["periodDescriptor"]["periodType"] for p in plays)))
    print("  SO events:", dict(Counter(p["typeDescKey"] for p in plays if p["periodDescriptor"]["periodType"] == "SO")))
    print("  OT situation codes:", dict(Counter(p.get("situationCode") for p in plays if p["periodDescriptor"]["periodType"] == "OT")))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--probe", action="store_true", help="check one week and one shootout game")
    mode.add_argument("--all", action="store_true", help=f"scrape seasons {SEASONS[0]}..{SEASONS[-1]}")
    mode.add_argument("--season", action="append", metavar="YYYYYYYY", help="scrape one season (repeatable)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if args.probe:
        probe()
    else:
        bad = [s for s in (args.season or []) if s not in SEASONS]
        if bad:
            raise SystemExit(f"Unknown season(s) {bad}; choose from {SEASONS}")
        scrape(SEASONS if args.all else args.season)


if __name__ == "__main__":
    main()
