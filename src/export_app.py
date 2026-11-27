"""Export the decision model for the interactive calculator (app/).

Usage (from the repo root, with the venv active):
    python -m src.export_app

Writes:
    app/data/model.json    thinned posterior draws + 2026-27 rosters, everything the page needs
    app/data/checks.json   test cases computed by src/decision.py on the same draws

The page recomputes only the decision formulas per draw (no model fitting in the browser):
    W0(t)       = 0.5 * (1 - surv[t]) + surv[t] * S             (equal 3-on-3 rates, no home term)
    P(win | PS) = p + (1 - p) * W0(t)
    P(win | PP) = ppGoal[d] + ppNone[d] * W0(t - d),  d = min(120, t)
    p*(t)       = (P(win | PP) - W0(t)) / (1 - W0(t))
with p = expit(alpha + delta + shooter effect + goalie effect). The shootout probability S is computed in the
page per draw (exact, as in src/model_ot_winprob.shootout_win_prob) from each team's top five shooters
(exported here) and the two goalies the user selects.

Contains model summaries only (no raw PWHL data), as allowed by CONVENTIONS.md.
"""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from src import decision as D

ROOT = Path(__file__).resolve().parents[1]
APP_DATA = ROOT / "app" / "data"
N_DRAWS = 500
DECIMALS = 5


def thin(post: D.Posterior, n: int) -> D.Posterior:
    """Every k-th posterior draw, so the page stays small."""
    idx = np.linspace(0, post.n - 1, n).round().astype(int)
    fields = {f: getattr(post, f)[idx] for f in ("alpha", "delta", "delta_nhl", "shooter_effect", "sigma_s",
                                                 "goalie_effect", "sigma_g", "pp_rate", "rho", "nu", "lam", "G")}
    return replace(post, n=n, **fields)


def r(x) -> list:
    return np.round(np.asarray(x, dtype=float), DECIMALS).tolist()


def team_names() -> dict[str, str]:
    sched = json.loads((ROOT / "data" / "raw" / "pwhl" / "schedules" / "season_11.json").read_text())["SiteKit"]["Schedule"]
    names = {}
    for g in sched:
        names[g["home_team_code"]] = g["home_team_name"]
        names[g["visiting_team_code"]] = g["visiting_team_name"]
    return names


def build(post: D.Posterior) -> tuple[dict, dict]:
    paths = D.pp_paths(post)
    pa, _, none = paths
    surv = np.exp(-2 * post.lam[:, None] * post.G)
    roster = D.load_rosters()
    lu = D.lineups(post, roster)
    names = team_names()

    # Players: shooters with PWHL shootout history get their effect draws; everyone else is league average
    shooter_eff, goalie_eff = {}, {}
    teams = []
    for code in sorted(roster["team"].unique()):
        t = roster[roster["team"] == code]
        skaters, goalies = [], []
        for row in t.itertuples(index=False):
            pid = row.player_id if isinstance(row.player_id, str) and row.player_id != "AMBIGUOUS" else None
            if row.position == "G":
                known = pid in post.goalie_ids
                if known:
                    goalie_eff[pid] = r(post.goalie_effect[:, post.goalie_ids.index(pid)])
                goalies.append({"name": row.name, "id": pid if known else None})
            else:
                known = pid in post.shooter_ids
                eff = post.shooter_effect[:, post.shooter_ids.index(pid)] if known else None
                if known:
                    shooter_eff[pid] = r(eff)
                pos = row.position if isinstance(row.position, str) else ""   # blank in the roster (e.g. Webster)
                skaters.append({"name": row.name, "id": pid if known else None, "pos": pos,
                                "mean": float(eff.mean()) if known else None})
        skaters.sort(key=lambda s: (s["id"] is None, -(s["mean"] or 0), s["name"]))
        for s in skaters:
            s.pop("mean")
        main = lu[code]["goalie"]
        goalies.sort(key=lambda g: (g["id"] != main, g["id"] is None, g["name"]))
        teams.append({"code": code, "name": names.get(code, code), "skaters": skaters, "goalies": goalies,
                      "mainGoalie": main, "shootoutShooters": lu[code]["shooters"]})
        for sid in lu[code]["shooters"]:          # every shootout shooter's effect must be in the export
            assert sid is None or sid in shooter_eff, (code, sid)

    model = {
        "meta": {
            "generated": date.today().isoformat(), "draws": post.n,
            "deltaCases": {"zero": D.DELTA_CASES["zero"], "nhl": D.DELTA_CASES["nhl"]},
            "credit": "Official statistics provided by Professional Women's Hockey League. NHL data via the NHL API.",
            "notes": "Shootout: each team's top five shooters (2026-27 rosters) against the selected goalies.",
        },
        "draws": {
            "alpha": r(post.alpha), "deltaNhl": r(post.delta_nhl),
            "ppGoal": [r(row) for row in pa], "ppNone": [r(row) for row in none],
            "surv": [r(row) for row in surv],
        },
        "shooterEffects": shooter_eff, "goalieEffects": goalie_eff,
        "teams": teams,
    }

    # Test cases from the Python engine on the same draws (the page must reproduce these)
    checks = []
    rng = np.random.default_rng(D.SEED)
    codes = [t["code"] for t in teams]
    by_code = {t["code"]: t for t in teams}
    scenarios = []
    for _ in range(35):
        a, b = (str(x) for x in rng.choice(codes, 2, replace=False))
        ta, tb = by_code[a], by_code[b]
        shooter = rng.choice([s["id"] for s in ta["skaters"][:5]] + [None])
        own_goalie = rng.choice([g["id"] for g in ta["goalies"]] + [None])     # any goalie, incl. backups
        goalie = rng.choice([g["id"] for g in tb["goalies"]] + [None])
        scenarios.append((a, shooter, own_goalie, b, goalie, int(rng.integers(1, 301)), str(rng.choice(["zero", "nhl"]))))
    for a, sid, own, b, gid, t, case in scenarios:
        S = D.shootout_S(post, tuple(lu[a]["shooters"]), own, tuple(lu[b]["shooters"]), gid)
        p = D.p_shot(post, sid, gid, delta=case)
        pstar = D.breakeven(post, S, paths)[:, t]
        checks.append({
            "team": a, "shooter": sid, "ownGoalie": own, "opponent": b, "goalie": gid, "seconds": t, "delta": case,
            "shootoutMean": float(S.mean()), "pShotMean": float(p.mean()), "pstarMean": float(pstar.mean()),
            "pShotBetter": float((p > pstar).mean()),
            "winPsMean": float(D.win_prob_ps(post, p, S)[:, t].mean()),
            "winPpMean": float(D.win_prob_pp(post, S, paths)[:, t].mean()),
        })
    return model, {"tolerance": 2e-3, "cases": checks}


def main() -> None:
    APP_DATA.mkdir(parents=True, exist_ok=True)
    post = thin(D.load_posterior(), N_DRAWS)
    model, checks = build(post)
    for name, obj in (("model.json", model), ("checks.json", checks)):
        path = APP_DATA / name
        path.write_text(json.dumps(obj, separators=(",", ":"), allow_nan=False), encoding="utf-8")  # strict JSON
        print(f"{path.relative_to(ROOT)}: {path.stat().st_size / 1e6:.2f} MB")
    print(f"{len(model['teams'])} teams, {len(model['shooterEffects'])} shooters and "
          f"{len(model['goalieEffects'])} goalies with history")


if __name__ == "__main__":
    main()
