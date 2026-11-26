# Project: PWHL overtime penalty-shot vs power-play decision model
Goal: CSAS 2027 student poster. Full plan in PROJECT_PLAN.md. Follow its steps in order.

## Conventions
- Python 3.12, pandas + pyarrow, PyMC + ArviZ for Bayesian models, matplotlib for figures.
- Raw data in data/raw is cached JSON and is never modified. Parsed data goes to data/interim, modeling tables to data/processed (parquet).
- All final logic lives in src/; notebooks are for exploration only.
- Every scraper must cache, resume and rate-limit (≤2 requests/second).
- Times are stored as seconds elapsed in the game; OT time remaining = 300 − seconds into OT.
- Any model reports posterior means with 90% credible intervals; check R-hat < 1.01 and zero divergences.
- Figures: colorblind-safe palette, large fonts (poster), saved as PNG at 300 dpi and SVG to outputs/figures/.

## Domain rules (PWHL)
- Regular-season OT: 5 minutes, 3-on-3, sudden death, then a shootout (5 rounds, then sudden death; shooters may repeat).
- OT minor penalty → 4-on-3 power play. Pilot rule: fouled team may instead take a penalty shot.
- No Escape rule (2024-25 onward): penalized team's skaters cannot change until after the next faceoff.
- Jailbreak: a shorthanded goal ends the penalty (in OT it ends the game anyway).
- Points: 3 for a regulation win, 2 for an OT/SO win, 1 for an OT/SO loss.

## Judging criteria (CSAS 2027: Motivation, Innovation, Execution, Contribution)
Lasting decisions made so the poster scores on all four criteria:
- **Public and reproducible.** The repo is public. One command (`make all`) rebuilds everything from scraping to figures, with pinned `requirements.txt` and fixed random seeds for every sampler and simulation. The README documents the full pipeline. PROJECT_PLAN.md stays local, so the README must stand on its own.
- **Data.** Raw and processed data are never committed or published; only code is. Aggregate model results appear on the poster and in the calculator. Credit the PWHL/HockeyTech feed ("Official statistics provided by Professional Women's Hockey League"), the NHL API and MoneyPuck.
- **PWHL scraping is done.** Do not run `src/scrape_pwhl.py` again.
- **Holdout validation.** Before the final fit, fit the penalty-shot model on PWHL seasons through 2024–25 and predict 2025–26 shootout attempts. Report log score and Brier score against a league-average baseline. Also run the NHL OT power-play calibration backtest (Step 10).
- **Motivation framing.** Lead with timing and the gap: the league is deciding whether to adopt the rule now, and teams will face the choice with no framework (prior work is descriptive only). Show the stakes per decision, not per season: the win-probability swing between the better and worse option at the moment of the penalty, since one sudden-death choice can decide the game. OT penalties are rare, so per-season standings points (RQ4) will be small; report them honestly in the league-impact or limitations section, not as the headline.
- **Case studies.** The PWHL does not publish overtime results for the 2026–27 preseason scrimmages (only regulation), so the pilot-rule uses are not available. Case studies are the 9 historical regular-season OT penalties: what the team should have chosen had the rule existed, and by how much. Add pilot-rule uses only if they become public. On the poster, frame the missing pilot data as motivation: the league is judging the rule on a handful of unpublished scrimmage overtimes, and the model answers the question without waiting for data.
- **Interactive calculator (leave-behind).** A static web page in `app/`, hosted on GitHub Pages and linked from the poster with a QR code. The user picks team, shooter, goalie and time remaining; the page shows `p*(t)`, the shooter's estimated success rate and P(penalty shot is better). It reads precomputed posterior summaries exported by `src/` to JSON; no server, no model code in the browser.
- **Write-up.** A short public write-up of the results (blog post or pitch to The Ice Garden) linked from the README and the poster.
- **Poster.** It needs a limitations panel and must cite Rachman (2026) as prior work.

## Model decisions from exploration (Step 6, Nov 2026)
Evidence for each is in `notebooks/01_eda.ipynb`. Numbers below are from that notebook; recompute rather than copy them into results.
- **Penalty-shot model.** Fit the NHL first and use its posterior (α, σ_s, σ_g, widened) as the PWHL prior: the leagues convert shootouts at about the same rate. Expect small σ_s: shooter success is barely repeatable (split-half reliability near 0 in the PWHL, about 0.1 in the NHL), goalies more so; report that shooter differences are small. Keep δ (in-game penalty shot vs shootout) with a prior centered slightly negative, from the NHL. Round and home terms only as small NHL-informed adjustments, kept only if they help in the holdout test.
- **Power-play model.** PWHL OT power-play rate = league 5-on-4 rate × r, with r from the NHL 4-on-3/5-on-4 ratio (about 2.0, stable across seasons), then updated with PWHL OT power plays. League-level rate with a strongly shrunk team effect: PWHL teams show no detectable 5-on-4 differences. Scoring rate within a power play is two-piece, early (before the first stoppage) and late, with the early/late ratio's prior centered near the NHL value (about 1.15); the No Escape effect is modest, not a spike. Include a small shorthanded-goal rate, mostly from the NHL.
- **Overtime model.** The 3-on-3 scoring rate rises through overtime, so `W0(t)` uses a time-varying rate (linear in OT time or piecewise by minute), shape from the NHL and level from the PWHL, computed numerically on the 1-second grid. This replaces the constant-rate formula in PROJECT_PLAN.md §5.3. Home-ice term starts at no advantage and is only lightly updated by PWHL data.
- **Validation targets.** The model must reproduce NHL OT power-play conversion by time left (called with 2:00+ left vs under 2:00) and the Rachman (2026) PWHL OT power-play rates, which our data matches.
- **Scope.** Preseason is excluded from the overtime models (most preseason OTs were not tied). NHL games with a missing goal are excluded from rates (it makes no difference, checked).

## Do not
- Do not invent data or hand-type results; always compute from data/processed.
- Do not change PROJECT_PLAN.md without asking.