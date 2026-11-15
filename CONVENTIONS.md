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

## Do not
- Do not invent data or hand-type results; always compute from data/processed.
- Do not change PROJECT_PLAN.md without asking.