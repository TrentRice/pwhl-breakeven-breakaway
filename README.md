# Breakeven Breakaway: When Should PWHL Teams Take the Penalty Shot?

A student poster project for the 2027 Connecticut Sports Analytics Symposium (CSAS), April 9–10, 2027, UConn Storrs.

**Status:** in progress (data collection). Results, figures and the interactive calculator will be linked here when ready.

## The question

For 2026–27 the PWHL is piloting an overtime rule in preseason play: when a minor penalty is called in 3-on-3 overtime, the fouled team may take a **penalty shot** instead of a **4-on-3 power play**. Which option gives the better chance to win, and how does the answer change with time remaining, shooter and goaltender?

This project estimates the **break-even penalty-shot success rate** `p*(t)`: the success probability a shooter needs for the penalty shot to beat the power play when the penalty is called with `t` seconds left in overtime.

## Approach

PWHL overtime penalties are rare (single digits over three seasons), so the models borrow strength from related situations:

- **Penalty-shot success:** Bayesian hierarchical logistic model on all PWHL shootout attempts and in-game penalty shots, with shooter and goalie effects and an NHL-informed prior.
- **Overtime power-play scoring:** each team's PWHL 5-on-4 rate, scaled by a 4-on-3 multiplier estimated from NHL 3-on-3 overtime (2015–16 onward), with a time-varying hazard for the PWHL's No Escape rule.
- **Win probability:** a continuous-time competing-hazards model of sudden-death 3-on-3 overtime plus a Monte Carlo shootout simulation.
- **Decision:** compare P(win | power play) with P(win | penalty shot) over posterior draws, and report P(penalty shot is the better choice).

Prior work: N. Rachman, "How applicable is the PWHL's overtime penalty shot rule actually?", *The Ice Garden*, Nov 9, 2026 (descriptive analysis). This project adds a quantitative decision model.

## Reproducing

Requires Python 3.12 (PyMC does not yet support newer versions).

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Pipeline (scripts are being built; this section will be finalized with a single `make all` command):

| Step | Script | Output |
|---|---|---|
| Scrape PWHL | `src/scrape_pwhl.py` | `data/raw/pwhl/` (cached JSON) |
| Scrape NHL | `src/scrape_nhl.py` | `data/raw/nhl/` (cached JSON) |
| Parse | `src/parse_pwhl.py`, `src/parse_nhl.py` | `data/interim/*.parquet` |
| Modeling tables | `src/build_tables.py` | `data/processed/*.parquet` |
| Models | `src/model_*.py` | posterior draws |
| Decision and backtest | `src/decision.py`, `src/backtest.py` | `outputs/tables/` |
| Figures | `src/figures.py` | `outputs/figures/` |

Data is not included in this repository; only code is.

## Data sources

- PWHL play-by-play and game summaries via the HockeyTech stats feed behind thepwhl.com. Official statistics provided by Professional Women's Hockey League.
- NHL play-by-play via the public NHL API (`api-web.nhle.com`)
- [MoneyPuck](https://moneypuck.com/data.htm) shot data (optional cross-check)

## Repository layout

```
data/       raw/ (cached JSON), interim/ (parsed), processed/ (modeling tables)
src/        all pipeline and model code
notebooks/  exploration only
outputs/    figures/ and tables/
app/        interactive break-even calculator (static page, GitHub Pages)
tests/      sanity checks on parsed data
poster/     poster files
```

See `CONVENTIONS.md` for coding and hockey-rule conventions.

## License

Code is released under the MIT License (see `LICENSE`). The license covers this project's code only, not the PWHL, NHL or MoneyPuck data, which stay subject to their sources' terms.
