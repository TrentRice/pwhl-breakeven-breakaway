# Breakeven Breakaway: When Should PWHL Teams Take the Penalty Shot?

A student poster project for the 2027 Connecticut Sports Analytics Symposium (CSAS), April 9–10, 2027, UConn Storrs.

**Try the calculator:** https://trentrice.github.io/pwhl-breakeven-breakaway/

Pick your team, shooter and goalie, the opponent and their goalie, and the time left in overtime; it tells you whether to take the penalty shot or the power play, with the uncertainty.

**Status:** analysis complete (data, models, decision engine, backtest, sensitivity analysis, figures and calculator).

## The question

For 2026–27 the PWHL is piloting an overtime rule in preseason play: when a minor penalty is called in 3-on-3 overtime, the fouled team may take a **penalty shot** instead of a **4-on-3 power play**. Which option gives the better chance to win, and how does the answer change with time remaining, shooter and goaltender?

This project estimates the **break-even penalty-shot success rate** `p*(t)`: the success probability a shooter needs for the penalty shot to beat the power play when the penalty is called with `t` seconds left in overtime.

## Approach

PWHL overtime penalties are rare (nine games over three seasons), so the models borrow strength from related situations and from the NHL:

- **Penalty-shot success:** Bayesian hierarchical logistic model of all 287 PWHL regular-season shootout attempts, with shooter and goalie effects and an NHL-informed prior.
- **Overtime power-play scoring:** the PWHL 5-on-4 scoring rate times a 4-on-3 multiplier learned from eleven NHL seasons of 3-on-3 overtime, plus shorthanded goals against.
- **Win probability:** a 3-on-3 scoring rate that rises as overtime goes on (learned from the NHL, level set from the PWHL), followed by an exact shootout calculation from each team's top five shooters against the opposing goalie.
- **Decision:** for each posterior draw, compare P(win | penalty shot) with P(win | power play), and report the break-even rate and P(penalty shot is the better choice).

Every extra model term (home ice, team effects, shooter scoring skill, a No Escape change) was tested once against a rule fixed in advance and kept only if it improved out-of-sample prediction; none did. Validation uses held-out seasons.

Prior work: N. Rachman, "How applicable is the PWHL's overtime penalty shot rule actually?", *The Ice Garden*, Nov 9, 2026 (descriptive analysis). This project adds a quantitative decision model.

## Main findings

- **The clock decides most of it.** With two full minutes left, the power play sets a break-even of about 31.5% for evenly matched teams, close to the average PWHL shooter (about 30%), so it is a close call. Inside about 1:50 the power play gets cut short by the end of overtime and the penalty shot becomes the better choice for almost any shooter.
- **Above 2:00, shootout strength tips it.** Teams that are weaker in the shootout lean toward the penalty shot; stronger shootout teams lean toward the power play.
- **Who shoots matters less than expected.** Shootout success is mostly luck, so the best and an average shooter differ by under three points.
- **The right call is worth up to about 13 points of win probability** late in overtime, but only about 0.01 standings points per team per season, because overtime penalties are rare.
- **Out-of-sample NHL backtest:** on 580 overtime power plays (leave one season out), the framework beats a constant rate (Brier 0.205 vs 0.218) and reproduces conversion falling from 38% (2:00+ left) to 9% (under 1:00).
- **Sensitivity:** "shoot inside about 1:00" holds in every case tested; the early-overtime answer depends on assumptions (shooter selection, the 4-on-3 multiplier).

Figures are in `outputs/figures/` and tables in `outputs/tables/`.

## Reproducing

Requires Python 3.12 (PyMC does not yet support newer versions).

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Pipeline (run each step from the repo root with `python -m src.<script>`):

| Step | Script | Output |
|---|---|---|
| Scrape PWHL | `src/scrape_pwhl.py` | `data/raw/pwhl/` (cached JSON) |
| Scrape NHL | `src/scrape_nhl.py` | `data/raw/nhl/` (cached JSON) |
| Parse | `src/parse_pwhl.py`, `src/parse_nhl.py` | `data/interim/*.parquet` |
| Modeling tables | `src/build_tables.py` | `data/processed/*.parquet` |
| Models | `src/model_penalty_shot.py`, `src/model_powerplay.py`, `src/model_ot_winprob.py` | posterior draws |
| Decision and backtest | `src/decision.py`, `src/backtest.py` | `outputs/tables/` |
| Figures | `src/figures.py` | `outputs/figures/` |
| Calculator data | `src/export_app.py` | `app/data/model.json` |
| Poster QR code | `src/make_qr.py` | `poster/qr_calculator.svg`, `.png` |

Tests: `pytest` (includes a check that the calculator reproduces `src/decision.py`). The calculator deploys to GitHub Pages from `app/` on every push to `main`.

Data is not included in this repository; only code is. The one exception is `app/data/`, which holds model summaries (posterior draws) for the calculator, not raw data.

## Data sources

- PWHL play-by-play and game summaries via the HockeyTech stats feed behind thepwhl.com. Official statistics provided by Professional Women's Hockey League.
- NHL play-by-play via the public NHL API (`api-web.nhle.com`)

## Repository layout

```
data/       raw/ (cached JSON), interim/ (parsed), processed/ (modeling tables)
src/        all pipeline and model code
notebooks/  exploratory data analysis
outputs/    figures/ and tables/
app/        interactive break-even calculator (static page, GitHub Pages)
tests/      parser, model and calculator checks
poster/     poster files
```

See `CONVENTIONS.md` for coding and hockey-rule conventions.

## License

Code is released under the MIT License (see `LICENSE`). The license covers this project's code only, not the PWHL or NHL data, which stay subject to their sources' terms.
