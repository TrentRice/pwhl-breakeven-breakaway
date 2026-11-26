"""Poster figures (PNG at 300 dpi + SVG in outputs/figures/). Reads only outputs/tables.

Usage (from the repo root, with the venv active):
    python -m src.figures
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "outputs" / "tables"
FIGS = ROOT / "outputs" / "figures"

# Reference palette (validated categorical slots), recessive axes and grid.
BLUE, ORANGE, AQUA = "#257de0", "#f47230", "#00b894"
INK, MUTED, GRID = "#191919", "#666666", "#e6e6e6"


def style() -> None:
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 300, "font.size": 15, "axes.titlesize": 18, "axes.labelsize": 15,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
        "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
        "lines.linewidth": 2.5, "legend.frameon": False,
    })


def save(fig, name: str) -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / f"{name}.png")
    fig.savefig(FIGS / f"{name}.svg")


def hero_breakeven() -> None:
    """Break-even penalty-shot success rate p*(t) vs time left in OT, with the shooters' success rates."""
    curve = pd.read_csv(TABLES / "decision_breakeven_average.csv")
    cases = pd.read_csv(TABLES / "decision_delta_cases.csv")
    avg = cases[cases["team"] == "league average"].set_index("delta_case")
    teams = cases[cases["team"] != "league average"]
    best_name = teams[teams["delta_case"] == "zero"].sort_values("p_shot_mean", ascending=False).iloc[0]["shooter"]
    best = teams[teams["shooter"] == best_name].set_index("delta_case")

    c = curve[curve["seconds_left"] >= 1]
    x = c["seconds_left"] / 60
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.fill_between(x, c["pstar_lo90"], c["pstar_hi90"], color=BLUE, alpha=0.18, linewidth=0)
    ax.plot(x, c["pstar_mean"], color=BLUE, label="Break-even: penalty shot must score this often")
    # shooters: line = main case (chosen shooter, delta = 0); band down to the cautious case (NHL delta)
    for row, color, ls, who in ((avg, ORANGE, "--", "Average PWHL shooter"), (best, AQUA, ":", f"Best PWHL shooter ({best_name})")):
        hi, lo = row.loc["zero", "p_shot_mean"], row.loc["nhl", "p_shot_mean"]
        ax.axhspan(lo, hi, color=color, alpha=0.15, linewidth=0)
        ax.axhline(hi, color=color, ls=ls, label=f"{who}: {hi:.0%} (cautious case {lo:.0%})")
    lo_s, hi_s = avg.loc["nhl", "crossover_seconds_left"] / 60, avg.loc["zero", "crossover_seconds_left"] / 60
    ax.axvspan(lo_s, hi_s, color=MUTED, alpha=0.12, linewidth=0)
    clock = lambda m: f"{int(round(m * 60)) // 60}:{int(round(m * 60)) % 60:02d}"
    ax.text(lo_s - 0.05, 0.43, f"Inside ~{clock(lo_s)}–{clock(hi_s)}:\ntake the penalty shot", ha="left", va="top",
            fontsize=13, color=INK)
    ax.text(4.95, 0.43, "Above 2:00: close call.\nPower play for an average shooter;\ntoss-up for a team's best shooter",
            ha="left", va="top", fontsize=13, color=INK)
    ax.set_xlim(5, 0)
    ax.set_ylim(0, 0.45)
    ax.set_xticks([5, 4, 3, 2, 1, 0], ["5:00", "4:00", "3:00", "2:00", "1:00", "0:00"])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xlabel("Time left in overtime when the penalty is called")
    ax.set_ylabel("Penalty-shot success rate")
    ax.set_title("When should a PWHL team take the penalty shot?", loc="left")
    ax.legend(loc="lower left", fontsize=12)
    fig.text(0.01, 0.01, "Evenly matched teams, league-average goalie. Blue band: 90% interval.\nShooter lines: team-chosen "
             "shooter (shootout-like); shaded down to the cautious case (full historical NHL penalty-shot gap).",
             fontsize=10, color=MUTED)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, "hero_breakeven")
    plt.close(fig)


def _clock(seconds: float) -> str:
    s = int(round(seconds))
    return f"{s // 60}:{s % 60:02d}"


def _diverging_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("pp_ps", [BLUE, "#eeeeee", ORANGE])


def heatmap_shooter_goalie() -> None:
    """P(penalty shot is better) for each team's top 3 shooters vs each team's main goalie, at two times."""
    m = pd.read_csv(TABLES / "decision_matchups.csv")
    m["row"] = m["team"] + "  " + m["shooter"]
    m["col"] = m["goalie"] + " (" + m["goalie_team"] + ")"
    rows = m.drop_duplicates("row").sort_values(["team", "rank"])["row"].tolist()
    col_order = m.groupby("col")["p_shot_mean"].mean().sort_values().index.tolist()   # strongest goalie first
    times = [c for c in m.columns if c.startswith("P_shot_better_")]
    fig, axes = plt.subplots(1, len(times), figsize=(22, 16), sharey=True)
    fig.subplots_adjust(left=0.14, right=0.88, bottom=0.2, top=0.92, wspace=0.08)
    cmap = _diverging_cmap()
    for ax, col in zip(axes, times):
        secs = int(col.split("_")[-1].rstrip("s"))
        grid = m.pivot(index="row", columns="col", values=col).loc[rows, col_order]
        own = m.pivot(index="row", columns="col", values="own_team").loc[rows, col_order].astype(bool)
        vals = grid.mask(own)
        im = ax.imshow(vals.to_numpy(), cmap=cmap, vmin=0, vmax=1, aspect="auto")
        for i in range(vals.shape[0]):
            for j in range(vals.shape[1]):
                v = vals.iat[i, j]
                ax.text(j, i, "–" if np.isnan(v) else f"{v:.0%}", ha="center", va="center", fontsize=8, color=INK)
        ax.set_xticks(range(len(col_order)), col_order, rotation=60, ha="right", fontsize=10)
        ax.set_yticks(range(len(rows)), rows, fontsize=10)
        ax.set_title(f"{_clock(secs)} left in OT", loc="left")
        ax.grid(False)
        for k in range(3, len(rows), 3):
            ax.axhline(k - 0.5, color="white", lw=2)
    cb = fig.colorbar(im, cax=fig.add_axes((0.9, 0.35, 0.015, 0.45)))
    cb.set_label("P(penalty shot is the better choice)")
    cb.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    fig.suptitle("Shooter vs goalie: how often the penalty shot beats the power play", x=0.01, ha="left", fontsize=20)
    fig.text(0.01, 0.01, "Each team's top 3 shooters (2026-27 rosters) vs each team's main goalie. Evenly matched teams; "
             "main case (team-chosen shooter).\nOrange: shoot. Blue: power play. Columns: strongest goalie first. "
             "– = own goalie.", fontsize=11, color=MUTED)
    save(fig, "heatmap_shooter_goalie")
    plt.close(fig)


def team_cards() -> None:
    """One panel per team: P(penalty shot better) vs time left for its best shooter, vs each opponent."""
    cards = pd.read_csv(TABLES / "decision_team_cards.csv")
    curves = pd.read_csv(TABLES / "decision_team_card_curves.csv")
    curves = curves[curves["seconds_left"] >= 1]
    fig, axes = plt.subplots(3, 4, figsize=(20, 13), sharex=True, sharey=True)
    for ax, (_, c) in zip(axes.ravel(), cards.iterrows()):
        cv = curves[curves["team"] == c["team"]]
        for opp, g in cv.groupby("opponent"):
            if opp != "league average":
                ax.plot(g["seconds_left"] / 60, g["P_shot_better"], color=MUTED, lw=0.8, alpha=0.5)
        avg = cv[cv["opponent"] == "league average"]
        ax.plot(avg["seconds_left"] / 60, avg["P_shot_better"], color=BLUE, lw=2.5)
        ax.axhline(0.5, color=INK, lw=0.8, ls="--")
        ax.set_xlim(5, 0)
        ax.set_ylim(0, 1)
        ax.set_title(f"{c['team']}  ·  {c['best_shooter']}", loc="left", fontsize=14)
        cross = c["crossover_vs_average"]
        early = c["P_shot_better_2m30_vs_average"]
        note = (f"Shoot inside {_clock(cross)}" if cross < 120 else "Close call all OT") + f"\nAt 2:30 left: {early:.0%}"
        ax.text(4.85, 0.95, note, va="top", fontsize=11, color=INK)
    for ax in axes[-1]:
        ax.set_xticks([5, 4, 3, 2, 1, 0], ["5:00", "4:00", "3:00", "2:00", "1:00", "0:00"])
        ax.set_xlabel("Time left in OT")
    for ax in axes[:, 0]:
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.set_ylabel("P(penalty shot better)")
    fig.suptitle("Team decision cards: when should each team's best shooter take the penalty shot?", x=0.01, ha="left", fontsize=20)
    fig.text(0.01, 0.005, "Blue: vs league-average goalie and opponent. Grey: vs each opponent (its main goalie, real shootout "
             "matchup). Above 50%: shoot. 2026-27 rosters; main case (team-chosen shooter).", fontsize=11, color=MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    save(fig, "team_cards")
    plt.close(fig)


def decision_cost() -> None:
    """Win-probability difference between the two options (penalty shot minus power play) by time left."""
    cost = pd.read_csv(TABLES / "decision_cost_by_time.csv")
    cost = cost[cost["seconds_left"] >= 1]
    fig, ax = plt.subplots(figsize=(12, 7))
    for (who, g), color, ls in zip(cost.groupby("shooter", sort=False), (ORANGE, AQUA), ("--", ":")):
        x = g["seconds_left"] / 60
        ax.fill_between(x, g["lo90"], g["hi90"], color=color, alpha=0.15, linewidth=0)
        ax.plot(x, g["ps_minus_pp_mean"], color=color, ls=ls, label=who[0].upper() + who[1:])
    ax.axhline(0, color=INK, lw=1)
    ax.set_xlim(5, 0)
    ax.set_xticks([5, 4, 3, 2, 1, 0], ["5:00", "4:00", "3:00", "2:00", "1:00", "0:00"])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:+.0%}"))
    ax.set_xlabel("Time left in overtime when the penalty is called")
    ax.set_ylabel("Win probability: penalty shot minus power play")
    ax.text(4.95, ax.get_ylim()[1] * 0.95, "Penalty shot better ↑", va="top", fontsize=13)
    ax.text(4.95, ax.get_ylim()[0] * 0.9, "Power play better ↓", va="bottom", fontsize=13)
    ax.set_title("What the choice is worth: the cost of choosing wrong", loc="left")
    ax.legend(loc="upper center", fontsize=12)
    fig.text(0.01, 0.01, "Evenly matched teams, league-average goalie, main case. Bands: 90% intervals.", fontsize=10, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    save(fig, "decision_cost")
    plt.close(fig)


def backtest_calibration() -> None:
    """NHL leave-one-season-out backtest: predicted vs observed OT power-play conversion by time left."""
    cal = pd.read_csv(TABLES / "backtest_nhl_calibration.csv")
    summ = pd.read_csv(TABLES / "backtest_nhl_summary.csv").set_index("method")
    fig, ax = plt.subplots(figsize=(11, 6.5))
    x = np.arange(len(cal))
    ax.errorbar(x - 0.08, cal["observed"], yerr=[cal["observed"] - cal["observed_lo90"], cal["observed_hi90"] - cal["observed"]],
                fmt="o", color=INK, capsize=0, elinewidth=1.5, markersize=9, label="Observed (90% interval)")
    ax.plot(x + 0.08, cal["predicted"], "D", color=BLUE, markersize=10, label="Model, predicted out of sample")
    ax.plot(x, cal["constant_baseline"], "_", color=ORANGE, markersize=40, mew=3, label="Constant-rate baseline")
    for i, n in enumerate(cal["n"]):
        ax.text(i, 0.02, f"n = {n}", ha="center", fontsize=12, color=MUTED)
    ax.set_xticks(x, [f"{t} left" for t in cal["time_left"]])
    ax.set_xlim(-0.5, len(cal) - 0.5)
    ax.set_ylim(0, 0.5)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_ylabel("OT power plays that score")
    ax.set_title("NHL backtest: the model predicts OT power plays it never saw", loc="left")
    ax.legend(loc="upper left", fontsize=12)
    b_m, b_c = summ.loc["model (leave-one-season-out)", "brier"], summ.loc["constant rate", "brier"]
    fig.text(0.01, 0.01, f"580 NHL OT power plays, 2015-16 to 2025-26; each season predicted from the other ten. "
             f"Brier score: model {b_m:.3f} vs constant rate {b_c:.3f} (lower is better).", fontsize=10, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    save(fig, "backtest_calibration")
    plt.close(fig)


def sensitivity_chart() -> None:
    """P(penalty shot better) for an average shooter at 2:30 and 1:30 left under each sensitivity case."""
    s = pd.read_csv(TABLES / "sensitivity.csv")
    s = s.iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(12, 7))
    y = np.arange(len(s))
    for col, color, marker, label in (("P_shot_better_2m30", BLUE, "o", "2:30 left"),
                                      ("P_shot_better_1m30", ORANGE, "D", "1:30 left"),
                                      ("P_shot_better_1m00", AQUA, "s", "1:00 left")):
        ax.plot(s[col], y, marker, color=color, markersize=10, label=label, ls="none")
    ax.axvline(0.5, color=INK, lw=1, ls="--")
    ax.set_yticks(y, s["case"])
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xlabel("P(penalty shot is the better choice), average shooter")
    fig.suptitle("Robustness: late OT always says shoot; early OT depends on assumptions", x=0.01, ha="left", fontsize=18)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=12, ncols=3)
    ax.grid(axis="y", visible=False)
    fig.text(0.01, 0.01, "Evenly matched teams, league-average goalie. Right of the dashed line: shoot.", fontsize=10, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    save(fig, "sensitivity")
    plt.close(fig)


def main() -> None:
    style()
    if (TABLES / "backtest_nhl_calibration.csv").exists():
        backtest_calibration()
    if (TABLES / "sensitivity.csv").exists():
        sensitivity_chart()
    hero_breakeven()
    heatmap_shooter_goalie()
    team_cards()
    decision_cost()
    print(f"Saved figures to {FIGS}")


if __name__ == "__main__":
    main()
