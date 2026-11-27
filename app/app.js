/* Breakeven Breakaway calculator.
 *
 * All decision math is in the pure functions below (also used by tests/app_check.js). For each
 * posterior draw i, with t seconds left, shootout probability S and penalty-shot success p:
 *   W0(t)       = 0.5 * (1 - surv[i][t]) + surv[i][t] * S
 *   P(win | PS) = p + (1 - p) * W0(t)
 *   P(win | PP) = ppGoal[i][d] + ppNone[i][d] * W0(t - d),   d = min(120, t)
 *   p*(t)       = (P(win | PP) - W0(t)) / (1 - W0(t))
 * S comes from shootoutDraws: each team's top five shooters against the other side's selected goalie.
 * Data come from src/export_app.py (thinned posterior draws of the fitted models).
 */
"use strict";

const PP_LEN = 120;
const OT_LEN = 300;

// ---------------------------------------------------------------- math

function expit(x) { return 1 / (1 + Math.exp(-x)); }

function quantile(sorted, q) {            // linear interpolation, same as numpy's default
  const pos = (sorted.length - 1) * q;
  const lo = Math.floor(pos), hi = Math.ceil(pos);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
}

function summarize(values) {
  const s = Float64Array.from(values).sort();
  let sum = 0;
  for (const v of values) sum += v;
  return { mean: sum / values.length, lo: quantile(s, 0.05), hi: quantile(s, 0.95) };
}

function effectAt(effects, id, i) { return id && effects[id] ? effects[id][i] : 0; }

/** P(team wins the shootout) per draw, exactly (same as src/model_ot_winprob.shootout_win_prob):
 *  five rounds (Poisson-binomial goal counts), then sudden death with each team's average shooter.
 *  Your shooters face the opponent's goalie; the opponent's shooters face your goalie. */
function shootoutDraws(model, team, ownGoalie, opponent, oppGoalie) {
  const d = model.draws, n = d.alpha.length;
  const sa = model.teams.find((t) => t.code === team).shootoutShooters;
  const sb = model.teams.find((t) => t.code === opponent).shootoutShooters;
  const out = new Float64Array(n);
  const dist = (shooters, goalie, i) => {
    const pr = new Float64Array(shooters.length + 1); pr[0] = 1;
    let mean = 0;
    shooters.forEach((sid, j) => {
      const q = expit(d.alpha[i] + effectAt(model.shooterEffects, sid, i) + effectAt(model.goalieEffects, goalie, i));
      mean += q / shooters.length;
      for (let k = j + 1; k >= 1; k--) pr[k] = pr[k] * (1 - q) + pr[k - 1] * q;
      pr[0] *= 1 - q;
    });
    return { pr, mean };
  };
  for (let i = 0; i < n; i++) {
    const A = dist(sa, oppGoalie, i), B = dist(sb, ownGoalie, i);
    let win = 0, tie = 0, below = 0;
    for (let a = 0; a < A.pr.length; a++) {
      win += A.pr[a] * below;            // B scored fewer than a
      tie += A.pr[a] * B.pr[a];
      below += B.pr[a];
    }
    const pa = A.mean, pb = B.mean;
    out[i] = win + tie * (pa * (1 - pb)) / (pa * (1 - pb) + pb * (1 - pa));
  }
  return out;
}

function pShotDraws(model, shooterId, goalieId, deltaCase) {
  const d = model.draws;
  const s = shooterId ? model.shooterEffects[shooterId] : null;
  const g = goalieId ? model.goalieEffects[goalieId] : null;
  const out = new Float64Array(d.alpha.length);
  for (let i = 0; i < out.length; i++) {
    const delta = deltaCase === "nhl" ? d.deltaNhl[i] : 0;
    out[i] = expit(d.alpha[i] + delta + (s ? s[i] : 0) + (g ? g[i] : 0));
  }
  return out;
}

function w0(model, i, t, S) {
  const sv = model.draws.surv[i][t];
  return 0.5 * (1 - sv) + sv * S;
}

function winPp(model, i, t, S) {
  const d = Math.min(PP_LEN, t);
  return model.draws.ppGoal[i][d] + model.draws.ppNone[i][d] * w0(model, i, t - d, S);
}

function breakevenDraw(model, i, t, S) {
  const W = w0(model, i, t, S);
  return W >= 1 ? 0 : (winPp(model, i, t, S) - W) / (1 - W);
}

/** Everything shown for one moment: per-draw comparison summarized. */
function evaluate(model, { team, shooter, ownGoalie, opponent, goalie, seconds, delta }, S) {
  S = S || shootoutDraws(model, team, ownGoalie, opponent, goalie);
  const p = pShotDraws(model, shooter, goalie, delta);
  const n = p.length;
  const pstar = new Float64Array(n), winPs = new Float64Array(n), winPpArr = new Float64Array(n), diff = new Float64Array(n);
  let better = 0;
  for (let i = 0; i < n; i++) {
    pstar[i] = breakevenDraw(model, i, seconds, S[i]);
    const W = w0(model, i, seconds, S[i]);
    winPs[i] = p[i] + (1 - p[i]) * W;
    winPpArr[i] = winPp(model, i, seconds, S[i]);
    diff[i] = winPs[i] - winPpArr[i];
    if (p[i] > pstar[i]) better++;
  }
  return {
    pShot: summarize(p), pstar: summarize(pstar), winPs: summarize(winPs), winPp: summarize(winPpArr),
    diff: summarize(diff), pShotBetter: better / n, shootout: summarize(S),
  };
}

/** Break-even curve over the whole overtime (depends on the shootout, not the penalty shooter). */
function breakevenCurve(model, S) {
  const n = model.draws.alpha.length;
  const curve = [];
  const buf = new Float64Array(n);
  for (let t = 1; t <= OT_LEN; t++) {
    for (let i = 0; i < n; i++) buf[i] = breakevenDraw(model, i, t, S[i]);
    curve.push({ t, ...summarize(buf) });
  }
  return curve;
}

/** Last second (scanning down from 5:00) at which the shot is better in at least half the draws. */
function crossover(model, S, p) {
  const n = p.length;
  for (let t = OT_LEN; t >= 1; t--) {
    let better = 0;
    for (let i = 0; i < n; i++) if (p[i] > breakevenDraw(model, i, t, S[i])) better++;
    if (better / n >= 0.5) return t;
  }
  return null;
}

// ---------------------------------------------------------------- formatting

const pct = (x, digits = 0) => `${(100 * x).toFixed(digits)}%`;
const clock = (s) => `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
const signedPts = (x) => `${x >= 0 ? "+" : "−"}${Math.abs(100 * x).toFixed(1)} pts`;

function verdict(prob) {
  if (prob >= 0.6) return { key: "shot", text: "Take the penalty shot" };
  if (prob <= 0.4) return { key: "pp", text: "Take the power play" };
  return { key: "close", text: prob >= 0.5 ? "Close call — slight lean to the penalty shot" : "Close call — slight lean to the power play" };
}

// ---------------------------------------------------------------- page

function init() {
  const $ = (id) => document.getElementById(id);
  const els = {
    team: $("team"), shooter: $("shooter"), ownGoalie: $("own-goalie"), opponent: $("opponent"), goalie: $("goalie"),
    time: $("time"), timeOut: $("time-out"), delta: document.querySelectorAll('input[name="delta"]'),
    verdict: $("verdict"), prob: $("prob"), details: $("details"), chart: $("chart"), note: $("chart-note"),
    status: $("status"),
  };
  let model = null;
  let curveKey = "", curve = null, shootout = null;

  fetch("data/model.json")
    .then((r) => { if (!r.ok) throw new Error(r.statusText); return r.json(); })
    .then((m) => { model = m; setup(); })
    .catch((e) => { els.status.textContent = `Could not load the model data (${e.message}).`; });

  function option(value, label) { const o = document.createElement("option"); o.value = value; o.textContent = label; return o; }

  function setup() {
    for (const sel of [els.team, els.opponent]) {
      for (const t of model.teams) sel.append(option(t.code, t.name));
    }
    els.team.value = model.teams[0].code;
    els.opponent.value = model.teams[1].code;
    els.team.addEventListener("change", () => {
      if (separate(els.team, els.opponent)) fillGoalies(els.goalie, els.opponent.value);
      fillShooters(); fillGoalies(els.ownGoalie, els.team.value); update();
    });
    els.opponent.addEventListener("change", () => {
      if (separate(els.opponent, els.team)) { fillShooters(); fillGoalies(els.ownGoalie, els.team.value); }
      fillGoalies(els.goalie, els.opponent.value); update();
    });
    for (const el of [els.shooter, els.ownGoalie, els.goalie, els.time]) el.addEventListener("input", update);
    els.delta.forEach((r) => r.addEventListener("change", update));
    let resizeTimer = null;
    window.addEventListener("resize", () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(update, 150); });
    fillShooters(); fillGoalies(els.ownGoalie, els.team.value); fillGoalies(els.goalie, els.opponent.value);
    els.status.textContent = "";
    $("meta").textContent = `Model updated ${model.meta.generated} · ${model.meta.draws} posterior draws`;
    update();
  }

  function teamOf(code) { return model.teams.find((t) => t.code === code); }

  // A team can't play itself: if both menus match, move the other one. Returns true if it moved.
  function separate(changed, other) {
    if (changed.value !== other.value) return false;
    other.value = model.teams.find((t) => t.code !== changed.value).code;
    return true;
  }

  function fillShooters() {
    els.shooter.replaceChildren(option("", "League-average shooter"));
    const t = teamOf(els.team.value);
    for (const s of t.skaters) {
      els.shooter.append(option(s.id || "", s.id ? s.name : `${s.name} (no shootout history: league average)`));
    }
    const best = t.skaters.find((s) => s.id);
    els.shooter.value = best ? best.id : "";
  }

  function fillGoalies(sel, code) {
    sel.replaceChildren(option("", "League-average goalie"));
    const t = teamOf(code);
    for (const g of t.goalies) {
      sel.append(option(g.id || "", g.id ? g.name : `${g.name} (no shootout history: league average)`));
    }
    sel.value = t.mainGoalie || "";
  }

  function state() {
    return {
      team: els.team.value, opponent: els.opponent.value,
      shooter: els.shooter.value || null, ownGoalie: els.ownGoalie.value || null, goalie: els.goalie.value || null,
      seconds: Math.max(1, Number(els.time.value)),
      delta: [...els.delta].find((r) => r.checked).value,
    };
  }

  function update() {
    if (!model) return;
    const st = state();
    els.timeOut.textContent = clock(st.seconds);
    const key = `${st.team}|${st.ownGoalie}|${st.opponent}|${st.goalie}`;
    if (key !== curveKey) {
      shootout = shootoutDraws(model, st.team, st.ownGoalie, st.opponent, st.goalie);
      curve = breakevenCurve(model, shootout);
      curveKey = key;
    }
    const res = evaluate(model, st, shootout);
    const v = verdict(res.pShotBetter);
    els.verdict.textContent = v.text;
    els.verdict.dataset.kind = v.key;
    els.prob.textContent = `The penalty shot is the better choice in ${pct(res.pShotBetter)} of model draws.`;
    const row = (label, s) => `<tr><th scope="row">${label}</th><td>${pct(s.mean, 1)}</td><td class="range">${pct(s.lo)}–${pct(s.hi)}</td></tr>`;
    els.details.innerHTML =
      `<table><thead><tr><th></th><th>Estimate</th><th class="range">90% range</th></tr></thead><tbody>` +
      row("Shooter scores on the penalty shot", res.pShot) +
      row(`Break-even rate with ${clock(st.seconds)} left`, res.pstar) +
      row("Win probability: penalty shot", res.winPs) +
      row("Win probability: power play", res.winPp) +
      `<tr><th scope="row">Gain from the penalty shot</th><td>${signedPts(res.diff.mean)}</td><td class="range">${signedPts(res.diff.lo)} to ${signedPts(res.diff.hi)}</td></tr>` +
      `<tr><th scope="row">Your team wins a shootout</th><td>${pct(res.shootout.mean, 1)}</td><td class="range">${pct(res.shootout.lo)}–${pct(res.shootout.hi)}</td></tr>` +
      `</tbody></table>`;
    const p = pShotDraws(model, st.shooter, st.goalie, st.delta);
    const cross = crossover(model, shootout, p);
    els.note.textContent = cross === null ? "With this shooter and goalie, the power play is better at every point in overtime."
      : cross >= OT_LEN ? "With this shooter and goalie, the penalty shot is at least even all overtime."
      : `With this shooter and goalie, the penalty shot becomes the better choice inside ${clock(cross)}.`;
    drawChart(curve, res.pShot, st.seconds, cross);
  }

  function drawChart(curve, shooter, seconds, cross) {
    // Drawn at the container's real width so text stays readable on phones.
    const W = Math.max(300, Math.round(els.chart.clientWidth || 640));
    const H = Math.round(Math.min(340, Math.max(230, W * 0.55)));
    const m = { l: 44, r: 12, t: 12, b: 40 };
    const yMax = Math.min(0.6, Math.ceil((Math.max(...curve.map((c) => c.hi), shooter.hi) + 0.03) * 20) / 20);
    const x = (t) => m.l + ((OT_LEN - t) / (OT_LEN - 1)) * (W - m.l - m.r);
    const y = (v) => H - m.b - (v / yMax) * (H - m.t - m.b);
    const band = curve.map((c) => `${x(c.t)},${y(c.hi)}`).join(" ") + " " +
      [...curve].reverse().map((c) => `${x(c.t)},${y(c.lo)}`).join(" ");
    const line = curve.map((c) => `${x(c.t)},${y(c.mean)}`).join(" ");
    let grid = "";
    for (let v = 0; v <= yMax + 1e-9; v += 0.1) {
      grid += `<line class="grid" x1="${m.l}" x2="${W - m.r}" y1="${y(v)}" y2="${y(v)}"/>` +
        `<text class="tick" x="${m.l - 6}" y="${y(v) + 4}" text-anchor="end">${pct(v)}</text>`;
    }
    const ticks = W < 420 ? [300, 180, 120, 60, 1] : [300, 240, 180, 120, 60, 1];
    for (const t of ticks) {
      grid += `<text class="tick" x="${x(t)}" y="${H - m.b + 18}" text-anchor="middle">${clock(t === 1 ? 0 : t)}</text>`;
    }
    const crossShade = cross && cross < OT_LEN ? `<rect class="shoot-zone" x="${x(cross)}" y="${m.t}" width="${x(1) - x(cross)}" height="${H - m.t - m.b}"/>` : "";
    els.chart.innerHTML =
      `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Break-even penalty-shot success rate by time left, compared with the selected shooter">` +
      crossShade + grid +
      `<rect class="shooter-band" x="${m.l}" y="${y(shooter.hi)}" width="${W - m.l - m.r}" height="${y(shooter.lo) - y(shooter.hi)}"/>` +
      `<line class="shooter-line" x1="${m.l}" x2="${W - m.r}" y1="${y(shooter.mean)}" y2="${y(shooter.mean)}"/>` +
      `<polygon class="be-band" points="${band}"/><polyline class="be-line" points="${line}"/>` +
      `<line class="now" x1="${x(seconds)}" x2="${x(seconds)}" y1="${m.t}" y2="${H - m.b}"/>` +
      `<circle class="now-dot" cx="${x(seconds)}" cy="${y(curve[seconds - 1].mean)}" r="5"/>` +
      `<text class="axis" x="${(W + m.l) / 2}" y="${H - 4}" text-anchor="middle">Time left in overtime</text>` +
      `</svg>`;
  }
}

if (typeof document !== "undefined") document.addEventListener("DOMContentLoaded", init);
if (typeof module !== "undefined") {
  module.exports = { evaluate, breakevenCurve, crossover, pShotDraws, shootoutDraws, breakevenDraw, summarize, quantile, verdict };
}
