// Checks that the calculator (app/app.js) reproduces src/decision.py on the exported draws.
// Run: node tests/app_check.js   (called by tests/test_app.py)
"use strict";
const fs = require("fs");
const path = require("path");
const app = require(path.join(__dirname, "..", "app", "app.js"));

const dataDir = path.join(__dirname, "..", "app", "data");
const model = JSON.parse(fs.readFileSync(path.join(dataDir, "model.json"), "utf8"));
const { tolerance, cases } = JSON.parse(fs.readFileSync(path.join(dataDir, "checks.json"), "utf8"));

let failures = 0;
for (const c of cases) {
  const r = app.evaluate(model, {
    team: c.team, opponent: c.opponent, shooter: c.shooter, ownGoalie: c.ownGoalie, goalie: c.goalie,
    seconds: c.seconds, delta: c.delta,
  });
  const pairs = [
    ["shootoutMean", r.shootout.mean],
    ["pShotMean", r.pShot.mean], ["pstarMean", r.pstar.mean], ["pShotBetter", r.pShotBetter],
    ["winPsMean", r.winPs.mean], ["winPpMean", r.winPp.mean],
  ];
  for (const [k, v] of pairs) {
    // pShotBetter counts draws: exported values are rounded to 5 decimals, so a borderline draw can flip.
    const tol = k === "pShotBetter" ? 3 / model.draws.alpha.length : tolerance;
    if (Math.abs(v - c[k]) > tol) {
      failures++;
      console.log(`MISMATCH ${c.team} vs ${c.opponent} t=${c.seconds} ${c.delta} ${k}: app ${v.toFixed(5)} vs python ${c[k].toFixed(5)}`);
    }
  }
}
// sanity: 5 seconds left, average shooter and goalies -> the shot should essentially always win
const [t1, t2] = model.teams;
const late = app.evaluate(model, { team: t1.code, opponent: t2.code, shooter: null, ownGoalie: null, goalie: null, seconds: 5, delta: "zero" });
if (late.pShotBetter < 0.99) { failures++; console.log(`SANITY late-OT pShotBetter = ${late.pShotBetter}`); }
// sanity: identical lineups and goalies on both sides -> a 50/50 shootout
const twin = { ...model, teams: [{ ...t1 }, { ...t1, code: "TWIN" }] };
const even = app.shootoutDraws(twin, t1.code, t1.mainGoalie, "TWIN", t1.mainGoalie);
if (even.some((v) => Math.abs(v - 0.5) > 1e-9)) { failures++; console.log("SANITY mirror shootout is not 50/50"); }
// sanity: a better opposing goalie lowers your shootout chance
const goalieMeans = Object.entries(model.goalieEffects).map(([id, e]) => [id, e.reduce((a, b) => a + b, 0) / e.length])
  .sort((a, b) => a[1] - b[1]);
const best = goalieMeans[0][0], worst = goalieMeans[goalieMeans.length - 1][0];
const mean = (a) => a.reduce((x, y) => x + y, 0) / a.length;
if (!(mean(app.shootoutDraws(model, t1.code, null, t2.code, best)) < mean(app.shootoutDraws(model, t1.code, null, t2.code, worst)))) {
  failures++; console.log("SANITY better opposing goalie should lower the shootout chance");
}
console.log(`${cases.length} cases checked, ${failures} failures`);
process.exit(failures ? 1 : 0);
