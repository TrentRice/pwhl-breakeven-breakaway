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
    team: c.team, opponent: c.opponent, shooter: c.shooter, goalie: c.goalie, seconds: c.seconds, delta: c.delta,
  });
  const pairs = [
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
// sanity: 5 seconds left, average everything -> the shot should essentially always win
const late = app.evaluate(model, { team: "AVG", opponent: "AVG", shooter: null, goalie: null, seconds: 5, delta: "zero" });
if (late.pShotBetter < 0.99) { failures++; console.log(`SANITY late-OT pShotBetter = ${late.pShotBetter}`); }
console.log(`${cases.length} cases checked, ${failures} failures`);
process.exit(failures ? 1 : 0);
