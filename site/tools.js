/* Live tool demos.
   Both run the real logic in the browser. The brake explorer replays 48
   measured episodes; the power calculator runs the same simulation evalgate
   runs. Nothing here is a mock-up, and nothing calls a model — the site is
   static and the numbers are already measured. */

/* ── 1. Brake trigger explorer ─────────────────────────────── */

let EPISODES = [];

const el = (id) => document.getElementById(id);

function evaluateTrigger(mode, tvCut, confCut) {
  const flagged = EPISODES.filter((e) =>
    mode === "instability" ? e.tv > tvCut
    : mode === "confidence" ? e.conf < confCut
    : e.tv > tvCut || e.conf < confCut
  );
  const errors = EPISODES.filter((e) => !e.c);
  const caught = flagged.filter((e) => !e.c);
  return {
    flagged: flagged.length,
    total: EPISODES.length,
    caught: caught.length,
    errors: errors.length,
    precision: flagged.length ? caught.length / flagged.length : 0,
    recall: errors.length ? caught.length / errors.length : 0,
    missed: errors.length - caught.length
  };
}

function renderBrake() {
  if (!EPISODES.length) return;
  const mode = document.querySelector('input[name="trigmode"]:checked').value;
  const tv = Number(el("tvCut").value);
  const cf = Number(el("confCut").value);
  el("tvCutVal").textContent = tv.toFixed(2);
  el("confCutVal").textContent = cf.toFixed(2);
  el("tvRow").style.opacity = mode === "confidence" ? 0.35 : 1;
  el("confRow").style.opacity = mode === "instability" ? 0.35 : 1;

  const r = evaluateTrigger(mode, tv, cf);
  el("brakeStats").innerHTML = `
    <div class="stat"><b>${((r.flagged / r.total) * 100).toFixed(0)}%</b><span>of decisions stopped</span></div>
    <div class="stat"><b>${((r.recall) * 100).toFixed(0)}%</b><span>of errors caught</span></div>
    <div class="stat"><b>${((r.precision) * 100).toFixed(0)}%</b><span>of stops were real errors</span></div>
    <div class="stat stat-warn"><b>${r.missed}</b><span>errors passed through</span></div>`;

  el("brakeGrid").replaceChildren(
    ...EPISODES.map((e) => {
      const stopped =
        mode === "instability" ? e.tv > tv
        : mode === "confidence" ? e.conf < cf
        : e.tv > tv || e.conf < cf;
      const d = document.createElement("div");
      d.className =
        "ep " + (stopped ? "ep-stop " : "") + (e.c ? "ep-ok" : "ep-err");
      d.title = `episode ${e.id} — ${e.c ? "correct" : "WRONG"}, reorder shift ${e.tv}, confidence ${e.conf}${stopped ? " — STOPPED" : ""}`;
      return d;
    })
  );
}

/* ── 2. Power calculator (the evalgate arithmetic) ─────────── */

function comb(n, k) {
  if (k < 0 || k > n) return 0;
  let r = 1;
  for (let i = 0; i < k; i++) r = (r * (n - i)) / (i + 1);
  return r;
}

/** P(both conditions give identical per-unit rates) under the null.
    This is the exact number that made TURN-1's margin unsatisfiable. */
function tieRate(base, reps) {
  let s = 0;
  for (let k = 0; k <= reps; k++) {
    const p = comb(reps, k) * base ** k * (1 - base) ** (reps - k);
    s += p * p;
  }
  return s;
}

function mulberry(seed) {
  return function () {
    seed |= 0; seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function gauss(rnd, mu, sd) {
  const u = Math.max(1e-9, rnd()), v = rnd();
  return mu + sd * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function permP(deltas, iters, rnd) {
  const obs = deltas.reduce((a, b) => a + b, 0) / deltas.length;
  let ge = 0;
  for (let i = 0; i < iters; i++) {
    let m = 0;
    for (const d of deltas) m += rnd() < 0.5 ? d : -d;
    if (m / deltas.length >= obs) ge++;
  }
  return (ge + 1) / (iters + 1);
}

function simPower(effect, units, reps, base, trials, alpha, rnd) {
  let hits = 0;
  for (let t = 0; t < trials; t++) {
    const ds = [];
    for (let u = 0; u < units; u++) {
      const p = Math.min(0.97, Math.max(0.03, gauss(rnd, base, 0.12)));
      const a = Math.min(1, p + effect / 2), b = Math.max(0, p - effect / 2);
      let fa = 0, fb = 0;
      for (let r = 0; r < reps; r++) { if (rnd() < a) fa++; if (rnd() < b) fb++; }
      ds.push(fa / reps - fb / reps);
    }
    if (permP(ds, 600, rnd) < alpha) hits++;
  }
  return hits / trials;
}

function renderPower() {
  const units = Number(el("pUnits").value);
  const reps = Number(el("pReps").value);
  const base = Number(el("pBase").value);
  const effect = Number(el("pEffect").value);
  el("pUnitsVal").textContent = units;
  el("pRepsVal").textContent = reps;
  el("pBaseVal").textContent = base.toFixed(2);
  el("pEffectVal").textContent = effect.toFixed(2);

  const tie = tieRate(base, reps);
  const pw = simPower(effect, units, reps, base, 120, 0.01, mulberry(7));
  const calls = units * 2 * reps * 12;
  const verdict = pw >= 0.8 ? "adequate" : "UNDERPOWERED";

  el("powerStats").innerHTML = `
    <div class="stat ${pw >= 0.8 ? "" : "stat-warn"}"><b>${(pw * 100).toFixed(0)}%</b><span>power to detect it</span></div>
    <div class="stat"><b>${(tie * 100).toFixed(0)}%</b><span>of units will tie</span></div>
    <div class="stat"><b>${calls.toLocaleString()}</b><span>model calls needed</span></div>
    <div class="stat ${pw >= 0.8 ? "" : "stat-warn"}"><b>${verdict}</b><span>at 80% target</span></div>`;

  el("powerNote").textContent =
    pw >= 0.8
      ? "This design can find the effect you are looking for. Ties still discard information, so a test that uses effect magnitudes beats a sign test."
      : "This design will probably miss a real effect of that size. Increase units or replicates before spending — the arithmetic is free and the run is not.";
}

/* ── wiring ────────────────────────────────────────────────── */

fetch("pid2-episodes.json")
  .then((r) => r.json())
  .then((d) => { EPISODES = d; renderBrake(); })
  .catch(() => {
    el("brakeStats").innerHTML =
      '<div class="stat stat-warn"><b>—</b><span>episode data unavailable</span></div>';
  });

for (const id of ["tvCut", "confCut"]) el(id).addEventListener("input", renderBrake);
for (const r of document.querySelectorAll('input[name="trigmode"]'))
  r.addEventListener("change", renderBrake);
for (const id of ["pUnits", "pReps", "pBase", "pEffect"])
  el(id).addEventListener("input", renderPower);

el("presetTurn1").addEventListener("click", () => {
  el("pUnits").value = 24; el("pReps").value = 3;
  el("pBase").value = 0.85; el("pEffect").value = 0.20;
  renderPower();
});
el("presetTurn2").addEventListener("click", () => {
  el("pUnits").value = 32; el("pReps").value = 5;
  el("pBase").value = 0.70; el("pEffect").value = 0.20;
  renderPower();
});

renderPower();
