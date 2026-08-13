/* FOIL — the order effect.
   Every number rendered here is measured, not modelled. Orderings that were
   never run render blank rather than interpolated: the point of the page is
   that this is data. */

const DATA = {
  truth: "south",
  actions: ["north", "south", "east", "west"],
  uncovered: ["east"],
  liar: "Charlie",
  sources: ["Alpha", "Bravo", "Charlie", "Delta"],
  reports: {
    Alpha:   "south is clear; north is blocked.",
    Bravo:   "south is clear; north is blocked.",
    Charlie: "west, south are blocked.",
    Delta:   "west, north are blocked."
  },
  // key = source order joined by "|"
  measured: {
    "Alpha|Bravo|Charlie|Delta": { south: 0.99,  east: 0.01  },
    "Delta|Bravo|Charlie|Alpha": { south: 0.875, east: 0.125 },
    "Bravo|Alpha|Charlie|Delta": { south: 0.97,  east: 0.03  },
    "Charlie|Delta|Bravo|Alpha": { south: 1.0 },
    "Bravo|Delta|Alpha|Charlie": { south: 1.0 },
    "Delta|Bravo|Alpha|Charlie": { south: 0.005, east: 0.995 }
  },
  n: 200,
  canonical: ["Alpha", "Bravo", "Charlie", "Delta"],
  heroA: ["Bravo", "Delta", "Alpha", "Charlie"],
  heroB: ["Delta", "Bravo", "Alpha", "Charlie"],
  flip:  ["Delta", "Bravo", "Alpha", "Charlie"]
};

const key = (o) => o.join("|");
const pCorrect = (d) => (d && d[DATA.truth]) || 0;

/* ── report rows ──────────────────────────────────────────── */

function row(name, opts = {}) {
  const li = document.createElement("li");
  if (name === DATA.liar) li.classList.add("liar");
  if (opts.moved) li.classList.add("moved");
  const who = document.createElement("span");
  who.className = "who";
  who.textContent = name;
  const txt = document.createElement("span");
  txt.textContent = DATA.reports[name];
  li.append(who, txt);
  return li;
}

function renderStatic(sel, order, movedSet) {
  const ol = document.querySelector(sel);
  ol.replaceChildren(...order.map((s) => row(s, { moved: movedSet.has(s) })));
}

/* The hero pair differs by swapping the first two rows; mark exactly those. */
renderStatic('[data-col="a"]', DATA.heroA, new Set(["Bravo", "Delta"]));
renderStatic('[data-col="b"]', DATA.heroB, new Set(["Bravo", "Delta"]));

/* ── live reorderable list ────────────────────────────────── */

let order = [...DATA.canonical];
const live = document.getElementById("live");

function renderLive() {
  live.replaceChildren(
    ...order.map((name, i) => {
      const li = row(name);
      li.draggable = true;
      li.dataset.i = i;

      const grip = document.createElement("span");
      grip.className = "grip";
      for (const [label, delta, disabled] of [
        ["↑", -1, i === 0],
        ["↓", 1, i === order.length - 1]
      ]) {
        const b = document.createElement("button");
        b.type = "button";
        b.textContent = label;
        b.disabled = disabled;
        b.setAttribute(
          "aria-label",
          `Move ${name} ${delta < 0 ? "up" : "down"}`
        );
        b.addEventListener("click", () => move(i, i + delta));
        grip.append(b);
      }
      li.append(grip);

      li.addEventListener("dragstart", (e) => {
        e.dataTransfer.effectAllowed = "move";
        e.dataTransfer.setData("text/plain", String(i));
        li.classList.add("dragging");
      });
      li.addEventListener("dragend", () => li.classList.remove("dragging"));
      li.addEventListener("dragover", (e) => {
        e.preventDefault();
        li.classList.add("over");
      });
      li.addEventListener("dragleave", () => li.classList.remove("over"));
      li.addEventListener("drop", (e) => {
        e.preventDefault();
        li.classList.remove("over");
        move(Number(e.dataTransfer.getData("text/plain")), i);
      });
      return li;
    })
  );
}

function move(from, to) {
  if (to < 0 || to >= order.length || from === to) return;
  const [x] = order.splice(from, 1);
  order.splice(to, 0, x);
  update();
}

function setOrder(next) {
  order = [...next];
  update();
}

/* ── readout panel ────────────────────────────────────────── */

const panel = document.getElementById("panel");
const bars = document.getElementById("bars");
const panelTitle = document.getElementById("panelTitle");
const panelN = document.getElementById("panelN");
const panelFoot = document.getElementById("panelFoot");

function renderPanel(dist) {
  if (!dist) {
    panel.classList.add("unmeasured");
    panelTitle.textContent = "Not measured";
    panelN.textContent = "n = 0";
    const b = document.createElement("div");
    b.className = "blank";
    b.textContent =
      "This ordering was never run. Six of the twenty-four permutations were measured; the rest are blank rather than estimated.";
    bars.replaceChildren(b);
    panelFoot.textContent = "";
    return;
  }

  panel.classList.remove("unmeasured");
  panelTitle.textContent = "Measured";
  panelN.textContent = `n = ${DATA.n}`;

  bars.replaceChildren(
    ...DATA.actions.map((a) => {
      const v = dist[a] || 0;
      const wrap = document.createElement("div");
      wrap.className =
        "bar " + (a === DATA.truth ? "bar-a" : v > 0 ? "bar-b" : "bar-z");

      const top = document.createElement("div");
      top.className = "bar-top";
      const nm = document.createElement("span");
      nm.className = "bar-name";
      nm.textContent = a;
      if (a === DATA.truth) {
        const m = document.createElement("span");
        m.className = "mark";
        m.textContent = "correct";
        nm.append(m);
      } else if (DATA.uncovered.includes(a)) {
        const m = document.createElement("span");
        m.className = "mark";
        m.textContent = "no scout covers";
        nm.append(m);
      }
      const val = document.createElement("span");
      val.className = "bar-val";
      val.textContent = v.toFixed(3);
      top.append(nm, val);

      const track = document.createElement("div");
      track.className = "bar-track";
      const fill = document.createElement("div");
      fill.className = "bar-fill";
      track.append(fill);
      wrap.append(top, track);
      requestAnimationFrame(() => {
        fill.style.width = `${v * 100}%`;
      });
      return wrap;
    })
  );

  const p = pCorrect(dist);
  panelFoot.textContent =
    p >= 0.5
      ? `Lands on the correct route ${(p * 100).toFixed(1)}% of the time.`
      : `Inverted — lands on the correct route only ${(p * 100).toFixed(1)}% of the time.`;
}

/* ── lattice ──────────────────────────────────────────────── */

function permutations(arr) {
  if (arr.length <= 1) return [arr];
  return arr.flatMap((x, i) =>
    permutations([...arr.slice(0, i), ...arr.slice(i + 1)]).map((r) => [x, ...r])
  );
}

const ALL = permutations(DATA.sources);
const lattice = document.getElementById("lattice");

function buildLattice() {
  lattice.replaceChildren(
    ...ALL.map((perm) => {
      const k = key(perm);
      const dist = DATA.measured[k];
      const cell = document.createElement(dist ? "button" : "div");
      cell.className = "cell";
      cell.dataset.k = k;
      cell.setAttribute("role", "listitem");

      const seq = document.createElement("span");
      seq.className = "seq";
      seq.textContent = perm.map((s) => s.slice(0, 2).toUpperCase()).join(" ");

      const out = document.createElement("span");
      out.className = "out";
      if (dist) {
        const p = pCorrect(dist);
        out.textContent = p.toFixed(3);
        cell.classList.add("m", p >= 0.5 ? "m-a" : "m-b");
        cell.type = "button";
        cell.setAttribute(
          "aria-label",
          `Load ordering ${perm.join(", ")}. Correct ${p.toFixed(3)}.`
        );
        cell.addEventListener("click", () => {
          setOrder(perm);
          document.getElementById("live").scrollIntoView({
            behavior: matchMedia("(prefers-reduced-motion: reduce)").matches
              ? "auto"
              : "smooth",
            block: "center"
          });
        });
      } else {
        out.textContent = "—";
        out.classList.add("none");
      }
      cell.append(seq, out);
      return cell;
    })
  );
}

/* ── glue ─────────────────────────────────────────────────── */

function update() {
  renderLive();
  renderPanel(DATA.measured[key(order)]);
  const k = key(order);
  for (const c of lattice.children) c.classList.toggle("sel", c.dataset.k === k);
}

document.getElementById("reset").addEventListener("click", () => setOrder(DATA.canonical));
document.getElementById("findflip").addEventListener("click", () => setOrder(DATA.flip));

buildLattice();
update();
