/* EBABY front-end: one START button drives the whole chain
   (upload -> rename -> colour -> barcode -> eBay), and only when all of
   that is sorted does the cropper open. The cropper NEVER renames files. */

const state = {
  batchName: null,
  pending: { used: [], new: [] },   // File objects picked before START
  quads: {},                        // filename -> quad (natural coords)
  ebayRows: [],                     // the fence-card rows, for in-place edits
};

/* ---------- tiny helpers ---------- */

const $ = (sel) => document.querySelector(sel);

async function api(path, opts = {}) {
  const res = await fetch(`/api${path}`, {
    headers: opts.body instanceof FormData ? {} : { "Content-Type": "application/json" },
    ...opts,
  });
  let data = null;
  try { data = await res.json(); } catch { /* non-JSON error body */ }
  if (!res.ok && !(data && data.error)) {
    throw new Error(`${path} failed (${res.status})`);
  }
  return data;
}

const RAIL = [
  ["upload", "LOAD UP", "loadup"],
  ["color", "CLEAN", "clean"],
  ["barcode", "INTERROGATE", "interrogate"],
  ["ebay", "FENCE", "fence"],
  ["crop", "CHOP", "chop"],
  ["done", "PAID", "paid"],
];

function setRail(stage) {
  const rail = $("#stage-rail");
  rail.innerHTML = "";
  const idx = RAIL.findIndex(([k]) => k === stage);
  RAIL.forEach(([k, label, icon], i) => {
    const chip = document.createElement("span");
    const img = document.createElement("img");
    // the active chip sits on amber — use the icons cut from the orange
    // sheet there; the black-sheet cuts stay for the dark chips
    img.src = i === idx ? `/img/icon-${icon}-on.png` : `/img/icon-${icon}.png`;
    img.alt = "";
    chip.appendChild(img);
    chip.appendChild(document.createTextNode(label));
    if (i < idx) chip.className = "done";
    if (i === idx) chip.className = "active";
    rail.appendChild(chip);
  });
}

/* ---------- sound: G-funk loop + synth stabs, one mute switch ---------- */

const AUDIO = {
  muted: localStorage.getItem("ebaby-muted") === "1",
  bgm: null,    // victory loop — only on a win
  intro: null,  // menu music — only on the opening screen
  _loop(key, file, vol) {
    if (this.muted) return;
    if (!this[key]) {
      this[key] = new Audio(`/sounds/${file}`);
      this[key].loop = true;
      this[key].volume = vol;
    }
    this[key].play().catch(() => {});  // browsers may block until a user gesture
  },
  startBgm() { this.stopIntro(); this._loop("bgm", "theme-loop.mp3", 0.22); },
  startIntro() { if (this.bgm) this.bgm.pause(); this._loop("intro", "intro-loop.mp3", 0.16); },
  stopIntro() { if (this.intro) this.intro.pause(); },
  stab(name) {
    if (this.muted) return;
    const a = new Audio(`/sounds/${name}.mp3`);
    a.volume = 0.5;
    a.play().catch(() => {});
  },
  toggle() {
    this.muted = !this.muted;
    localStorage.setItem("ebaby-muted", this.muted ? "1" : "0");
    if (this.muted) {
      if (this.bgm) this.bgm.pause();
      if (this.intro) this.intro.pause();
    } else if (!$("#screen-upload").hidden) this.startIntro();
    else if (!$("#screen-crop").hidden) this.startBgm();
    $("#btn-sound").innerHTML = this.muted ? "&#128263;" : "&#128266;";
  },
};

$("#btn-sound").addEventListener("click", () => AUDIO.toggle());
if (AUDIO.muted) $("#btn-sound").innerHTML = "&#128263;";

// Menu music from the jump. Autoplay is usually blocked until the user
// touches the page, so also arm a one-shot: first click/keypress starts it
// (skipped if the job has already moved past the opening screen).
AUDIO.startIntro();
document.addEventListener("pointerdown", () => {
  if (!$("#screen-upload").hidden && (!AUDIO.intro || AUDIO.intro.paused)) AUDIO.startIntro();
}, { once: true });

function show(id) {
  for (const el of document.querySelectorAll("section[id^='screen-']")) {
    el.hidden = el.id !== id;
  }
}

/* While a slow stage runs, poll /state so the mission log counts 3/7
   instead of hanging silently for minutes. */
async function withProgress(promise, step, text) {
  const timer = setInterval(async () => {
    try {
      const s = await api(`/batches/${state.batchName}/state`);
      const p = s && s.progress;
      if (p && p.total) step.li.textContent = `${text} — ${p.done}/${p.total}`;
      if (s && s.stage) setRail(s.stage);
    } catch { /* polling is best-effort */ }
  }, 1500);
  try {
    return await promise;
  } finally {
    clearInterval(timer);
  }
}

function logStep(text) {
  const li = document.createElement("li");
  li.textContent = text;
  li.className = "doing";
  $("#mission-log").appendChild(li);
  return {
    li,
    done(extra) { li.className = "done"; li.textContent = extra ? `${text} — ${extra}` : text; },
    fail(extra) { li.className = "fail"; if (extra) li.textContent = `${text} — ${extra}`; },
  };
}

/* ---------- 1. upload screen ---------- */

/* used discs need sets of 3 shots, new discs sets of 2 — catch a bad count
   HERE instead of letting the run start and come back BUSTED */
const SHOTS_PER_SET = { used: 3, new: 2 };

function refreshStartButton() {
  const total = state.pending.used.length + state.pending.new.length;
  const problems = [];
  for (const zone of ["used", "new"]) {
    const n = state.pending[zone].length;
    const per = SHOTS_PER_SET[zone];
    if (n % per) problems.push(
      `${zone.toUpperCase()}: ${n} photos isn't full sets of ${per} — add or remove some`);
  }
  $("#btn-start").disabled = total === 0 || problems.length > 0;
  $("#upload-error").textContent = total ? problems.join("  •  ") : "";
}

function renderFileList(zone) {
  const box = $(`#files-${zone}`);
  box.innerHTML = "";
  state.pending[zone].forEach((f, i) => {
    const chip = document.createElement("span");
    chip.className = "file-chip";
    chip.textContent = f.name;
    const x = document.createElement("button");
    x.textContent = "✕";
    x.title = "remove this photo";
    x.addEventListener("click", () => {
      state.pending[zone].splice(i, 1);
      renderFileList(zone);
    });
    chip.appendChild(x);
    box.appendChild(chip);
  });
  $(`#count-${zone}`).textContent = `${state.pending[zone].length} photos`;
  refreshStartButton();
}

function addFiles(zone, fileList) {
  if (fileList.length) AUDIO.stab("camera-shutter");
  for (const f of fileList) state.pending[zone].push(f);
  renderFileList(zone);
}

for (const zone of ["used", "new"]) {
  const input = $(`#input-${zone}`);
  input.addEventListener("change", (e) => { addFiles(zone, e.target.files); input.value = ""; });
  document.querySelector(`[data-pick="${zone}"]`).addEventListener("click", () => input.click());

  const dz = document.querySelector(`.dropzone[data-zone="${zone}"]`);
  dz.addEventListener("dragover", (e) => { e.preventDefault(); dz.classList.add("dragover"); });
  dz.addEventListener("dragleave", () => dz.classList.remove("dragover"));
  dz.addEventListener("drop", (e) => {
    e.preventDefault();
    dz.classList.remove("dragover");
    addFiles(zone, e.dataTransfer.files);
  });
}

/* ---------- 2. the automated job ---------- */

function busted(err) {
  AUDIO.stab("siren-busted");
  if (AUDIO.bgm) AUDIO.bgm.pause();
  // straight back to the opening screen — photos stay picked, ready to re-run
  show("screen-upload");
  setTimeout(() => { if (!$("#screen-upload").hidden) AUDIO.startIntro(); }, 2500);
  $("#upload-error").textContent = `BUSTED — ${err.message}`;
  $("#btn-start").disabled = false;
}

function checked(r, what) {
  if (r && r.error) throw new Error(`${what}: ${r.error}`);
  return r;
}

$("#btn-start").addEventListener("click", () => {
  const btn = $("#btn-start");
  if (btn.disabled) return;
  btn.disabled = true; // double-click = two batches
  AUDIO.stopIntro(); // menu music out, the job runs in silence
  $("#upload-error").textContent = "";
  runJob().catch(busted);
});

async function runJob() {
  show("screen-progress");
  $("#mission-log").innerHTML = "";

  // create batch
  let step = logStep("Casing the joint");
  state.batchName = `run-${new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19)}`;
  const created = await api("/batches", {
    method: "POST", body: JSON.stringify({ name: state.batchName }),
  });
  if (created && created.error) throw new Error(created.error);
  step.done(state.batchName);

  // upload
  step = logStep("Loading the merchandise into the van");
  for (const zone of ["used", "new"]) {
    for (const file of state.pending[zone]) {
      const form = new FormData();
      form.append("file", file);
      await api(`/batches/${state.batchName}/upload/${zone}`, { method: "POST", body: form });
    }
  }
  step.done(`${state.pending.used.length} used + ${state.pending.new.length} new`);

  // sequential rename (a_front... / 01_front...)
  step = logStep("Forging the first set of papers");
  for (const zone of ["used", "new"]) {
    const r = await api(`/batches/${state.batchName}/rename/apply`, {
      method: "POST", body: JSON.stringify({ zone }),
    });
    if (r && r.error) throw new Error(`${zone}: ${r.error}`);
  }
  step.done();

  await runFromColor();
}

/* the automated chain from colour onward — also the resume entry point */
async function runFromColor() {
  show("screen-progress");
  setRail("color");
  let step = logStep("Cleaning the goods so they look legit (the slow burn)");
  const c = checked(await withProgress(
    api(`/batches/${state.batchName}/color/run`, { method: "POST" }),
    step, "Cleaning the goods"), "colour stage");
  step.done(`${c.colored} processed`);
  await runFromBarcode();
}

async function runFromBarcode() {
  show("screen-progress");
  setRail("barcode");
  const step = logStep("Interrogating the barcodes");
  const b = checked(await withProgress(
    api(`/batches/${state.batchName}/barcode/run`, { method: "POST" }),
    step, "Interrogating the barcodes"), "barcode stage");
  const results = b.results || {};
  const missed = Object.entries(results).filter(([, v]) => !v);
  step.done(`${Object.keys(results).length - missed.length} hit, ${missed.length} missed`);

  if (missed.length) {
    renderBarcodeTable(results, b.crops || {});
    show("screen-barcode");
    return; // continues from the ROLL OUT button
  }
  await finishJob();
}

/* ---------- 3. barcode fixes (only when needed) ---------- */

function renderBarcodeTable(results, crops) {
  const tbody = $("#barcode-table tbody");
  tbody.innerHTML = "";
  for (const [key, digits] of Object.entries(results)) {
    const tr = document.createElement("tr");
    const status = digits
      ? `<span class="got">${digits}</span>`
      : `<span class="miss">MISSED</span>`;
    const fix = digits ? "" : `<input type="text" inputmode="numeric" data-fix-key="${key}" placeholder="type the digits" />`;
    // show the barcode crop so the digits can be read OFF THE SCREEN
    const evidence = crops && crops[key]
      ? `<img class="evidence" src="/api/files/${state.batchName}/3_barcodes/${crops[key]}" onerror="this.remove()" />`
      : "";
    tr.innerHTML = `<td>${key.toUpperCase()}</td><td>${evidence}</td><td>${status}</td><td>${fix}</td>`;
    tbody.appendChild(tr);
  }
}

$("#btn-barcode-continue").addEventListener("click", async () => {
  const btn = $("#btn-barcode-continue");
  if (btn.disabled) return; // double-click would run finishJob twice
  btn.disabled = true;
  AUDIO.stab("stab-passed"); // barcodes talked — rolling out
  try {
    for (const input of document.querySelectorAll("[data-fix-key]")) {
      const digits = input.value.trim();
      if (digits) {
        const r = await api(`/batches/${state.batchName}/barcode/manual`, {
          method: "POST",
          body: JSON.stringify({ key: input.dataset.fixKey, digits }),
        });
        if (r && r.error) {
          alert(`Barcode for ${input.dataset.fixKey.toUpperCase()}: ${r.error}`);
          return;
        }
      }
    }
    show("screen-progress");
    await finishJob();
  } catch (err) {
    busted(err);
  } finally {
    btn.disabled = false;
  }
});

/* ---------- 4. eBay + title rename, then open the cropper ---------- */

async function finishJob() {
  setRail("ebay");
  let step = logStep("Fencing the goods on eBay + issuing final identities");
  const e = await api(`/batches/${state.batchName}/ebay/run`, { method: "POST" });
  if (e && e.error) throw new Error(e.error);
  if (e.warning) step.fail(e.warning); else step.done();

  setRail("crop");
  step = logStep("Running everything through the chop shop");
  const cr = checked(await withProgress(
    api(`/batches/${state.batchName}/crop/run`, { method: "POST" }),
    step, "Chopping"), "chop shop");
  state.quads = cr.quads || {};
  step.done(`${Object.keys(state.quads).length} photos`);

  setRail("done");
  renderEbayPanel(e);
  renderCropGrid();
  AUDIO.startBgm();            // job approved — the G-funk only plays on a win
  show("screen-crop");
}

function priceTag(v) {
  const n = parseFloat(v);
  return Number.isFinite(n) ? `$${n.toFixed(2)}` : "—";
}

function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function renderFenceCards(rows) {
  const wrap = $("#fence-cards");
  wrap.innerHTML = "";
  for (const row of rows) {
    // one disc is sold as new OR used — the row only ever carries that
    // condition's numbers (collapsed server-side)
    const label = (row.Condition || (row.Stock === "new" ? "New" : "Used")).toUpperCase();
    const found = priceTag(row["Lowest Price (AUD)"]);
    const mine = priceTag(row["Your Price (AUD)"]);
    const title = row.Title || row["Image Set Name"] || row.Barcode || "UNKNOWN DISC";
    const card = document.createElement("div");
    card.className = "score-card";
    card.innerHTML =
      `<div class="score-title">${esc(title)}</div>` +
      `<div class="score-prices">` +
        `<button class="fix-btn" data-fix="${esc(row["Image Set Name"] || "")}" title="Wrong disc? Fix the title / price">✎ FIX</button>` +
        `<div class="score-small"><span class="label">LOWEST ${label}</span>${found}</div>` +
      `</div>` +
      `<div class="score-move">` +
        `<span class="move-tag">YOUR MOVE · 10% UNDER</span>` +
        `<div class="move-prices">` +
          `<div class="move-big"><span class="label">LIST ${label} AT</span>${mine}</div>` +
        `</div>` +
      `</div>`;
    wrap.appendChild(card);
  }
  if (!rows.length) { wrap.innerHTML = `<div class="hint">The fence had nothing to say.</div>`; return; }
  // overall total = sum of the 10%-under prices only (what you'd list the
  // whole pile at), not the lowest-comp prices
  const total = rows.reduce((sum, row) => {
    const v = parseFloat(row["Your Price (AUD)"]);
    return sum + (Number.isFinite(v) ? v : 0);
  }, 0);
  const totalRow = document.createElement("div");
  totalRow.className = "score-total";
  totalRow.innerHTML =
    `<span class="score-total-label">TOTAL TAKE · YOUR PRICES</span>` +
    `<span class="score-total-value">$${total.toFixed(2)}</span>`;
  wrap.appendChild(totalRow);
}

function renderEbayTable(rows) {
  const table = $("#ebay-table");
  if (!rows.length) { table.innerHTML = "<tr><td>No eBay matches found.</td></tr>"; return; }
  const cols = Object.keys(rows[0]).filter((c) => !c.startsWith("_") && c !== "Stock");
  table.innerHTML =
    `<thead><tr>${cols.map((c) => `<th>${esc(c)}</th>`).join("")}</tr></thead>` +
    `<tbody>${rows.map((r) =>
      `<tr>${cols.map((c) => `<td>${esc(r[c] ?? "")}</td>`).join("")}</tr>`).join("")}</tbody>`;
}

function renderEbayPanel(e) {
  // paths live in the txt/CSV files themselves — the user doesn't want them
  // on screen, so the panel only appears when there's a warning to show
  $("#listing-info").hidden = !e.warning;
  $("#listing-info").innerHTML = e.warning
    ? `<div class="miss">${esc(e.warning)} — photos are named by barcode instead.</div>` : "";
  state.ebayRows = e.rows || [];
  renderFenceCards(state.ebayRows);
  renderEbayTable(state.ebayRows);
}

/* inline "FIX" editor on a fence card — correct a wrong eBay match's title
   (and list price); the server re-slugs, renames the photos, and rewrites the
   CSV + listing text to match. */
function openFenceEdit(card, row) {
  card.classList.add("editing");
  card.innerHTML =
    `<div class="fix-form">` +
      `<label class="fix-field">TITLE<input class="fix-title" type="text"></label>` +
      `<label class="fix-field">LIST PRICE (AUD)<input class="fix-price" type="text" placeholder="e.g. 12.99"></label>` +
      `<div class="fix-actions">` +
        `<button class="fix-save">SAVE</button>` +
        `<button class="fix-cancel">CANCEL</button>` +
      `</div>` +
    `</div>`;
  card.querySelector(".fix-title").value = row.Title || row["Image Set Name"] || "";
  const p = row["Your Price (AUD)"];
  card.querySelector(".fix-price").value = (p === "" || p == null) ? "" : p;
  card.querySelector(".fix-title").focus();
}

async function saveFenceEdit(card, slug) {
  const title = card.querySelector(".fix-title").value.trim();
  const price = card.querySelector(".fix-price").value.trim();
  const saveBtn = card.querySelector(".fix-save");
  saveBtn.disabled = true;
  saveBtn.textContent = "SAVING…";
  try {
    const r = await api(`/batches/${state.batchName}/title/edit`, {
      method: "POST",
      body: JSON.stringify({ image_set_name: slug, title, price }),
    });
    if (r && r.error) {
      alert(r.error);
      saveBtn.disabled = false; saveBtn.textContent = "SAVE";
      return;
    }
    state.ebayRows = r.rows || state.ebayRows;
    if (r.quads) { state.quads = r.quads; renderCropGrid(); } // filenames changed
    if (r.warning) alert(r.warning);
    renderFenceCards(state.ebayRows);
    renderEbayTable(state.ebayRows);
  } catch (err) {
    alert(`Save failed: ${err.message}`);
    saveBtn.disabled = false; saveBtn.textContent = "SAVE";
  }
}

$("#fence-cards").addEventListener("click", (ev) => {
  const fixBtn = ev.target.closest(".fix-btn");
  if (fixBtn) {
    const card = fixBtn.closest(".score-card");
    const slug = fixBtn.dataset.fix;
    const row = (state.ebayRows || []).find((r) => r["Image Set Name"] === slug);
    if (row) { card.dataset.slug = slug; openFenceEdit(card, row); }
    return;
  }
  if (ev.target.closest(".fix-cancel")) {
    renderFenceCards(state.ebayRows || []);
    return;
  }
  const saveBtn = ev.target.closest(".fix-save");
  if (saveBtn) saveFenceEdit(saveBtn.closest(".score-card"), saveBtn.closest(".score-card").dataset.slug);
});

/* ---------- 5. the cropper (view + adjust only; NEVER renames) ---------- */

function croppedUrl(filename) {
  const jpg = filename.replace(/\.[^.]+$/, ".jpg");
  return `/api/files/${state.batchName}/5_cropped/${jpg}?t=${Date.now()}`;
}

function renderCropGrid() {
  const grid = $("#crop-grid");
  grid.innerHTML = "";
  for (const filename of Object.keys(state.quads)) {
    const fig = document.createElement("figure");
    const img = document.createElement("img");
    img.src = croppedUrl(filename);
    img.title = "Click to adjust corners";
    img.addEventListener("click", () => openEditor(filename));
    const cap = document.createElement("figcaption");
    cap.textContent = filename;
    fig.append(img, cap);
    grid.appendChild(fig);
  }
}

/* corner-drag editor */
const editor = {
  canvas: $("#crop-editor"),
  ctx: $("#crop-editor").getContext("2d"),
  img: null, filename: null, quad: null,   // quad in NATURAL image coords
  scale: 1, ox: 0, oy: 0, dragIndex: -1,
};

function openEditor(filename) {
  $("#editor-title").textContent = `ADJUSTING: ${filename}`;
  editor.filename = filename;
  editor.quad = (state.quads[filename] || []).map((pt) => [...pt]);
  const img = new Image();
  img.onload = () => {
    editor.img = img;
    const { canvas } = editor;
    editor.scale = Math.min(canvas.width / img.naturalWidth, canvas.height / img.naturalHeight);
    editor.ox = (canvas.width - img.naturalWidth * editor.scale) / 2;
    editor.oy = (canvas.height - img.naturalHeight * editor.scale) / 2;
    if (!editor.quad.length) {
      editor.quad = [[0, 0], [img.naturalWidth, 0],
                     [img.naturalWidth, img.naturalHeight], [0, img.naturalHeight]];
    }
    $("#editor-wrap").hidden = false;
    drawEditor();
    $("#editor-wrap").scrollIntoView({ behavior: "smooth" });
  };
  img.src = `/api/files/${state.batchName}/4_renamed/${filename}`;
}

const toCanvas = ([x, y]) => [x * editor.scale + editor.ox, y * editor.scale + editor.oy];
const toNatural = (cx, cy) => [(cx - editor.ox) / editor.scale, (cy - editor.oy) / editor.scale];

function drawEditor() {
  const { ctx, canvas, img } = editor;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(img, editor.ox, editor.oy,
    img.naturalWidth * editor.scale, img.naturalHeight * editor.scale);
  ctx.strokeStyle = "#9ec724";
  ctx.lineWidth = 3;
  ctx.beginPath();
  editor.quad.forEach((pt, i) => {
    const [px, py] = toCanvas(pt);
    i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
  });
  ctx.closePath();
  ctx.stroke();
  for (const pt of editor.quad) {
    const [px, py] = toCanvas(pt);
    ctx.fillStyle = "#ff2ec4";
    ctx.beginPath();
    ctx.arc(px, py, 8, 0, Math.PI * 2);
    ctx.fill();
  }
}

function canvasPos(e) {
  const r = editor.canvas.getBoundingClientRect();
  return [
    (e.clientX - r.left) * (editor.canvas.width / r.width),
    (e.clientY - r.top) * (editor.canvas.height / r.height),
  ];
}

editor.canvas.addEventListener("mousedown", (e) => {
  const [cx, cy] = canvasPos(e);
  editor.dragIndex = editor.quad.findIndex((pt) => {
    const [px, py] = toCanvas(pt);
    return Math.hypot(px - cx, py - cy) < 18;
  });
});
editor.canvas.addEventListener("mousemove", (e) => {
  if (editor.dragIndex < 0) return;
  const [cx, cy] = canvasPos(e);
  editor.quad[editor.dragIndex] = toNatural(cx, cy);
  drawEditor();
});
window.addEventListener("mouseup", () => { editor.dragIndex = -1; });

$("#btn-crop-cancel").addEventListener("click", () => { $("#editor-wrap").hidden = true; });

$("#btn-rechop").addEventListener("click", async () => {
  if (!confirm("Re-chop ALL photos? Any corners you fixed by hand get re-detected from scratch.")) {
    return;
  }
  const btn = $("#btn-rechop");
  btn.disabled = true;
  btn.textContent = "CHOPPING…";
  try {
    const cr = await api(`/batches/${state.batchName}/crop/run`, { method: "POST" });
    if (cr && cr.error) { alert(cr.error); return; }
    state.quads = cr.quads || state.quads;
    renderCropGrid();
  } catch (err) {
    alert(`Re-chop failed: ${err.message}`);
  } finally {
    btn.disabled = false;
    btn.textContent = "RE-CHOP EVERYTHING";
  }
});

$("#btn-crop-save").addEventListener("click", async () => {
  try {
    const r = await api(`/batches/${state.batchName}/crop/manual`, {
      method: "POST",
      body: JSON.stringify({ filename: editor.filename, quad: editor.quad }),
    });
    if (r && r.error) { alert(r.error); return; } // e.g. box dragged too small
    state.quads[editor.filename] = editor.quad.map((pt) => [...pt]);
    $("#editor-wrap").hidden = true;
    renderCropGrid(); // cache-busted thumbs pick up the recrop
  } catch (err) {
    alert(`Save failed: ${err.message}`);
  }
});

$("#btn-new-job").addEventListener("click", () => location.reload());

/* ---------- "ACCEPT HUSTLE?" — final cleanup, then close the app ---------- */

$("#btn-accept-hustle").addEventListener("click", () => {
  $("#accept-modal").hidden = false;
});

$("#btn-accept-no").addEventListener("click", () => {
  $("#accept-modal").hidden = true;
});

$("#btn-accept-yes").addEventListener("click", async () => {
  const yesBtn = $("#btn-accept-yes");
  const noBtn = $("#btn-accept-no");
  yesBtn.disabled = true; noBtn.disabled = true;
  yesBtn.textContent = "CLEANING UP…";
  try {
    const r = await api(`/batches/${state.batchName}/accept`, { method: "POST" });
    if (r && r.error) {
      alert(r.error);
      yesBtn.disabled = false; noBtn.disabled = false; yesBtn.textContent = "YES";
      return;
    }
    AUDIO.stab("cash-register"); // hustle accepted — money in the register
    // Tell the desktop wrapper (if any) to close its window — it polls this
    // flag from its own background thread, so a plain browser tab (no
    // wrapper watching) just leaves it set (cleared on next wrapper launch).
    // wait out the ~1s cha-ching before the desktop window drops
    setTimeout(() => api(`/quit`, { method: "POST" }).catch(() => {}), 1400);
    // In the desktop app the window closes itself in <1s; in a browser tab
    // it can't, so ALWAYS offer a way out instead of a dead-end modal.
    $("#accept-modal .modal-box").innerHTML =
      `<h2 class="mission passed">CLEANED UP</h2>` +
      `<p class="hint">Only the cropped photos and eBay files remain.</p>` +
      `<button id="btn-after-accept" class="big-btn">START ANOTHER JOB</button>`;
    $("#btn-after-accept").addEventListener("click", () => location.reload());
  } catch (err) {
    alert(`Cleanup failed: ${err.message}`);
    yesBtn.disabled = false; noBtn.disabled = false; yesBtn.textContent = "YES";
  }
});

setRail("upload");
