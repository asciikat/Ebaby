/* EBABY front-end: one START button drives the whole chain
   (upload -> rename -> colour -> barcode -> eBay), and only when all of
   that is sorted does the cropper open. The cropper NEVER renames files. */

const state = {
  batchName: null,
  pending: { used: [], new: [] },   // File objects picked before START
  quads: {},                        // filename -> quad (natural coords)
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
  ["upload", "LOAD UP"],
  ["color", "CLEAN"],
  ["barcode", "INTERROGATE"],
  ["ebay", "FENCE"],
  ["crop", "CHOP"],
  ["done", "PAID"],
];

function setRail(stage) {
  const rail = $("#stage-rail");
  rail.innerHTML = "";
  const idx = RAIL.findIndex(([k]) => k === stage);
  RAIL.forEach(([k, label], i) => {
    const chip = document.createElement("span");
    chip.textContent = label;
    if (i < idx) chip.className = "done";
    if (i === idx) chip.className = "active";
    rail.appendChild(chip);
  });
}

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

function refreshStartButton() {
  const total = state.pending.used.length + state.pending.new.length;
  $("#btn-start").disabled = total === 0;
}

function addFiles(zone, fileList) {
  for (const f of fileList) state.pending[zone].push(f);
  $(`#count-${zone}`).textContent = `${state.pending[zone].length} photos`;
  refreshStartButton();
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
  show("screen-progress"); // the error must land on a VISIBLE screen
  logStep("BUSTED").fail(err.message);
  const retry = document.createElement("button");
  retry.className = "big-btn";
  retry.style.marginTop = "1rem";
  retry.textContent = "BACK TO THE GARAGE";
  retry.addEventListener("click", () => location.reload());
  $("#screen-progress").appendChild(retry);
}

function checked(r, what) {
  if (r && r.error) throw new Error(`${what}: ${r.error}`);
  return r;
}

$("#btn-start").addEventListener("click", () => {
  const btn = $("#btn-start");
  if (btn.disabled) return;
  btn.disabled = true; // double-click = two batches
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
  show("screen-crop");
}

function priceTag(v) {
  const n = parseFloat(v);
  return Number.isFinite(n) ? `$${n.toFixed(2)}` : "—";
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
    const card = document.createElement("div");
    card.className = "score-card";
    card.innerHTML =
      `<div class="score-title">${row.Title || row["Image Set Name"] || row.Barcode || "UNKNOWN DISC"}</div>` +
      `<div class="score-prices">` +
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

function renderEbayPanel(e) {
  const warn = e.warning
    ? `<div class="miss">${e.warning} — photos are named by barcode instead.</div>` : "";
  $("#listing-info").innerHTML = warn +
    `Listing text file: <code>${e.listing_txt || "?"}</code><br>` +
    `Batch folder: <code>${e.folder || "?"}</code>`;
  renderFenceCards(e.rows || []);
  const rows = e.rows || [];
  const table = $("#ebay-table");
  if (!rows.length) { table.innerHTML = "<tr><td>No eBay matches found.</td></tr>"; return; }
  const cols = Object.keys(rows[0]);
  table.innerHTML =
    `<thead><tr>${cols.map((c) => `<th>${c}</th>`).join("")}</tr></thead>` +
    `<tbody>${rows.map((r) =>
      `<tr>${cols.map((c) => `<td>${r[c] ?? ""}</td>`).join("")}</tr>`).join("")}</tbody>`;
}

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

/* ---------- 6. resume: a page refresh must never orphan a job ---------- */

function ebayPanelFromState(s) {
  return { rows: s.ebay_rows || [], warning: s.ebay_warning,
           listing_txt: s.listing_txt, folder: s.folder };
}

async function resumeBatch(name, s) {
  state.batchName = name;
  if (s.stage === "color") { await runFromColor(); return; }
  if (s.stage === "barcode") { await runFromBarcode(); return; }
  if (s.stage === "ebay") {
    const barcodes = s.barcodes || {};
    const missed = Object.entries(barcodes).filter(([, v]) => !v);
    if (missed.length) {
      renderBarcodeTable(barcodes, s.barcode_crops || {});
      show("screen-barcode");
      setRail("barcode");
      return;
    }
    show("screen-progress");
    await finishJob();
    return;
  }
  // crop / done: rebuild the results + chop shop from persisted state
  renderEbayPanel(ebayPanelFromState(s));
  if (s.quads && Object.keys(s.quads).length) {
    state.quads = s.quads;
  } else {
    const cr = await api(`/batches/${name}/crop/run`, { method: "POST" });
    state.quads = (cr && cr.quads) || {};
  }
  renderCropGrid();
  setRail("done");
  show("screen-crop");
}

async function offerResume() {
  try {
    const names = await api("/batches");
    if (!Array.isArray(names) || !names.length) return;
    const latest = names[names.length - 1]; // run-<timestamp> sorts by time
    const s = await api(`/batches/${latest}/state`);
    if (!s || s.error || s.stage === "upload") return;
    $("#resume-text").textContent =
      `Unfinished business: ${latest} (stage: ${s.stage}). `;
    $("#resume-banner").hidden = false;
    $("#btn-resume").addEventListener("click", () => {
      $("#resume-banner").hidden = true;
      resumeBatch(latest, s).catch(busted);
    }, { once: true });
  } catch { /* no server-side history — fresh start */ }
}

setRail("upload");
offerResume();
