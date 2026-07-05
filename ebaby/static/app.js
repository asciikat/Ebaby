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

function show(id) {
  for (const el of document.querySelectorAll("section[id^='screen-']")) {
    el.hidden = el.id !== id;
  }
}

function logStep(text) {
  const li = document.createElement("li");
  li.textContent = text;
  li.className = "doing";
  $("#mission-log").appendChild(li);
  return {
    done(extra) { li.className = "done"; if (extra) li.textContent = `${text} — ${extra}`; },
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

$("#btn-start").addEventListener("click", () => {
  runJob().catch((err) => {
    logStep("BUSTED").fail(err.message);
  });
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

  // colour correction
  step = logStep("Cleaning the goods so they look legit (the slow burn)");
  const c = await api(`/batches/${state.batchName}/color/run`, { method: "POST" });
  step.done(`${c.colored} processed`);

  // barcode scan
  step = logStep("Interrogating the barcodes");
  const b = await api(`/batches/${state.batchName}/barcode/run`, { method: "POST" });
  const results = b.results || {};
  const missed = Object.entries(results).filter(([, v]) => !v);
  step.done(`${Object.keys(results).length - missed.length} hit, ${missed.length} missed`);

  if (missed.length) {
    renderBarcodeTable(results);
    show("screen-barcode");
    return; // continues from the CARRY ON button
  }
  await finishJob();
}

/* ---------- 3. barcode fixes (only when needed) ---------- */

function renderBarcodeTable(results) {
  const tbody = $("#barcode-table tbody");
  tbody.innerHTML = "";
  for (const [key, digits] of Object.entries(results)) {
    const tr = document.createElement("tr");
    const status = digits
      ? `<span class="got">${digits}</span>`
      : `<span class="miss">MISSED</span>`;
    const fix = digits ? "" : `<input type="text" inputmode="numeric" data-fix-key="${key}" placeholder="type the digits" />`;
    tr.innerHTML = `<td>${key.toUpperCase()}</td><td>${status}</td><td>${fix}</td>`;
    tbody.appendChild(tr);
  }
}

$("#btn-barcode-continue").addEventListener("click", async () => {
  for (const input of document.querySelectorAll("[data-fix-key]")) {
    const digits = input.value.trim();
    if (digits) {
      await api(`/batches/${state.batchName}/barcode/manual`, {
        method: "POST",
        body: JSON.stringify({ key: input.dataset.fixKey, digits }),
      });
    }
  }
  show("screen-progress");
  await finishJob().catch((err) => logStep("BUSTED").fail(err.message));
});

/* ---------- 4. eBay + title rename, then open the cropper ---------- */

async function finishJob() {
  let step = logStep("Fencing the goods on eBay + issuing final identities");
  const e = await api(`/batches/${state.batchName}/ebay/run`, { method: "POST" });
  if (e && e.error) throw new Error(e.error);
  if (e.warning) step.fail(e.warning); else step.done();

  step = logStep("Running everything through the chop shop");
  const cr = await api(`/batches/${state.batchName}/crop/run`, { method: "POST" });
  state.quads = cr.quads || {};
  step.done(`${Object.keys(state.quads).length} photos`);

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
    const isNew = row.Stock === "new";
    const bigLabel = isNew ? "NEW" : "USED";
    const bigPrice = priceTag(isNew ? row["Lowest Price New (AUD)"]
                                    : row["Lowest Price Used (AUD)"]);
    const smallLabel = isNew ? "USED" : "NEW";
    const smallPrice = priceTag(isNew ? row["Lowest Price Used (AUD)"]
                                      : row["Lowest Price New (AUD)"]);
    const card = document.createElement("div");
    card.className = "score-card";
    card.innerHTML =
      `<div class="score-title">${row.Title || row["Image Set Name"] || row.Barcode || "UNKNOWN DISC"}</div>` +
      `<div class="score-prices">` +
        `<div class="score-big"><span class="label">${bigLabel}</span>${bigPrice}</div>` +
        `<div class="score-small"><span class="label">${smallLabel}</span>${smallPrice}</div>` +
      `</div>`;
    wrap.appendChild(card);
  }
  if (!rows.length) wrap.innerHTML = `<div class="hint">The fence had nothing to say.</div>`;
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
  const btn = $("#btn-rechop");
  btn.disabled = true;
  btn.textContent = "CHOPPING…";
  const cr = await api(`/batches/${state.batchName}/crop/run`, { method: "POST" });
  state.quads = cr.quads || state.quads;
  renderCropGrid();
  btn.disabled = false;
  btn.textContent = "RE-CHOP EVERYTHING";
});

$("#btn-crop-save").addEventListener("click", async () => {
  await api(`/batches/${state.batchName}/crop/manual`, {
    method: "POST",
    body: JSON.stringify({ filename: editor.filename, quad: editor.quad }),
  });
  state.quads[editor.filename] = editor.quad.map((pt) => [...pt]);
  $("#editor-wrap").hidden = true;
  renderCropGrid(); // cache-busted thumbs pick up the recrop
});
