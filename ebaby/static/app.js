const state = { batchName: null, zoneCounts: { used: 0, new: 0 } };

async function api(path, opts = {}) {
  const res = await fetch(`/api${path}`, {
    headers: opts.body instanceof FormData ? {} : { "Content-Type": "application/json" },
    ...opts,
  });
  return res.json();
}

function showStage(id) {
  for (const el of document.querySelectorAll("section[id^='stage-']")) {
    el.hidden = el.id !== id;
  }
}

async function ensureBatch() {
  if (state.batchName) return state.batchName;
  const name = `run-${new Date().toISOString().replace(/[:.]/g, "-")}`;
  await api("/batches", { method: "POST", body: JSON.stringify({ name }) });
  state.batchName = name;
  return name;
}

async function uploadFiles(zone, files) {
  const name = await ensureBatch();
  for (const file of files) {
    const form = new FormData();
    form.append("file", file);
    await api(`/batches/${name}/upload/${zone}`, { method: "POST", body: form });
    state.zoneCounts[zone] += 1;
  }
  document.querySelector(`[data-zone-count="${zone}"]`).textContent =
    `${state.zoneCounts[zone]} files`;
}

for (const zone of ["used", "new"]) {
  document.querySelector(`[data-zone-input="${zone}"]`).addEventListener("change", (e) => {
    uploadFiles(zone, e.target.files);
  });
}

document.getElementById("btn-rename-plan").addEventListener("click", async () => {
  const name = await ensureBatch();
  const output = document.getElementById("rename-plan-output");
  output.textContent = "";
  for (const zone of ["used", "new"]) {
    const result = await api(`/batches/${name}/rename/plan`, {
      method: "POST", body: JSON.stringify({ zone }),
    });
    if (result.error) {
      output.textContent += `${zone}: ${result.error}\n`;
    } else {
      output.textContent += `${zone}:\n` +
        result.plan.map((p) => `  ${p.old_name} -> ${p.new_name}`).join("\n") + "\n";
    }
  }
  document.getElementById("btn-rename-apply").disabled = false;
});

document.getElementById("btn-rename-apply").addEventListener("click", async () => {
  const name = await ensureBatch();
  for (const zone of ["used", "new"]) {
    await api(`/batches/${name}/rename/apply`, {
      method: "POST", body: JSON.stringify({ zone }),
    });
  }
  await runColorAndBarcode(name);
});

async function runColorAndBarcode(name) {
  await api(`/batches/${name}/color/run`, { method: "POST" });
  const barcodeResult = await api(`/batches/${name}/barcode/run`, { method: "POST" });
  renderBarcodeTable(barcodeResult.results);
  showStage("stage-barcode");
}

function renderBarcodeTable(results) {
  const tbody = document.querySelector("#barcode-table tbody");
  tbody.innerHTML = "";
  for (const [key, digits] of Object.entries(results)) {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${key}</td><td>${digits ?? "MISS"}</td>` +
      `<td><input type="text" data-fix-key="${key}" placeholder="type digits" /></td>`;
    tbody.appendChild(tr);
  }
}

document.getElementById("btn-barcode-continue").addEventListener("click", async () => {
  const name = state.batchName;
  for (const input of document.querySelectorAll("[data-fix-key]")) {
    if (input.value.trim()) {
      await api(`/batches/${name}/barcode/manual`, {
        method: "POST",
        body: JSON.stringify({ key: input.dataset.fixKey, digits: input.value.trim() }),
      });
    }
  }
  const ebayResult = await api(`/batches/${name}/ebay/run`, { method: "POST" });
  renderEbayTable(ebayResult.rows);
  showStage("stage-ebay");
});

function renderEbayTable(rows) {
  const table = document.getElementById("ebay-table");
  if (!rows.length) { table.innerHTML = "<p>No eBay matches found.</p>"; return; }
  const cols = Object.keys(rows[0]);
  table.innerHTML = `<thead><tr>${cols.map((c) => `<th>${c}</th>`).join("")}</tr></thead>` +
    `<tbody>${rows.map((r) => `<tr>${cols.map((c) => `<td>${r[c] ?? ""}</td>`).join("")}</tr>`).join("")}</tbody>`;
}

document.getElementById("btn-ebay-continue").addEventListener("click", async () => {
  const name = state.batchName;
  const cropResult = await api(`/batches/${name}/crop/run`, { method: "POST" });
  renderCropGrid(cropResult.quads);
  showStage("stage-crop");
});

let currentCropFile = null;
let currentQuad = null;

function renderCropGrid(quads) {
  const grid = document.getElementById("crop-grid");
  grid.innerHTML = "";
  for (const filename of Object.keys(quads)) {
    const img = document.createElement("img");
    img.src = `/files/${state.batchName}/5_cropped/${filename.replace(/\.png$/, ".jpg")}`;
    img.dataset.filename = filename;
    img.addEventListener("click", () => openCropEditor(filename, quads[filename]));
    grid.appendChild(img);
  }
}

function openCropEditor(filename, quad) {
  currentCropFile = filename;
  currentQuad = quad.map((pt) => [...pt]);
  const canvas = document.getElementById("crop-editor");
  canvas.hidden = false;
  document.getElementById("btn-crop-save").hidden = false;
  const ctx = canvas.getContext("2d");
  const img = new Image();
  img.onload = () => {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    drawQuad(ctx, canvas);
  };
  img.src = `/files/${state.batchName}/4_renamed/${filename}`;
}

function drawQuad(ctx, canvas) {
  ctx.strokeStyle = "lime";
  ctx.lineWidth = 3;
  ctx.beginPath();
  currentQuad.forEach(([x, y], i) => {
    const px = (x / 3000) * canvas.width;   // scale factor placeholder — real
    const py = (y / 4000) * canvas.height;  // image dimensions come from state
    i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
  });
  ctx.closePath();
  ctx.stroke();
}

document.getElementById("btn-crop-save").addEventListener("click", async () => {
  await api(`/batches/${state.batchName}/crop/manual`, {
    method: "POST",
    body: JSON.stringify({ filename: currentCropFile, quad: currentQuad }),
  });
});
