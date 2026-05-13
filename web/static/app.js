const $ = (sel) => document.querySelector(sel);

let debounceTimer = null;
let plotRenderToken = 0;
let plotHasRendered = false;
let lastMeta = null;

const PRESETS = {
  iris: [
    { id: "a", label: "Iris setosa", values: [5.1, 3.5, 1.4, 0.2] },
    { id: "b", label: "Iris versicolor", values: [5.9, 2.8, 4.5, 1.3] },
    { id: "c", label: "Iris virginica", values: [6.3, 3.3, 6.0, 2.5] },
  ],
  weather: [
    { id: "a", label: "Weather: likely Skip", values: [32, 90, 20, 1004] },
    { id: "b", label: "Weather: likely Play", values: [22, 53, 7, 1018] },
    { id: "c", label: "Weather: mixed", values: [26, 70, 10, 1011] },
  ],
  loan: [
    { id: "a", label: "Loan: likely Reject", values: [24, 0.56, 520, 1] },
    { id: "b", label: "Loan: likely Approve", values: [82, 0.19, 742, 9] },
    { id: "c", label: "Loan: borderline", values: [45, 0.34, 650, 4] },
  ],
};

const DATASET_FIELDS = {
  iris: [
    { name: "Sepal length", unit: "cm" },
    { name: "Sepal width", unit: "cm" },
    { name: "Petal length", unit: "cm" },
    { name: "Petal width", unit: "cm" },
  ],
  weather: [
    { name: "Temperature", unit: "°C" },
    { name: "Humidity", unit: "%" },
    { name: "Wind speed", unit: "km/h" },
    { name: "Pressure", unit: "hPa" },
  ],
  loan: [
    { name: "Income", unit: "k$" },
    { name: "Debt ratio", unit: "ratio" },
    { name: "Credit score", unit: "score" },
    { name: "Employment years", unit: "years" },
  ],
};

function getActiveDataset() {
  const ds = ($("#dataset")?.value || lastMeta?.dataset || "iris").toLowerCase();
  return PRESETS[ds] ? ds : "iris";
}

function updatePresetButtons(dataset) {
  const ds = PRESETS[dataset] ? dataset : "iris";
  const rows = PRESETS[ds];
  const buttons = [
    $("#preset-setosa"),
    $("#preset-versicolor"),
    $("#preset-virginica"),
  ];
  for (let i = 0; i < buttons.length; i++) {
    const btn = buttons[i];
    const row = rows[i];
    if (!btn || !row) continue;
    btn.textContent = row.label;
    btn.dataset.presetValues = JSON.stringify(row.values);
  }
}

function applyPresetFromButton(btn) {
  if (!btn) return;
  let vals = null;
  try {
    vals = JSON.parse(btn.dataset.presetValues || "null");
  } catch (_e) {}
  if (!Array.isArray(vals) || vals.length !== 4) return;
  setPresets(vals[0], vals[1], vals[2], vals[3]);
}

function updateFeatureLabels(dataset) {
  const ds = DATASET_FIELDS[dataset] ? dataset : "iris";
  const fields = DATASET_FIELDS[ds];
  for (let i = 0; i < 4; i++) {
    const row = fields[i];
    const nameEl = $(`#f${i}-name`);
    const unitEl = $(`#f${i}-unit`);
    if (nameEl && row) nameEl.textContent = row.name;
    if (unitEl && row) unitEl.textContent = row.unit;
  }
}

function getModelMode() {
  const el = $("#model-mode");
  const v = el?.value || "cn2";
  return v === "tree" ? "tree" : "cn2";
}

function formatImportances(fi) {
  if (!fi || typeof fi !== "object") return "";
  const parts = Object.entries(fi)
    .map(([k, v]) => [k, Number(v)])
    .filter(([, v]) => Number.isFinite(v))
    .sort((a, b) => b[1] - a[1])
    .map(([k, v]) => `${k}: ${v.toFixed(3)}`);
  return parts.length ? `Feature importance: ${parts.join(" · ")}` : "";
}

function applyModelChrome() {
  const mode = getModelMode();
  const intro = $("#intro-copy");
  const explain = $("#explain-title");
  const blurb = $("#plot-blurb");
 if (intro) {
  intro.textContent =
    mode === "tree"
      ? "Input features for the selected dataset. The Decision Tree uses continuous feature values without discretization."
      : "Input features for the selected dataset. CN2 converts continuous values into symbolic bins such as low, middle and high.";
}
  if (explain) {
    explain.textContent =
      mode === "tree"
        ? "Decision path (root → leaf)"
        : "First matching rule (ordered CN2 list)";
  }
  if (blurb) {
    blurb.textContent =
      "Your instance vs training mean of the predicted class (same chart for CN2 and tree).";
  }
}

function readTrainConfig() {
  return {
    dataset: $("#dataset")?.value || "iris",
    cn2_beam_width: Number($("#cn2-beam")?.value || 6),
    cn2_min_rule_coverage: Number($("#cn2-mincov")?.value || 4),
    tree_max_depth: Number($("#tree-depth")?.value || 4),
    tree_min_samples_leaf: Number($("#tree-leaf")?.value || 2),
  };
}

function setTheme(theme) {
  const t = ["neon", "minimal", "contrast"].includes(theme) ? theme : "neon";
  document.body.dataset.theme = t;
  const sel = $("#theme");
  if (sel) sel.value = t;
  try {
    localStorage.setItem("ml_theme", t);
  } catch (_e) {}
}

function loadTheme() {
  let t = "neon";
  try {
    t = localStorage.getItem("ml_theme") || "neon";
  } catch (_e) {}
  setTheme(t);
}

function setTab(tabId) {
  const tabs = Array.from(document.querySelectorAll('.tab[role="tab"]'));
  const panels = Array.from(document.querySelectorAll("[data-tab-panel]"));
  const valid = new Set(panels.map((p) => p.getAttribute("data-tab-panel")));
  const t = valid.has(tabId) ? tabId : "inputs";

  for (const btn of tabs) {
    const isActive = btn.getAttribute("data-tab") === t;
    btn.setAttribute("aria-selected", String(isActive));
  }
  for (const p of panels) {
    p.hidden = p.getAttribute("data-tab-panel") !== t;
  }
  try {
    localStorage.setItem("ml_cn2_tab", t);
  } catch (_e) {}
}

function loadTab() {
  let t = "inputs";
  try {
    t = localStorage.getItem("ml_cn2_tab") || "inputs";
  } catch (_e) {}
  setTab(t);
}

function readFeaturesFromUI() {
  const out = [];
  for (let i = 0; i < 4; i++) {
    const el = document.getElementById(`f${i}`);
    if (!el) return { error: "Input field not found." };
    const n = Number(el.value);
    if (!Number.isFinite(n)) return { error: `Feature ${i + 1}: enter a valid number.` };
    out.push(n);
  }
  return { features: out };
}

function setLoading(isLoading) {
  const btn = $("#run");
  btn.disabled = isLoading;
  $("#btn-text").textContent = isLoading ? "Working…" : "Classify";
  $("#btn-spin").hidden = !isLoading;
}

function setError(msg) {
  const el = $("#error");
  if (!msg) {
    el.hidden = true;
    el.textContent = "";
    return;
  }
  el.hidden = false;
  el.textContent = msg;
}

function setPlot(figureJson) {
  renderPlot("#plot", figureJson);
}

function renderPlot(selector, figureJson) {
  const plotEl = $(selector);
  if (!plotEl) return;

  if (!figureJson) {
    plotEl.textContent = "No chart data.";
    return;
  }

  if (!window.Plotly) {
    plotEl.textContent = "Plotly failed to load. Check your connection or the CDN.";
    return;
  }

  if (!plotHasRendered) {
    plotEl.innerHTML = "";
  }

  let fig;
  try {
    fig = JSON.parse(figureJson);
  } catch (_e) {
    plotEl.textContent = "Could not parse chart data.";
    return;
  }

  const layout = {
    ...fig.layout,
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: { color: "rgba(255,255,255,.86)" },
    autosize: true,
  };
  delete layout.height;
  delete layout.width;

  const myToken = ++plotRenderToken;
  const doRender = () => {
    if (myToken !== plotRenderToken) return;
    if (plotEl.offsetWidth === 0) {
      requestAnimationFrame(doRender);
      return;
    }
    try {
      window.Plotly.react(plotEl, fig.data, layout, {
        displayModeBar: false,
        responsive: true,
      });
      plotHasRendered = true;
      window.requestAnimationFrame(() => {
        try {
          window.Plotly.Plots.resize(plotEl);
        } catch (_e) {}
      });
    } catch (_e) {
      plotEl.textContent = "Chart render error.";
    }
  };

  requestAnimationFrame(doRender);
}

function setPresets(f0, f1, f2, f3) {
  $("#f0").value = String(f0);
  $("#f1").value = String(f1);
  $("#f2").value = String(f2);
  $("#f3").value = String(f3);
}

async function loadMetaPanels() {
  const listEl = $("#rules-list");
  const treePre = $("#tree-export");
  if (!listEl) return;
  try {
    const r = await fetch("/api/meta");
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const data = await r.json();
    lastMeta = data;
    if ($("#dataset") && data.dataset) {
      $("#dataset").value = data.dataset;
    }
    const active = (data.dataset || "iris").toLowerCase();
    updatePresetButtons(active);
    updateFeatureLabels(active);
    listEl.innerHTML = "";
    for (const row of data.rules || []) {
      const li = document.createElement("li");
      li.textContent = row.text;
      listEl.appendChild(li);
    }
    if (treePre) {
      treePre.textContent = data.tree_export_text || "(no tree export)";
    }
    const img = $("#tree-img");
    if (img) {
      const url = data.tree_plot_url || "/api/tree-plot.png";
      img.src = `${url}?t=${Date.now()}`;
    }
  } catch (e) {
    lastMeta = null;
    listEl.innerHTML = "";
    const li = document.createElement("li");
    li.textContent = e?.message ? `Could not load rules: ${e.message}` : "Could not load rules.";
    listEl.appendChild(li);
    if (treePre) treePre.textContent = e?.message ? `Could not load tree: ${e.message}` : "Could not load tree.";
    const img = $("#tree-img");
    if (img) img.src = `/api/tree-plot.png?t=${Date.now()}`;
  }
}

function setCompare(data) {
  const cn2 = data?.cn2 || {};
  const tree = data?.tree || {};
  $("#cmp-cn2-class").textContent = cn2.class_name ?? "—";
  $("#cmp-cn2-conf").textContent = Number.isFinite(cn2.confidence) ? cn2.confidence.toFixed(3) : "—";
  $("#cmp-tree-class").textContent = tree.class_name ?? "—";
  $("#cmp-tree-conf").textContent = Number.isFinite(tree.confidence) ? tree.confidence.toFixed(3) : "—";
  const st = $("#compare-status");
  if (!st) return;
  st.textContent = data?.note || "—";
  st.className = `compare-status ${data?.disagree ? "bad" : "ok"}`;
  if (getModelMode() === "tree" && tree?.tree_plot_highlight) {
    const img = $("#tree-img");
    if (img) img.src = tree.tree_plot_highlight;
  }
}

async function classify() {
  setError(null);

  const built = readFeaturesFromUI();
  if (built.error) {
    setError(built.error);
    return;
  }

  setLoading(true);
  try {
    const mode = getModelMode();
    const r = await fetch("/api/classify", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ features: built.features, model: mode }),
    });
    const data = await r.json().catch(() => ({}));
    if (!r.ok) {
      throw new Error(data.error || `Server error (${r.status})`);
    }

    $("#result").textContent = data.class_name ?? "—";

    const discEl = $("#disc-line");
    if (data.model === "tree") {
      const fiLine =
        formatImportances(data.tree_feature_importances) ||
        formatImportances(lastMeta?.tree_feature_importances);
      discEl.textContent = fiLine || "Decision tree — continuous features.";
      if (data.tree_plot_highlight) {
        const img = $("#tree-img");
        if (img) img.src = data.tree_plot_highlight;
      }
    } else {
      const discParts = (data.discretized_labels || []).map((d) => `${d.attr}: ${d.label}`);
      discEl.textContent = discParts.length > 0 ? `Discretized: ${discParts.join(" · ")}` : "";
    }

    $("#rule-text").textContent = data.rule_text || "";

    setPlot(data.figure ?? null);

    const cr = await fetch("/api/compare", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ features: built.features }),
    });
    const cmp = await cr.json().catch(() => ({}));
    if (cr.ok) setCompare(cmp);
  } catch (e) {
    setError(e?.message || "Unknown request error.");
  } finally {
    setLoading(false);
  }
}

async function trainModels() {
  setError(null);
  const cfg = readTrainConfig();
  const btn = $("#train-btn");
  if (btn) btn.disabled = true;
  try {
    const r = await fetch("/api/train", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(cfg),
    });
    const data = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(data.error || `Server error (${r.status})`);
    $("#tree-depth-value").textContent = String(cfg.tree_max_depth);
    if ($("#dataset") && data?.meta?.dataset) $("#dataset").value = data.meta.dataset;
    const active = (data?.meta?.dataset || cfg.dataset || "iris").toLowerCase();
    updatePresetButtons(active);
    updateFeatureLabels(active);
    $("#train-summary").textContent =
      `Dataset: ${data.meta.dataset} | CN2 rules: ${data.cn2_rule_count} | Tree depth: ${data.tree_depth} | ` +
      `CN2 acc train/test: ${data.meta.cn2_train_accuracy.toFixed(3)}/${data.meta.cn2_test_accuracy.toFixed(3)} | ` +
      `Tree acc train/test: ${data.meta.tree_train_accuracy.toFixed(3)}/${data.meta.tree_test_accuracy.toFixed(3)}`;
    if ($("#tree-export")) $("#tree-export").textContent = data.tree_export_text || "";
    if ($("#tree-img")) $("#tree-img").src = `/api/tree-plot.png?t=${Date.now()}`;

    const curve = data.overfitting_curve || [];
    const fig = {
      data: [
        { x: curve.map((r) => r.max_depth), y: curve.map((r) => r.train_accuracy), mode: "lines+markers", name: "Train accuracy" },
        { x: curve.map((r) => r.max_depth), y: curve.map((r) => r.test_accuracy), mode: "lines+markers", name: "Test accuracy" },
      ],
      layout: { title: "Overfitting demo: depth vs accuracy", xaxis: { title: "max_depth" }, yaxis: { title: "accuracy", range: [0, 1.05] } },
    };
    renderPlot("#overfit-plot", JSON.stringify(fig));

    await loadMetaPanels();
    await classify();
  } catch (e) {
    setError(e?.message || "Training failed.");
  } finally {
    if (btn) btn.disabled = false;
  }
}

function runDebounced() {
  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(() => classify(), 300);
}

async function init() {
  loadTheme();
  $("#theme")?.addEventListener("change", (e) => setTheme(e.target.value));

  let savedModel = "cn2";
  try {
    savedModel = localStorage.getItem("ml_model_mode") || "cn2";
  } catch (_e) {}
  const modeSel = $("#model-mode");
  if (modeSel) modeSel.value = savedModel === "tree" ? "tree" : "cn2";
  modeSel?.addEventListener("change", () => {
    try {
      localStorage.setItem("ml_model_mode", getModelMode());
    } catch (_e) {}
    applyModelChrome();
    classify();
  });
  $("#tree-depth")?.addEventListener("input", (e) => {
    $("#tree-depth-value").textContent = String(e.target.value);
  });

  loadTab();
  document.querySelectorAll('.tab[role="tab"]').forEach((btn) => {
    btn.addEventListener("click", () => setTab(btn.getAttribute("data-tab")));
  });

  applyModelChrome();
  updatePresetButtons(getActiveDataset());
  updateFeatureLabels(getActiveDataset());
  applyPresetFromButton($("#preset-setosa"));

  for (let i = 0; i < 4; i++) {
    document.getElementById(`f${i}`)?.addEventListener("input", runDebounced);
  }

  $("#run").addEventListener("click", () => classify());

  $("#dataset")?.addEventListener("change", () => {
    const ds = getActiveDataset();
    updatePresetButtons(ds);
    updateFeatureLabels(ds);
    applyPresetFromButton($("#preset-setosa"));
  });
  $("#preset-setosa")?.addEventListener("click", () => {
    applyPresetFromButton($("#preset-setosa"));
    classify();
  });
  $("#preset-versicolor")?.addEventListener("click", () => {
    applyPresetFromButton($("#preset-versicolor"));
    classify();
  });
  $("#preset-virginica")?.addEventListener("click", () => {
    applyPresetFromButton($("#preset-virginica"));
    classify();
  });
  $("#train-btn")?.addEventListener("click", () => trainModels());

  await loadMetaPanels();
  classify();
}

window.addEventListener("DOMContentLoaded", init);

window.addEventListener(
  "resize",
  debounceResize(() => {
    const plotEl = $("#plot");
    if (!plotHasRendered || !window.Plotly || !plotEl) return;
    try {
      window.Plotly.Plots.resize(plotEl);
    } catch (_e) {}
  }, 120)
);

function debounceResize(fn, ms) {
  let t = null;
  return () => {
    clearTimeout(t);
    t = setTimeout(fn, ms);
  };
}
