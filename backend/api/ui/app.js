// ML Factory control console — a thin client. It holds no business logic and computes no metric;
// it only calls the /api routes and renders what the engine returns. The one rule it enforces
// visually: deterministic values look like data (monospace cobalt), LLM output is marked suggested.

"use strict";

const api = {
  async call(method, path, body) {
    const opts = { method, headers: {} };
    if (body !== undefined) {
      opts.headers["Content-Type"] = "application/json";
      opts.body = JSON.stringify(body);
    }
    const res = await fetch(`/api${path}`, opts);
    const text = await res.text();
    const data = text ? JSON.parse(text) : null;
    if (!res.ok) {
      const message = (data && data.message) || `Request failed (${res.status})`;
      const err = new Error(message);
      err.code = data && data.code;
      throw err;
    }
    return data;
  },
  get: (p) => api.call("GET", p),
  post: (p, b) => api.call("POST", p, b || {}),
};

const state = { experimentId: null, step: 0, record: null };

const STEPS = [
  { key: "create", label: "Start experiment" },
  { key: "data", label: "Point at data" },
  { key: "eda", label: "EDA & problem type" },
  { key: "features", label: "Engineered features" },
  { key: "train", label: "Train" },
  { key: "versions", label: "Model versions" },
  { key: "validate", label: "Validate & analyze" },
];

// --- small render helpers ----------------------------------------------------------------
const el = (tag, attrs = {}, ...kids) => {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v;
    else if (k === "html") node.innerHTML = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else if (v !== null && v !== undefined) node.setAttribute(k, v);
  }
  for (const kid of kids.flat()) {
    if (kid === null || kid === undefined) continue;
    node.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
  }
  return node;
};
const val = (x) => el("span", { class: "val" }, x === null || x === undefined ? "—" : x);
const fmt = (x, d = 4) => (typeof x === "number" ? x.toFixed(d) : x);

let toastTimer;
function toast(message, isError = false) {
  const node = document.getElementById("toast");
  node.textContent = message;
  node.className = `toast show${isError ? " error" : ""}`;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (node.className = "toast"), 3600);
}

async function guard(fn) {
  try {
    await fn();
  } catch (err) {
    toast(err.message, true);
  }
}

// --- the step rail -----------------------------------------------------------------------
function renderRail() {
  const list = document.getElementById("steps");
  list.replaceChildren(
    ...STEPS.map((s, i) => {
      const unlocked = i === 0 || state.experimentId;
      const btn = el(
        "button",
        {
          class: `step${i < state.step ? " done" : ""}`,
          "aria-current": i === state.step ? "true" : "false",
          disabled: unlocked ? null : "true",
          onclick: () => unlocked && go(i),
        },
        el("span", { class: "num" }, String(i + 1).padStart(2, "0")),
        el("span", {}, s.label)
      );
      return el("li", {}, btn);
    })
  );
}

function go(step) {
  state.step = step;
  renderRail();
  VIEWS[STEPS[step].key]();
}

const view = () => document.getElementById("view");
const title = (t, sub) => {
  document.getElementById("view-title").textContent = t;
  document.getElementById("view-sub").textContent = sub || "";
};

// --- views -------------------------------------------------------------------------------
const VIEWS = {};

VIEWS.create = () => {
  title("Start an experiment", "Name it, point it at a dataset, and name the column to predict.");
  const name = el("input", { placeholder: "Quarterly churn" });
  const uri = el("input", { placeholder: "s3://bucket/path/data.csv" });
  const target = el("input", { placeholder: "churned" });
  view().replaceChildren(
    el(
      "div",
      { class: "panel" },
      el("h3", {}, "New experiment"),
      el("label", { class: "field" }, el("span", {}, "Name"), name),
      el("label", { class: "field" }, el("span", {}, "Dataset S3 URI"), uri),
      el("label", { class: "field" }, el("span", {}, "Target column"), target),
      el(
        "button",
        {
          class: "primary",
          onclick: () =>
            guard(async () => {
              const rec = await api.post("/experiments", {
                name: name.value,
                dataset_uri: uri.value,
                target_column: target.value,
              });
              state.experimentId = rec.experiment_id;
              state.record = rec;
              toast(`Created ${rec.experiment_id}`);
              go(2); // EDA runs on create; skip straight to it
            }),
        },
        "Create & start EDA"
      )
    ),
    el(
      "div",
      { class: "panel" },
      el("h3", {}, "Open an existing experiment"),
      el("div", { class: "row" }, (() => {
        const idInput = el("input", { placeholder: "exp-…" });
        return el(
          "div",
          { class: "row", style: "width:100%" },
          idInput,
          el(
            "button",
            {
              class: "ghost",
              onclick: () =>
                guard(async () => {
                  const rec = await api.get(`/experiments/${idInput.value}`);
                  state.experimentId = rec.experiment_id;
                  state.record = rec;
                  go(2);
                }),
            },
            "Open"
          )
        );
      })())
    )
  );
};

VIEWS.data = () => {
  title("Point at data", "The dataset is already attached. Preview is bounded — never the full file.");
  view().replaceChildren(el("div", { class: "panel" }, el("p", { class: "empty" }, "Loading preview…")));
  guard(async () => {
    const p = await api.get(`/experiments/${state.experimentId}/dataset-preview`);
    const rows = p.columns.map((c) =>
      el("tr", {}, el("td", {}, c.name), el("td", { class: "id" }, c.dtype))
    );
    view().replaceChildren(
      el(
        "div",
        { class: "panel" },
        el("h3", {}, "Dataset"),
        el("p", {}, "URI ", el("span", { class: "id" }, p.dataset_uri)),
        el("p", {}, "Format ", val(p.file_format), "  Size ", val(p.size_bytes)),
        el("table", {}, el("thead", {}, el("tr", {}, el("th", {}, "Column"), el("th", {}, "Type"))), el("tbody", {}, ...rows)),
        p.columns_truncated ? el("p", { class: "hint" }, "Columns truncated for preview.") : null
      )
    );
  });
};

VIEWS.eda = () => {
  title("EDA & problem type", "What the platform found in the data before any engineering.");
  view().replaceChildren(el("div", { class: "panel" }, el("p", { class: "empty" }, "Loading EDA…")));
  guard(async () => {
    const eda = await api.get(`/experiments/${state.experimentId}/eda`);
    const problem = eda.target && eda.target.inferred_problem_type;
    view().replaceChildren(
      el(
        "div",
        { class: "panel" },
        el("h3", {}, "Overview"),
        el("p", {}, "Rows ", val(eda.dataset && eda.dataset.row_count), "  Columns ", val(eda.dataset && eda.dataset.column_count)),
        el("p", {}, "Inferred problem type ", val(problem)),
        el("p", { class: "hint" }, "You can override the problem type when you start training.")
      ),
      el(
        "div",
        { class: "panel" },
        el("h3", {}, "Next"),
        el("button", { class: "primary", onclick: () => go(3) }, "Review features")
      )
    );
  });
};

VIEWS.features = () => {
  title("Engineered features", "Keep or drop each column. Suggestions from the model are marked.");
  view().replaceChildren(el("div", { class: "panel" }, el("p", { class: "empty" }, "Loading feature review…")));
  guard(async () => {
    const review = await api.get(`/experiments/${state.experimentId}/feature-review`);
    const boxes = review.features.map((f) =>
      el(
        "label",
        {},
        el("input", { type: "checkbox", checked: f.selected ? "true" : null, "data-feature": f.feature }),
        el("span", {}, f.feature),
        el("span", { class: "id" }, `  ${f.semantic_type} · ${f.leakage_risk} risk`)
      )
    );
    const panel = el(
      "div",
      { class: "panel" },
      el("h3", {}, `Features — target ${review.target_column}`),
      el("div", { class: "checks" }, ...boxes),
      el("div", { class: "row", style: "margin-top:16px" }, el("div", { class: "spacer" }),
        el(
          "button",
          {
            class: "primary",
            onclick: (e) =>
              guard(async () => {
                const selected = [...panel.querySelectorAll("input:checked")].map((i) => i.dataset.feature);
                await api.post(`/experiments/${state.experimentId}/features`, { selected_features: selected });
                toast(`Saved ${selected.length} feature(s)`);
                go(4);
              }),
          },
          "Approve & continue"
        )
      )
    );
    view().replaceChildren(panel);
  });
};

VIEWS.train = () => {
  title("Train", "Pick algorithms and start. Cross-validation and a held-out test fold are optional.");
  const models = el("input", { placeholder: "logistic_regression, random_forest, xgboost" });
  const cv = el("input", { type: "checkbox" });
  const folds = el("input", { type: "number", value: "5", min: "2", max: "10", style: "max-width:90px" });
  view().replaceChildren(
    el(
      "div",
      { class: "panel" },
      el("h3", {}, "Training request"),
      el("label", { class: "field" }, el("span", {}, "Models (comma-separated; blank = all)"), models),
      el("label", {}, el("div", { class: "row" }, cv, el("span", {}, " Cross-validate"), el("span", { class: "hint" }, " folds "), folds)),
      el("div", { class: "row", style: "margin-top:16px" },
        el(
          "button",
          {
            class: "primary",
            onclick: () =>
              guard(async () => {
                const list = models.value.split(",").map((m) => m.trim()).filter(Boolean);
                const body = { models: list };
                if (cv.checked) { body.evaluation_mode = "cv"; body.cv_folds = Number(folds.value); }
                const started = await api.post(`/experiments/${state.experimentId}/training`, body);
                const mode = started.evaluation_mode + (started.cv_folds ? ` · K=${started.cv_folds}` : "");
                toast(`Training started (${mode})`);
                go(5);
              }),
          },
          "Start training"
        )
      ),
      el("p", { class: "hint", style: "margin-top:14px" }, "All guards (instance, budget, concurrency, fold ceiling) run on the platform; nothing is bypassed here.")
    )
  );
};

VIEWS.versions = () => {
  title("Model versions", "Every model from every run, kept and independently validatable.");
  view().replaceChildren(el("div", { class: "panel" }, el("p", { class: "empty" }, "Loading versions…")));
  guard(async () => {
    const data = await api.get(`/experiments/${state.experimentId}/model-versions`);
    if (!data.model_versions.length) {
      view().replaceChildren(el("div", { class: "panel" }, el("p", { class: "empty" }, "No model versions yet. Train first, then refresh.")));
      return;
    }
    const rows = data.model_versions.map((v) =>
      el(
        "tr",
        {},
        el("td", { class: "id" }, v.version_id),
        el("td", {}, v.model_name),
        el("td", {}, v.run_id ? el("span", { class: "badge" }, "archived") : el("span", { class: "badge run" }, "current")),
        el("td", { class: "num" }, val(fmt(v.primary_score)))
      )
    );
    view().replaceChildren(
      el(
        "div",
        { class: "panel" },
        el("h3", {}, `${data.count} version(s)`),
        el("table", {}, el("thead", {}, el("tr", {}, el("th", {}, "Version id"), el("th", {}, "Model"), el("th", {}, "Run"), el("th", { class: "num" }, "Primary score"))), el("tbody", {}, ...rows)),
        el("div", { class: "row", style: "margin-top:16px" }, el("button", { class: "primary", onclick: () => go(6) }, "Validate on a dataset"))
      )
    );
  });
};

VIEWS.validate = () => {
  title("Validate & analyze", "Score the saved models on a labeled dataset. Labels are kept only for metrics.");
  const uri = el("input", { placeholder: "s3://bucket/path/validation.csv" });
  const results = el("div", {});
  view().replaceChildren(
    el(
      "div",
      { class: "panel" },
      el("h3", {}, "Validation dataset"),
      el("label", { class: "field" }, el("span", {}, "Labeled dataset S3 URI (all model versions are scored)"), uri),
      el(
        "button",
        {
          class: "primary",
          onclick: () =>
            guard(async () => {
              results.replaceChildren(el("p", { class: "empty" }, "Running validation…"));
              const out = await api.post(`/experiments/${state.experimentId}/validations`, { dataset_uri: uri.value });
              renderValidation(results, out);
            }),
        },
        "Validate all versions"
      )
    ),
    el("div", { class: "panel" }, el("h3", {}, "Results"), results)
  );
  results.replaceChildren(el("p", { class: "empty" }, "No validation yet."));
};

function renderValidation(container, out) {
  if (out.status !== "completed") {
    container.replaceChildren(el("p", { class: "empty" }, `Status: ${out.status}. ${out.failure_message || ""}`));
    return;
  }
  const report = out.report;
  const kids = [
    el("p", {}, "Rows ", val(report.row_count), "  Labels present ", val(String(report.labels_present))),
  ];
  for (const r of report.results) {
    kids.push(el("h3", { style: "margin-top:18px" }, `${r.model_name} `, el("span", { class: "id" }, r.version_id)));
    if (r.status !== "completed") {
      kids.push(el("p", { class: "empty" }, r.failure_message || "failed"));
      continue;
    }
    if (r.metrics_all_rows) {
      const metricRows = Object.entries(r.metrics_all_rows.values || r.metrics_all_rows).map(([k, v]) =>
        el("tr", {}, el("td", {}, k), el("td", { class: "num" }, val(fmt(v))))
      );
      kids.push(el("table", {}, el("thead", {}, el("tr", {}, el("th", {}, "Metric (all rows)"), el("th", { class: "num" }, "Value"))), el("tbody", {}, ...metricRows)));
    } else {
      kids.push(el("p", { class: "hint" }, "No labels on this dataset, so no metrics — predictions only."));
    }
    // Threshold analysis, computed on THIS dataset's labels (binary classification).
    if (r.threshold_analysis && r.threshold_analysis.curve) {
      kids.push(el("p", { style: "margin-top:12px" }, el("strong", {}, "Threshold analysis "), el("span", { class: "hint" }, "— which threshold yields what, on the validation labels")));
      const t = r.threshold_analysis;
      const curveRows = t.curve.map((p) =>
        el("tr", {}, el("td", { class: "num" }, val(fmt(p.threshold, 2))), el("td", { class: "num" }, val(fmt(p.precision, 3))), el("td", { class: "num" }, val(fmt(p.recall, 3))), el("td", { class: "num" }, val(fmt(p.f1, 3))))
      );
      kids.push(
        el(
          "table",
          {},
          el("thead", {}, el("tr", {}, el("th", { class: "num" }, "Threshold"), el("th", { class: "num" }, "Precision"), el("th", { class: "num" }, "Recall"), el("th", { class: "num" }, "F1"))),
          el("tbody", {}, ...curveRows)
        )
      );
      if (t.best_f1) {
        kids.push(el("p", { class: "hint" }, "Best F1 at threshold ", val(fmt(t.best_f1.threshold, 2)), " — F1 ", val(fmt(t.best_f1.f1, 3))));
      }
    }
  }
  container.replaceChildren(...kids);
}

// --- boot --------------------------------------------------------------------------------
renderRail();
VIEWS.create();
