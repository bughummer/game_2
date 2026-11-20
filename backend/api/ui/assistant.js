// ML Assistant chat panel (ml-assistant design C19). Loaded after app.js and reuses its globals
// (api, el, toast, title, view). It holds no agent logic: it calls /api/agent/... and renders
// the SessionView. Every string from the model, a tool, a file or a column goes into the page as
// a text node (el() children or textContent). Nothing here assigns HTML markup.

"use strict";

(() => {
  const TERMINAL = new Set(["completed", "cancelled", "failed"]);
  const STATUS_TEXT = {
    running: "Running",
    waiting_for_approval: "Waiting for your approval",
    waiting_for_job: "Waiting for jobs",
    waiting_for_user: "Waiting for you",
    completed: "Completed",
    failed: "Failed",
    cancelled: "Cancelled",
  };
  const REVIEW_LABELS = {
    file_name: "File name",
    size_bytes: "Size (bytes)",
    destination_prefix: "Destination",
    content_hash: "Content hash",
    draft_id: "Draft",
    revision: "Revision",
    assumptions: "Assumptions",
    file_id: "File id",
  };
  const MAX_BATCH = 10;
  const MAX_FILES = 100;

  // One open panel at a time. `sid` is null on the session list.
  const S = {
    config: null,
    root: null,
    sid: null,
    timer: null,
    view: null,
    lastSeq: 0,
    seen: new Set(),
    entries: [],
    resumed: false,
    cards: new Map(),
    keys: {},
    files: new Map(), // `${name}\0${sha256}` -> File picked in this page
    uploads: new Set(), // action ids started (or failed and parked until "Retry")
    missing: null, // { action_id, key, name } when the page no longer holds the file
    uploadNote: "",
    failedAction: null,
    N: null,
    railBtn: null,
  };

  // --- helpers ---------------------------------------------------------------------------
  const json = (v) => JSON.stringify(v, null, 2);
  const money = (x) => (typeof x === "number" ? `$${x.toFixed(2)}` : "—");
  const pre = (v) => el("pre", { class: "as-pre" }, typeof v === "string" ? v : json(v));
  const badge = (text, cls) => el("span", { class: `badge${cls ? ` ${cls}` : ""}` }, text);
  const label = (text) => el("div", { class: "as-label" }, text);
  const isObj = (v) => v !== null && typeof v === "object" && !Array.isArray(v);
  const isScalar = (v) => v === null || ["string", "number", "boolean"].includes(typeof v);
  const cell = (v) => (isScalar(v) ? (v === null ? "—" : String(v)) : JSON.stringify(v));
  const short = (v, n = 500) => {
    const t = typeof v === "string" ? v : JSON.stringify(v);
    return t.length > n ? `${t.slice(0, n - 3)}...` : t;
  };
  const fileKey = (name, sha) => `${name}\0${String(sha).toLowerCase()}`;
  const alive = () => Boolean(S.root && S.root.isConnected && S.sid);

  async function attempt(fn) {
    try {
      return await fn();
    } catch (err) {
      toast(err.message, true);
      return undefined;
    }
  }

  // --- rail entry ------------------------------------------------------------------------
  function mountRail() {
    const steps = document.getElementById("steps");
    if (!steps || document.getElementById("as-rail")) return;
    S.railBtn = el(
      "button",
      { class: "step", "aria-current": "false", onclick: () => openPanel() },
      el("span", { class: "num" }, "AI"),
      el("span", {}, "Assistant")
    );
    steps.after(el("div", { class: "as-rail", id: "as-rail" }, S.railBtn));
    // app.js re-renders its own steps; once its view replaces ours, drop our highlight.
    steps.addEventListener("click", () =>
      setTimeout(() => {
        if (!S.root || !S.root.isConnected) {
          stopPolling();
          S.railBtn.setAttribute("aria-current", "false");
        }
      }, 0)
    );
  }

  async function openPanel() {
    stopPolling();
    S.railBtn.setAttribute("aria-current", "true");
    document.querySelectorAll("#steps .step").forEach((b) => b.setAttribute("aria-current", "false"));
    title("Assistant", "Describe a goal. Steps that change or cost something wait for your approval.");
    S.sid = null;
    S.root = el("div", { class: "as" }, el("div", { class: "panel" }, el("p", { class: "empty" }, "Loading…")));
    view().replaceChildren(S.root);
    S.config = await attempt(() => api.get("/agent/config"));
    if (!S.config) {
      S.root.replaceChildren(
        el("div", { class: "panel" }, el("p", { class: "empty" }, "The assistant is not available on this deployment."))
      );
      return;
    }
    renderSessions();
  }

  // --- session list ----------------------------------------------------------------------
  async function renderSessions() {
    stopPolling();
    S.sid = null;
    const goal = el("textarea", { rows: "3", placeholder: "Train a churn model on my CSV and tell me which model is best." });
    const list = el("div", {}, el("p", { class: "empty" }, "Loading sessions…"));
    S.root.replaceChildren(
      el(
        "div",
        { class: "panel" },
        el("h3", {}, "New session"),
        el("label", { class: "field" }, el("span", {}, "Goal"), goal),
        el(
          "button",
          {
            class: "primary",
            onclick: () =>
              attempt(async () => {
                const v = await api.post("/agent/sessions", { goal: goal.value, mode: "supervised" });
                openSession(v.session_id, v);
              }),
          },
          "Start session"
        ),
        el("p", { class: "hint", style: "margin-top:10px" }, "Sessions start in Supervised mode.")
      ),
      el("div", { class: "panel" }, el("h3", {}, "Your sessions"), list)
    );
    const page = await attempt(() => api.get("/agent/sessions?limit=50"));
    if (!page) return;
    if (!page.items.length) {
      list.replaceChildren(el("p", { class: "empty" }, "No sessions yet."));
      return;
    }
    const rows = page.items.map((s) =>
      el(
        "tr",
        {},
        el("td", { class: "id" }, s.session_id),
        el("td", {}, statusBadge(s.status)),
        el("td", {}, s.mode),
        el("td", { class: "id" }, s.last_activity_at),
        el("td", {}, el("button", { class: "ghost", onclick: () => openSession(s.session_id) }, "Open"))
      )
    );
    list.replaceChildren(
      el(
        "table",
        {},
        el("thead", {}, el("tr", {}, el("th", {}, "Session"), el("th", {}, "Status"), el("th", {}, "Mode"), el("th", {}, "Last activity"), el("th", {}, ""))),
        el("tbody", {}, ...rows)
      )
    );
  }

  function statusBadge(status) {
    const cls = status === "completed" ? "ok" : status === "failed" || status === "cancelled" ? "fail" : "run";
    return badge(STATUS_TEXT[status] || status, cls);
  }

  // --- session panel ---------------------------------------------------------------------
  function openSession(sid, initial) {
    stopPolling();
    Object.assign(S, {
      sid,
      view: null,
      lastSeq: 0,
      seen: new Set(),
      entries: [],
      resumed: false,
      cards: new Map(),
      keys: {},
      uploads: new Set(),
      missing: null,
      uploadNote: "",
      failedAction: null,
    });
    buildLayout();
    if (initial) {
      apply(initial);
      schedule();
    } else {
      poll();
    }
  }

  function buildLayout() {
    const N = {
      head: el("div", {}),
      cancel: cancelControls(),
      mode: el("div", { class: "panel" }),
      limit: el("div", { class: "panel as-attention", hidden: "true" }),
      approvals: el("div", { class: "panel as-attention", hidden: "true" }),
      transcript: el("div", { class: "as-transcript", "aria-live": "polite" }),
      answer: el("div", { class: "panel", hidden: "true" }),
      files: el("div", {}),
      audit: el("div", {}),
    };
    S.N = N;
    const text = el("textarea", { rows: "3", placeholder: "Message the assistant…" });
    const send = () =>
      attempt(async () => {
        if (!text.value.trim()) return;
        const v = await api.post(`/agent/sessions/${S.sid}/messages`, { text: text.value });
        text.value = "";
        apply(v);
        restart();
      });
    text.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) send();
    });
    S.root.replaceChildren(
      el("div", { class: "row as-back" }, el("button", { class: "ghost", onclick: () => renderSessions() }, "← All sessions")),
      el("div", { class: "panel" }, N.head, N.cancel),
      N.mode,
      N.limit,
      N.approvals,
      el(
        "div",
        { class: "panel" },
        el("h3", {}, "Transcript"),
        N.transcript,
        el("div", { class: "as-composer" }, text, el("div", { class: "row" }, el("span", { class: "hint" }, "Ctrl+Enter to send"), el("div", { class: "spacer" }), el("button", { class: "primary", onclick: send }, "Send")))
      ),
      N.answer,
      el("div", { class: "panel" }, el("h3", {}, "Local files"), filePicker(), N.files),
      el(
        "div",
        { class: "panel" },
        el("h3", {}, "Audit trail"),
        el("button", { class: "ghost", onclick: () => loadAudit() }, "Load audit trail"),
        N.audit
      )
    );
  }

  // --- polling ---------------------------------------------------------------------------
  function stopPolling() {
    clearTimeout(S.timer);
    S.timer = null;
  }

  function schedule() {
    if (!alive() || S.timer) return;
    if (S.view && TERMINAL.has(S.view.status)) return; // nothing changes without a user action
    const secs = Number(S.config && S.config.web_poll_seconds) > 0 ? Number(S.config.web_poll_seconds) : 1;
    S.timer = setTimeout(poll, secs * 1000);
  }

  function restart() {
    stopPolling();
    schedule();
  }

  async function poll() {
    S.timer = null;
    if (!alive()) return;
    const sid = S.sid;
    try {
      const v = await api.get(`/agent/sessions/${sid}?after_seq=${S.lastSeq}`);
      if (sid !== S.sid || !alive()) return;
      apply(v);
    } catch (err) {
      toast(err.message, true);
    }
    schedule();
  }

  function apply(v) {
    S.view = v;
    const box = S.N.transcript;
    const atBottom = box.scrollHeight - box.scrollTop - box.clientHeight < 40;
    for (const e of v.transcript || []) {
      if (S.seen.has(e.seq)) continue;
      S.seen.add(e.seq);
      S.entries.push(e);
      box.append(entryNode(e));
    }
    if (atBottom) box.scrollTop = box.scrollHeight;
    S.lastSeq = Math.max(S.lastSeq, v.last_seq || 0);
    renderHead(v);
    renderMode(v);
    renderLimit(v);
    renderApprovals(v);
    renderAnswer(v);
    renderFiles(v);
    if (!v.jobs_ready) S.resumed = false;
    else if (!S.resumed) {
      S.resumed = true; // once per jobs_ready episode
      attempt(async () => {
        const r = await api.post(`/agent/sessions/${S.sid}/resume`);
        apply(r);
        restart();
      });
    }
    const ca = v.client_action;
    if (ca && ca.kind === "upload" && ca.status === "ready") upload(ca);
  }

  // --- header: status, counters, jobs, cancel --------------------------------------------
  function renderHead(v) {
    const c = v.counters;
    const l = v.limits;
    const counter = (name, text) => el("div", { class: "as-counter" }, el("span", { class: "hint" }, name), el("span", { class: "val" }, text));
    const kids = [
      el("div", { class: "row" }, el("h3", { style: "margin:0" }, "Session"), el("span", { class: "id" }, v.session_id), el("div", { class: "spacer" }), statusBadge(v.status)),
      el("p", { class: "as-goal" }, v.goal),
      el(
        "div",
        { class: "as-counters" },
        counter("Tool calls", `${c.tool_calls} / ${l.max_tool_calls}`),
        counter("Costly calls", `${c.costly_tool_calls} / ${l.max_costly_tool_calls}`),
        counter("Estimated spend", `${money(c.estimated_spend_usd)} / ${money(l.max_estimated_spend_usd)}`),
        counter("Turns", `${c.converse_turns} / ${l.max_converse_turns}`)
      ),
    ];
    if (v.plan && v.plan.length) {
      kids.push(label("Plan"), el("ol", { class: "as-plan" }, ...v.plan.map((s) => el("li", {}, s))));
    }
    if (v.status === "waiting_for_job") {
      kids.push(el("p", { class: "hint" }, "Waiting for the jobs below. No model call is open; the assistant resumes when they finish."));
    }
    if (v.running_jobs && v.running_jobs.length) {
      kids.push(
        label("Running jobs"),
        el("ul", { class: "as-list" }, ...v.running_jobs.map((j) => el("li", {}, el("span", { class: "id" }, j.execution_id), ` ${j.kind} · experiment `, el("span", { class: "id" }, j.experiment_id))))
      );
    }
    if (v.experiment_ids && v.experiment_ids.length) {
      kids.push(el("p", { class: "hint" }, "Experiments ", ...v.experiment_ids.map((id) => el("span", { class: "id" }, `${id} `))));
    }
    if (v.failure) {
      kids.push(
        el(
          "div",
          { class: "as-failure" },
          el("strong", {}, v.failure.category),
          el("span", {}, ` ${v.failure.message}`),
          v.retriable
            ? el("button", { class: "ghost", style: "margin-left:10px", onclick: () => attempt(async () => { apply(await api.post(`/agent/sessions/${S.sid}/resume`)); restart(); }) }, "Retry")
            : null
        )
      );
    }
    S.N.head.replaceChildren(...kids);
    const finished = v.status === "completed" || v.status === "cancelled";
    S.N.cancel.hidden = finished;
  }

  function cancelControls() {
    const stopJobs = el("input", { type: "checkbox" });
    const jobsRow = el("label", { class: "as-check" }, stopJobs, el("span", {}, "Also stop the running jobs"));
    const box = el(
      "div",
      { class: "as-confirm", hidden: "true" },
      el("p", {}, "Cancel this session? Pending approvals are rejected and the assistant stops."),
      jobsRow,
      el(
        "div",
        { class: "row" },
        el(
          "button",
          {
            class: "ghost as-danger",
            onclick: () =>
              attempt(async () => {
                const v = await api.post(`/agent/sessions/${S.sid}/cancel`, { stop_jobs: stopJobs.checked });
                box.hidden = true;
                apply(v);
                restart();
              }),
          },
          "Cancel session"
        ),
        el("button", { class: "ghost", onclick: () => (box.hidden = true) }, "Keep running")
      )
    );
    const open = el(
      "button",
      {
        class: "ghost",
        onclick: () => {
          stopJobs.checked = false; // default: leave running jobs alone
          jobsRow.hidden = !(S.view && S.view.running_jobs && S.view.running_jobs.length);
          box.hidden = false;
        },
      },
      "Cancel session…"
    );
    return el("div", { class: "as-cancel" }, open, box);
  }

  // --- mode switch -----------------------------------------------------------------------
  function renderMode(v) {
    const allowed = Boolean(S.config.autopilot_allowed);
    const finished = v.status === "completed" || v.status === "cancelled";
    const key = [v.mode, v.autopilot_max_estimated_spend_usd, allowed, finished, v.limits.max_estimated_spend_usd].join("|");
    if (S.keys.mode === key) return;
    S.keys.mode = key;
    const cap = el("input", { type: "number", min: "0.01", step: "0.01", max: String(v.limits.max_estimated_spend_usd), placeholder: `≤ ${v.limits.max_estimated_spend_usd}`, style: "max-width:140px" });
    const capRow = el(
      "div",
      { class: "row as-cap", hidden: "true" },
      el("span", { class: "hint" }, "Max estimated spend (USD)"),
      cap,
      el(
        "button",
        {
          class: "primary",
          onclick: () => {
            const amount = Number(cap.value);
            if (!(amount > 0)) return toast("Enter a max spend above 0.", true);
            setMode("autopilot", amount);
          },
        },
        "Enable Autopilot"
      )
    );
    const isAuto = v.mode === "autopilot";
    const seg = (name, on, onclick, disabled) =>
      el("button", { class: `as-seg${on ? " on" : ""}`, "aria-pressed": on ? "true" : "false", disabled: disabled ? "true" : null, onclick }, name);
    S.N.mode.replaceChildren(
      el("h3", {}, "Mode"),
      el(
        "div",
        { class: "row" },
        el(
          "div",
          { class: "as-switch", role: "group", "aria-label": "Mode" },
          seg("Supervised", !isAuto, () => !isAuto || setMode("supervised"), finished),
          seg("Autopilot", isAuto, () => { capRow.hidden = false; cap.focus(); }, finished || !allowed)
        ),
        isAuto && v.autopilot_max_estimated_spend_usd != null
          ? el("span", { class: "hint" }, "Max estimated spend ", el("span", { class: "val" }, money(v.autopilot_max_estimated_spend_usd)))
          : null
      ),
      capRow,
      el(
        "p",
        { class: "hint", style: "margin-top:10px" },
        allowed
          ? "Supervised: every step that changes or costs something waits for you. Autopilot: those steps run without asking, up to the spend you set."
          : "Supervised: every step that changes or costs something waits for you. Autopilot is turned off on this deployment."
      )
    );
  }

  function setMode(mode, maxSpend) {
    return attempt(async () => {
      const body = { mode };
      if (mode === "autopilot") body.max_estimated_spend_usd = maxSpend;
      apply(await api.post(`/agent/sessions/${S.sid}/mode`, body));
      restart();
    });
  }

  // --- limit stop ------------------------------------------------------------------------
  function reachedLimit(v) {
    if (v.status !== "waiting_for_user") return null;
    for (let i = S.entries.length - 1; i >= 0; i--) {
      const e = S.entries[i];
      if (e.kind === "limit_stop") return typeof (e.data || {}).limit === "string" ? e : null;
      if (e.kind === "user_message" || (e.kind === "status" && (e.data || {}).status === "running")) return null;
    }
    return null;
  }

  function renderLimit(v) {
    const stop = reachedLimit(v);
    const key = stop ? String(stop.seq) : "";
    if (S.keys.limit === key) return;
    S.keys.limit = key;
    S.N.limit.hidden = !stop;
    if (!stop) return S.N.limit.replaceChildren();
    const name = stop.data.limit;
    const amount = el("input", { type: "number", min: "0", step: "any", style: "max-width:140px" });
    S.N.limit.replaceChildren(
      el("h3", {}, "Session limit reached"),
      el("p", {}, "Stopped at ", el("span", { class: "id" }, name), stop.text ? ` — ${stop.text}` : ""),
      el(
        "div",
        { class: "row" },
        el("span", { class: "hint" }, "Raise by"),
        amount,
        el(
          "button",
          {
            class: "primary",
            onclick: () =>
              attempt(async () => {
                const n = Number(amount.value);
                if (!(n > 0)) return toast("Enter an amount above 0.", true);
                apply(await api.post(`/agent/sessions/${S.sid}/limits/continue`, { limit: name, amount: n }));
                restart();
              }),
          },
          "Raise and continue"
        )
      )
    );
  }

  // --- approvals -------------------------------------------------------------------------
  const shownHash = (a) => (a.status === "edit_proposed" && a.edited_hash ? a.edited_hash : a.args_hash);

  function renderApprovals(v) {
    const pending = v.pending_approvals || [];
    const panel = S.N.approvals;
    panel.hidden = !pending.length;
    const ids = new Set();
    const cards = pending.map((a) => {
      const key = `${a.approval_id}|${a.status}|${a.edited_hash || ""}|${a.args_hash}`;
      ids.add(key);
      if (!S.cards.has(key)) S.cards.set(key, approvalCard(a));
      return S.cards.get(key);
    });
    for (const k of [...S.cards.keys()]) if (!ids.has(k)) S.cards.delete(k);
    const batchKey = pending.map((a) => `${a.approval_id}:${shownHash(a)}`).join(",");
    if (S.keys.approvals === batchKey && panel.childElementCount) return;
    S.keys.approvals = batchKey;
    const shown = pending.slice(0, MAX_BATCH);
    panel.replaceChildren(
      el(
        "div",
        { class: "row" },
        el("h3", { style: "margin:0" }, `Approval needed (${pending.length})`),
        el("div", { class: "spacer" }),
        pending.length > 1
          ? el(
              "button",
              {
                class: "ghost",
                onclick: () =>
                  attempt(async () => {
                    const items = shown.map((a) => ({ approval_id: a.approval_id, args_hash: shownHash(a) }));
                    apply(await api.post(`/agent/sessions/${S.sid}/approvals/batch`, { items }));
                    restart();
                  }),
              },
              `Approve the ${shown.length} shown`
            )
          : null
      ),
      ...cards
    );
  }

  function approvalCard(a) {
    const hash = shownHash(a);
    const edited = a.status === "edit_proposed";
    const reason = el("input", { placeholder: "Reason (optional, sent with a rejection)" });
    const editor = el("textarea", { rows: "8", class: "as-code" });
    editor.value = json(a.edited_arguments || a.arguments);
    const editBox = el(
      "div",
      { class: "as-edit", hidden: "true" },
      label("Edit the arguments (JSON object)"),
      editor,
      el(
        "div",
        { class: "row" },
        el(
          "button",
          {
            class: "ghost",
            onclick: () => {
              let parsed;
              try {
                parsed = JSON.parse(editor.value);
              } catch (err) {
                return toast("The edited arguments are not valid JSON.", true);
              }
              if (!isObj(parsed)) return toast("The edited arguments must be a JSON object.", true);
              decide(a, "edit", { edited_args: parsed });
            },
          },
          "Propose edit"
        ),
        el("span", { class: "hint" }, "After an edit, approve again to run the edited arguments.")
      )
    );
    const kids = [
      el("div", { class: "row" }, el("strong", { class: "as-tool" }, a.tool), badge(a.kind), edited ? badge("edit awaiting your approval", "run") : null, el("div", { class: "spacer" }), el("span", { class: "id" }, a.approval_id)),
      label(edited ? "Original arguments" : "Arguments"),
      pre(a.arguments),
    ];
    if (edited && a.edited_arguments) kids.push(label("Edited arguments (these run if you approve)"), pre(a.edited_arguments));
    kids.push(label("Rationale (written by the model)"), el("p", { class: "as-untrusted" }, a.rationale || "—"));
    kids.push(
      el(
        "p",
        {},
        "Estimate ",
        a.estimate_usd === null || a.estimate_usd === undefined ? el("span", { class: "hint" }, "no estimate available") : el("span", { class: "val" }, money(a.estimate_usd)),
        a.estimate_basis ? el("span", { class: "hint" }, ` (${a.estimate_basis})`) : null
      )
    );
    if (a.review) kids.push(reviewBlock(a.review));
    kids.push(el("p", { class: "hint" }, "Arguments hash ", el("span", { class: "id" }, hash), " · expires ", el("span", { class: "id" }, a.expires_at)));
    kids.push(
      el(
        "div",
        { class: "row as-actions" },
        el("button", { class: "primary", onclick: () => decide(a, "approve") }, edited ? "Approve edited" : "Approve"),
        el("button", { class: "ghost as-danger", onclick: () => decide(a, "reject", reason.value ? { reason: reason.value } : {}) }, "Reject"),
        el("button", { class: "ghost", onclick: () => (editBox.hidden = !editBox.hidden) }, "Edit…"),
        reason
      ),
      editBox
    );
    return el("div", { class: "as-card" }, ...kids);
  }

  function reviewBlock(review) {
    const rows = Object.keys(review)
      .filter((k) => k !== "content")
      .sort()
      .map((k) => el("tr", {}, el("td", {}, REVIEW_LABELS[k] || k), el("td", { class: "id" }, Array.isArray(review[k]) ? review[k].map(cell).join("; ") : cell(review[k]))));
    return el(
      "div",
      { class: "as-review" },
      label("Review"),
      rows.length ? el("table", {}, el("tbody", {}, ...rows)) : null,
      "content" in review ? el("div", {}, label("Script"), pre(String(review.content))) : null
    );
  }

  function decide(a, decision, extra = {}) {
    return attempt(async () => {
      const body = { decision, args_hash: shownHash(a), ...extra };
      apply(await api.post(`/agent/sessions/${S.sid}/approvals/${a.approval_id}`, body));
      restart();
    });
  }

  // --- transcript ------------------------------------------------------------------------
  function entryNode(e) {
    const data = e.data || {};
    const seq = el("span", { class: "as-seq" }, String(e.seq));
    const row = (cls, ...kids) => el("div", { class: `as-entry ${cls}` }, seq, el("div", { class: "as-body" }, ...kids));
    switch (e.kind) {
      case "user_message":
        return row("user", el("div", { class: "as-who" }, "You"), el("div", { class: "as-text" }, e.text || ""));
      case "assistant_text":
        return row("assistant", el("div", { class: "as-who" }, "Assistant"), el("div", { class: "as-text" }, e.text || ""));
      case "plan":
        return row("event", el("div", { class: "as-who" }, "Plan"), el("ol", { class: "as-plan" }, ...(data.steps || []).map((s) => el("li", {}, String(s)))));
      case "tool_call":
        return row(
          "tool",
          el("details", {}, el("summary", {}, "→ ", el("span", { class: "as-tool" }, data.tool || "tool")), "arguments" in data ? pre(data.arguments) : pre(data))
        );
      case "tool_result": {
        const parts = [data.state, data.error_code ? `error ${data.error_code}` : null].filter(Boolean).join(" · ");
        return row(
          "tool",
          el(
            "details",
            {},
            el("summary", {}, "← ", el("span", { class: "as-tool" }, data.tool || "result"), parts ? el("span", { class: "hint" }, ` ${parts}`) : null),
            e.text ? el("div", { class: "as-text" }, e.text) : null,
            Object.keys(data).length ? pre(data) : null
          )
        );
      }
      case "final_answer":
        return row("event", el("span", { class: "hint" }, "Final answer (shown below)."));
      case "error":
        return row("error", el("span", {}, e.text || short(data)));
      default: {
        const name = String(e.kind || "event").replace(/_/g, " ");
        return row("event", el("span", { class: "hint" }, `${name}: `), el("span", {}, e.text || short(data)));
      }
    }
  }

  // --- final answer ----------------------------------------------------------------------
  function renderAnswer(v) {
    const fa = v.final_answer;
    const key = fa ? JSON.stringify(fa) : "";
    if (S.keys.answer === key) return;
    S.keys.answer = key;
    S.N.answer.hidden = !fa;
    if (!fa) return S.N.answer.replaceChildren();
    const kids = [
      el(
        "div",
        { class: "row" },
        el("h3", { style: "margin:0" }, "Answer"),
        el("div", { class: "spacer" }),
        ...(fa.labels || []).map((l) => badge(l, l === "Advisory" ? "as-advisory" : ""))
      ),
      ...(fa.sentences || []).map(sentenceNode),
    ];
    for (const block of fa.verbatim || []) {
      kids.push(
        el(
          "div",
          { class: "as-verbatim" },
          el("div", { class: "row" }, el("strong", {}, block.title), el("span", { class: "id" }, `from ${block.tool_call_id}`), el("span", { class: "hint" }, "verbatim from the stored result")),
          dataNode(block.content)
        )
      );
    }
    if (fa.notes && fa.notes.length) kids.push(el("ul", { class: "as-notes" }, ...fa.notes.map((n) => el("li", {}, n))));
    S.N.answer.replaceChildren(...kids);
  }

  function attrTag(a) {
    const source = a.experiment_id || a.tool_call_id;
    const where = `${source} · ${a.path}${a.artifact_ref ? ` · ${a.artifact_ref}` : ""}`;
    return el(
      "span",
      { class: "as-attr", title: where },
      `${source} · ${a.path}`,
      a.metric_label ? el("span", { class: `as-metric ${a.metric_label}` }, a.metric_label === "self_reported" ? "self-reported" : "engine") : null
    );
  }

  function sentenceNode(s) {
    let text = s.text || "";
    const flagged = s.status === "unverified" || s.status === "overclaim";
    const marker = `[${s.status}] `;
    if (flagged && text.startsWith(marker)) text = text.slice(marker.length);
    const p = el("p", { class: `as-sentence ${s.status}` });
    if (flagged) p.append(badge(s.status, "as-warn"), " ");
    let cursor = 0;
    let last = null;
    const unplaced = [];
    for (const a of s.attributions || []) {
      const at = a.value_text ? text.indexOf(a.value_text, cursor) : -1;
      if (at < 0) {
        if (last !== null && a.value_text === last) p.append(attrTag(a)); // another source for the same number
        else unplaced.push(a);
        continue;
      }
      p.append(text.slice(cursor, at), el("span", { class: "as-num" }, a.value_text), attrTag(a));
      cursor = at + a.value_text.length;
      last = a.value_text;
    }
    p.append(text.slice(cursor));
    for (const a of unplaced) p.append(" ", el("span", { class: "as-num" }, a.value_text), attrTag(a));
    return p;
  }

  function dataNode(v) {
    if (Array.isArray(v) && v.length && v.every(isObj)) {
      const cols = [...new Set(v.flatMap((r) => Object.keys(r)))].slice(0, 40);
      return el(
        "div",
        { class: "as-scroll" },
        el(
          "table",
          {},
          el("thead", {}, el("tr", {}, ...cols.map((c) => el("th", {}, c)))),
          el("tbody", {}, ...v.map((r) => el("tr", {}, ...cols.map((c) => el("td", { class: typeof r[c] === "number" ? "num" : "" }, cell(r[c]))))))
        )
      );
    }
    if (isObj(v) && Object.values(v).every(isScalar)) {
      return el("table", {}, el("tbody", {}, ...Object.entries(v).map(([k, x]) => el("tr", {}, el("td", {}, k), el("td", { class: typeof x === "number" ? "num" : "" }, cell(x))))));
    }
    if (isObj(v)) {
      return el("div", {}, ...Object.entries(v).map(([k, x]) => el("div", { class: "as-vblock" }, label(k), dataNode(x))));
    }
    if (isScalar(v)) return el("span", { class: "as-num" }, cell(v));
    return pre(v);
  }

  // --- local files and uploads -----------------------------------------------------------
  async function sha256Hex(file) {
    const digest = await crypto.subtle.digest("SHA-256", await file.arrayBuffer());
    return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, "0")).join("");
  }

  const isoSeconds = (ms) => new Date(ms).toISOString().replace(/\.\d{3}Z$/, "Z");

  function filePicker() {
    const input = el("input", { type: "file", accept: ".csv,.parquet", multiple: "true" });
    const status = el("span", { class: "hint" });
    const share = el(
      "button",
      {
        class: "primary",
        onclick: () =>
          attempt(async () => {
            const picked = [...input.files]
              .filter((f) => /\.(csv|parquet)$/i.test(f.name))
              .sort((x, y) => (x.name < y.name ? -1 : x.name > y.name ? 1 : 0));
            if (!picked.length) return toast("Pick a .csv or .parquet file.", true);
            share.disabled = true;
            try {
              status.textContent = "Computing SHA-256…";
              const manifest = [];
              for (const f of picked.slice(0, MAX_FILES)) {
                const sha = await sha256Hex(f);
                S.files.set(fileKey(f.name, sha), f);
                manifest.push({ name: f.name, size_bytes: f.size, modified_at: isoSeconds(f.lastModified), sha256: sha });
              }
              // Picked again for an upload the page could no longer perform: just upload it.
              if (S.missing && S.files.has(S.missing.key)) {
                const action = S.view && S.view.client_action;
                S.missing = null;
                S.uploadNote = "";
                status.textContent = "";
                if (action) {
                  S.uploads.delete(action.action_id);
                  upload(action);
                }
                return;
              }
              const v = await api.post(`/agent/sessions/${S.sid}/local-files`, { files: manifest });
              status.textContent = picked.length > MAX_FILES ? `Shared the first ${MAX_FILES} of ${picked.length} files.` : `Shared ${manifest.length} file(s).`;
              apply(v);
              restart();
            } finally {
              share.disabled = false;
            }
          }),
      },
      "Share file details"
    );
    return el(
      "div",
      { class: "as-picker" },
      el("div", { class: "row" }, input, share, status),
      el("p", { class: "hint" }, "Only each file's name, size, modified time and SHA-256 are sent now. The file itself is uploaded only after you approve the upload.")
    );
  }

  function renderFiles(v) {
    const kids = [];
    if (v.datasets && v.datasets.length) {
      kids.push(
        label("Datasets in this session"),
        el(
          "table",
          {},
          el("thead", {}, el("tr", {}, el("th", {}, "File"), el("th", {}, "Format"), el("th", { class: "num" }, "Bytes"), el("th", { class: "num" }, "Rows"), el("th", { class: "num" }, "Columns"))),
          el(
            "tbody",
            {},
            ...v.datasets.map((d) =>
              el("tr", {}, el("td", {}, d.filename), el("td", { class: "id" }, d.file_format), el("td", { class: "num" }, cell(d.size_bytes)), el("td", { class: "num" }, cell(d.row_count)), el("td", { class: "num" }, cell(d.column_count)))
            )
          )
        )
      );
    }
    const ca = v.client_action;
    if (ca && ca.status === "ready") kids.push(el("p", { class: "hint" }, "Upload approved. ", S.uploadNote || "Uploading…"));
    else if (S.uploadNote) kids.push(el("p", { class: "hint" }, S.uploadNote));
    if (S.failedAction && ca && ca.action_id === S.failedAction.action_id && ca.status === "ready") {
      kids.push(
        el(
          "button",
          {
            class: "ghost",
            onclick: () => {
              S.uploads.delete(ca.action_id);
              S.failedAction = null;
              S.uploadNote = "";
              upload(ca);
            },
          },
          "Retry upload"
        )
      );
    }
    const key = JSON.stringify([v.datasets, ca, S.uploadNote, Boolean(S.failedAction)]);
    if (S.keys.files === key) return;
    S.keys.files = key;
    S.N.files.replaceChildren(...kids);
  }

  async function upload(ca) {
    if (S.uploads.has(ca.action_id)) return;
    S.uploads.add(ca.action_id);
    const sid = S.sid;
    try {
      const t = await api.get(`/agent/sessions/${sid}/client-action`);
      if (!t || t.action_id !== ca.action_id) {
        S.uploads.delete(ca.action_id);
        return;
      }
      const file = S.files.get(fileKey(t.file_name, t.sha256));
      if (!file) {
        // Parked: the page no longer holds the file (for example after a reload).
        S.missing = { action_id: t.action_id, key: fileKey(t.file_name, t.sha256), name: t.file_name };
        S.uploadNote = `Pick ${t.file_name} again above to upload it.`;
        renderFiles(S.view);
        return;
      }
      if (file.size !== t.size_bytes) throw new Error(`${t.file_name} changed size since it was shared; share it again.`);
      S.uploadNote = `Uploading ${t.file_name}…`;
      renderFiles(S.view);
      let res;
      try {
        res = await fetch(t.upload_url, { method: t.method || "PUT", headers: { ...(t.headers || {}) }, body: file });
      } catch (err) {
        throw new Error(`Upload of ${t.file_name} failed. The bucket may not allow uploads from this page (CORS).`);
      }
      if (!res.ok) throw new Error(`Upload of ${t.file_name} failed (${res.status}).`);
      const v = await api.post(`/agent/sessions/${sid}/client-action/${t.action_id}/complete`);
      S.uploadNote = `Uploaded ${t.file_name}.`;
      if (sid === S.sid && alive()) {
        apply(v);
        restart();
      }
      toast(`Uploaded ${t.file_name}`);
    } catch (err) {
      // Stays in S.uploads so polling does not retry it; the user retries explicitly.
      S.failedAction = ca;
      S.uploadNote = err.message;
      toast(err.message, true);
      if (S.view) renderFiles(S.view);
    }
  }

  // --- audit -----------------------------------------------------------------------------
  function loadAudit() {
    return attempt(async () => {
      const out = await api.get(`/agent/sessions/${S.sid}/audit`);
      const records = out.records || [];
      if (!records.length) return S.N.audit.replaceChildren(el("p", { class: "empty" }, "No audit records."));
      S.N.audit.replaceChildren(
        el(
          "div",
          { class: "as-scroll" },
          el(
            "table",
            {},
            el("thead", {}, el("tr", {}, ...["Seq", "At", "Event", "Actor", "Tool", "Decision", "Result"].map((h) => el("th", {}, h)))),
            el(
              "tbody",
              {},
              ...records.map((r) =>
                el("tr", {}, el("td", { class: "num" }, cell(r.seq)), el("td", { class: "id" }, r.at), el("td", {}, r.event_type), el("td", {}, r.decided_by || r.actor), el("td", {}, r.tool || "—"), el("td", {}, r.decision || "—"), el("td", {}, r.result_status))
              )
            )
          )
        )
      );
    });
  }

  mountRail();
})();
