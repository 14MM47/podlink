// podlink frontend: render the two buttons from server-pushed state, and POST
// up/down. All enablement policy lives on the server; we only reflect it.

const upBtn = document.getElementById("up");       // the POD UP button element
const downBtn = document.getElementById("down");   // the POD DOWN button element
const badge = document.getElementById("badge");    // the state pill (IDLE/RUNNING/…)
const phaseEl = document.getElementById("phase");  // the progress line
const errorEl = document.getElementById("error");  // the error line
const metaEl = document.getElementById("meta");    // pod id / proxy url line
const podSelect = document.getElementById("podSelect");  // target-pod dropdown
const profileSelect = document.getElementById("profileSelect");  // stack-profile dropdown
const refreshBtn = document.getElementById("refresh");   // reload-pod-list button
const volwarn = document.getElementById("volwarn");      // "no Network Volume" banner
const costEl = document.getElementById("cost");          // uptime + running cost line
const autotermEl = document.getElementById("autoterm");  // auto-terminate countdown row
const autotermText = document.getElementById("autotermText");  // its text span
const keepaliveBtn = document.getElementById("keepalive");     // cancel auto-terminate
const tilesEl = document.getElementById("tiles");        // per-service health tiles row
const tileEls = {                                        // service -> tile element
  llm: document.getElementById("tile-llm"),
  embedder: document.getElementById("tile-embedder"),
  reranker: document.getElementById("tile-reranker"),
};
const feedEls = {                                        // event feed split by source/function
  lifecycle: document.getElementById("feed-lifecycle"),  // provisioning + timeline
  health: document.getElementById("feed-health"),        // per-service transitions
  system: document.getElementById("feed-system"),        // errors / auto-terminate
};
const volinfoEl = document.getElementById("volinfo");    // network-volume info line
const actionsEl = document.getElementById("actions");    // per-pod action buttons
const copyenvBtn = document.getElementById("copyenv");   // copy the client config
const copiedEl = document.getElementById("copied");      // "copied ✓" flash
const envviewEl = document.getElementById("envview");    // the .env block, viewable
const teststackBtn = document.getElementById("teststack");    // run the stack test
const testresultEl = document.getElementById("testresult");   // its result rows

let token = null;                                   // per-process CSRF token (fetched once)
let lastState = null;                               // to reload pods on return to IDLE
let lastSnap = null;                                // most recent status snapshot (for the Down guard)
let envKey = null;                                  // pod_id:dim the .env block was last fetched for
// Change-detection keys: only rebuild (and repaint) a panel when ITS data changed.
// The cost meter ticks every second, so render() runs every second — without these
// we'd rebuild the frosted-glass log panels every second for nothing.
let mBadge, mPhase, mError, mMeta, mVol, mWarn, mTiles, mTest, mFeeds;

// Fetch the per-process CSRF token once (same-origin; cross-origin JS can't read it).
async function loadToken() {
  const r = await fetch("/config");                 // GET the token endpoint
  token = (await r.json()).token;                   // stash the token string
}

// Fetch the account's pods and (re)populate the dropdown, preserving selection.
async function loadPods() {
  if (!token) await loadToken();                    // need the token to call /pods
  try {
    const r = await fetch("/pods", { headers: { "X-Podlink-Token": token } });
    if (!r.ok) return;                              // leave the Auto-only list on failure
    const { pods } = await r.json();
    const cur = podSelect.value;                    // remember current choice
    podSelect.innerHTML =
      '<option value="">Auto — create/resume the podlink pod</option>';
    for (const p of pods) {                         // one option per account pod
      const o = document.createElement("option");
      o.value = p.id;
      const cost = p.cost_per_hr != null ? ` · $${p.cost_per_hr}/hr` : "";
      o.textContent =
        `${p.name || "(unnamed)"} · ${p.status || "?"} · ${p.gpu || "?"}${cost} · ${p.id}`;
      podSelect.appendChild(o);
    }
    podSelect.value = cur;                          // restore selection if still present
  } catch {}                                        // network hiccup — keep prior list
}

// Fetch the available stack profiles and (re)populate the dropdown, marking the
// active one. Names only — the server never exposes conf contents.
async function loadProfiles() {
  if (!token) await loadToken();                    // need the token to call /profiles
  try {
    const r = await fetch("/profiles", { headers: { "X-Podlink-Token": token } });
    if (!r.ok) return;                              // leave the base-only list on failure
    const { profiles, active } = await r.json();
    profileSelect.innerHTML = '<option value="">base — no profile</option>';
    for (const name of profiles) {                  // one option per conf in profiles/
      const o = document.createElement("option");
      o.value = name;
      o.textContent = `profile: ${name}`;
      profileSelect.appendChild(o);
    }
    profileSelect.value = active || "";             // reflect the server's active profile
  } catch {}                                        // network hiccup — keep prior list
}

// Format a whole number of seconds as M:SS or H:MM:SS.
function fmtDuration(sec) {
  const h = Math.floor(sec / 3600), m = Math.floor((sec % 3600) / 60), s = sec % 60;
  const pad = (n) => String(n).padStart(2, "0");
  return h > 0 ? `${h}:${pad(m)}:${pad(s)}` : `${m}:${pad(s)}`;
}

// Local wall-clock time (HH:MM:SS) for a unix-epoch-seconds event timestamp.
function fmtClock(epochSec) {
  return new Date(epochSec * 1000).toLocaleTimeString();
}

// Render the per-service health tiles. Always visible; grey (unknown) until a pod
// exists, per the "indicators present from startup" rule.
function renderTiles(services) {
  for (const [name, el] of Object.entries(tileEls)) {
    const st = (services && services[name]) || "unknown";
    el.dataset.status = st;
    const stEl = el.querySelector(".st");
    if (stEl) stEl.textContent = st === "unknown" ? "" : st;   // show the word, blank when unknown
  }
}

// Render one feed panel (oldest→newest, auto-scrolled). withDelta adds a per-step
// "+M:SS" delta from the previous row — that's the provisioning timeline.
function renderPanel(el, rows, withDelta) {
  if (!rows.length) {
    el.innerHTML = '<div class="feed-row muted"><span class="msg">— none yet —</span></div>';
    return;
  }
  let prev = null;
  const last = rows.length - 1;
  el.innerHTML = rows.map((e, i) => {
    let dt = "";
    if (withDelta && prev !== null) {
      dt = `<span class="dt">+${fmtDuration(Math.max(0, Math.round(e.t - prev)))}</span>`;
    }
    prev = e.t;
    const cls = i === last ? "feed-row latest" : "feed-row";   // highlight the newest line
    return `<div class="${cls}"><span class="ts">${fmtClock(e.t)}</span>${dt}<span class="msg">${escapeHtml(e.msg)}</span></div>`;
  }).join("");
  el.scrollTop = el.scrollHeight;          // keep the latest line in view
}

// Render the stack-test state: "running…", a per-service pass/fail + latency
// table, or nothing when no test has run for this pod.
function renderTest(s) {
  if (s.test_running) {
    testresultEl.innerHTML = '<span class="muted">running stack test…</span>';
    return;
  }
  const tr = s.test_result;
  if (!tr) { testresultEl.innerHTML = '<span class="muted">— not run —</span>'; return; }
  if (tr.error) {
    testresultEl.innerHTML = `<div class="tr-row" style="color:#ff8a8a">test error: ${escapeHtml(tr.error)}</div>`;
    return;
  }
  const svc = tr.services || {};
  const row = (name) => {
    const r = svc[name] || {};
    const ck = r.ok ? '<span class="ck">✓</span>' : '<span class="ck" style="color:#ff9a9a">✗</span>';
    return `<div class="tr-row">${ck}<span class="svc">${name}</span>` +
           `<span class="ms">${r.latency_ms ?? "?"}ms</span>` +
           `<span class="note">· ${escapeHtml(r.detail || "")}</span></div>`;
  };
  let html = ["llm", "embedder", "reranker"].map(row).join("");
  if (tr.embedding_dim) {
    html += `<div class="tr-row muted"><span class="ck"> </span><span class="svc"></span>` +
            `<span class="note">embedding dimension: ${tr.embedding_dim}</span></div>`;
  }
  testresultEl.innerHTML = html;
}

// Split the event feed into its three source/function panels.
function renderFeeds(events) {
  events = events || [];
  const by = (cat) => events.filter((e) => (e.cat || "lifecycle") === cat);
  renderPanel(feedEls.lifecycle, by("lifecycle"), true);   // timeline deltas here
  renderPanel(feedEls.health, by("health"), false);
  renderPanel(feedEls.system, by("system"), false);
}

// Minimal HTML-escape so a phase/message string can't inject markup.
function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

// Reflect a status snapshot into the UI.
function render(s) {
  lastSnap = s;                                     // remember it for the POD DOWN guard

  // "no Network Volume" banner — only toggle on change.
  const warn = s.network_volume_configured === false;
  if (warn !== mWarn) { volwarn.classList.toggle("show", warn); mWarn = warn; }

  // Badge / phase / error — tiny text, but gated to avoid churn (and animation restarts).
  if (s.state !== mBadge) { badge.textContent = s.state; badge.dataset.state = s.state; mBadge = s.state; }
  const ph = s.phase || ""; if (ph !== mPhase) { phaseEl.textContent = ph; mPhase = ph; }
  const er = s.error || ""; if (er !== mError) { errorEl.textContent = er; mError = er; }

  // Pod id + service URLs — rebuild only when the pod changes.
  const metaKey = `${s.pod_id || ""}|${s.proxy_url || ""}`;
  if (metaKey !== mMeta) {
    if (s.pod_id) {
      const pid = escapeHtml(s.pod_id);
      let m = `<div><span class="k">pod</span> <span class="pid">${pid}</span></div>`;
      if (s.proxy_url) {
        const base = (port) => `https://${pid}-${port}.proxy.runpod.net`;
        m += `<div class="urls">` +
             `<span><b>llm</b> ${base(8000)}/v1</span>` +
             `<span><b>embed</b> ${base(8080)}/v1</span>` +
             `<span><b>rerank</b> ${base(8081)}</span></div>`;
      }
      metaEl.innerHTML = m;
    } else {
      metaEl.innerHTML = '<span class="muted">— no pod running —</span>';
    }
    mMeta = metaKey;
  }

  // Cost meter + auto-terminate countdown — these DO tick every second by design;
  // they're single-line text so the per-second update is cheap.
  if (s.uptime_s != null) {
    let c = `up ${fmtDuration(s.uptime_s)}`;
    if (s.session_cost_usd != null) c += ` · <b>$${s.session_cost_usd.toFixed(2)}</b>`;
    if (s.cost_per_hr != null) c += ` ($${Number(s.cost_per_hr).toFixed(2)}/hr)`;
    costEl.innerHTML = c;
  } else {
    costEl.innerHTML = '<span class="muted">— no pod running —</span>';
  }
  if (s.auto_terminate_in_s != null) {
    autotermText.textContent = `auto-terminate in ${fmtDuration(s.auto_terminate_in_s)}`;
    autotermText.classList.remove("muted");
    keepaliveBtn.style.display = "";
    keepaliveBtn.disabled = false;
  } else {
    autotermText.textContent = "auto-terminate: off";
    autotermText.classList.add("muted");
    keepaliveBtn.style.display = "none";
  }

  // Deploy-info lines (profile / model / volume are process constants) —
  // rebuild only when they change.
  const volKey = `${s.active_profile || ""}|${s.llm_model_id || ""}|${s.volume_id || ""}`;
  if (volKey !== mVol) {
    profileSelect.value = s.active_profile || "";   // keep the dropdown in sync with the server
    const profileLine = s.active_profile
      ? `Profile: <b>${escapeHtml(s.active_profile)}</b> · ${escapeHtml(s.llm_model_id || "")}<br>`
      : "";
    const volLine = s.volume_id
      ? `Network Volume: <b>${escapeHtml(s.volume_id)}</b> · persistence <span class="on">ON</span>`
      : 'Network Volume: <span class="muted">none — Data-Volume mode (weights not persisted)</span>';
    volinfoEl.innerHTML = profileLine + volLine;
    mVol = volKey;
  }

  // Per-pod action button states — cheap toggles, every frame.
  const up = !!(s.pod_id && s.proxy_url);
  teststackBtn.disabled = !up || !!s.test_running;
  copyenvBtn.disabled = !up;

  // Outputs rail: auto-populate the client config block (already change-gated by envKey).
  const dim = s.test_result && s.test_result.embedding_dim;
  const key = up ? `${s.pod_id}:${dim || ""}` : null;
  if (key && key !== envKey) { envKey = key; fetchEnv(); }
  else if (!up && envKey !== "idle") {
    envKey = "idle"; copiedEl.style.display = "none";
    envviewEl.innerHTML = '<span class="muted">— start a pod to generate the client config —</span>';
  }

  // Expensive panels (glass, many rows) — rebuild ONLY when their data changed.
  const testKey = JSON.stringify(s.test_result) + "|" + s.test_running;
  if (testKey !== mTest) { renderTest(s); mTest = testKey; }
  const tilesKey = JSON.stringify(s.services || {});
  if (tilesKey !== mTiles) { renderTiles(s.services); mTiles = tilesKey; }
  const feedsKey = JSON.stringify(s.events || []);
  if (feedsKey !== mFeeds) { renderFeeds(s.events); mFeeds = feedsKey; }

  // Enablement flags — cheap, every frame.
  upBtn.disabled = !s.up_enabled;
  downBtn.disabled = !s.down_enabled;
  podSelect.disabled = !s.up_enabled;
  refreshBtn.disabled = !s.up_enabled;
  // Mirrors the server's /profile/select guard: switchable only with no pod at all.
  profileSelect.disabled = !s.up_enabled || !!s.pod_id;
  if (s.state !== lastState) {
    if (s.up_enabled) { loadPods(); loadProfiles(); }  // IDLE or ERROR -> lists may have changed
    lastState = s.state;
  }
}

// Fetch the client config block for the running pod and show it in the outputs rail.
async function fetchEnv() {
  if (!token) await loadToken();
  try {
    const r = await fetch("/pod/env", { headers: { "X-Podlink-Token": token } });
    if (!r.ok) return;                              // 409 before a pod is up — leave the placeholder
    envviewEl.textContent = (await r.json()).env;   // plain text (selectable, copyable)
  } catch { /* transient — a later frame retries */ }
}

// POST a control action with the token header (+ optional JSON body).
async function send(path, body) {
  if (!token) await loadToken();                    // ensure we have a token first
  const headers = { "X-Podlink-Token": token };
  if (body) headers["Content-Type"] = "application/json";
  return await fetch(path, {                        // callers may check .ok; most ignore it
    method: "POST",
    headers,
    body: body ? JSON.stringify(body) : undefined,
  });
}

upBtn.addEventListener("click", () => {             // when POD UP is clicked
  upBtn.disabled = true;          // optimistic; SSE will confirm
  send("/pod/up", { target: podSelect.value || null });  // pass the chosen target (or Auto)
});
downBtn.addEventListener("click", () => {           // when POD DOWN is clicked
  // Destructive-action guard, fail CLOSED: only skip the warning when we
  // positively know a Network Volume is set. Unknown (no snapshot yet, or the
  // field absent) is treated as unsafe so an early click can't slip through.
  const safe = lastSnap && lastSnap.network_volume_configured === true;
  if (!safe) {
    const ok = confirm(
      "No RunPod Network Volume is confirmed configured.\n\n" +
      "POD DOWN will TERMINATE the pod and may DESTROY the downloaded " +
      "model weights — the next POD UP would re-download them.\n\nTerminate anyway?"
    );
    if (!ok) return;                                // aborted; leave the button live
  }
  downBtn.disabled = true;                          // optimistic; SSE will confirm
  send("/pod/down", { confirm: true });            // explicit confirmation for the server-side guard
});
refreshBtn.addEventListener("click", loadPods);     // manual pod-list refresh
profileSelect.addEventListener("change", async () => {  // switch the stack profile
  const chosen = profileSelect.value || null;       // "" = base (no profile)
  profileSelect.disabled = true;                    // optimistic; SSE re-enables
  let r = null;
  try { r = await send("/profile/select", { profile: chosen }); } catch {}
  if (!r || !r.ok) {                                // rejected (409/400) or network error
    profileSelect.value = (lastSnap && lastSnap.active_profile) || "";  // snap back
    profileSelect.disabled = false;                 // no snapshot change will re-enable it
  }
});
keepaliveBtn.addEventListener("click", () => {      // cancel the pending auto-terminate
  keepaliveBtn.disabled = true;                     // optimistic; SSE re-enables on next frame
  send("/pod/keepalive");
});
teststackBtn.addEventListener("click", () => {      // run a real completion+embedding+rerank
  teststackBtn.disabled = true;                     // optimistic; SSE reflects test_running
  send("/pod/test");
});
copyenvBtn.addEventListener("click", async () => {  // copy the shown client config block
  await fetchEnv();                                 // ensure it's current (fills envview)
  const env = envviewEl.textContent || "";
  if (!env || env.trim().startsWith("—")) return;   // still the placeholder — nothing to copy
  try { await navigator.clipboard.writeText(env); } catch { /* clipboard blocked; text is shown to select */ }
  copiedEl.style.display = "inline";
  setTimeout(() => { copiedEl.style.display = "none"; }, 2000);
});

// Live updates via SSE, with a polling fallback if the stream drops.
function connect() {
  const es = new EventSource("/events");            // open the SSE stream
  es.onmessage = (e) => render(JSON.parse(e.data)); // render each pushed snapshot
  es.onerror = () => {                              // stream dropped (e.g. server restart)
    es.close();                                     // close the broken stream
    setTimeout(connect, 2000);                      // retry after 2s
  };
}

// Get the token, load the pod + profile lists, then start the live status stream.
loadToken().then(() => { loadPods(); loadProfiles(); connect(); });
