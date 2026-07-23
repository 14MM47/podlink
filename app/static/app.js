// podlink frontend: render the two buttons from server-pushed state, and POST
// up/down. All enablement policy lives on the server; we only reflect it.

const upBtn = document.getElementById("up");       // the POD UP button element
const downBtn = document.getElementById("down");   // the POD DOWN button element
const badge = document.getElementById("badge");    // the state pill (IDLE/RUNNING/…)
const phaseEl = document.getElementById("phase");  // the progress line
const errorEl = document.getElementById("error");  // the error line
const metaEl = document.getElementById("meta");    // pod id / proxy url line
const podSelect = document.getElementById("podSelect");  // target-pod dropdown
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
const feedEl = document.getElementById("feed");          // streamed status event feed

let token = null;                                   // per-process CSRF token (fetched once)
let lastState = null;                               // to reload pods on return to IDLE
let lastSnap = null;                                // most recent status snapshot (for the Down guard)

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
    el.dataset.status = (services && services[name]) || "unknown";
  }
}

// Render the streamed status event feed (oldest→newest), auto-scrolled to bottom.
// Always visible; shows a muted placeholder until the first event arrives.
function renderFeed(events) {
  if (!events || !events.length) {
    feedEl.innerHTML = '<div class="feed-row muted"><span class="msg">waiting for activity…</span></div>';
    return;
  }
  feedEl.innerHTML = events.map(
    (e) => `<div class="feed-row"><span class="ts">${fmtClock(e.t)}</span><span class="msg">${escapeHtml(e.msg)}</span></div>`
  ).join("");
  feedEl.scrollTop = feedEl.scrollHeight;  // keep the latest line in view
}

// Minimal HTML-escape so a phase/message string can't inject markup.
function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

// Reflect a status snapshot into the UI.
function render(s) {
  lastSnap = s;                                     // remember it for the POD DOWN guard
  // Warn when no Network Volume is set: POD DOWN (a terminate) destroys weights.
  // `=== false` so an older frame without the field never flashes the banner.
  volwarn.classList.toggle("show", s.network_volume_configured === false);
  badge.textContent = s.state;                      // show the current state
  phaseEl.textContent = s.phase || "";              // show progress text (or blank)
  errorEl.textContent = s.error || "";              // show error (or blank)
  // Show the pod id, and — for our three-service stack (proxy_url set on the
  // Auto path) — the three ragline URLs, which are just pod-id + fixed ports.
  if (s.pod_id) {
    let m = `pod ${s.pod_id}`;
    if (s.proxy_url) {
      const base = (port) => `https://${s.pod_id}-${port}.proxy.runpod.net`;
      m += ` · llm ${base(8000)}/v1 · embed ${base(8080)}/v1 · rerank ${base(8081)}`;
    }
    metaEl.textContent = m;
  } else {
    metaEl.textContent = "";                        // no pod -> blank
  }
  // Cost meter — live while a pod is up; a muted placeholder when idle (the line
  // is always present, per the "indicators from startup" rule).
  if (s.uptime_s != null) {
    let c = `up ${fmtDuration(s.uptime_s)}`;
    if (s.session_cost_usd != null) c += ` · <b>$${s.session_cost_usd.toFixed(2)}</b>`;
    if (s.cost_per_hr != null) c += ` ($${Number(s.cost_per_hr).toFixed(2)}/hr)`;
    costEl.innerHTML = c;
  } else {
    costEl.innerHTML = '<span class="muted">— no pod running —</span>';
  }
  // Auto-terminate row — always shown: a live countdown + Keep alive when armed,
  // a muted "off" when not.
  if (s.auto_terminate_in_s != null) {
    autotermText.textContent = `auto-terminate in ${fmtDuration(s.auto_terminate_in_s)}`;
    autotermText.classList.remove("muted");
    keepaliveBtn.style.display = "";
    keepaliveBtn.disabled = false;                  // re-enable each armed frame
  } else {
    autotermText.textContent = "auto-terminate: off";
    autotermText.classList.add("muted");
    keepaliveBtn.style.display = "none";
  }
  // Per-service health tiles and the streamed status event feed.
  renderTiles(s.services);
  renderFeed(s.events);
  // Server is the single source of truth for enablement.
  upBtn.disabled = !s.up_enabled;                   // grey Up unless server allows it
  downBtn.disabled = !s.down_enabled;               // grey Down unless server allows it
  // The target can only be changed before a run starts (i.e. when Up is live).
  podSelect.disabled = !s.up_enabled;
  refreshBtn.disabled = !s.up_enabled;
  // Refresh the pod list whenever we settle back into a selectable state.
  if (s.state !== lastState) {
    if (s.up_enabled) loadPods();                   // IDLE or ERROR -> statuses may have changed
    lastState = s.state;
  }
}

// POST a control action with the token header (+ optional JSON body).
async function send(path, body) {
  if (!token) await loadToken();                    // ensure we have a token first
  const headers = { "X-Podlink-Token": token };
  if (body) headers["Content-Type"] = "application/json";
  await fetch(path, {
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
      "POD DOWN will TERMINATE the pod and may DESTROY the ~36 GB of downloaded " +
      "model weights — the next POD UP would re-download them.\n\nTerminate anyway?"
    );
    if (!ok) return;                                // aborted; leave the button live
  }
  downBtn.disabled = true;                          // optimistic; SSE will confirm
  send("/pod/down", { confirm: true });            // explicit confirmation for the server-side guard
});
refreshBtn.addEventListener("click", loadPods);     // manual pod-list refresh
keepaliveBtn.addEventListener("click", () => {      // cancel the pending auto-terminate
  keepaliveBtn.disabled = true;                     // optimistic; SSE re-enables on next frame
  send("/pod/keepalive");
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

// Get the token, load the pod list, then start the live status stream.
loadToken().then(() => { loadPods(); connect(); });
