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

let token = null;                                   // per-process CSRF token (fetched once)
let lastState = null;                               // to reload pods on return to IDLE

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

// Reflect a status snapshot into the UI.
function render(s) {
  badge.textContent = s.state;                      // show the current state
  phaseEl.textContent = s.phase || "";              // show progress text (or blank)
  errorEl.textContent = s.error || "";              // show error (or blank)
  metaEl.textContent = s.pod_id                     // show pod id + proxy when known
    ? `pod ${s.pod_id}${s.proxy_url ? " · " + s.proxy_url : ""}`
    : "";                                           // otherwise blank
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
  downBtn.disabled = true;                          // optimistic; SSE will confirm
  send("/pod/down");                                // ask the server to stop
});
refreshBtn.addEventListener("click", loadPods);     // manual pod-list refresh

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
