// podlink frontend: render the two buttons from server-pushed state, and POST
// up/down. All enablement policy lives on the server; we only reflect it.

const upBtn = document.getElementById("up");       // the POD UP button element
const downBtn = document.getElementById("down");   // the POD DOWN button element
const badge = document.getElementById("badge");    // the state pill (IDLE/RUNNING/…)
const phaseEl = document.getElementById("phase");  // the progress line
const errorEl = document.getElementById("error");  // the error line
const metaEl = document.getElementById("meta");    // pod id / proxy url line

let token = null;                                   // per-process CSRF token (fetched once)

// Fetch the per-process CSRF token once (same-origin; cross-origin JS can't read it).
async function loadToken() {
  const r = await fetch("/config");                 // GET the token endpoint
  token = (await r.json()).token;                   // stash the token string
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
}

// POST a control action with the token header; ignore benign 409 races.
async function send(path) {
  if (!token) await loadToken();                    // ensure we have a token first
  await fetch(path, { method: "POST", headers: { "X-Podlink-Token": token } });  // fire it
}

upBtn.addEventListener("click", () => {             // when POD UP is clicked
  upBtn.disabled = true;          // optimistic; SSE will confirm
  send("/pod/up");                                  // ask the server to start
});
downBtn.addEventListener("click", () => {           // when POD DOWN is clicked
  downBtn.disabled = true;                          // optimistic; SSE will confirm
  send("/pod/down");                                // ask the server to stop
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

loadToken().then(connect);                          // get the token, then start streaming
