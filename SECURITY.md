# Security policy

## Reporting a vulnerability

Please report suspected vulnerabilities **privately** — do not open a public issue.

Use GitHub's **[Report a vulnerability](https://github.com/14MM47/podlink/security/advisories/new)**
(Security → Advisories) to open a private advisory. You'll get an acknowledgement,
and a fix or mitigation will be coordinated before any public disclosure.

## Threat model

podlink is a **local, single-user** tool. It binds to `127.0.0.1` only and is not
designed to be exposed on a network. Its defenses target the browser boundary:

- State-changing requests require a per-process token (`X-Podlink-Token`), which
  blocks browser-based CSRF from other origins.
- POD DOWN has a server-side destructive-action guard (won't terminate without a
  Network Volume unless the caller explicitly confirms).
- Secrets are read only by the RunPod driver, never sent to the browser; error
  messages carry the exception *type*, never `str(e)` (which can embed keys); and
  the SDK's env-echoing `create_pod` stdout is captured and discarded.

podlink **cannot** defend against a malicious process already running as your user
— that process can read `~/.config/podlink/` directly. Keep those secret files at
mode `0600` (podlink refuses to read anything more permissive).

## Scope

In scope: the podlink app (`app/`, `pod_control/`, `start.sh`, `run.sh`). Out of
scope: RunPod itself, the models/images you deploy, and issues that require an
already-compromised local user account.
