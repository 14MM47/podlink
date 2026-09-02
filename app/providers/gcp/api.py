"""A thin, audited client for the two Google APIs the provider needs.

Deliberately REST over the raw Compute Engine v1 and Secret Manager v1 endpoints
rather than the google-cloud-* SDKs:

  * every control-plane call goes through the vendored egress_logger client, so
    it is redacted and recorded like all of podlink's other outbound traffic
    (the SDKs would bypass that audit trail);
  * the only dependency is google-auth, for Application Default Credentials —
    no JSON key files, and the access token is registered for redaction the
    moment it is minted;
  * the request bodies are the same JSON the SDKs build, and a fake client is
    one class in the tests.

Errors surface as GcpApiError carrying the HTTP status and, for a failed
long-running operation, the operation's own error code — which is what tells a
transient zone stockout (retry) from a quota or permission failure (stop).
"""
from __future__ import annotations

import base64
import time
from typing import Protocol

from ...vendored import egress_logger

COMPUTE = "https://compute.googleapis.com/compute/v1"
SECRETS = "https://secretmanager.googleapis.com/v1"
CLOUD_PLATFORM_SCOPE = "https://www.googleapis.com/auth/cloud-platform"

# Operation error codes that mean "no capacity right now" — a failed insert
# allocates nothing and bills nothing, so these are safe to retry.
STOCKOUT_CODES = frozenset({
    "ZONE_RESOURCE_POOL_EXHAUSTED",
    "ZONE_RESOURCE_POOL_EXHAUSTED_WITH_DETAILS",
})
# HTTP statuses that are transient at the API layer (rate limit, backend hiccup).
TRANSIENT_HTTP = frozenset({429, 503})


class GcpApiError(RuntimeError):
    """A failed API call or a long-running operation that finished with an error."""

    def __init__(self, message: str, *, http_status: int | None = None, code: str | None = None):
        super().__init__(message)
        self.http_status = http_status
        self.code = code

    @property
    def retryable(self) -> bool:
        """True for a zone stockout or a transient API failure — never for quota,
        permission or a bad request, which retrying would only hide."""
        return self.code in STOCKOUT_CODES or self.http_status in TRANSIENT_HTTP


class TokenSource(Protocol):
    def token(self) -> str: ...


class AdcTokenSource:
    """Application Default Credentials via google-auth, refreshed on demand.

    Imports google.auth lazily so the provider module loads (for preflight, for
    the registry) even before requirements-gcp.txt is installed; the ImportError
    then surfaces where it is actionable.
    """

    def __init__(self) -> None:
        self._creds = None

    def _load(self):
        import google.auth                                   # noqa: PLC0415 — lazy on purpose
        from google.auth.exceptions import DefaultCredentialsError
        try:
            creds, _project = google.auth.default(scopes=[CLOUD_PLATFORM_SCOPE])
        except DefaultCredentialsError as e:
            raise RuntimeError("no Application Default Credentials — run "
                               "`gcloud auth application-default login`") from e
        return creds

    def token(self) -> str:
        if self._creds is None:
            self._creds = self._load()
        if not self._creds.valid:
            from google.auth.transport.requests import Request   # noqa: PLC0415
            self._creds.refresh(Request())                   # token refresh only; no payload of ours
            egress_logger.register_secret(self._creds.token)  # never let the bearer into a log line
        return self._creds.token


class GcpApi:
    """Compute Engine + Secret Manager calls scoped to one project."""

    def __init__(self, project: str, tokens: TokenSource, timeout: float = 30.0) -> None:
        self.project = project
        self._tokens = tokens
        self._timeout = timeout

    # --- transport ---------------------------------------------------------

    def _request(self, method: str, url: str, *, json: dict | None = None,
                 params: dict | None = None, none_on_404: bool = False,
                 timeout: float | None = None) -> dict | None:
        headers = {"Authorization": f"Bearer {self._tokens.token()}"}
        with egress_logger.client(timeout=timeout or self._timeout) as c:   # audited client
            r = c.request(method, url, headers=headers, json=json, params=params)
        if r.status_code == 404 and none_on_404:
            return None
        if r.status_code >= 400:
            raise GcpApiError(_error_text(r), http_status=r.status_code, code=_error_reason(r))
        if r.status_code == 204 or not r.content:
            return {}
        return r.json()

    # --- compute: instances ------------------------------------------------

    def _zone_url(self, zone: str) -> str:
        return f"{COMPUTE}/projects/{self.project}/zones/{zone}"

    def get_instance(self, zone: str, name: str) -> dict | None:
        return self._request("GET", f"{self._zone_url(zone)}/instances/{name}", none_on_404=True)

    def list_instances(self, zone: str) -> list[dict]:
        page = self._request("GET", f"{self._zone_url(zone)}/instances") or {}
        return list(page.get("items") or [])

    def insert_instance(self, zone: str, body: dict) -> dict:
        """Create and WAIT for the operation — an insert that fails with a
        stockout does so here, as a GcpApiError with the operation's code."""
        op = self._request("POST", f"{self._zone_url(zone)}/instances", json=body)
        self.wait_operation(zone, op)
        return self.get_instance(zone, body["name"]) or {"name": body["name"], "zone": zone}

    def delete_instance(self, zone: str, name: str) -> None:
        """Issue the delete; the driver verifies by polling until 404."""
        self._request("DELETE", f"{self._zone_url(zone)}/instances/{name}")

    def stop_instance(self, zone: str, name: str) -> None:
        self._request("POST", f"{self._zone_url(zone)}/instances/{name}/stop")

    def start_instance(self, zone: str, name: str) -> None:
        op = self._request("POST", f"{self._zone_url(zone)}/instances/{name}/start")
        self.wait_operation(zone, op)

    def wait_operation(self, zone: str, op: dict, deadline_s: float = 600.0) -> dict:
        """Block until a zonal operation is DONE, raising on an operation error.

        Uses the server-side `wait` (returns within ~2 min or when done) so a
        create takes one or two round-trips rather than a polling loop.
        """
        name = (op or {}).get("name")
        if not name:                                         # already-complete or malformed
            return op or {}
        end = time.time() + deadline_s
        current = op
        while current.get("status") != "DONE":
            if time.time() > end:
                raise GcpApiError(f"operation {name} did not complete within {deadline_s:.0f}s",
                                  code="OPERATION_TIMEOUT")
            current = self._request("POST", f"{self._zone_url(zone)}/operations/{name}/wait",
                                    timeout=150.0) or {}
        err = (current.get("error") or {}).get("errors") or []
        if err:
            first = err[0]
            raise GcpApiError(first.get("message") or first.get("code") or "operation failed",
                              code=first.get("code"))
        return current

    # --- secret manager ----------------------------------------------------

    def _secret_url(self, name: str) -> str:
        return f"{SECRETS}/projects/{self.project}/secrets/{name}"

    def secret_latest(self, name: str) -> str | None:
        """The latest version's value, or None if the secret/version is absent."""
        data = self._request("GET", f"{self._secret_url(name)}/versions/latest:access", none_on_404=True)
        if not data:
            return None
        payload = (data.get("payload") or {}).get("data") or ""
        return base64.b64decode(payload).decode() if payload else None

    def secret_put(self, name: str, value: str, location: str) -> None:
        """Add a version, creating the secret (replicated ONLY in `location`) if missing."""
        body = {"payload": {"data": base64.b64encode(value.encode()).decode()}}
        try:
            self._request("POST", f"{self._secret_url(name)}:addVersion", json=body)
        except GcpApiError as e:
            if e.http_status != 404:
                raise
            self._request("POST", f"{SECRETS}/projects/{self.project}/secrets",
                          params={"secretId": name},
                          json={"replication": {"userManaged": {"replicas": [{"location": location}]}}})
            self._request("POST", f"{self._secret_url(name)}:addVersion", json=body)


def _error_reason(r) -> str | None:
    try:
        errors = r.json()["error"]["errors"]
        return errors[0].get("reason")
    except Exception:  # noqa: BLE001
        return None


def _error_text(r) -> str:
    try:
        return f"HTTP {r.status_code}: {r.json()['error']['message']}"
    except Exception:  # noqa: BLE001
        return f"HTTP {r.status_code}"
