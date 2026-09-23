"""The provider contract: everything the driver needs a cloud to do.

app/driver.py owns the parts of POD UP / POD DOWN that are the same everywhere —
the state machine, the cost meter, the cancel-aware waits, the three-service
readiness gate, the health tiles, the event feed and the stack test. A Provider
supplies only the cloud-shaped pieces underneath: how an instance is created,
found, described, reached and destroyed.

Two rules keep the seam honest:

  * **Naming.** The driver speaks in podlink's own vocabulary — "instance" for
    the compute unit whatever the cloud calls it, and normalised booleans
    (`is_running`, `is_up`) rather than raw status strings, so the driver never
    branches on a provider's enum spelling.
  * **Cheapness.** `service_urls`, `snapshot_fields`, `stack_config` and
    `persistence_configured` are called on every SSE frame (~1/s). They must be
    pure local computation — no network calls, no blocking.

Instances are passed around as opaque values: the driver only ever hands one
back to the provider that produced it, or reads it through the accessors below.
"""
from __future__ import annotations

from typing import Any, Protocol


class Provider(Protocol):
    """What app/driver.py requires of a cloud. See app/providers/runpod.py."""

    #: Short lowercase id, echoed to the UI (e.g. "runpod").
    name: str

    #: Exception types create_once() may raise for an *expected* create failure.
    #: The driver catches exactly these and asks is_retryable_create_error();
    #: anything else propagates immediately, unretried.
    create_error_types: tuple[type[Exception], ...]

    # --- configuration -----------------------------------------------------

    def reload_config(self) -> None:
        """Re-read every PODLINK_* setting from the environment.

        Called after a profile switch has swapped the process environment. Must
        leave the provider fully consistent or raise, in which case the caller
        restores the previous environment and calls this again.
        """

    def snapshot_fields(self) -> dict:
        """Provider-specific fields merged into every status snapshot.

        Required keys, because the UI renders them generically:
          provider              — self.name
          persistence_configured— bool; False makes POD DOWN destructive
          persistence_id        — the storage id, or None
          persistence_label     — what this cloud calls it ("Network Volume")
          persistence_off_hint  — one line explaining what "off" costs the user
          llm_model_id          — which stack this launch serves
        """

    def preflight(self) -> list[tuple[str, str]]:
        """Cloud-specific launch checks for `start.sh --check`, spending nothing.

        Rows are (level, message) with level in ok / warn / fail / info. Report
        what would stop a launch on THIS cloud — credentials, quota, the image
        registry, the persistence mode. The neutral checks (the stack's own
        bearer + HF token, the venv) belong to app/preflight.py, not here. Any
        "fail" row makes the preflight exit non-zero.
        """

    # --- authentication ----------------------------------------------------

    def authenticate(self) -> None:
        """Load credentials for the calls below. Called before each work batch."""

    # --- inventory ---------------------------------------------------------

    def valid_instance_id(self, instance_id: str) -> bool:
        """True when a client-supplied id is well-formed for this cloud.

        Checked before the id reaches any SDK call, so a malformed or injected
        target cannot be passed through.
        """

    def list_instances(self) -> list[dict]:
        """WHITELISTED rows for the instance selector.

        Each row: id, name, status, gpu, cost_per_hr. Never return a raw
        instance record — those embed env/config that can carry credentials.
        """

    def find_existing(self) -> dict | None:
        """The instance podlink manages (matched by its configured name), or None."""

    def get_instance(self, instance_id: str) -> Any | None:
        """Fetch one instance, or None if it no longer exists. May raise."""

    def instance_id(self, instance: Any) -> str:
        """The id of an instance record returned by this provider."""

    def status_of(self, instance: Any) -> str | None:
        """The cloud's own status string — display only; never branched on."""

    def is_running(self, instance: Any) -> bool:
        """True when the instance is meant to be running (may still be booting)."""

    def is_up(self, instance: Any) -> bool:
        """True when the instance is running AND its container is actually up."""

    def cost_per_hr(self, instance: Any) -> float | None:
        """Hourly rate for the cost meter, or None when the cloud doesn't say."""

    # --- lifecycle ---------------------------------------------------------

    def resume(self, instance: Any) -> None:
        """Bring a stopped instance back. Only used on the adopt-a-selection path."""

    def prepare_create(self, session) -> Any | None:
        """Do the pre-create lookups, returning an opaque context for create_once.

        Must check `session.cancel` between steps and return None the moment it
        is set, and report progress with session.update(phase=…) so a slow
        lookup is visible in the UI.
        """

    def create_once(self, ctx: Any, secrets: dict, attempt: int) -> Any:
        """ONE create attempt. Raises on failure; the driver owns the retry loop.

        `secrets` carries {"bearer": …, "hf": …} — the tokens the container
        needs. How they reach the instance is the provider's business.
        `attempt` is 1-based so a provider can vary placement per attempt
        (zone rotation on stockout) without keeping state in ctx.
        """

    def is_retryable_create_error(self, exc: Exception) -> bool:
        """True when a create failure is transient capacity, safe to retry.

        Quota, permission and bad-spec failures must return False: retrying
        those only hides the real error behind ten minutes of waiting.
        """

    def create_phase(self, ctx: Any, attempt: int, retries: int) -> str:
        """Phase line for one create attempt (shown live in the UI)."""

    def capacity_note(self, attempt: int, retries: int, delay: int) -> str:
        """Event-feed line explaining why an attempt is being retried."""

    def create_exhausted_message(self, retries: int, delay: int) -> str:
        """Error text when every attempt failed — say what the user can do next."""

    def terminate(self, instance_id: str) -> None:
        """Destroy the instance and release its GPU. Verification is the driver's."""

    # --- access lifecycle ---------------------------------------------------

    def ensure_access(self, instance_id: str) -> None:
        """Open the local path to this instance's service ports. Idempotent.

        Some clouds publish the ports themselves (RunPod's HTTPS proxy); others
        require podlink to establish the path — a tunnel process, a port forward
        — before anything can be probed. That work cannot live in
        `service_urls()` (contractually cheap and pure, called every SSE frame)
        nor in `create_once()`, because an ADOPTED instance is never created:
        the driver calls this on both the create and the adopt paths, right
        after the instance comes up and before the first probe.

        Raising here fails the launch, which is correct — an instance nothing
        can reach is not up.
        """

    def access_events(self) -> list[str]:
        """Messages from the access layer since the last call, for the event feed.

        A cloud whose access path is a supervised process (a tunnel) reports
        its transitions — up, dropped, restored, given up — here; the driver
        drains this on every health pass. Clouds with nothing to report return
        an empty list. Must be cheap and must not raise.
        """

    def release_access(self) -> None:
        """Tear down whatever ensure_access opened. Idempotent; must never raise.

        Called on every terminal path — a verified terminate, a failed start, a
        stop that found nothing — so an interrupted launch cannot leave a tunnel
        process behind. Safe to call when nothing was ever opened.
        """

    # --- endpoints + local state -------------------------------------------

    def service_urls(self, instance_id: str) -> dict[str, str]:
        """Base URLs keyed llm / embedder / reranker. Cheap and pure."""

    def stack_config(self) -> dict:
        """{"llm_served_name": …, "embed_model_id": …, "rerank_backend": …} for the
        stack test and preflight. A missing rerank_backend means "tei"."""

    def client_env(self, instance_id: str, embedding_dim: int | None) -> str:
        """The RAG-client .env block for this instance.

        The bearer MUST be left as a placeholder — this string reaches the
        browser, and the real token must not.
        """

    def persistence_configured(self) -> bool:
        """True when storage survives POD DOWN (weights are not destroyed).

        False makes POD DOWN destructive, which the server's 428 guard and the
        UI banner both key off.
        """

    def write_state(self, instance_id: str) -> None:
        """Persist the on-disk record for a fully-up instance (CLI compatibility)."""

    def clear_state(self) -> None:
        """Remove that record after a verified terminate. Best-effort, never fatal."""

    def state_file_instance_id(self) -> str | None:
        """The instance id from the on-disk record — last-resort id source on stop."""
