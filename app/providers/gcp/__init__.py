"""The Google Cloud provider package.

Layout:
  provider.py  — GcpProvider, the class app/driver.py drives
  config.py    — PODLINK_GCP_* settings, validated, re-read on profile switch
  api.py       — audited REST client for Compute Engine + Secret Manager
  bootstrap.py — the cloud-init the VM boots with (no secrets in it)
  launch.sh    — sourced by start.sh before the venv exists

Selected by PODLINK_PROVIDER=gcp. Nothing outside this package may know GCP's
shape; tests/test_siloing.py enforces that.
"""
from __future__ import annotations

from .provider import GcpProvider  # noqa: F401  re-exported

#: The class the registry instantiates for PODLINK_PROVIDER=gcp.
PROVIDER = GcpProvider

__all__ = ["PROVIDER", "GcpProvider"]
