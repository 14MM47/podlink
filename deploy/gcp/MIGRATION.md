# Standalone now, organization later — the migration checklist

The podlink GCP project starts under **No organization** (a personal account,
no domain yet) and is migrated into a Cloud organization and an Assured
Workloads **UK Data Boundary** folder once the domain and Cloud Identity tenant
exist. **No client data touches the project before this checklist is complete.**
Until then, only the local test corpus is used.

Everything `deploy/gcp/setup.sh` builds and everything the provider refuses to
launch without (`PODLINK_GCP_HARDENING=strict`) exists so that this migration is
an afternoon, not a rebuild. The two things it cannot fix are listed last.

## Before: what the standalone project already has

- [x] All resources in `europe-west2` (config-enforced; `PODLINK_GCP_ALLOWED_REGIONS`)
- [x] `_Default` log sink routed to a regional bucket
- [x] CMEK on disks and the registry, key in-region, 90-day rotation
- [x] Dedicated VM service account; no use of the default compute SA
- [x] Secrets user-managed-replicated in-region only
- [x] The personal account is the **only** consumer principal, as Owner only

## The migration

1. **Domain + Cloud Identity.** Buy the domain; sign up for Cloud Identity Free
   with a new admin identity (`cloud-admin@domain`); verify the domain (TXT);
   sign in to the console as that identity and accept the terms — the
   organization is created at that moment. Enrol 2-step verification. Forward
   the admin address's mail (Cloud Identity has no mailbox).
2. **Org roles for the admin identity:** Organization Administrator (automatic),
   Billing Account Creator, Folder Admin, Project Creator, Organization Policy
   Administrator, Assured Workloads Administrator.
3. **Billing.** Prefer moving the existing billing account into the org
   (`gcloud billing accounts` / console → *Change organization*) over creating a
   new one: quota reviewers weigh the account's spend history, and a fresh
   account is a "young account" again. Verify the current mechanics first.
4. **Grant the admin identity Owner on the project** — from the personal
   account. Do not remove the personal account yet.
5. **Move the project into the org**, as the admin identity:
   `gcloud projects move PROJECT --organization ORG_ID`. Needs Owner on the
   project + Project Creator on the org.
6. **Create the Assured Workloads folder** (console → Compliance → Assured
   Workloads → Create; control package **"UK Data Boundary"**; resource
   location `europe-west2`; say yes to the CMEK key-ring project in-region).
7. **Move the project into the folder** and read the compliance evaluation.
   Expected: clean, because of the "Before" list. Anything flagged is a
   resource created outside the rules — fix it, do not waive it.
8. **Org policies** now apply from the folder. Confirm the ones the provider
   was standing in for: `gcp.resourceLocations`, `compute.vmExternalIpAccess`
   (deny), `compute.requireShieldedVm`, `compute.restrictNonCmekServices`.
   Consider `iam.allowedPolicyMemberDomains` (domain-restricted sharing).
9. **Remove the personal account** from the project's IAM. If step 8 enabled
   domain-restricted sharing, this is mandatory, not tidy.
10. **Local tooling:** `gcloud auth login` and
    `gcloud auth application-default login` as the admin identity; keep the
    personal account as a separate `gcloud config configurations` profile or
    drop it. Re-run `./start.sh --check --profile <gcp profile>`.
11. **Quota.** The grant travels with the project — nothing to redo. (If you
    had chosen a *fresh* production project instead, you would request it
    again here, and wait again.)
12. **Re-run the setup script** — idempotent — so anything the folder's key
    project should own is reconciled, then run a full POD UP / POD DOWN.

## What migration does not fix — accepted, on record

- **`_Required` audit logs** (admin-activity) for a standalone project live in
  a **global** bucket and cannot be relocated or reconfigured for an existing
  project. Only projects *born* under an org with a default storage location
  get a regional `_Required`. Everything written to `_Default` is already
  regional; the admin-activity trail from before the migration is not.
- **History.** Anything run before the move ran under a personal identity in
  a project outside any compliance folder. That is why no client data is used
  before the checklist is complete — there is nothing to explain later.
- **Assured Workloads attestation** covers the project from the date it enters
  the folder, not before.

## Reference

- `gcloud projects move` — Resource Manager, "Migrating projects into an organization"
- Assured Workloads → "Create a new Assured Workloads folder", "Migrate existing projects"
- Cloud Billing → "Migrate a billing account into an organization"
