# Manifests

Manifests are used to track the current public API of the Arm backend. They are
generated with
`python backends/arm/scripts/public_api_manifest/generate_public_api_manifest.py`.

## Running manifest

There is always one running manifest which has the main purpose of tracking the
API surface inbetween releases.

## Static manifests

At any given time there may be up to two static manifests. These are generated
in conjunction with a release and are used to track the API surface of that
release. The main purpose of these is to make sure backwards compatibility.

A static manifest may never be changed. It belongs to a release and must be kept
as is.

A static manifest should not live longer than 2 releases. It may then be
removed.

# On release

With each release, check that the running manifest is up to date and reflects
the API surface of the release. Then, copy the running manifest to a new static
manifest for the release. This can be done by running
`cp <running_manifest> <static_manifest>`. The new static manifest should be
named according to the release, e.g. `api_manifest_1_3.toml` for release 1.3 and
so on. If there are now more than two static manifests, remove the oldest one in
the same commit.

# API changes

When introducing an API change, the running manifest must be updated to reflect
the change. This is done by running the manifest generation script,
`python backends/arm/scripts/public_api_manifest/generate_public_api_manifest.py`.
This updates the running manifest.

To validate the running manifest directly, run
`python backends/arm/scripts/public_api_manifest/validate_public_api_manifest.py`.

To validate all manifests, use `backends/arm/scripts/pre-push`. This is the
check that must pass before the change is ready to merge.

Manifest validation only checks the API surface and signatures. Workflow-level
backward compatibility is covered separately by the scenario runner described
below.

Running-manifest validation uses exact signature matching. Any intentional API
change must update `api_manifest_running.toml`.

Static-manifest validation uses backward-compatibility matching. The old
release signature must still be callable against the current API. For example,
adding a trailing optional parameter is accepted for static manifests, while
removing a parameter, reordering parameters, or adding a new required
parameter still fails validation.

## Backward-compatibility scenarios

Workflow-level backward compatibility is checked by
`python backends/arm/test/public_api_bc/run_public_api_bc_scenarios.py`.

The runner hardcodes the current canonical public API workflow scripts:

- `backends/arm/test/public_api_bc/test_ethosu_flow.py`
- `backends/arm/test/public_api_bc/test_vgf_fp_flow.py`
- `backends/arm/test/public_api_bc/test_vgf_int_flow.py`

These scripts should be updated continuously to reflect the current public API.
The runner materializes those same paths into a temporary harness and executes
them there with pytest so they import the latest installed
`executorch.backends.arm` package instead of the repository source tree.

The rolling support window is controlled by the `OLDEST_SUPPORTED_REF` constant
in `backends/arm/test/public_api_bc/run_public_api_bc_scenarios.py`:

- If `OLDEST_SUPPORTED_REF` is empty, the runner uses the current workspace.
  This is the bootstrap mode until a release contains the scenario scripts.
- Once a release contains the scripts, the release epic should update
  `OLDEST_SUPPORTED_REF` to the oldest still-supported release ref.
- At that point the runner uses `git show <ref>:<path>` to fetch the old
  release's scripts and run them against the latest code.

When an old release falls out of the support window, update
`OLDEST_SUPPORTED_REF` to the next newer supported release. That is how the
backward-compatibility window rolls forward.
Reasons for passing validation may include:
- Adding a new API symbol and adding it to the running manifest.
- Removing an API that was marked as deprecated and no longer exists in any
  manifest.
- Deprecated symbols do not break backward compatibility with static
  manifests.
- Deprecating a symbol removes it from the running manifest, but it can only be
  removed fully once it no longer appears in any static manifest.
- Extending a static-manifest signature in a backward-compatible way, such as
  adding a trailing optional parameter.

Reasons for failing validation may include:
- Removing an API symbol without deprecation.
- Changing a running-manifest signature without regenerating the running
  manifest.
- Changing a static-manifest signature in a non-backward-compatible way.
- New API symbol added but not added to the running manifest.
