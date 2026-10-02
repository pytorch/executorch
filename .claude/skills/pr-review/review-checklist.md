# PR Review Checklist

This checklist covers areas that CI cannot check. Skip items related to linting, formatting, type checking, and import ordering.

## Code Quality

### Abstractions and Design

- [ ] **Clear abstractions** - State management is explicit; no dynamic attribute setting/getting
- [ ] **No side-channel communication** - If behavior changes based on a hidden flag or dynamically-set attribute, the interface itself should change instead (different function signature, different class, different code path). Side-channel patterns (set a private flag in one place, check it in another via `getattr`) create undocumented behavioral modes
- [ ] **Proper interface, not on/off flags** - A private boolean that switches between two fundamentally different behaviors should be two separate code paths or a proper interface change, not a flag
- [ ] **Interface documentation** - New internal calling conventions, protocols, or contracts between components must have concrete documentation: what the caller provides, what the callee receives, what invariants hold, and cleanup responsibilities. Motivational comments ("this allows X") are not interface documentation
- [ ] **Match existing patterns in the same file** - Before accepting new code in a file, read how similar features are already implemented in that same file. If the file uses class attributes for boolean flags, new boolean flags must use class attributes. If the file uses a specific setter pattern, new setters must use the same pattern
- [ ] **No over-engineering** - Only requested changes are made; no speculative features
- [ ] **No premature abstraction** - Helpers and utilities are only created when reused; three similar lines is better than a one-use helper
- [ ] **No trivial helpers** - Avoid 1-2 LOC helper functions used only once (unless significantly improves readability)

## Infrastructure

### GitHub Actions Storage

**Intent.** Artifacts and caches stored on GitHub are billed to the PyTorch
org. ExecuTorch CI has a cheaper alternative: the `gha-artifacts` S3 bucket.
This check catches PRs that add GitHub-hosted storage nobody needs, or that
put large data on GitHub when S3 would work. It is about cost, not
correctness. Required, bounded storage is not a finding.

#### What counts as GitHub-hosted storage

In scope:
- `actions/upload-artifact` steps.
- The `upload-artifact: <name>` input to the pytorch/test-infra reusable
  workflows (`linux_job_v2.yml`, `macos_job.yml`, `windows_job.yml`) **without**
  `upload-artifact-to-s3: true`. This uploads everything the job writes to
  `${RUNNER_ARTIFACT_DIR}` (plus `dist/*.whl` and `artifacts-to-be-uploaded/`)
  to GitHub, with the repository's default retention.
- `actions/cache` steps and the `cache:` option of `actions/setup-*` actions.

Out of scope, so never flag these:
- S3 uploads: `upload-artifact-to-s3: true`, `seemethere/upload-artifact-s3`,
  test-infra's `upload-artifact-s3` action, or `aws s3 cp` to `gha-artifacts`.
- Upload steps whose stored contents the PR does not change.

#### Constraints to know before suggesting S3

- S3 uploads only work on PyTorch's AWS runners (e.g. `linux.*` and
  `macos-*` runners used through test-infra). Jobs with `runs-on:
  ubuntu-latest` or other GitHub-hosted runners cannot use S3. Do not
  suggest S3 for those jobs.
- test-infra's `download-artifact:` input and `actions/download-artifact`
  only read GitHub storage. If a downstream job reads the artifact that way,
  switching the upload to S3 also means changing the consumer to fetch
  `https://gha-artifacts.s3.amazonaws.com/${{ github.repository }}/${{ github.run_id }}/artifacts/<file>`.
  `android-release-artifacts.yml` follows this pattern. Include the consumer
  change in your suggestion.
- The S3 bucket is publicly readable, so never suggest it for anything
  sensitive.

#### What to flag

For each in-scope line the PR adds or changes:

- [ ] **No consumer** - Search `.github/` for consumers of the artifact name:
  `download-artifact: <name>`, `actions/download-artifact` with `name: <name>`,
  or a matching `pattern:`. If nothing in the repo consumes it and the PR
  description gives no use for it, ask whether it is needed. If it is only for
  occasional manual debugging, suggest `upload-artifact-to-s3: true` (on AWS
  runners) or dropping it.
- [ ] **Large payload on GitHub** - Models (`.pte`, `.ptd`, `.bin`,
  `.safetensors`), checkpoints, wheels, app bundles (`.apk`, `.aar`, `.ipa`,
  `.app`), and whole build trees should go to S3 unless a GitHub-only consumer
  needs them. As a rough threshold, anything over about 100 MB per upload
  deserves this question.
- [ ] **Path broader than the consumer needs** - For example, the job copies
  all of `cmake-out/`, a build directory, or a model cache into
  `${RUNNER_ARTIFACT_DIR}`, but the consumer only uses one binary or report.
  Name the files the consumer actually reads.
- [ ] **Fan-out and duplication** - Look for an upload inside a `matrix:` whose
  payload does not vary with the matrix value, multiple jobs uploading the same
  files, or a new upload on high-frequency triggers (`pull_request`, `push` to
  `main`, short `schedule:` crons) where `workflow_dispatch` or a nightly run
  would do.
- [ ] **Retention** - An explicit `retention-days` longer than the consumer
  needs. An artifact consumed by a later job in the same run needs 1 day.
- [ ] **Cache churn** - `actions/cache` keys that include `github.sha`,
  `github.run_id`, or timestamps, so each entry is written once and never
  restored. Also flag large caches on jobs that rarely run.
- [ ] **Indirect growth** - Non-workflow changes, such as a script or test that
  starts writing models, logs, or build outputs into `${RUNNER_ARTIFACT_DIR}`,
  `artifacts-to-be-uploaded/`, or `dist/`. These grow an existing,
  unchanged GitHub upload.

#### When a finding is confirmed

A storage finding is **confirmed** only when all of these hold:
1. The responsible line is in this PR's diff.
2. You searched `.github/` for consumers and found none, or found one that
   needs only part of the payload.
3. The PR description, nearby comments, and the workflow give no stated use
   (release, HUD, debugging, another repo).
4. A cheaper alternative exists under the constraints above.

If any condition is unclear, for example because the consumer might be
outside this repo, write the finding as a **question to the author** under
Infrastructure. Do not count it as confirmed, and do not request changes for
it alone. Not being able to find a consumer is not proof that none exists.

Confirmed waste is a must-fix Infrastructure finding and requires
**Request Changes**. Write it up as follows:
- cite the line;
- say what is stored;
- estimate the footprint (size per upload × matrix or job fan-out × run
  frequency × retention), stating your assumptions;
- give the concrete fix: set `upload-artifact-to-s3: true` (plus the consumer
  change if one is needed), narrow the path, deduplicate, shorten retention,
  or remove the upload.

If storage is necessary and bounded, do not mention this check in the review.

## Testing

### Test Existence

- [ ] **Tests exist** - New functionality has corresponding tests
- [ ] **Regression tests for bug fixes** - Bug fixes must include a test that reproduces the bug before the fix
- [ ] **Tests are in the right place** - Tests should be added to an existing test file next to other related tests
- [ ] **New test file is rare** - New test file should only be added when new major features are added

### Test Quality

- [ ] **Edge cases covered** - Tests include boundary conditions, empty inputs, error cases
- [ ] **Error conditions tested** - Expected exceptions are tested with `assertRaisesRegex`, not bare `assertRaises`. `assertRaisesRegex` verifies both the exception type and message, catching cases where the right exception is raised for the wrong reason. Bare `assertRaises` should be flagged — always require a message pattern match
- [ ] **No duplicated test logic** - Similar tests share a private helper method called from individual tests with different configs
- [ ] **Prefer xfail over skip** - PR disables a test on a platform/config with `skip` (e.g. `@skipIf`, `@unittest.skip`, `self.skipTest`) when the test merely fails rather than crashing. Prefer expected-failure (`@unittest.expectedFailure`) instead. A skip silently hides the test forever — once the underlying bug is fixed or the platform gains support, the test stays disabled and the new coverage is lost. An xfail flips to a hard failure the moment the test starts passing, forcing the author to remove the marker and re-enable the test. Only accept a `skip` when the test would hard-crash the process (segfault, fatal abort that takes down the whole test binary), hang, or is genuinely flaky (non-deterministic pass/fail); in those cases the author should say so explicitly. A plain deterministic assertion failure or unsupported-op error is always an xfail, never a skip
- [ ] **Use weakref for lifetime testing** - PR uses `sys.getrefcount()` to test whether objects are kept alive. Use `weakref.ref()` instead — create a weak reference, delete the strong references, then check if the weakref is dead (`wr() is None`). `sys.getrefcount` is a CPython implementation detail that varies across versions and is fragile

## Security

### CI/CD and Workflow Security

When reviewing changes to workflows, build scripts, or CI configuration:

- [ ] **No secrets exposed to untrusted code** - Secrets must not be passed to jobs that run fork-controlled code or run on non-ephemeral runners; secrets there can be compromised via reverse shell attacks
- [ ] **Ephemeral runners for sensitive jobs** - Binary builds, uploads, and merge actions must run on ephemeral runners only
- [ ] **No cache-dependent binaries in sensitive contexts** - sccache-backed builds are susceptible to cache corruption; these artifacts should not access sensitive info or be published for general use
- [ ] **Protected branch rules respected** - Changes to merge rules, release workflows, or deployment environments require extra scrutiny
- [ ] **Immutable artifact references** - Docker images use immutable tags; no overwriting of published artifacts

## Thread Safety & Concurrency

### Python Threading

- [ ] **No unprotected shared mutable state** - Shared data structures accessed from multiple threads are protected by locks or are inherently thread-safe
- [ ] **Lock ordering** - When multiple locks are acquired, ordering is consistent to avoid deadlocks
- [ ] **No GIL-reliant correctness** - Code that mutates shared state should not rely on the GIL for thread safety, since the GIL may not be present in free-threaded builds

### C++ Threading

- [ ] **No data races** - Shared mutable state is protected by mutexes or uses atomics with appropriate memory ordering
- [ ] **RAII lock guards** - Prefer `std::lock_guard` or `std::unique_lock` over manual `lock()`/`unlock()` to ensure exception-safe unlocking
- [ ] **No lock-order inversions** - When acquiring multiple locks, a consistent global ordering is followed
- [ ] **Correct atomic memory ordering** - `std::memory_order_relaxed` is only used when ordering with other operations is genuinely unnecessary; default to `seq_cst` or use `acquire`/`release` pairs
