# Buck vs CMake Parity — QNN backend

This guide reproduces the `test-qnn-buck-build-linux` CI check locally and iteratively fixes it. OSS contributors typically work with CMake. Internal CI also exercises a Buck build of `//backends/qualcomm/...`. The two systems can drift — a new include, source file, or dep that CMake handles via global include paths can break Buck's strict per-target header visibility.

Audience: an OSS contributor (likely Qualcomm-side) editing files under `backends/qualcomm/`. All paths in this guide are repo-root-relative (no `fbcode/` or `xplat/` prefix).

## When to run

There are two entry points — both use the exact same workflow below:

**Pre-PR (proactive):** any of the following just touched `backends/qualcomm/`:

- a new `#include` in a `.cpp` or `.h`
- a new `.cpp` file (especially under `runtime/backends/**`, `aot/**`, `aot/wrappers/**`)
- a change to `targets.bzl`, `BUCK`, `TARGETS`, or `CMakeLists.txt`
- preparing to push a PR

**Post-PR (fix red CI):** the GitHub check `test-qnn-buck-build-linux` is failing on the contributor's PR and they want to fix it locally before pushing again. Pull the failing branch (`git checkout <branch>`), then run this flow. The same buck command, same recipes, same iteration loop — the only difference is the contributor already has a known-failing baseline.

## How to invoke

This workflow is reachable in three ways:

```
/qualcomm buck-fix       # explicit trigger — full iterative fix loop (default)
/qualcomm buck-cmake     # synonym
/qualcomm buck-parity    # synonym
/qualcomm <any natural-language request mentioning buck CI or drift>   # routed by SKILL.md
```

To skip the auto-fix loop and only diagnose, append `check` or `diagnose`:

```
/qualcomm buck-fix check
/qualcomm buck diagnose
```

In `check`/`diagnose` mode, run pre-flight 0 + 1, run buck once, report the failure (or success) verbatim, and stop without applying any recipe.

For all other invocations, run the full iterative loop documented below.

## Pre-flight 0 — buck2 must be on $PATH

`./install_executorch.sh` does **not** install buck2. Check:

```bash
command -v buck2 || echo "MISSING"
```

If missing, install the same pinned version OSS CI uses. The pin lives at `.ci/docker/ci_commit_pins/buck2.txt`. Read it, then download the matching release from `https://github.com/facebook/buck2/releases`. Pattern (Linux x86_64; adjust the asset name for macOS / arm64):

```bash
BUCK2_VERSION=$(cat .ci/docker/ci_commit_pins/buck2.txt)
ASSET=buck2-x86_64-unknown-linux-gnu.zst   # or buck2-aarch64-apple-darwin.zst on macOS
wget -q "https://github.com/facebook/buck2/releases/download/${BUCK2_VERSION}/${ASSET}"
zstd -d "${ASSET}" -o buck2
chmod +x buck2
sudo mv buck2 /usr/local/bin/   # any directory on $PATH
```

Reference: `.ci/docker/common/install_buck.sh` (the script CI itself uses). If the contributor declines to install, halt — they will only catch this drift when CI fails after pushing.

## Pre-flight 1 — QNN_SDK_ROOT and `.buckconfig`

The Buck build reads `[qualcomm] qnn_sdk_root` from `.buckconfig`. The OSS `.buckconfig` typically resolves it from `${QNN_SDK_ROOT}`. Verify:

```bash
echo "${QNN_SDK_ROOT:?set me to your QNN SDK install root}"
test -d "${QNN_SDK_ROOT}/include/QNN" || echo "QNN_SDK_ROOT does not look right"
```

If `QNN_SDK_ROOT` isn't set, the existing `/qualcomm` skill (see this same directory's `SKILL.md`) covers SDK setup.

## The CI command we replicate

The OSS CI job `test-qnn-buck-build-linux` runs (from `.github/workflows/pull.yml`):

```bash
buck2 build //backends/qualcomm/...
```

The skill runs exactly this. If it succeeds, the CI signal will pass. If it fails, the iteration loop below applies a recipe-driven fix and re-runs.

## Iteration loop

```
loop:
  run: buck2 build //backends/qualcomm/...
  if green:                                  → goto cmake-recheck
  if iterations >= 3:                        → halt and ask the user
  parse first error line / span
  match against the recipes table below
  if recipe is "auto":                       apply the fix silently
  if recipe is "confirm":                    print the proposed edit (file, diff, rationale), wait for user confirmation
  if no recipe matches:                      halt, print the error, ask the user
  re-run buck
cmake-recheck:
  run: ./backends/qualcomm/scripts/build.sh --skip_linux_android   # x86_64 cmake re-verification
  on failure: report the cmake delta — the buck-side fix probably broke parity in the other direction
```

Hard cap: 3 automatic iterations. On the 4th attempt, stop and surface the full state to the user — which recipes were tried, which file changes are pending, the latest buck error — and ask whether to keep iterating, try a different recipe, or hand off.

Output policy: **quiet during the loop**, single summary at the end. Each iteration only logs `iter N: <recipe>` so the contributor can scroll the final transcript. Errors and proposed edits are printed only at confirm-points and on final report.

## Recipes

These cover ~80% of OSS-contributor drift. Each recipe lists: the buck-side error pattern to match, the auto-vs-confirm classification, and the fix.

### R1. `fatal error: 'X.h' file not found` (strict-header-visibility)

Symptom — the Buck preprocessor can't see a header that CMake's global include paths reach.

Decision tree:

1. **Find every use of symbols from `X.h` in the consuming source file**. If every use is inside a single `#ifdef <PLATFORM>` block (typically `#ifdef __hexagon__`), and the include itself is unconditional, the cheapest fix is to **gate the include** in the same `#ifdef`. (auto)

   ```cpp
   #ifdef __hexagon__
   #include "X.h"
   #endif
   ```

2. **If the only symbol used from `X.h` is a single small constant (a `#define <NAME> <number>` or similar)**, prefer **inlining the constant as a local `constexpr`** at the use site, then drop the include entirely. (auto, when use site is single)

   ```cpp
   #ifdef __hexagon__
       constexpr size_t kFooAlignment = 64;   // was: #define FOO_ALIGNMENT 64 in X.h
       use(kFooAlignment, …);
   #endif
   ```

   Why: removes the cross-target dependency entirely, follows fbcode-cpp.md's "constexpr everywhere possible". If the macro in `X.h` becomes unused after this, leave a trailing comment for the original author to clean up — don't unilaterally edit a header you don't own in this PR.

3. **If `X.h` exports symbols truly needed unconditionally** in the consumer, a Buck dep edge is required. Find the target that exports `X.h` (look at the `exported_headers` of nearby `targets.bzl` files). Then **check for a cycle before adding the dep** (see R4 — many `runtime` ↔ `aot/wrappers` edges already exist). (confirm)

### R2. New `.cpp` file, undefined-symbol or missing-rule error

Symptom — a freshly added `.cpp` is invisible to Buck because the consuming target's source list doesn't pick it up. CMake handles this because it lists files explicitly in `CMakeLists.txt`; Buck typically uses globs in `targets.bzl`.

Decision tree:

1. **Identify the directory of the new file.** Read the relevant `targets.bzl` (e.g. `backends/qualcomm/runtime/targets.bzl`) and look at its `glob([...])` patterns.
2. **If the directory pattern already covers the file** (e.g. file is `runtime/backends/lpai/foo.cpp` and the glob has `"backends/lpai/*.cpp"`), nothing to do — the failure is something else, look again.
3. **If the directory is new** (e.g. `runtime/backends/lpai/target/`), extend the glob: add the new pattern to the appropriate target's `srcs`. (confirm — glob extension changes target surface)
4. **If the file lives in a directory that has its own `BUCK` / `TARGETS`**, add it to that target's `srcs` list explicitly.

Special case: `runtime/backends/direct_mode/` — see R5.

### R3. Undefined reference at link time

Symptom — Buck successfully compiles each translation unit, then fails at link with `undefined reference to <symbol>`.

Most common cause: a target that uses a symbol declared in another target now lacks the dep edge. The lookup pattern:

1. Find the symbol's definition (its `.cpp` file).
2. Find which Buck target's `srcs` glob covers that `.cpp`.
3. Add that target as a dep of the failing target. (confirm — dep edges affect the build graph)
4. **Before adding**, check for a cycle (R4).

### R4. Cycle detected

Symptom — Buck refuses to add an edge because it would create a cycle in the target graph. In `backends/qualcomm/`, the most common trap is `runtime/targets.bzl` already lists `aot/wrappers:wrappers` as a dep, so adding `runtime:runtime` to wrappers' deps cycles.

Do NOT auto-resolve. Halt and surface the cycle to the user. Three known-good escapes:

- **Factor a header-only target.** Move the header in question into a small `cxx_library` with `header_only = True` (or equivalent), and have both sides depend on it.
- **Inline the symbol.** If the consumer only needs one constant or trivial helper, inline it (see R1.2).
- **Conditional gate.** If the symbol use is platform-conditional, gate the consuming code path so the dep edge isn't needed in linux/Buck builds.

### R5. `runtime/backends/direct_mode/` — Buck has no target there, by design

Symptom — a contributor added a file under `runtime/backends/direct_mode/` and is trying to wire it into Buck.

`direct_mode/` is **CMake-only**, gated on `CMAKE_SYSTEM_PROCESSOR MATCHES Hexagon` in `backends/qualcomm/CMakeLists.txt`. Buck has no Hexagon platform configured for this backend, so `runtime/targets.bzl` deliberately does not glob `direct_mode/`.

Action: **do not add a Buck target for `direct_mode/`**. If the contributor's CMake change works, that's all the OSS build needs. The `test-qnn-buck-build-linux` job won't compile this directory regardless. Tell the user this and skip.

### R6. Adjacent-but-non-Buck error: missing `__init__.py`, missing schema codegen, etc.

If the buck error mentions a file outside `backends/qualcomm/` (e.g. a third-party header path, a flatbuffer codegen target, an `executorch/runtime/...` header), the drift is upstream of this skill's scope. Surface the error to the user with a one-line hint ("this looks unrelated to QNN backend parity — it's likely an `executorch/runtime` or `third-party` issue") and stop. Do not attempt to repair files outside `backends/qualcomm/`.

## On final green

After buck goes green, **re-run cmake** to confirm we didn't break parity in the other direction:

```bash
./backends/qualcomm/scripts/build.sh --skip_linux_android
```

If cmake still passes: print the final summary — files modified, recipes applied, and a one-line hint suggesting the contributor mention this in their PR description ("Buck-vs-CMake parity verified locally with `buck2 build //backends/qualcomm/...`").

If cmake fails: the buck-side fix introduced a parity regression. Surface the cmake error to the user and ask whether to revert.

## On halt (4th iteration or unmatched recipe)

Print:

1. The final unfixed buck error (the first ~30 lines of the actual compiler/linker error, not the buck-tool noise).
2. The recipes that were tried, which iteration applied each, and the resulting error after each.
3. The current `git status` / pending edits.
4. A suggestion: "Open a comment on your PR mentioning the QNN reviewers, link this transcript, and ask for help. The internal `test-qnn-buck-build-linux` reviewers will know the buck graph."

## Out of scope

- Authoring brand-new Buck targets (only minor edits to existing `targets.bzl`/`BUCK`/`TARGETS`).
- Fixes outside `backends/qualcomm/`.
- Hexagon Buck platform setup — there is no Hexagon Buck target in this backend today (R5).
- Generic backend coverage. This guide is QNN-only by design; the same patterns generalize to other backends but recipes would differ.
