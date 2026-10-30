# Releasing the Arduino library

The published library at
[meta-pytorch/executorch-arduino](https://github.com/meta-pytorch/executorch-arduino)
is generated from this directory. Everything under `src/`, `examples/` and
`extras/` comes out of `build_arduino_library.sh` and is never hand-edited —
make the change here, then regenerate.

Library Manager indexes new git tags hourly, so pushing a tag publishes the
release.

## Versions

The generator fills `library.properties.in` from the root `version.txt`. The
publishing repository's **Sync from ExecuTorch** workflow uses that version by
default. Generating from ExecuTorch `v1.5.1`, for example, produces
`version=1.5.1`, aligning the Arduino package with the runtime it bundles.

Syncs accept a branch, tag or SHA and default to `main`. A development checkout
can already carry the next version. Keep `executorch_pin.txt` and
`extras/PROVENANCE.txt`: they record the exact source commit and ExecuTorch
version independently of the Arduino package version.

For a separate Arduino package release, maintainers can set the workflow's
optional `library_version` input to an explicit `X.Y.Z` version. Numbering
Arduino-only releases remains a maintainer decision. Choose an unused package
version before publishing. Syncing opens a PR; pushing a tag publishes it.

### Moving from 0.2.0

Leave the published `0.x` releases intact. The next release can jump directly
to `1.5.1` after regenerating and validating against ExecuTorch `v1.5.1`.
Changing the old package's version alone does not update its runtime or models.

First update the publishing repository's **Sync from ExecuTorch** workflow to
derive the default version from `version.txt` and remove its automatic bump.
The workflow also handles older ExecuTorch tags whose generator still writes
`0.x`, so this migration does not require changing an existing upstream tag:

```bash
gh workflow run sync.yml --repo meta-pytorch/executorch-arduino \
  -f executorch_ref=v1.5.1
```

The workflow builds, verifies model schemas and opens a PR. Review its CI and
test on a board before merging and tagging. The manual procedure below uses a
checkout containing the version-aware generator. Use the sync workflow for
older tags such as `v1.5.1` or an explicitly selected package version.

## 1. Make your changes

Edit anything under `examples/arduino/` in this repository and land it as a
normal pull request.

## 2. Select the ExecuTorch sources

```bash
# From the ExecuTorch repository root:
git fetch origin <branch-tag-or-sha>
git checkout --detach FETCH_HEAD
examples/arduino/build_arduino_library.sh --version
```

Use a clean checkout. The printed version comes from that checkout's
`version.txt`; confirm it is the package version you intend to publish.

## 3. Build

The exporter must be the same ExecuTorch as the runtime, or the shipped models
will not match the library:

```bash
./install_executorch.sh                      # from the repository root
cd examples/arduino && ./build_arduino_library.sh
```

## 4. Verify

```bash
# every example compiles -- static link mode is required
for s in arduino_lib/ExecuTorch/examples/*/; do
  arduino-cli compile --fqbn arduino:zephyr:unoq:link_mode=static "$s"
done

# shipped models match the library that will run them
python verify_models.py arduino_lib/ExecuTorch

# Library Manager's own rules ("update", not "submit" -- the name is indexed)
cd arduino_lib/ExecuTorch
arduino-lint --project-type library --library-manager update --compliance strict
cd -
```

Then flash an Uno Q and run all three examples. Compiling is not enough on its
own; failures here reach the board before they reach the compiler.

## 5. Copy into the published repository

Delete the generated paths first — a plain `cp` leaves behind files the build no
longer emits:

```bash
cd <executorch-arduino>
git rm -r --quiet src examples extras library.properties executorch_pin.txt
cp -r <executorch>/examples/arduino/arduino_lib/ExecuTorch/. .
git add -A
```

`README.md`, `CHANGELOG.md`, `LICENSE` and `.github/` belong to that repository
and are not generated.

## 6. Check what you are about to publish

```bash
grep ^version= library.properties               # the intended Arduino package version
cat executorch_pin.txt                          # matches: git -C <executorch> rev-parse HEAD
ls src/executorch/runtime/platform/default/     # exactly one .cpp
git status --short | grep '^D '                 # deletions are expected, not surprising ones
```

## 7. Open a release pull request

The repository requires pull requests, so push a branch:

```bash
git push origin main:release-<version>
```

Open it against `main` and merge.

## 8. Tag the merged commit

Fetch the merged release commit from remote `main` and tag that exact commit.
Set `ARDUINO_VERSION` to the intended package version; the check below verifies
it before publishing:

```bash
(
  set -e
  ARDUINO_VERSION='<version>'
  git fetch origin main
  RELEASE_COMMIT=$(git rev-parse FETCH_HEAD)
  test "$(git show "${RELEASE_COMMIT}:library.properties" | sed -n 's/^version=//p')" = "$ARDUINO_VERSION"
  git tag "v${ARDUINO_VERSION}" "$RELEASE_COMMIT"
  git push origin "v${ARDUINO_VERSION}"
)
```

Published tags are immutable. The tag must match the version in
`library.properties`, and that package version must not already be published.

## 9. Write the release notes

Draft a release against the tag at
`https://github.com/meta-pytorch/executorch-arduino/releases/new?tag=v<version>`.

Arduino library releases are short. Use GitHub's **Generate release notes** for
the `What's Changed` list, then write a few lines above it:

```markdown
<One or two sentences: what this release is for.>

**Upgrading:** <anything a user must change -- an include that moved, an arena
that needs raising, a board core version. This goes first, not last.>

## Highlights
* <User-visible change, naming the API or #define involved>
* <Board or core support, by marketing name>
* <Footprint change, if measured>

## What's Changed
<generated>

**Full Changelog**: .../compare/v<previous>...v<version>
```

Title the release with what a user cares about, for example
`v1.5.1 — Arduino library for ExecuTorch 1.5.1`.

Add a matching `CHANGELOG.md` entry in that repository's existing format.

## 10. Confirm it indexed

Indexing takes an hour or two:

```bash
arduino-cli lib update-index
arduino-cli lib search ExecuTorch                # should list the new version
```

Rejections are silent. If the version has not appeared after a few hours:

```
http://downloads.arduino.cc/libraries/logs/github.com/meta-pytorch/executorch-arduino/
```

## Record what you tested against

Board core versions change what the library needs — an arena size that worked on
one core has failed on the next. Note the core version in the release notes, and
keep the pin in CI matching it.
