# Releasing the Arduino library

The published library at
[meta-pytorch/executorch-arduino](https://github.com/meta-pytorch/executorch-arduino)
is generated from this directory. Everything under `src/`, `examples/` and
`extras/` comes out of `build_arduino_library.sh` and is never hand-edited —
make the change here, then regenerate.

Library Manager indexes new git tags hourly, so pushing a tag publishes the
release.

## 1. Make your changes

Edit anything under `examples/arduino/` in this repository and land it as a
normal pull request. Then release from a clean checkout of `main`.

## 2. Bump the version

```bash
cd examples/arduino
./build_arduino_library.sh --bump minor      # major | minor | patch
```

Minor for new public API such as `ETModel.h`, patch for fixes. The version must
differ from every previously published one or Library Manager drops the release.

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
grep ^version= library.properties               # the version you bumped to
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

Tag last. A tag pushed before the release commit is on `main` points at a commit
no branch contains, and the fix is to delete a published tag:

```bash
git fetch origin && git checkout main && git reset --hard origin/main
grep ^version= library.properties               # confirms the merge landed
git tag v<version>
git push origin v<version>
git rev-parse v<version> HEAD                   # both must print the same SHA
```

If a tag already exists on the wrong commit, move it:

```bash
git push origin :refs/tags/v<version>           # delete on the remote
git tag -d v<version>                           # delete locally
git tag v<version>                              # recreate on the current HEAD
git push origin v<version>
```

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
`v0.2.0 — ETModel helper and Arduino platform layer`.

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
