<!-- Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree. -->

# Offline validation evidence

Use absolute `REPO` and `BENCH` variables as defined in `$BENCH/README.md`.
Local logs and snapshots below are ignored by `$BENCH/.gitignore`; retain them
for review, not as benchmark results. These logs cover the original offline
package validation; later SM80 HTTP integration is recorded in the main README.

- `$BENCH/validation/pytest.log`: all 10 copied helper tests passed with socket
  creation and connection blocked; zero network attempts. Production export and
  serving modules were not invoked.
- `$BENCH/validation/integrity.log`: original freeze hashes matched; all four
  payloads matched both the original JSON fixture and statically parsed raw
  source. Each raw source was byte-identical and SHA256-identical. Client
  non-import AST was exactly identical to the frozen original. Portable fixture
  SHA256 is `2cfa9faf7eb595fa24908353a53c233d68e88ea8c41faf7ef8d8d8f916e1dd64`.
- `$BENCH/validation/arc-lint.log`: requested narrow `arc lint -a` attempt could
  not run because the OSS checkout has no configured lint engine. No production
  lint config was added or changed.
- `$BENCH/validation/lintrunner.log`: final FLAKE8, UFMT, NEWLINE, LICENSELINT,
  and BANDIT run exited 0. Narrow inherited-code exceptions are explained in
  `$BENCH/validation/flake8.ini`. No dependency installation/init was run.
- `$BENCH/validation/production_python_before.sha256` and
  `$BENCH/validation/production_python_after.sha256`: tracked and nonignored
  existing Python baseline and final snapshot, excluding this new directory.
  `$BENCH/validation/scope.log` records their equality and original-hash checks.
- `$BENCH/validation/worktree_before.txt` and
  `$BENCH/validation/worktree_after.txt`: existing dirty-checkout status and
  final status. The pre-existing branch changes were left untouched.

The first copied test run caught a docstring-copy typo, corrected before the
successful runs. Initial lint attempts exposed inherited warnings and runner
CLI compatibility issues; the retained first/format logs document those steps.
The final portable client differs only by license header, import formatting,
and newline formatting. Its functional AST was checked against the original.

Scope exception: installed lintrunner 0.12.7 wrote automatic metadata in its
pre-existing default user-data area during early attempts, despite the supplied
`--data-path`. No out-of-scope metadata was removed because deletion was not
authorized. Subsequent runs set `XDG_DATA_HOME` and `TMPDIR` inside this directory.
This exception is runner metadata, not a production source edit, install,
model artifact, HTTP request, or GPU workload. Do not claim literally zero
out-of-directory filesystem writes for this session.

RTX 5090 re-export, native build, serving and measurement remain unrun.
The main README now records the subsequent SM80 real HTTP measurements, sampled
VRAM and response/numerical-parity limitations. SM80 cubin cannot be treated as
an SM120 artifact. Generated code and raw fixture scripts were never executed.
