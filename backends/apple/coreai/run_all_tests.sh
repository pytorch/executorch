#!/usr/bin/env bash
# Dev convenience: run all Core AI backend tests (test/ + passes/test/).
#
# WIP/under-construction backend -- a local helper, not a landing artifact.
#
# Uses unittest (NOT pytest): pytest's default discovery puts backends/apple/ on
# sys.path, so `import coreai` resolves to THIS backend dir and shadows the real
# Apple `coreai` SDK (coreai_torch then fails to import coreai.authoring).
# unittest imports via the full executorch.backends.apple.coreai.* path, so the
# SDK is not shadowed.
#
# Run inside the Core AI conda env, e.g.:
#     conda run -n coreai backends/apple/coreai/run_all_tests.sh
# Extra args are forwarded to unittest, e.g. `... run_all_tests.sh -v`.
set -euo pipefail

cd "$(sl root 2>/dev/null || git rev-parse --show-toplevel)"

# Discover every test_*.py under the backend and turn it into a dotted module
# name (executorch.backends.apple.coreai...<file without .py>).
modules=$(
  find backends/apple/coreai -name 'test_*.py' \
    | sed -E 's#^#executorch.#; s#/#.#g; s#\.py$##' \
    | sort
)

exec python -m unittest ${modules} "$@"
