#!/bin/bash
# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# This is a script to test run the zephyrOS Ethos-U tests and verify that the README.md
# instructions are correct.by running the tests content of test-arm-backend-zephyr in
# github/workflows/trunk.yml as a bash script, (ignoring the conda commands)
# This is motly useful for testing this before upstreaming or for debugging CI issues.

set -eu pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}" && pwd)"

SCRIPT_CONTENT="$(python - <<'PY'
from ruamel.yaml import YAML
with open(".github/workflows/trunk.yml") as f:
    data = YAML().load(f)
script = data["jobs"]["test-arm-backend-zephyr"]["with"]["script"]
filtered = []
con_lines = ("CONDA_ENV=", "conda activate")
for line in script.splitlines():
    if any(line.strip().startswith(prefix) for prefix in con_lines):
        continue
    filtered.append(line)
print("\n".join(filtered))
PY
)"

printf "%s\n" "${SCRIPT_CONTENT}" | /bin/bash
