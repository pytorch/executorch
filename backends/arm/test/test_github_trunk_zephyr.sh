#!/bin/bash
# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# This is a script to test run the zephyrOS Ethos-U tests and verify that the README.md
# instructions are correct.by running the tests content of test-arm-backend-zephyr in
# github/workflows/trunk.yml as a bash script, (ignoring the conda commands)
# This is mostly useful for testing this before upstreaming or for debugging CI issues.
# Or a way to setup zephyrOS so you can play with it.
# Parse optional argument to set TARGET_LIST environment variable

# Target list (--target-list) adds a comma separated list of targets you want to test
# matching the zephyr/README.md tags:
#    <!-- RUN test_XX_generate_pte -->
#    <!-- RUN test_XX_build_and_run -->
# e.g. one or more of this: ethos-u55,cortex-m55,ethos-u85

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}" && pwd)"

TARGET_LIST="${TARGET_LIST:-ethos-u55,cortex-m55,ethos-u85}"
usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Options:
  -t, --target-list LIST      Comma-separated targets (default: ${TARGET_LIST})
  -h, --help                  Show this help and exit

You can also set TARGET_LIST environment variable.
Examples:
  $(basename "$0")
  $(basename "$0") -t ethos-u55,cortex-m55
  $(basename "$0") --target-list=ethos-u85
EOF
}

# No-arg support: loop simply won't run and defaults are used
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        -t|--target-list)
            if [[ -z "${2:-}" || "$2" == -* ]]; then
                echo "Error: $1 requires a non-empty argument."
                usage
                exit 2
            fi
            TARGET_LIST="$2"
            shift 2
            ;;
        --target-list=*)
            TARGET_LIST="${1#*=}"
            shift
            ;;
        *)
            echo "Error: Unknown option: $1"
            usage
            exit 2
            ;;
    esac
done
export TARGET_LIST

echo "Running .github/workflows/trunk.yml testing ${TARGET_LIST} from zephyr/README.md"

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
