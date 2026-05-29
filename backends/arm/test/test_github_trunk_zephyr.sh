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
# Parse optional arguments to select README and target list.

# Target list (--target-list) adds a comma separated list of targets you want to test
# matching the zephyr/README.md tags:
#    <!-- RUN test_XX_generate_pte -->
#    <!-- RUN test_XX_build_and_run -->
# e.g. one or more of this: ethos-u55,cortex-m55,ethos-u85

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

TARGET_LIST_OVERRIDDEN=0
if [[ -n "${TARGET_LIST:-}" ]]; then
    TARGET_LIST_OVERRIDDEN=1
fi
TARGET_LIST="${TARGET_LIST:-ethos-u55,cortex-m55,ethos-u85}"
README_PATH="${README_PATH:-}"
HELLO_README_PATH="zephyr/samples/hello-executorch/README.md"
MV2_README_PATH="zephyr/samples/mv2-ethosu/README.md"
DEFAULT_MV2_TARGET_LIST="ethos-u55,ethos-u85"
usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Options:
  -t, --target-list LIST      Comma-separated targets (default: ${TARGET_LIST})
  -r, --readme PATH           Run only one README path
  -h, --help                  Show this help and exit

When --readme is used, --target-list or TARGET_LIST is required.
You can also set TARGET_LIST or README_PATH environment variable.
Examples:
  $(basename "$0")
  $(basename "$0") -t ethos-u55,cortex-m55
  $(basename "$0") --target-list=ethos-u85
  $(basename "$0") --readme zephyr/samples/mv2-ethosu/README.md --target-list ethos-u85
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
            TARGET_LIST_OVERRIDDEN=1
            shift 2
            ;;
        --target-list=*)
            TARGET_LIST="${1#*=}"
            TARGET_LIST_OVERRIDDEN=1
            shift
            ;;
        -r|--readme)
            if [[ -z "${2:-}" || "$2" == -* ]]; then
                echo "Error: $1 requires a non-empty argument."
                usage
                exit 2
            fi
            README_PATH="$2"
            shift 2
            ;;
        --readme=*)
            README_PATH="${1#*=}"
            shift
            ;;
        *)
            echo "Error: Unknown option: $1"
            usage
            exit 2
            ;;
    esac
done

if [[ -n "${README_PATH}" && ${TARGET_LIST_OVERRIDDEN} -eq 0 ]]; then
    echo "Error: --readme requires --target-list or TARGET_LIST." >&2
    usage >&2
    exit 2
fi

cd "${ROOT_DIR}"

run_zephyr_readme() {
    local readme_path="$1"
    local targets="$2"

    echo "Running ${readme_path} targets: ${targets}"
    .ci/scripts/test_zephyr.sh \
        --zephyr-samples-readme-path "${readme_path}" \
        --targets "${targets}"
}

if [[ -n "${README_PATH}" ]]; then
    run_zephyr_readme "${README_PATH}" "${TARGET_LIST}"
else
    run_zephyr_readme "${HELLO_README_PATH}" "${TARGET_LIST}"
    run_zephyr_readme "${MV2_README_PATH}" "${DEFAULT_MV2_TARGET_LIST}"
fi
