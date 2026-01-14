#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Run instruction from zephyr/README.md
#
# This contains some CI helper runtime functions to dig out and run commands
# from zephyr/README.md
# It also try to verify that zephyr/executorch.yaml is in sync and the snippets
# in the README are sane with various regexps.
#
# Main functions is run_command_block_from_zephyr_readme it will dig out the code between
# the two ''' after a blockheader text and run it
#
# .e.g. from this README.md snippet
#
#   Install requirements
#   ```
#   pip install something
#   pip install something_else
#   ```
#
# The blockheader is 'Install requirements' and the code block is 'pip install something ... \npip install something_else'
# so if we run
#   run_command_block_from_zephyr_readme 'Install requirements' '^(pip|pip3)[[:space:]]+install([[:space:]].*)?$'
# it will make sure each lite start with pip or pip3 and then run them one by one



# Resolve and cache this script's directory; if sourced, and remember it globally
# So functions in this script can be used even after sourced and still find
# the correct paths
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
    export ZEPHYR_UTILS_DIR="${script_dir}"
fi

# Internal utility functions
_zephyr_utils_root_dir () {
    local script_dir resolved

    if [[ -n "${ZEPHYR_UTILS_DIR:-}" ]]; then
        script_dir="${ZEPHYR_UTILS_DIR}"
    else
        resolved="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
        if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
            ZEPHYR_UTILS_DIR="${resolved}"
        fi
        script_dir="${resolved}"
    fi

    (cd "${script_dir}/../.." && pwd)
}

_zephyr_utils_readme_path () {
  local root_dir
  root_dir="$(_zephyr_utils_root_dir)"
  printf '%s/zephyr/README.md\n' "${root_dir}"
}

_zephyr_utils_ensure_file () {
  local path="$1"
  local description="$2"
  if [[ ! -f "${path}" ]]; then
    echo "ERROR: ${description} not found at ${path}" >&2
    return 1
  fi
}

_zephyr_utils_extract_block () {
  local readme_path="$1"
  local marker="$2"
  awk -v marker="${marker}" '
    $0 ~ marker { section=1; next }
    section && /^```/ {
      if (in_block) { exit }
      in_block=1
      next
    }
    in_block && /^```/ { exit }
    in_block { sub(/\r$/, ""); print }
  ' "${readme_path}"
}

_zephyr_utils_run_simple_commands () {
  local block="$1"
  local allowedpattern="$2"
  local description="$3"
  temp_dir="$(mktemp -d)"
  if [[ ! -d "${temp_dir}" ]]; then
    echo "ERROR: Failed to create temporary directory for west init" >&2
    return 1
  fi
  trap "rm -rf '${temp_dir}'; trap - RETURN" RETURN

  local ran=0 cmd
  while IFS= read -r cmd; do
    cmd="${cmd#"${cmd%%[![:space:]]*}"}"
    cmd="${cmd%"${cmd##*[![:space:]]}"}"
    [[ -z "${cmd}" ]] && continue

    if [[ ! "${cmd}" =~ ${allowedpattern} ]]; then
      echo "ERROR: Unexpected command in '${description}' readme block: ${cmd} must match pattern: ${allowedpattern}" >&2
      return 1
    fi

    if [[ "${cmd}" == *";"* || "${cmd}" == *"&&"* || "${cmd}" == *"||"* || "${cmd}" == *"|"* ]]; then
      echo "ERROR: Command chaining is not allowed in '${description}' readme block: ${cmd}" >&2
      return 1
    fi

    echo "Setup: conda:${CONDA_PREFIX:-}    or venv:${VIRTUAL_ENV:-}"
    echo "Running: ${cmd}"
    if ! eval "${cmd}"; then
      return 1
    fi
    ran=1
  done <<< "${block}"

  if [[ ${ran} -eq 0 ]]; then
    echo "ERROR: No commands found in block after ${description}" >&2
    return 1
  fi

  return 0
}

run_command_block_from_zephyr_readme () {
  local blockheader="$1"
  local allowedpattern="$2"

  echo "Run block '${blockheader}' from zephyr/README.md"

  readme_path="$(_zephyr_utils_readme_path)"
  _zephyr_utils_ensure_file "${readme_path}" "README.md" || return 1

  block="$(_zephyr_utils_extract_block "${readme_path}" "${blockheader}")"

  if [[ -z "${block}" ]]; then
    echo "ERROR: Failed to locate ${blockheader} block in ${readme_path}" >&2
    return 1
  fi
  _zephyr_utils_run_simple_commands "${block}" "${allowedpattern}" "${blockheader}"
}

# Check that zephyr/executorch.yaml match zephyr/README.md
verify_zephyr_readme () {
  local readme_path manifest_path snippet

  readme_path="$(_zephyr_utils_readme_path)"
  manifest_path="$(_zephyr_utils_root_dir)/zephyr/executorch.yaml"

  _zephyr_utils_ensure_file "${readme_path}" "README" || return 1
  _zephyr_utils_ensure_file "${manifest_path}" "Manifest" || return 1

snippet="$(
    _zephyr_utils_extract_block "${readme_path}" '<zephyr_build_root>/zephyr/submanifests/executorch\.yaml'
)"

  if [[ -z "${snippet}" ]]; then
    echo "ERROR: Failed to extract executorch.yaml snippet from ${readme_path}" >&2
    return 1
  fi

  if diff -u <(printf '%s\n' "${snippet}") "${manifest_path}"; then
    echo "zephyr/README.md executorch.yaml snippet is in sync with zephyr/executorch.yaml" >&2
    return 0
  fi

  echo "ERROR: ${readme_path} executorch.yaml snippet is out of sync with ${manifest_path}" >&2
  return 1
}
