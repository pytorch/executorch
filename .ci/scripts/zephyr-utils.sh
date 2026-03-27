#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Run instruction from a README.md file
#
# This contains CI helper runtime functions to dig out and run commands
# from a README file.
#
# Generic helper:
# - run_command_block_from_readme
#
# Zephyr-specific validator:
# - verify_zephyr_readme (checks zephyr/executorch.yaml snippet sync)
#
# Main function is run_command_block_from_readme. It digs out the code between
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
#   run_command_block_from_readme path/to/README.md 'Install requirements'
# it will run them one by one



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

_utils_path_from_root () {
  local path="$1"
  local root_dir

  if [[ -z "${path}" || "${path}" == "." ]]; then
    echo "ERROR: Path argument must be a non-empty file path" >&2
    return 1
  fi

  if [[ "${path}" = /* ]]; then
    printf '%s\n' "${path}"
    return 0
  fi

  root_dir="$(_zephyr_utils_root_dir)"
  printf '%s/%s\n' "${root_dir}" "${path}"
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
  local description="$2"
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

run_command_block_from_readme () {
  local readme_path="$1"
  local blockheader="$2"
  local block

  if [[ -z "${readme_path}" || -z "${blockheader}" ]]; then
    echo "ERROR: Usage: run_command_block_from_readme <readme_path> <block_header_regex>" >&2
    return 1
  fi

  readme_path="$(_utils_path_from_root "${readme_path}")" || return 1

  echo "Run block '${blockheader}' from ${readme_path}"

  _zephyr_utils_ensure_file "${readme_path}" "README.md" || return 1

  block="$(_zephyr_utils_extract_block "${readme_path}" "${blockheader}")"
  if [[ -n "${block}" ]]; then
    _zephyr_utils_run_simple_commands "${block}" "${blockheader}"
    return $?
  fi

  echo "ERROR: Failed to locate ${blockheader} block in ${readme_path}" >&2
  return 1
}

# Check that zephyr/executorch.yaml match zephyr/README.md
verify_zephyr_readme () {
  local readme_path manifest_path snippet

  readme_path="$(_utils_path_from_root "zephyr/README.md")" || return 1
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
