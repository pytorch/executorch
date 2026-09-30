#!/usr/bin/env bash
# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Run a command with Khronos Vulkan validation and convert validation
# diagnostics into a failing command status.
#
# This intentionally does not depend on VK_DBG_LAYER_ACTION_FAIL because
# older Vulkan Validation Layer releases do not support that action.

set -uo pipefail


vulkan_validation_enabled() {
    case "${EXECUTORCH_VGF_VULKAN_VALIDATION:-0}" in
        ""|0|false|False|FALSE|off|Off|OFF|no|No|NO)
            return 1
            ;;
        *)
            return 0
            ;;
    esac
}


if [[ $# -eq 0 ]]; then
    echo "Usage: $0 <command> [args ...]" >&2
    exit 2
fi


# Preserve normal behavior when validation was explicitly disabled.
if ! vulkan_validation_enabled; then
    exec "$@"
fi


tmp_dir="$(mktemp -d "${TMPDIR:-/tmp}/executorch-vulkan-validation.XXXXXX")"
command_log="${tmp_dir}/command.log"

cleanup() {
    rm -rf "${tmp_dir}"
}

trap cleanup EXIT


# Do not rely on VK_DBG_LAYER_ACTION_FAIL. Older VVL versions understand
# LOG_MSG. Route diagnostics to stdout so every Vulkan subprocess contributes
# to the same wrapper-captured stream. A shared log file is unsafe because a
# later process can truncate diagnostics written by an earlier process.
export VK_KHRONOS_VALIDATION_REPORT_FLAGS="error"
export VK_KHRONOS_VALIDATION_DEBUG_ACTION="VK_DBG_LAYER_ACTION_LOG_MSG"
export VK_KHRONOS_VALIDATION_LOG_FILENAME="stdout"


# Keep normal command output visible in CI while retaining a copy that can be
# inspected as a fallback. PIPESTATUS[0] is the status of the command, not tee.
"$@" 2>&1 | tee "${command_log}"
command_status="${PIPESTATUS[0]}"


# Diagnostics from all Vulkan descendants are routed through stdout/stderr and
# captured by tee above, so this scan covers the entire wrapped command.
validation_error_pattern='Validation Error:|ERROR[[:space:]]*:[[:space:]]*VALIDATION|VUID-[A-Za-z0-9_]'

if grep -Eq "${validation_error_pattern}" "${command_log}"; then
    echo >&2
    echo "Vulkan validation error detected." >&2

    # Use a deterministic non-zero status. The underlying command may have
    # succeeded because VVL only logged the invalid Vulkan usage.
    exit 1
fi


# No validation error: preserve the underlying command's actual status.
exit "${command_status}"
