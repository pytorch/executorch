#!/bin/bash
# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Usage:
#   EXECUTORCH_CI=$(.ci/scripts/detect_ci.sh)
#
# Test whether any supported CI was detected:
#   [[ -n "${EXECUTORCH_CI}" ]]
#
# Test for a specific CI provider:
#   [[ "${EXECUTORCH_CI}" == "JENKINS" ]]
#   [[ "${EXECUTORCH_CI}" == "GITHUB" ]]
#
# Test for CI debug mode:
#   [[ -n "$(.ci/scripts/detect_ci.sh --and-debug)" ]]
#
# Test for CI but not debug mode:
#   [[ -n "$(.ci/scripts/detect_ci.sh --and-not-debug)" ]]

filter="${1:-}"
if [[ "${#}" -gt 1 ]]; then
    exit 0
fi

ci_provider=""
debug_env=""

if [[ "${GITHUB_ACTIONS:-}" == "true" ]]; then
    ci_provider="GITHUB"
    debug_env="${GITHUB_DEBUG:-}"
elif [[ -n "${JENKINS_URL:-}" ||
        -n "${JENKINS_HOME:-}" ||
        -n "${JENKINS_SERVER_COOKIE:-}" ||
        -n "${HUDSON_URL:-}" ||
        -n "${HUDSON_HOME:-}" ||
        "${BUILD_TAG:-}" == jenkins-* ]]; then
    ci_provider="JENKINS"
    # Jenkins and other pipelines do not have a standard debug-mode variable.
    # Pipelines that need debug-aware behavior should set this manually.
    debug_env="${EXECUTORCH_CI_DEBUG:-}"
elif [[ -n "${CI:-}" ]]; then
    ci_provider="UNKNOWN_CI"
    # Jenkins and other pipelines do not have a standard debug-mode variable.
    # Pipelines that need debug-aware behavior should set this manually.
    debug_env="${EXECUTORCH_CI_DEBUG:-}"
else
    exit 0
fi

debug_enabled=false
if [[ "${debug_env}" == "1" || "${debug_env}" == "true" || "${debug_env}" == "TRUE" ]]; then
    debug_enabled=true
fi

case "${filter}" in
    "")
        ;;
    --and-debug)
        [[ "${debug_enabled}" == "true" ]] || exit 0
        ;;
    --and-not-debug)
        [[ "${debug_enabled}" != "true" ]] || exit 0
        ;;
    *)
        exit 0
        ;;
esac

echo "${ci_provider}"
exit 0
