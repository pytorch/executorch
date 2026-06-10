#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Batch end-to-end inference orchestrator for the CUDA models that share a
# single A100 runner.
#
# Loops over a shared config file (see .ci/scripts/cuda_a100_models.txt) and
# runs each model's e2e test by delegating to the existing per-model script
# .ci/scripts/test_model_e2e.sh. The shared runtime setup
# (./install_requirements.sh) runs once up front; each per-model invocation is
# told to skip it via SKIP_INSTALL_REQUIREMENTS.
#
# Models are tested independently: a failure (or a missing/failed export) in
# one model never aborts the others. For each model we first consult the export
# status written by export_model_artifact_batch.sh under
# <artifact_root>/_status/<model_safe>.export.status:
#   - if export did not succeed, the model is recorded as "skipped: export
#     failed" (its export error is surfaced) and the run continues;
#   - otherwise the per-model e2e test runs.
#
# A summary section is printed at the end (and mirrored to $GITHUB_STEP_SUMMARY
# when available). This script exits non-zero unless EVERY model both exported
# and passed its e2e test, so the CI job is green only when all models pass.

show_help() {
  cat << EOF
Usage: test_model_e2e_batch.sh <device> <config_file> <artifact_root>

Run end-to-end inference tests for every model listed in <config_file>, reading
each model's artifacts from <artifact_root>/<model_safe>/ and its export status
from <artifact_root>/_status/.

Arguments:
  device         cuda, metal, or xnnpack (required) — passed through to
                 test_model_e2e.sh.
  config_file    Path to the shared model list (required). One model per line:
                   <hf_repo>/<hf_name> <quant>
                 Blank lines and lines starting with '#' are ignored.
  artifact_root  Root directory containing per-model artifacts + _status/
                 (required). Typically \${RUNNER_ARTIFACT_DIR} in CI.

Example:
  test_model_e2e_batch.sh cuda .ci/scripts/cuda_a100_models.txt "\${RUNNER_ARTIFACT_DIR}"
EOF
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  show_help
  exit 0
fi

DEVICE="${1:-}"
CONFIG_FILE="${2:-}"
ARTIFACT_ROOT="${3:-}"

if [ -z "$DEVICE" ] || [ -z "$CONFIG_FILE" ] || [ -z "$ARTIFACT_ROOT" ]; then
  echo "Error: missing required argument(s)"
  show_help
  exit 1
fi

if [ ! -f "$CONFIG_FILE" ]; then
  echo "Error: config file not found: $CONFIG_FILE"
  exit 1
fi

# Locate the per-model e2e script and the repo root from this script's location.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
E2E_SCRIPT="$SCRIPT_DIR/test_model_e2e.sh"
EXECUTORCH_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
if [ ! -f "$E2E_SCRIPT" ]; then
  echo "Error: test_model_e2e.sh not found at $E2E_SCRIPT"
  exit 1
fi

STATUS_DIR="$ARTIFACT_ROOT/_status"

# Shared runtime setup — run once for all models.
echo "::group::Setup ExecuTorch Requirements (shared)"
pushd "$EXECUTORCH_ROOT" > /dev/null
./install_requirements.sh
pip list
popd > /dev/null
echo "::endgroup::"

# Track per-model results (parallel arrays; bash 3.2 compatible).
MODELS=()
QUANTS=()
EXPORT_STATUSES=()
E2E_STATUSES=()

OVERALL_RC=0

while IFS= read -r raw_line || [ -n "$raw_line" ]; do
  line="$(echo "$raw_line" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
  case "$line" in
    ''|\#*) continue ;;
  esac

  HF_MODEL="$(echo "$line" | awk '{print $1}')"
  QUANT="$(echo "$line" | awk '{print $2}')"
  if [ -z "$QUANT" ]; then
    QUANT="non-quantized"
  fi

  MODEL_SAFE="$(echo "$HF_MODEL" | tr '/' '_')"
  MODEL_DIR="$ARTIFACT_ROOT/$MODEL_SAFE"
  EXPORT_STATUS_FILE="$STATUS_DIR/$MODEL_SAFE.export.status"
  E2E_LOG_FILE="$STATUS_DIR/$MODEL_SAFE.e2e.log"

  EXPORT_STATUS="missing"
  if [ -f "$EXPORT_STATUS_FILE" ]; then
    EXPORT_STATUS="$(cat "$EXPORT_STATUS_FILE")"
  fi

  MODELS+=("$HF_MODEL")
  QUANTS+=("$QUANT")
  EXPORT_STATUSES+=("$EXPORT_STATUS")

  if [ "$EXPORT_STATUS" != "success" ]; then
    echo "::group::Skip $HF_MODEL ($QUANT) — export status: $EXPORT_STATUS"
    echo "Skipping e2e for $HF_MODEL: export did not succeed (status: $EXPORT_STATUS)"
    EXPORT_LOG_FILE="$STATUS_DIR/$MODEL_SAFE.export.log"
    if [ -f "$EXPORT_LOG_FILE" ]; then
      echo "----- Export error tail -----"
      tail -n 40 "$EXPORT_LOG_FILE"
    fi
    echo "::endgroup::"
    E2E_STATUSES+=("skipped: export failed")
    OVERALL_RC=1
    continue
  fi

  echo "::group::E2E $HF_MODEL ($QUANT)"
  # Run the existing per-model script as a subprocess so its `set -e`, pushd,
  # and cwd changes are isolated and a failure never aborts this loop. The
  # shared runtime setup already ran, so skip the per-model install.
  set +e
  SKIP_INSTALL_REQUIREMENTS=1 bash "$E2E_SCRIPT" "$DEVICE" "$HF_MODEL" "$QUANT" "$MODEL_DIR" 2>&1 | tee "$E2E_LOG_FILE"
  RC=${PIPESTATUS[0]}
  set -e
  echo "::endgroup::"

  if [ "$RC" -eq 0 ]; then
    E2E_STATUSES+=("success")
    echo "E2E succeeded: $HF_MODEL ($QUANT)"
  else
    E2E_STATUSES+=("failed")
    OVERALL_RC=1
    echo "E2E FAILED: $HF_MODEL ($QUANT) (exit $RC)"
  fi
done < "$CONFIG_FILE"

if [ "${#MODELS[@]}" -eq 0 ]; then
  echo "Error: no models found in config file: $CONFIG_FILE"
  exit 1
fi

# ----------------------------------------------------------------------------
# Summary section
# ----------------------------------------------------------------------------
emit_summary() {
  # $1 = output sink: "stdout" or "github"
  local sink="$1"
  local i export_icon e2e_label
  if [ "$sink" = "github" ]; then
    echo "## A100 e2e summary"
    echo ""
    echo "| Model | Quant | Export | E2E |"
    echo "| --- | --- | --- | --- |"
    for i in "${!MODELS[@]}"; do
      if [ "${EXPORT_STATUSES[$i]}" = "success" ]; then export_icon="✅"; else export_icon="❌ ${EXPORT_STATUSES[$i]}"; fi
      case "${E2E_STATUSES[$i]}" in
        success) e2e_label="✅ success" ;;
        *) e2e_label="❌ ${E2E_STATUSES[$i]}" ;;
      esac
      echo "| ${MODELS[$i]} | ${QUANTS[$i]} | ${export_icon} | ${e2e_label} |"
    done
    echo ""
  else
    echo "============================================================"
    echo "A100 e2e summary"
    echo "============================================================"
    for i in "${!MODELS[@]}"; do
      printf '  %-40s %-26s export=%-8s e2e=%s\n' \
        "${MODELS[$i]}" "${QUANTS[$i]}" "${EXPORT_STATUSES[$i]}" "${E2E_STATUSES[$i]}"
    done
    echo "============================================================"
  fi
}

echo ""
emit_summary stdout

# Print error tails and emit GitHub annotations for any non-success model.
for i in "${!MODELS[@]}"; do
  if [ "${E2E_STATUSES[$i]}" = "success" ]; then
    continue
  fi
  MODEL_SAFE="$(echo "${MODELS[$i]}" | tr '/' '_')"
  echo ""
  echo "----- Error tail for ${MODELS[$i]} (${QUANTS[$i]}) -----"
  if [ "${EXPORT_STATUSES[$i]}" != "success" ]; then
    # Export failed/missing: surface the export log.
    EXPORT_LOG_FILE="$STATUS_DIR/$MODEL_SAFE.export.log"
    if [ -f "$EXPORT_LOG_FILE" ]; then tail -n 40 "$EXPORT_LOG_FILE"; fi
    echo "::error::E2E skipped for ${MODELS[$i]} (${QUANTS[$i]}): export ${EXPORT_STATUSES[$i]}"
  else
    # Export succeeded but e2e failed: surface the e2e log.
    E2E_LOG_FILE="$STATUS_DIR/$MODEL_SAFE.e2e.log"
    ERR_LINE=""
    if [ -f "$E2E_LOG_FILE" ]; then
      tail -n 40 "$E2E_LOG_FILE"
      ERR_LINE="$(grep -E -i 'error|fail|expected' "$E2E_LOG_FILE" | tail -n 1)"
      if [ -z "$ERR_LINE" ]; then ERR_LINE="$(tail -n 1 "$E2E_LOG_FILE")"; fi
    fi
    echo "::error::E2E failed for ${MODELS[$i]} (${QUANTS[$i]}): ${ERR_LINE}"
  fi
done

# Mirror summary to the GitHub job summary panel when available.
if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then
  {
    emit_summary github
    for i in "${!MODELS[@]}"; do
      if [ "${E2E_STATUSES[$i]}" = "success" ]; then continue; fi
      MODEL_SAFE="$(echo "${MODELS[$i]}" | tr '/' '_')"
      echo "<details><summary>Error tail: ${MODELS[$i]} (${QUANTS[$i]})</summary>"
      echo ""
      echo '```'
      if [ "${EXPORT_STATUSES[$i]}" != "success" ]; then
        EXPORT_LOG_FILE="$STATUS_DIR/$MODEL_SAFE.export.log"
        if [ -f "$EXPORT_LOG_FILE" ]; then tail -n 40 "$EXPORT_LOG_FILE"; fi
      else
        E2E_LOG_FILE="$STATUS_DIR/$MODEL_SAFE.e2e.log"
        if [ -f "$E2E_LOG_FILE" ]; then tail -n 40 "$E2E_LOG_FILE"; fi
      fi
      echo '```'
      echo "</details>"
      echo ""
    done
  } >> "$GITHUB_STEP_SUMMARY"
fi

if [ "$OVERALL_RC" -ne 0 ]; then
  echo "One or more models failed (export or e2e)."
else
  echo "All models passed export and e2e."
fi

exit "$OVERALL_RC"
