#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Batch export orchestrator for the CUDA models that share a single A100 runner.
#
# Loops over a shared config file (see .ci/scripts/cuda_a100_models.txt) and
# exports each model by delegating to the existing per-model script
# .ci/scripts/export_model_artifact.sh. Each model is exported independently:
# a failure in one model never aborts the others. Per-model outcomes are
# recorded under <output_root>/_status/ so the downstream e2e batch job can
# decide which models to run, and a summary section is printed at the end (and
# mirrored to $GITHUB_STEP_SUMMARY when available).
#
# This script exits non-zero if ANY model failed to export, so the CI job is
# green only when every model exported successfully.

show_help() {
  cat << EOF
Usage: export_model_artifact_batch.sh <device> <config_file> <output_root>

Export every model listed in <config_file> to <device> format, writing each
model's artifacts into <output_root>/<model_safe>/ and per-model status into
<output_root>/_status/.

Arguments:
  device        cuda, metal, or xnnpack (required) — passed through to
                export_model_artifact.sh.
  config_file   Path to the shared model list (required). One model per line:
                  <hf_repo>/<hf_name> <quant>
                Blank lines and lines starting with '#' are ignored.
  output_root   Root directory for artifacts + status (required). Typically
                \${RUNNER_ARTIFACT_DIR} in CI.

Example:
  export_model_artifact_batch.sh cuda .ci/scripts/cuda_a100_models.txt "\${RUNNER_ARTIFACT_DIR}"
EOF
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  show_help
  exit 0
fi

DEVICE="${1:-}"
CONFIG_FILE="${2:-}"
OUTPUT_ROOT="${3:-}"

if [ -z "$DEVICE" ] || [ -z "$CONFIG_FILE" ] || [ -z "$OUTPUT_ROOT" ]; then
  echo "Error: missing required argument(s)"
  show_help
  exit 1
fi

if [ ! -f "$CONFIG_FILE" ]; then
  echo "Error: config file not found: $CONFIG_FILE"
  exit 1
fi

# Locate the per-model export script next to this one.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPORT_SCRIPT="$SCRIPT_DIR/export_model_artifact.sh"
if [ ! -f "$EXPORT_SCRIPT" ]; then
  echo "Error: export_model_artifact.sh not found at $EXPORT_SCRIPT"
  exit 1
fi

STATUS_DIR="$OUTPUT_ROOT/_status"
mkdir -p "$STATUS_DIR"

# Track per-model results (parallel arrays; bash 3.2 compatible).
MODELS=()
QUANTS=()
STATUSES=()

OVERALL_RC=0

while IFS= read -r raw_line || [ -n "$raw_line" ]; do
  # Strip leading/trailing whitespace.
  line="$(echo "$raw_line" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
  # Skip blanks and comments.
  case "$line" in
    ''|\#*) continue ;;
  esac

  # First field = hf_model, second field = quant (default non-quantized).
  HF_MODEL="$(echo "$line" | awk '{print $1}')"
  QUANT="$(echo "$line" | awk '{print $2}')"
  if [ -z "$QUANT" ]; then
    QUANT="non-quantized"
  fi

  MODEL_SAFE="$(echo "$HF_MODEL" | tr '/' '_')"
  MODEL_OUT="$OUTPUT_ROOT/$MODEL_SAFE"
  LOG_FILE="$STATUS_DIR/$MODEL_SAFE.export.log"
  STATUS_FILE="$STATUS_DIR/$MODEL_SAFE.export.status"
  mkdir -p "$MODEL_OUT"

  echo "::group::Export $HF_MODEL ($QUANT)"
  # Run the existing per-model script as a subprocess so its `set -e`, traps,
  # and cwd changes are isolated and a failure never aborts this loop.
  set +e
  bash "$EXPORT_SCRIPT" "$DEVICE" "$HF_MODEL" "$QUANT" "$MODEL_OUT" 2>&1 | tee "$LOG_FILE"
  RC=${PIPESTATUS[0]}
  set -e
  echo "::endgroup::"

  MODELS+=("$HF_MODEL")
  QUANTS+=("$QUANT")
  if [ "$RC" -eq 0 ]; then
    echo "success" > "$STATUS_FILE"
    STATUSES+=("success")
    echo "Export succeeded: $HF_MODEL ($QUANT)"
  else
    echo "failed" > "$STATUS_FILE"
    STATUSES+=("failed")
    OVERALL_RC=1
    echo "Export FAILED: $HF_MODEL ($QUANT) (exit $RC)"
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
  local i status_icon
  if [ "$sink" = "github" ]; then
    echo "## A100 export summary"
    echo ""
    echo "| Model | Quant | Export |"
    echo "| --- | --- | --- |"
    for i in "${!MODELS[@]}"; do
      if [ "${STATUSES[$i]}" = "success" ]; then status_icon="✅ success"; else status_icon="❌ failed"; fi
      echo "| ${MODELS[$i]} | ${QUANTS[$i]} | ${status_icon} |"
    done
    echo ""
  else
    echo "============================================================"
    echo "A100 export summary"
    echo "============================================================"
    for i in "${!MODELS[@]}"; do
      printf '  %-45s %-28s %s\n' "${MODELS[$i]}" "${QUANTS[$i]}" "${STATUSES[$i]}"
    done
    echo "============================================================"
  fi
}

echo ""
emit_summary stdout

# Print the error tail for each failed model and emit GitHub annotations.
for i in "${!MODELS[@]}"; do
  if [ "${STATUSES[$i]}" != "success" ]; then
    MODEL_SAFE="$(echo "${MODELS[$i]}" | tr '/' '_')"
    LOG_FILE="$STATUS_DIR/$MODEL_SAFE.export.log"
    echo ""
    echo "----- Error tail for ${MODELS[$i]} (${QUANTS[$i]}) -----"
    if [ -f "$LOG_FILE" ]; then
      tail -n 40 "$LOG_FILE"
    fi
    # One-line annotation (last meaningful log line) for the CI UI.
    ERR_LINE=""
    if [ -f "$LOG_FILE" ]; then
      ERR_LINE="$(grep -E -i 'error|fail' "$LOG_FILE" | tail -n 1)"
      if [ -z "$ERR_LINE" ]; then
        ERR_LINE="$(tail -n 1 "$LOG_FILE")"
      fi
    fi
    echo "::error::Export failed for ${MODELS[$i]} (${QUANTS[$i]}): ${ERR_LINE}"
  fi
done

# Mirror summary to the GitHub job summary panel when available.
if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then
  {
    emit_summary github
    for i in "${!MODELS[@]}"; do
      if [ "${STATUSES[$i]}" != "success" ]; then
        echo "<details><summary>Error tail: ${MODELS[$i]} (${QUANTS[$i]})</summary>"
        echo ""
        echo '```'
        MODEL_SAFE="$(echo "${MODELS[$i]}" | tr '/' '_')"
        LOG_FILE="$STATUS_DIR/$MODEL_SAFE.export.log"
        if [ -f "$LOG_FILE" ]; then tail -n 40 "$LOG_FILE"; fi
        echo '```'
        echo "</details>"
        echo ""
      fi
    done
  } >> "$GITHUB_STEP_SUMMARY"
fi

if [ "$OVERALL_RC" -ne 0 ]; then
  echo "One or more models failed to export."
else
  echo "All models exported successfully."
fi

exit "$OVERALL_RC"
