#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

SAMSUNG_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SAMSUNG_COMMAND_PID=""
SAMSUNG_COMMAND_FILE=""
SAMSUNG_CLEANUP_DONE=0
SAMSUNG_COMMAND_STARTING=0
SAMSUNG_PENDING_SIGNAL=0
export DEVICE_REQUESTED=0
export DEVICE_QUEUED=0
export DEVICE_ACQUIRED=0

samsung_device_signal() {
  if [[ "$SAMSUNG_COMMAND_STARTING" == 1 ]]; then
    if [[ "$SAMSUNG_PENDING_SIGNAL" == 0 ]]; then
      SAMSUNG_PENDING_SIGNAL=$1
    fi
  else
    exit "$1"
  fi
}

samsung_device_command() {
  local duration="$1" rc=0
  shift
  SAMSUNG_COMMAND_FILE=$(mktemp)
  SAMSUNG_COMMAND_STARTING=1
  python "$SAMSUNG_SCRIPT_DIR/samsung-device-command.py" "$duration" "$@" >"$SAMSUNG_COMMAND_FILE" &
  SAMSUNG_COMMAND_PID=$!
  SAMSUNG_COMMAND_STARTING=0
  if [[ "$SAMSUNG_PENDING_SIGNAL" != 0 ]]; then
    exit "$SAMSUNG_PENDING_SIGNAL"
  fi
  wait "$SAMSUNG_COMMAND_PID" || rc=$?
  SAMSUNG_COMMAND_PID=""
  SAMSUNG_COMMAND_OUTPUT=$(cat "$SAMSUNG_COMMAND_FILE")
  rm -f "$SAMSUNG_COMMAND_FILE"
  SAMSUNG_COMMAND_FILE=""
  return "$rc"
}

samsung_device_cleanup() {
  local rc=$?
  trap '' TERM INT
  if [[ "$SAMSUNG_CLEANUP_DONE" == 0 ]]; then
    SAMSUNG_CLEANUP_DONE=1
    if [[ -n "$SAMSUNG_COMMAND_PID" ]]; then
      kill -TERM "$SAMSUNG_COMMAND_PID" 2>/dev/null || true
    fi
    if [[ -n "$SAMSUNG_COMMAND_FILE" ]]; then
      rm -f "$SAMSUNG_COMMAND_FILE"
    fi
    # Start remote cancellation without waiting out the command's kill grace.
    bash "$SAMSUNG_SCRIPT_DIR/cleanup-samsung-linux-deps.sh" || true
    if [[ -n "$SAMSUNG_COMMAND_PID" ]]; then
      wait "$SAMSUNG_COMMAND_PID" || true
    fi
  fi
  return "$rc"
}

trap samsung_device_cleanup EXIT
trap 'samsung_device_signal 143' TERM
trap 'samsung_device_signal 130' INT
