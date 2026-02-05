#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Retry function for flaky commands (e.g., PyTorch CDN rate limiting)
# Usage: retry_with_backoff <command> [args...]
# Example: retry_with_backoff pip install torch
retry_with_backoff() {
  local max_attempts=${RETRY_MAX_ATTEMPTS:-3}
  local delay=${RETRY_INITIAL_DELAY:-30}
  local attempt=1
  while [ $attempt -le $max_attempts ]; do
    echo "Attempt $attempt of $max_attempts..."
    if "$@"; then
      return 0
    fi
    if [ $attempt -lt $max_attempts ]; then
      echo "Command failed, retrying in ${delay}s..."
      sleep $delay
      delay=$((delay * 2))
    fi
    attempt=$((attempt + 1))
  done
  echo "All $max_attempts attempts failed"
  return 1
}
