#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -euo pipefail

if [[ -z "${MUSE_GLIMMER_SERVER_URL:-}" ]]; then
  echo "SKIP: set MUSE_GLIMMER_SERVER_URL=host:port to run the multi-session smoke test"
  exit 0
fi

exec "$(dirname "$0")/test_multisession.sh" "$MUSE_GLIMMER_SERVER_URL" "${MUSE_GLIMMER_MODEL_ID:-muse_glimmer}" "${MUSE_GLIMMER_NUM_CONCURRENT:-8}"
