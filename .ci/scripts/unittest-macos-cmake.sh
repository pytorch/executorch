#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
set -eux

# =============================================================================
# AOTI HANG DIAGNOSIS — run each AOTI test individually to find the culprit
# =============================================================================

export TORCHINDUCTOR_CACHE_DIR="$(mktemp -d "${RUNNER_TEMP:-/tmp}/torchinductor_cache_XXXXXX")"
trap 'rm -rf "${TORCHINDUCTOR_CACHE_DIR}"' EXIT
export PYTHONUNBUFFERED=1

AOTI_TESTS=(
  "examples/models/llama3_2_vision/preprocess/test_preprocess.py"
  "examples/models/llama3_2_vision/vision_encoder/test/test_vision_encoder.py"
  "examples/models/llama3_2_vision/text_decoder/test/test_text_decoder.py"
  "extension/llm/modules/test/test_position_embeddings.py::TilePositionalEmbeddingTest::test_tile_positional_embedding_aoti"
  "extension/llm/modules/test/test_position_embeddings.py::TiledTokenPositionalEmbeddingTest::test_tiled_token_positional_embedding_aoti"
  "extension/llm/modules/test/test_attention.py::AttentionTest::test_attention_aoti"
)

TIMEOUT=600  # 10 min per test — generous, but short enough to not burn the whole job

for test in "${AOTI_TESTS[@]}"; do
  echo ""
  echo "================================================================"
  echo "=== $(date): STARTING: ${test}"
  echo "================================================================"

  # Run pytest in the background so we can attach a watchdog
  ${CONDA_RUN} --no-capture-output python -m pytest "${test}" -v -x \
    --timeout=${TIMEOUT} --timeout-method=thread \
    -o faulthandler_timeout=120 -p no:xdist &
  TEST_PID=$!

  # Watchdog: sample native stacks every 60s
  (
    while kill -0 "$TEST_PID" 2>/dev/null; do
      sleep 60
      if kill -0 "$TEST_PID" 2>/dev/null; then
        echo ""
        echo "===== WATCHDOG $(date): sampling PID ${TEST_PID} for ${test} ====="
        sample "$TEST_PID" 1 2>&1 | head -200 || true
        echo "===== END WATCHDOG ====="
        echo ""
      fi
    done
  ) &
  WATCHDOG_PID=$!

  set +e
  wait "$TEST_PID"
  EXIT_CODE=$?
  set -e

  kill "$WATCHDOG_PID" 2>/dev/null || true
  wait "$WATCHDOG_PID" 2>/dev/null || true

  echo "================================================================"
  echo "=== $(date): FINISHED: ${test} (exit code ${EXIT_CODE})"
  echo "================================================================"

  if [ "$EXIT_CODE" -ne 0 ]; then
    echo "*** TEST FAILED OR TIMED OUT: ${test} ***"
  fi
done

echo ""
echo "=== All AOTI tests completed ==="
