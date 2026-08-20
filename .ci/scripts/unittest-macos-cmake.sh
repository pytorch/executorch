#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
set -eux

# Keep AOTInductor precompiled headers scoped to this job. The default cache
# location can persist across macOS self-hosted runner jobs and produce stale
# PCH failures after PyTorch is reinstalled.
export TORCHINDUCTOR_CACHE_DIR="$(mktemp -d "${RUNNER_TEMP:-/tmp}/torchinductor_cache_XXXXXX")"
trap 'rm -rf "${TORCHINDUCTOR_CACHE_DIR}"' EXIT

# TODO(SS-JIA): AOTI tests hang on macOS CI runners — the thread blocks in
# native C/C++ code (dlopen / inductor compilation) so faulthandler cannot
# even produce a traceback. Diagnosis ongoing in #19886.
AOTI_SKIPS=(
  --ignore=examples/models/llama3_2_vision/preprocess/test_preprocess.py
  --ignore=examples/models/llama3_2_vision/vision_encoder/test/test_vision_encoder.py
  --ignore=examples/models/llama3_2_vision/text_decoder/test/test_text_decoder.py
  --deselect=extension/llm/modules/test/test_position_embeddings.py::TilePositionalEmbeddingTest::test_tile_positional_embedding_aoti
  --deselect=extension/llm/modules/test/test_position_embeddings.py::TiledTokenPositionalEmbeddingTest::test_tiled_token_positional_embedding_aoti
  --deselect=extension/llm/modules/test/test_attention.py::AttentionTest::test_attention_aoti
)

${CONDA_RUN} pytest -n auto --cov=./ --cov-report=xml "${AOTI_SKIPS[@]}"
# Run gtest
LLVM_PROFDATA="xcrun llvm-profdata" LLVM_COV="xcrun llvm-cov" \
${CONDA_RUN} test/run_oss_cpp_tests.sh
