#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Install snakeviz for cProfile flamegraph
# Install sentencepiece for llama tokenizer
pip install snakeviz sentencepiece

# Install torchao.
TORCHAO_VERSION=$(cat "$(dirname "$0")"/../../../.ci/docker/ci_commit_pins/torchao.txt)
pip install --no-use-pep517 "git+https://github.com/pytorch/ao.git@${TORCHAO_VERSION}"

# Install lm-eval for Model Evaluation with lm-evalution-harness
# Install tiktoken for tokenizer
pip install lm_eval==0.4.2
pip install tiktoken blobfile

# Call the install helper for further setup
python examples/models/llama2/install_requirement_helper.py
