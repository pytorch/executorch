#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Install sentencepiece for llama tokenizer.
# Install tiktoken for tokenizer.
# Install tokenizers for hf .json tokenizer.
# Install snakeviz for cProfile flamegraph
# Install lm-eval for Model Evaluation with lm-evalution-harness.
pip install hydra-core huggingface_hub tiktoken torchtune sentencepiece tokenizers snakeviz lm_eval==0.4.5 blobfile

# Call the install helper for further setup
python examples/models/llama/install_requirement_helper.py
