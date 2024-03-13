#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Install snakeviz for cProfile flamegraph
# Install sentencepiece for llama tokenizer
pip install snakeviz sentencepiece
pip install torchao-nightly

# Install datasets for HuggingFace dataloader
# v2.14.0 is intentional to force lm-eval v0.3.0 compatibility
pip install datasets==2.14.0

# Install lm-eval for Model Evaluation with lm-evalution-harness
# v0.3.0 is intentional
pip install lm-eval==0.3.

# Call the install helper for further setup
python examples/models/llama2/install_requirement_helper.py
