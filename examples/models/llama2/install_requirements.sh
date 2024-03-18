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

# Install lm-eval for Model Evaluation with lm-evalution-harness
pip install lm-eval

# Call the install helper for further setup
python examples/models/llama2/install_requirement_helper.py
