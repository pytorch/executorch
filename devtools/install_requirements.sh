#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Conflict: this requires numpy<2 whereas ExecuTorch core requires numpy>=2
# Follow https://github.com/google-ai-edge/model-explorer/issues/277 for potential
# resolution
pip install ai-edge-model-explorer>=0.1.16
