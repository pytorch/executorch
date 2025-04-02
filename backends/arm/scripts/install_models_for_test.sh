#!/usr/bin/env bash
# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -e

# Install diffusers for Stable Diffusion model test
pip install "diffusers[torch]==0.33.1"
