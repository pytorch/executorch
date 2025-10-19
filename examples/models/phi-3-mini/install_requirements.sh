#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -x

pip install sentencepiece accelerate

EXECUTORCH_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"

OPTIMUM_ET_VERSION=$(cat "${EXECUTORCH_ROOT}/.ci/docker/ci_commit_pins/optimum-executorch.txt")
pip install git+https://github.com/huggingface/optimum-executorch.git@${OPTIMUM_ET_VERSION}

pip list
