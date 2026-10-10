#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ex

pip install evaluate 'accelerate>=1.1.0'
pip install --upgrade-strategy only-if-needed \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  -r examples/models/yolo26/requirements.txt

python -m unittest discover -v -s backends/samsung/test/unit -p "test_*.py"
# Fail fast on compiler regressions before running fine-tuning tests.
python -m unittest -fv \
  executorch.backends.samsung.test.models.test_yolo26_primitives \
  executorch.backends.samsung.test.models.test_yolo26 \
  executorch.backends.samsung.test.ops.test_mul.TestMul.test_fp32_attention_scale \
  executorch.backends.samsung.test.ops.test_topk
python -m unittest discover -fv -s backends/samsung/test/models -p "test_*.py"
