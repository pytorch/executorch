# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import os
import unittest

import torch

from executorch.devtools.tests.xnnpack.xnnpack_test_utils import (
    check_disturbance,
    check_numeric_gap,
    generate_etrecord_and_etdump,
)

from torchvision import models


class TestViTModel(unittest.TestCase):
    def setUp(self):
        vit = models.vision_transformer.vit_b_16(weights="IMAGENET1K_V1")
        self.model = vit.eval()
        self.model_inputs = (torch.randn(1, 3, 224, 224),)

    def test_numeric_gap(self):
        etrecord_path, etdump_path, debug_buffer_path = generate_etrecord_and_etdump(
            self.model,
            self.model_inputs,
        )

        # Check if the output files exist
        self.assertTrue(
            os.path.exists(etrecord_path), f"ETRecord not found: {etrecord_path}"
        )
        self.assertTrue(os.path.exists(etdump_path), f"ETDump not found: {etdump_path}")
        self.assertTrue(
            os.path.exists(debug_buffer_path),
            f"Debug buffer not found: {debug_buffer_path}",
        )

        metric = "MSE"
        max_allowed_gap = 1e-6
        is_within_threshold, max_gap = check_numeric_gap(
            etdump_path,
            etrecord_path,
            debug_buffer_path,
            metric=metric,
            max_allowed_gap=max_allowed_gap,
        )

        # Check if the numeric gap is within threshold
        self.assertTrue(
            is_within_threshold,
            f"Numeric gap {max_gap} exceeds allowed threshold {max_allowed_gap}",
        )

    def test_numeric_gap_with_disturbance(self):
        # Check if we can detect the first numeric gap directly affected by the disturbance
        etrecord_path, etdump_path, debug_buffer_path = generate_etrecord_and_etdump(
            self.model,
            self.model_inputs,
            disturb=True,
        )

        metric = "MSE"
        max_allowed_gap = 1e-6
        disturbance_threshold = 1e-3
        is_within_thresholds = check_disturbance(
            etdump_path,
            etrecord_path,
            debug_buffer_path,
            metric=metric,
            row=1,
            max_allowed_gap=max_allowed_gap,
            disturbance_threshold=disturbance_threshold,
        )

        self.assertTrue(is_within_thresholds)
