# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""DFlash2 runtime regressions; see test_dflash_runtime.py for runner setup.

Reuse an exported fixture through MUSE_GLIMMER_DFLASH_TEST_ARTIFACT.
"""

import unittest
from dataclasses import replace

from executorch.examples.models.muse_glimmer.tests.test_dflash_runtime import (
    DFlashRuntimeTestMixin,
)


class DFlash2RuntimeTest(DFlashRuntimeTestMixin, unittest.TestCase):
    artifact_env_var = "MUSE_GLIMMER_DFLASH_TEST_ARTIFACT"

    @classmethod
    def make_draft_config(cls):
        return replace(
            super().make_draft_config(),
            block_size=16,
            conv_kernel_size=2,
            conv_group_size=16,
            selector_rank=16,
            selector_top_k=16,
            output_multiplier=0.19611613513,
            final_logit_softcapping=20.0,
        )
