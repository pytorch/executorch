# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from executorch.backends.qualcomm.genai_pipeline.inference.runner_constants import (
    DECODER_MODEL_VERSION,
    EVAL_MODE,
    MODALITY_INPUT_FLAG_MAP,
)


class TestRunnerConstants(unittest.TestCase):

    def test_constants_match_legacy_decoder_constants(self):
        """The runner flag values equal the legacy flow's, so both drive the runner alike."""
        from executorch.examples.qualcomm.oss_scripts.llama import decoder_constants

        self.assertEqual(EVAL_MODE, decoder_constants.EVAL_MODE)
        self.assertEqual(DECODER_MODEL_VERSION, decoder_constants.DECODER_MODEL_VERSION)
        self.assertEqual(
            MODALITY_INPUT_FLAG_MAP, decoder_constants.MODALITY_INPUT_FLAG_MAP
        )


if __name__ == "__main__":
    unittest.main()
