# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from executorch.backends.qualcomm.genai_pipeline.artifact_keys import (
    ALL_ARTIFACT_KEYS,
    ARTIFACT_ATTENTION_SINK_EVICTOR,
    ARTIFACT_AUDIO_ENCODER,
    ARTIFACT_TEXT_DECODER,
    ARTIFACT_TEXT_ENCODER,
    ARTIFACT_TOK_EMBEDDING,
    ARTIFACT_VISION_ENCODER,
    DECODE_QDQ_FILENAME,
)


class TestArtifactKeys(unittest.TestCase):

    def test_keys_match_legacy_decoder_constants(self):
        """Every artifact key equals the legacy runner's pte_paths key."""
        from executorch.examples.qualcomm.oss_scripts.llama import decoder_constants

        pairs = {
            ARTIFACT_ATTENTION_SINK_EVICTOR: decoder_constants.ATTENTION_SINK_EVICTOR,
            ARTIFACT_AUDIO_ENCODER: decoder_constants.AUDIO_ENCODER,
            ARTIFACT_TEXT_DECODER: decoder_constants.TEXT_DECODER,
            ARTIFACT_TEXT_ENCODER: decoder_constants.TEXT_ENCODER,
            ARTIFACT_TOK_EMBEDDING: decoder_constants.TOK_EMBEDDING,
            ARTIFACT_VISION_ENCODER: decoder_constants.VISION_ENCODER,
        }

        for ours, legacy in pairs.items():
            with self.subTest(key=ours):
                self.assertEqual(ours, legacy)

    def test_all_artifact_keys_is_complete(self):
        """ALL_ARTIFACT_KEYS holds every individually exported key."""
        individual = {
            ARTIFACT_ATTENTION_SINK_EVICTOR,
            ARTIFACT_AUDIO_ENCODER,
            ARTIFACT_TEXT_DECODER,
            ARTIFACT_TEXT_ENCODER,
            ARTIFACT_TOK_EMBEDDING,
            ARTIFACT_VISION_ENCODER,
        }

        self.assertEqual(ALL_ARTIFACT_KEYS, individual)


class TestDecodeQdqFilename(unittest.TestCase):
    """The QDQ export is a file the A-side stages exchange, not an artifact."""

    def test_matches_legacy_filename(self):
        """A2's SQNR path reads what llama.py writes, so the name must agree."""
        from executorch.examples.qualcomm.oss_scripts.llama import decoder_constants

        self.assertEqual(DECODE_QDQ_FILENAME, decoder_constants.DECODE_QDQ_FILENAME)

    def test_is_not_an_artifact_key(self):
        """It is not a ``.pte``, so it must never reach ``artifact_paths``."""
        self.assertNotIn(DECODE_QDQ_FILENAME, ALL_ARTIFACT_KEYS)


if __name__ == "__main__":
    unittest.main()
