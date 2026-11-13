# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from executorch.examples.qualcomm.oss_scripts.lfm3.speech_policy import (
    require_speech_calibration,
    speech_sampling_metadata,
)


def config(**overrides):
    return {
        "do_sample": False,
        "temperature": 0,
        "top_k": 1,
        "repetition_penalty": 1.0,
        "audio_temperature": 0.4,
        "audio_top_k": None,
        "audio_repetition_penalty": 1.2,
        "audio_repetition_window": 64,
        **overrides,
    }


class SpeechPolicyTest(unittest.TestCase):
    def test_checkpoint_greedy_policy_and_generated_history_windows(self):
        metadata = speech_sampling_metadata(config())
        self.assertEqual(metadata["get_text_temperature"], 0)
        self.assertEqual(metadata["get_text_top_k"], 0)
        self.assertEqual(metadata["get_text_repetition_penalty"], 1.0)
        self.assertEqual(metadata["get_text_repetition_window"], 32)
        self.assertEqual(metadata["get_audio_repetition_window"], 64)
        self.assertEqual(metadata["get_audio_top_k"], 0)

    def test_explicit_sampling_and_penalty_are_not_replaced_by_cli_defaults(self):
        metadata = speech_sampling_metadata(
            config(
                do_sample=True,
                temperature=0.7,
                top_k=20,
                top_p=0.9,
                min_p=0.05,
                repetition_penalty=1.1,
            )
        )
        self.assertEqual(metadata["get_text_temperature"], 0.7)
        self.assertEqual(metadata["get_text_top_k"], 20)
        self.assertEqual(metadata["get_text_top_p"], 0.9)
        self.assertEqual(metadata["get_text_min_p"], 0.05)
        self.assertEqual(metadata["get_text_repetition_penalty"], 1.1)

    def test_do_sample_false_disables_warpers_but_not_penalty(self):
        metadata = speech_sampling_metadata(
            config(temperature=0.7, top_k=20, repetition_penalty=1.1)
        )
        self.assertEqual(metadata["get_text_temperature"], 0)
        self.assertEqual(metadata["get_text_top_k"], 0)
        self.assertEqual(metadata["get_text_repetition_penalty"], 1.1)

    def test_zero_window_disables_history_instead_of_requesting_full_history(self):
        metadata = speech_sampling_metadata(
            config(text_repetition_window=0, audio_repetition_window=0)
        )
        self.assertEqual(metadata["get_text_repetition_window"], 0)
        self.assertEqual(metadata["get_audio_repetition_window"], 0)

    def test_invalid_or_missing_policy_is_rejected_before_weight_loading(self):
        with self.assertRaisesRegex(ValueError, "explicitly set"):
            speech_sampling_metadata({})
        for settings in (
            {"do_sample": "false"},
            {"temperature": float("nan")},
            {"top_k": -1},
            {"repetition_penalty": 0},
            {"top_p": 1.1},
            {"min_p": -1},
            {"text_repetition_window": 2.5},
            {"audio_repetition_window": -1},
        ):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                speech_sampling_metadata(config(**settings))

    def test_only_fresh_speech_output_export_requires_embeddings(self):
        with self.assertRaisesRegex(ValueError, "requires --calib_embeddings"):
            require_speech_calibration(True, None)
        self.assertIsNone(require_speech_calibration(True, "captured.pt"))
        self.assertIsNone(require_speech_calibration(False, None))
        self.assertIsNone(require_speech_calibration(True, None, "existing_pte"))

    def test_unquantized_export_does_not_require_unused_calibration(self):
        self.assertIsNone(require_speech_calibration(True, None, use_fp16=True))
        with self.assertRaisesRegex(ValueError, "requires --calib_embeddings"):
            require_speech_calibration(True, None, use_fp16=False)
