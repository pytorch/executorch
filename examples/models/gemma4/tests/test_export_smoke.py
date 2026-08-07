# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from types import SimpleNamespace
from unittest import mock

from executorch.examples.models.gemma4.text_decoder.gemma4_model import Gemma4Model


def _model(max_seq_len: int = 8960) -> Gemma4Model:
    model = Gemma4Model.__new__(Gemma4Model)
    model.config = SimpleNamespace(
        enable_dynamic_shape=True,
        max_seq_len=max_seq_len,
        use_kv_cache=True,
    )
    return model


class ExportSmokeTest(unittest.TestCase):
    def test_input_bound_is_independent_from_kv_capacity(self) -> None:
        dim = object()
        with mock.patch("torch.export.Dim", return_value=dim) as dim_factory:
            dynamic_shapes = _model().get_dynamic_shapes(max_input_len=512)

        dim_factory.assert_called_once_with("seq_len", min=1, max=512)
        self.assertIs(dynamic_shapes["input_ids"][1], dim)
        self.assertIs(dynamic_shapes["input_pos"][0], dim)

    def test_input_bound_fails_closed(self) -> None:
        for invalid in (1, 8960, 8961):
            with self.subTest(max_input_len=invalid):
                with self.assertRaisesRegex(ValueError, "max_input_len"):
                    _model().get_dynamic_shapes(max_input_len=invalid)
