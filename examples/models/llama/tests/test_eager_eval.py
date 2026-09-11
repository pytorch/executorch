# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from types import SimpleNamespace
from typing import Optional

import pytest

pytest.importorskip("lm_eval", reason="requires lm-evaluation-harness")

from executorch.examples.models.llama.evaluate.eager_eval import (  # noqa: E402
    EagerEvalWrapper,
)


class TestEagerEvalWrapperTokenIds(unittest.TestCase):
    @staticmethod
    def _wrapper(**token_ids: Optional[int]) -> EagerEvalWrapper:
        # HFLM initialization loads a model and is unrelated to these properties.
        wrapper = object.__new__(EagerEvalWrapper)
        wrapper._tokenizer = SimpleNamespace(**token_ids)  # pyre-ignore[8]
        return wrapper

    def test_token_id_fallbacks(self):
        cases = (
            ({"bos_id": 1, "eot_id": 2, "eos_id": 3}, 2, 1),
            ({"bos_id": 0, "eot_id": 2, "eos_id": 3}, 2, 0),
            ({"bos_id": None, "eot_id": 2, "eos_id": 3}, 2, 2),
            ({"bos_id": None, "eot_id": 0, "eos_id": 3}, 0, 0),
            ({"bos_id": None, "eot_id": None, "eos_id": 0}, 0, 0),
            ({"eos_id": 0}, 0, 0),
        )

        for token_ids, expected_eot, expected_prefix in cases:
            with self.subTest(token_ids=token_ids):
                wrapper = self._wrapper(**token_ids)
                self.assertEqual(wrapper.eot_token_id, expected_eot)
                self.assertEqual(wrapper.prefix_token_id, expected_prefix)
