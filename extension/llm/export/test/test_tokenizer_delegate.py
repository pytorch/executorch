# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from executorch.exir.scalar_type import ScalarType
from executorch.exir.schema import (
    DataLocation,
    DelegateCall,
    Program,
    String,
    SubsegmentOffsets,
    Tensor,
    TensorShapeDynamism,
)
from executorch.extension.llm.export.tokenizer_delegate import (
    append_tokenizer_delegate_method,
    TOKENIZER_BACKEND_ID,
    TOKENIZER_METHOD_NAME,
)


class TestTokenizerDelegate(unittest.TestCase):
    def _make_program_manager(self) -> Any:
        program = Program(
            version=0,
            execution_plan=[],
            constant_buffer=[],
            backend_delegate_data=[],
            segments=[],
            constant_segment=SubsegmentOffsets(segment_index=0, offsets=[]),
        )
        return SimpleNamespace(
            _emitter_output=SimpleNamespace(program=program),
            _backend_config=object(),
            _data_serializer=None,
            _named_data=None,
            _pte_data=None,
            _tensor_data=None,
            _buffer=b"stale",
        )

    def test_appends_tokenizer_execution_plan(self) -> None:
        tokenizer_bytes = b"llama-stories-tokenizer-bytes"
        manager = self._make_program_manager()

        with tempfile.TemporaryDirectory() as tmpdir:
            tokenizer_path = Path(tmpdir) / "tokenizer.model"
            tokenizer_path.write_bytes(tokenizer_bytes)
            with patch(
                "executorch.extension.llm.export.tokenizer_delegate"
                ".serialize_for_executorch",
                return_value=(b"serialized", {"tensor": b"data"}),
            ) as serialize:
                append_tokenizer_delegate_method(
                    manager,
                    tokenizer_path=str(tokenizer_path),
                    max_context_length=16,
                )

        program = manager._emitter_output.program
        self.assertEqual(manager._pte_data, b"serialized")
        self.assertEqual(manager._tensor_data, {"tensor": b"data"})
        self.assertIsNone(manager._buffer)
        serialize.assert_called_once()

        self.assertEqual(len(program.backend_delegate_data), 1)
        self.assertEqual(program.backend_delegate_data[0].data, tokenizer_bytes)
        self.assertEqual(len(program.execution_plan), 1)

        plan = program.execution_plan[0]
        self.assertEqual(plan.name, TOKENIZER_METHOD_NAME)
        self.assertEqual(plan.inputs, [0])
        self.assertEqual(plan.outputs, [1])
        self.assertEqual(plan.non_const_buffer_sizes, [0, 16 * 8])

        self.assertIsInstance(plan.values[0].val, String)
        self.assertEqual(plan.values[0].val.string_val, "")
        self.assertIsInstance(plan.values[1].val, Tensor)
        token_tensor = plan.values[1].val
        self.assertEqual(token_tensor.scalar_type, ScalarType.LONG)
        self.assertEqual(token_tensor.sizes, [16])
        self.assertEqual(
            token_tensor.shape_dynamism, TensorShapeDynamism.DYNAMIC_BOUND
        )

        self.assertEqual(len(plan.delegates), 1)
        delegate = plan.delegates[0]
        self.assertEqual(delegate.id, TOKENIZER_BACKEND_ID)
        self.assertEqual(delegate.processed.location, DataLocation.INLINE)
        self.assertEqual(delegate.processed.index, 0)
        self.assertEqual(
            {spec.key: spec.value for spec in delegate.compile_specs},
            {
                "max_context_length": b"16",
                "bos": b"0",
                "eos": b"0",
            },
        )

        self.assertEqual(len(plan.chains), 1)
        delegate_call = plan.chains[0].instructions[0].instr_args
        self.assertIsInstance(delegate_call, DelegateCall)
        self.assertEqual(delegate_call.delegate_index, 0)
        self.assertEqual(delegate_call.args, [0, 1])


if __name__ == "__main__":
    unittest.main()
