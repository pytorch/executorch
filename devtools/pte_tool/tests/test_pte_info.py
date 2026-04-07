# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import json
import tempfile
import unittest

from executorch.devtools.pte_tool.pte_info import (
    format_delegate_infos,
    get_delegate_infos_from_pte,
)
from executorch.exir._serialize._program import PTEFile, serialize_pte_binary
from executorch.exir.backend.compile_spec_schema import CompileSpec
from executorch.exir.schema import (
    BackendDelegate,
    BackendDelegateDataReference,
    BackendDelegateInlineData,
    Buffer,
    Chain,
    ContainerMetadata,
    DataLocation,
    ExecutionPlan,
    Program,
    SubsegmentOffsets,
)


def _make_program() -> Program:
    return Program(
        version=0,
        execution_plan=[
            ExecutionPlan(
                name="forward",
                container_meta_type=ContainerMetadata(
                    encoded_inp_str="[]", encoded_out_str="[]"
                ),
                values=[],
                inputs=[],
                outputs=[],
                chains=[Chain(inputs=[], outputs=[], instructions=[], stacktrace=None)],
                operators=[],
                delegates=[
                    BackendDelegate(
                        id="BackendA",
                        processed=BackendDelegateDataReference(
                            location=DataLocation.INLINE, index=0
                        ),
                        compile_specs=[
                            CompileSpec(key="mode", value=b"fast"),
                            CompileSpec(key="binary", value=bytes([0, 255])),
                        ],
                    ),
                    BackendDelegate(
                        id="BackendB",
                        processed=BackendDelegateDataReference(
                            location=DataLocation.INLINE, index=0
                        ),
                        compile_specs=[CompileSpec(key="config", value=b"small")],
                    ),
                ],
                non_const_buffer_sizes=[0],
            )
        ],
        constant_buffer=[Buffer(storage=b"")],
        backend_delegate_data=[BackendDelegateInlineData(data=b"delegate-data")],
        segments=[],
        constant_segment=SubsegmentOffsets(segment_index=0, offsets=[]),
        mutable_data_segments=[],
        named_data=[],
    )


class PteInfoTest(unittest.TestCase):
    def test_get_delegate_infos_from_pte(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pte_path = f"{tmpdir}/model.pte"
            with open(pte_path, "wb") as pte_file:
                pte_file.write(bytes(serialize_pte_binary(PTEFile(_make_program()))))

            delegate_infos = get_delegate_infos_from_pte(pte_path)

        self.assertEqual(len(delegate_infos), 2)
        self.assertEqual(delegate_infos[0].plan_index, 0)
        self.assertEqual(delegate_infos[0].plan_name, "forward")
        self.assertEqual(delegate_infos[0].delegate_index, 0)
        self.assertEqual(delegate_infos[0].delegate_id, "BackendA")
        self.assertEqual(delegate_infos[0].compile_specs[0].key, "mode")
        self.assertEqual(delegate_infos[0].compile_specs[0].value, b"fast")
        self.assertEqual(delegate_infos[1].delegate_id, "BackendB")

    def test_format_delegate_infos(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pte_path = f"{tmpdir}/model.pte"
            with open(pte_path, "wb") as pte_file:
                pte_file.write(bytes(serialize_pte_binary(PTEFile(_make_program()))))

            delegate_infos = get_delegate_infos_from_pte(pte_path)

        pretty_output = format_delegate_infos(delegate_infos)
        self.assertIn("plan 0 forward, delegate 0 BackendA:", pretty_output)
        self.assertIn('  mode="fast"', pretty_output)
        self.assertIn("  binary=0x00ff", pretty_output)

        json_output = json.loads(
            format_delegate_infos(delegate_infos, output_format="json")
        )
        self.assertEqual(json_output[0]["delegate_id"], "BackendA")
        self.assertEqual(json_output[0]["compile_specs"][0]["value_text"], "fast")
        self.assertEqual(json_output[0]["compile_specs"][1]["value_hex"], "00ff")
