# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import difflib
import json
import unittest
from pprint import pformat
from typing import List

import executorch.sdk.etdump.schema_flatcc as flatcc
from executorch.exir._serialize._dataclass import _DataclassEncoder

from executorch.sdk.etdump.serialize import (
    deserialize_from_etdump_flatcc,
    serialize_to_etdump_flatcc,
)


def diff_jsons(a: str, b: str) -> List[str]:
    data_a = json.loads(a)
    data_b = json.loads(b)

    return list(
        difflib.unified_diff(pformat(data_a).splitlines(), pformat(data_b).splitlines())
    )


def get_sample_etdump_flatcc() -> flatcc.ETDumpFlatCC:
    return flatcc.ETDumpFlatCC(
        version=0,
        run_data=[
            flatcc.RunData(
                name="test_block",
                allocators=[
                    flatcc.Allocator(
                        name="test_allocator",
                    )
                ],
                events=[
                    flatcc.Event(
                        profile_event=flatcc.ProfileEvent(
                            name="test_profile_event",
                            chain_id=1,
                            instruction_id=1,
                            delegate_debug_id_str="",
                            delegate_debug_id_int=-1,
                            delegate_debug_metadata="",
                            start_time=1001,
                            end_time=2002,
                        ),
                        allocation_event=None,
                        debug_event=None,
                    ),
                    flatcc.Event(
                        profile_event=flatcc.ProfileEvent(
                            name="test_profile_event_delegated",
                            chain_id=1,
                            instruction_id=1,
                            delegate_debug_id_str="",
                            delegate_debug_id_int=13,
                            delegate_debug_metadata="",
                            start_time=1001,
                            end_time=2002,
                        ),
                        allocation_event=None,
                        debug_event=None,
                    ),
                    flatcc.Event(
                        profile_event=None,
                        allocation_event=flatcc.AllocationEvent(
                            allocator_id=1,
                            allocation_size=8,
                        ),
                        debug_event=None,
                    ),
                    flatcc.Event(
                        profile_event=None,
                        allocation_event=None,
                        debug_event=flatcc.DebugEvent(
                            chain_idx=1,
                            debug_handle=0,
                            debug_entries=[
                                flatcc.Value(
                                    val=flatcc.ValueType.TENSOR.value,
                                    offset=12345,
                                )
                            ],
                        ),
                    ),
                ],
            )
        ],
    )


class TestSerializeFlatCC(unittest.TestCase):
    def test_serialize(self) -> None:
        import json

        program = get_sample_etdump_flatcc()

        flatcc_from_py = serialize_to_etdump_flatcc(program)
        deserialized_obj = deserialize_from_etdump_flatcc(
            flatcc_from_py, size_prefixed=False
        )
        self.assertEqual(
            program,
            deserialized_obj,
            msg="\n".join(
                diff_jsons(
                    json.dumps(program, cls=_DataclassEncoder, indent=4),
                    json.dumps(deserialized_obj, cls=_DataclassEncoder, indent=4),
                )
            ),
        )
