# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

import difflib
import json
import unittest
from pprint import pformat
from typing import List

from executorch.exir._serialize._dataclass import _DataclassEncoder

from executorch.sdk.etdump.fb.schema import (
    AllocationEvent,
    Allocator,
    DebugBlock,
    DebugEvent,
    ETDump,
    ProfileBlock,
    ProfileEvent,
    RunData,
    ScalarType,
    Tensor,
    Value,
)
from executorch.sdk.etdump.fb.serialize import (
    deserialize_from_etdump,
    serialize_to_etdump,
)


def diff_jsons(a: str, b: str) -> List[str]:
    data_a = json.loads(a)
    data_b = json.loads(b)

    return list(
        difflib.unified_diff(pformat(data_a).splitlines(), pformat(data_b).splitlines())
    )


def get_sample_etdump() -> ETDump:
    return ETDump(
        version=0,
        run_data=[
            RunData(
                debug_blocks=[
                    DebugBlock(
                        name="test_debug_block",
                        debug_events=[
                            DebugEvent(
                                debug_handle=0,
                                debug_entries=[
                                    Value(
                                        val=Tensor(
                                            scalar_type=ScalarType.FLOAT,
                                            sizes=[1, 1],
                                            strides=[1, 1],
                                            data=b"datadump",
                                        ),
                                    )
                                ],
                            )
                        ],
                    )
                ],
                profile_blocks=[
                    ProfileBlock(
                        name="test_profile_block",
                        allocators=[Allocator("test_allocator")],
                        profile_events=[
                            ProfileEvent(
                                name="test_profile_event",
                                debug_handle=1,
                                start_time=1001,
                                end_time=2002,
                            )
                        ],
                        allocation_events=[
                            AllocationEvent(
                                allocator_id=1,
                                allocation_size=8,
                            )
                        ],
                    )
                ],
            )
        ],
    )


class TestSerialize(unittest.TestCase):
    def test_serialize(self) -> None:
        program = get_sample_etdump()
        flatbuffer_from_py = serialize_to_etdump(program)
        deserialized_obj = deserialize_from_etdump(flatbuffer_from_py)
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
