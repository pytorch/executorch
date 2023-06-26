# pyre-strict

import unittest

from executorch.sdk.etdump.schema import (
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
from executorch.sdk.etdump.serialize import deserialize_from_etdump, serialize_to_etdump


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
        self.assertEqual(program, deserialize_from_etdump(flatbuffer_from_py))
