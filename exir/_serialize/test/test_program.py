#!/usr/bin/env fbpython
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import copy
import difflib
import json
import unittest

from typing import List, Sequence

from executorch.exir._serialize._flatbuffer import _program_flatbuffer_to_json
from executorch.exir._serialize._program import (
    _ExtendedHeader,
    _get_extended_header,
    _json_to_program,
    _program_to_json,
    deserialize_pte_binary,
    serialize_pte_binary,
)

from executorch.exir.schema import (
    BackendDelegate,
    BackendDelegateDataReference,
    BackendDelegateInlineData,
    Buffer,
    ContainerMetadata,
    DataLocation,
    DataSegment,
    ExecutionPlan,
    Program,
    SubsegmentOffsets,
)
from executorch.exir.tests.common import get_test_program

SEGMENT_ALIGNMENT: int = 128

CONSTANT_TENSOR_ALIGNMENT: int = 16


def add_constant_data(program: Program, blobs: Sequence[bytes]) -> None:
    """Adds the provided constant data blobs to the program."""
    for blob in blobs:
        program.constant_buffer.append(Buffer(storage=blob))


def add_delegate_data(
    program: Program, plan: ExecutionPlan, blobs: Sequence[bytes]
) -> None:
    """Adds the provided delegate data blobs to the execution plan."""
    di = len(plan.delegates)
    for blob in blobs:
        data_index: int = len(program.backend_delegate_data)
        program.backend_delegate_data.append(
            BackendDelegateInlineData(
                data=blob,
            )
        )
        delegate = BackendDelegate(
            id=f"delegate{di}",
            processed=BackendDelegateDataReference(
                location=DataLocation.INLINE,
                index=data_index,
            ),
            compile_specs=[],
        )
        plan.delegates.append(delegate)
        di += 1


def canonicalize_delegate_indices(program: Program) -> Program:
    """Returns a copy of the program with the backend delegate data list in
    a predictable order.
    """
    program = copy.deepcopy(program)

    # Original index and its data.
    delegate_entries: list[tuple[int, bytes]] = [
        (i, entry.data) for i, entry in enumerate(program.backend_delegate_data)
    ]

    # Sort by the contents of the data, which is the second entry in the tuple.
    # NOTE: This is unstable if multiple entries have the same data contents.
    delegate_entries.sort(key=lambda x: x[1])

    # Build up the sorted Program.backend_delegate_data list, and a mapping from
    # the old index to the new index.
    old_to_new_index: dict[int, int] = {}
    program.backend_delegate_data = []
    for i, data in delegate_entries:
        old_to_new_index[i] = len(program.backend_delegate_data)
        print(f">>> Mapping [{i}]: {old_to_new_index[i]} '{data}'")
        program.backend_delegate_data.append(BackendDelegateInlineData(data=data))

    # Patch up the index pointers from the BackendDelegate entries.
    for plan in program.execution_plan:
        for delegate in plan.delegates:
            delegate.processed.index = old_to_new_index[delegate.processed.index]

    return program


class TestProgram(unittest.TestCase):
    def assert_file_magic_present(self, program_data: bytes) -> None:
        self.assertEqual(program_data[4:6], b"ET")
        # Ignore the other bytes, which can change over time and are not
        # important for this test.

    def assert_programs_equal(self, program1: Program, program2: Program) -> None:
        def prepare_json_string(j: str) -> List[str]:
            """Formats the JSON and splits it into lines."""
            return json.dumps(json.loads(j), indent=2, sort_keys=True).splitlines(
                keepends=True
            )

        # This JSON comparison is fragile: some parts of the program do not care
        # about order (like the operators list), so those are technically free
        # to be reordered. If they become a problem, we can canonicalize them
        # like we do for the backend delegate data list.
        json1 = _program_to_json(canonicalize_delegate_indices(program1))
        json2 = _program_to_json(canonicalize_delegate_indices(program2))

        # Use unified_diff so it only prints the differences instead of the
        # entire string.
        diff: str = "".join(
            difflib.unified_diff(
                prepare_json_string(json1),
                prepare_json_string(json2),
            )
        )
        if diff:
            self.fail(msg="Programs are not equal\n" + diff)

    def get_and_validate_extended_header(self, pte_data: bytes) -> _ExtendedHeader:
        """When an extended header is expected, check that it exists and is valid.
        Does not check correctness of the contents."""
        eh = _get_extended_header(pte_data)
        self.assertIsNotNone(eh)
        self.assertTrue(eh.is_valid())
        self.assertLess(eh.program_size, len(pte_data))
        return eh

    def constant_segment_with_tensor_alignment(
        self, constant_tensor_alignment: int
    ) -> None:
        """Utility to test constant segment with varying alignment.
        Args:
            constant_tensor_alignment: Alignment of constant tensor data.
                Must be a multiple of 2.
                Must be > 8 for the purposes of the test, which checks +- 3 bytes on the edges of each tensor.
        """
        # Create a program with some constant tensor data.
        program = get_test_program()
        blobs = (
            b"",  # Empty tensor.
            self.gen_blob_data(constant_tensor_alignment // 2, b"\x10\x11\x01"),
            self.gen_blob_data(constant_tensor_alignment - 1, b"\x20\x22\x02"),
            self.gen_blob_data(constant_tensor_alignment, b"\x30\x33\x03"),
            self.gen_blob_data(constant_tensor_alignment + 1, b"\x40\x44\x04"),
        )
        add_constant_data(program, blobs)

        # Extract blobs into constant segment during serialization.
        pte_data = bytes(
            serialize_pte_binary(
                program,
                segment_alignment=SEGMENT_ALIGNMENT,
                constant_tensor_alignment=constant_tensor_alignment,
            )
        )

        # The input Program should not be modified.
        self.assertEqual(program.segments, [])

        # Extended header should be present in the serialized data.
        eh = self.get_and_validate_extended_header(pte_data)

        # Segment offset should be non-zero since there are segments. It
        # should point past the end of the program data, but not beyond
        # the end of the file.
        self.assertGreaterEqual(eh.segment_base_offset, eh.program_size)
        self.assertLess(eh.segment_base_offset, len(pte_data))

        # Peek inside the actual flatbuffer data to see the segments.
        program_with_segments = _json_to_program(_program_flatbuffer_to_json(pte_data))

        # The constant tensor data should appear as the only segment.
        self.assertEqual(len(program_with_segments.segments), 1)

        # The constant buffer should appear now as a constant segment.
        segment_table: List[DataSegment] = program_with_segments.segments
        self.assertEqual(len(segment_table), 1)
        # Tensor sizes
        # - tensor[0]: 0
        # - tensors[1,2,3]: constant_tensor_alignment
        # - tensor[4]: constant_tensor_alignment + 1 (no padding on the last tensor)
        self.assertEqual(
            segment_table[0].size,
            constant_tensor_alignment * 3 + (constant_tensor_alignment + 1),
        )

        # Check constant_segment index and offsets.
        subsegment_offsets: SubsegmentOffsets = program_with_segments.constant_segment
        self.assertEqual(subsegment_offsets.segment_index, 0)
        self.assertEqual(
            subsegment_offsets.offsets,
            [
                0,  # Start at offset 0.
                0,  # tensor[0] is empty.
                constant_tensor_alignment,  # tensor[1] has size constant_tensor_alignment // 2. Round up.
                constant_tensor_alignment
                * 2,  # tensor[2] has size constant_tensor_alignment - 1. Round up.
                constant_tensor_alignment
                * 3,  # tensor[3] has size constant_tensor_alignment. No padding needed.
            ],
        )

        # Check constant_buffer is empty, because the data was moved into the segment.
        self.assertEqual(len(program_with_segments.constant_buffer), 0)

        # Check segment data.
        offsets = subsegment_offsets.offsets
        segment_data: bytes = pte_data[eh.segment_base_offset :]

        # tensor[1]: padding.
        self.assertEqual(
            segment_data[offsets[1] : offsets[1] + 3],
            # Tensor data.
            b"\x10\x11\x11",
        )
        self.assertEqual(
            segment_data[
                offsets[1]
                + constant_tensor_alignment // 2 : offsets[1]
                + constant_tensor_alignment // 2
                + 3
            ],
            # Padding.
            b"\x00\x00\x00",
        )

        # tensor[3]: no padding.
        self.assertEqual(
            segment_data[offsets[4] - 3 : offsets[4] + 3],
            # End of tensor 3.
            b"\x33\x33\x03"
            # Start of tensor 4.
            + b"\x40\x44\x44",
        )

        # tensor[4]: no padding for last tensor.
        self.assertEqual(
            segment_data[
                offsets[4]
                + constant_tensor_alignment
                - 3 : offsets[4]
                + constant_tensor_alignment
                + 1
            ],
            b"\x44\x44\x44\x04",
        )

        # The final segment should not point past the end of the file.
        self.assertLessEqual(
            segment_table[-1].offset + segment_table[-1].size,
            len(pte_data),
            f"{segment_table}",
        )

        # Convert back.
        program2 = deserialize_pte_binary(pte_data)
        # Programs are the same besides constant_buffer, as deserialization
        # does not preserve constant segment; padding may be added
        # during serialization.
        self.assertEqual(program2.execution_plan, program.execution_plan)
        # Number of constant tensors should be the same.
        self.assertEqual(len(program2.constant_buffer), len(program.constant_buffer))

    def test_canonicalize_delegate_indices(self) -> None:
        def make_execution_plan(
            name: str, delegates: List[BackendDelegate]
        ) -> ExecutionPlan:
            return ExecutionPlan(
                name=name,
                container_meta_type=ContainerMetadata(
                    encoded_inp_str="encoded_inp_str",
                    encoded_out_str="encoded_out_str",
                ),
                values=[],
                inputs=[],
                outputs=[],
                chains=[],
                operators=[],
                delegates=delegates,
                non_const_buffer_sizes=[],
            )

        # A program with three delegates across two execution plans. To start
        # with, the data indices in the delegates are in a non-canonical order.
        program = Program(
            version=0,
            execution_plan=[
                make_execution_plan(
                    name="forward0",
                    delegates=[
                        BackendDelegate(
                            id="delegate0",
                            processed=BackendDelegateDataReference(
                                location=DataLocation.INLINE, index=2
                            ),
                            compile_specs=[],
                        ),
                        BackendDelegate(
                            id="delegate1",
                            processed=BackendDelegateDataReference(
                                location=DataLocation.INLINE, index=1
                            ),
                            compile_specs=[],
                        ),
                    ],
                ),
                make_execution_plan(
                    name="forward1",
                    delegates=[
                        BackendDelegate(
                            id="delegate2",
                            processed=BackendDelegateDataReference(
                                location=DataLocation.INLINE, index=0
                            ),
                            compile_specs=[],
                        ),
                    ],
                ),
            ],
            constant_buffer=[],
            backend_delegate_data=[
                # Data is in non-canonical (unsorted) order.
                BackendDelegateInlineData(data=b"CC delegate [1,0] data"),
                BackendDelegateInlineData(data=b"BB delegate [0,1] data"),
                BackendDelegateInlineData(data=b"AA delegate [0,0] data"),
            ],
            segments=[],
            constant_segment=SubsegmentOffsets(segment_index=0, offsets=[]),
        )

        # Demonstrate which data each delegate points to.
        self.assertEqual(
            program.backend_delegate_data[
                program.execution_plan[0].delegates[0].processed.index
            ].data,
            b"AA delegate [0,0] data",
        )
        self.assertEqual(
            program.backend_delegate_data[
                program.execution_plan[0].delegates[1].processed.index
            ].data,
            b"BB delegate [0,1] data",
        )
        self.assertEqual(
            program.backend_delegate_data[
                program.execution_plan[1].delegates[0].processed.index
            ].data,
            b"CC delegate [1,0] data",
        )

        # Canonicalize the program.
        canonical_program: Program = canonicalize_delegate_indices(program)

        # The delegate data list should be sorted by contents.
        self.assertListEqual(
            canonical_program.backend_delegate_data,
            [
                # Should have been sorted.
                BackendDelegateInlineData(data=b"AA delegate [0,0] data"),
                BackendDelegateInlineData(data=b"BB delegate [0,1] data"),
                BackendDelegateInlineData(data=b"CC delegate [1,0] data"),
            ],
        )

        # Demonstrate that the delegate entries still point to the correct data.
        self.assertEqual(
            canonical_program.backend_delegate_data[
                canonical_program.execution_plan[0].delegates[0].processed.index
            ].data,
            b"AA delegate [0,0] data",
        )
        self.assertEqual(
            canonical_program.backend_delegate_data[
                canonical_program.execution_plan[0].delegates[1].processed.index
            ].data,
            b"BB delegate [0,1] data",
        )
        self.assertEqual(
            canonical_program.backend_delegate_data[
                canonical_program.execution_plan[1].delegates[0].processed.index
            ].data,
            b"CC delegate [1,0] data",
        )

    def test_round_trip_no_header_no_segments(self) -> None:
        """Tests that a Program remains the same after serializing and
        deserializing.
        """
        program = get_test_program()
        pte_data = bytes(serialize_pte_binary(program))
        self.assertGreater(len(pte_data), 16)

        # File magic should be present at the expected offset.
        self.assert_file_magic_present(pte_data)

        # Extended header should not be present.
        eh = _get_extended_header(pte_data)
        self.assertIsNone(eh)

        # Convert back.
        program2 = deserialize_pte_binary(pte_data)

        # Programs should be the same.
        self.assert_programs_equal(program, program2)

    def test_round_trip_large_buffer_sizes(self) -> None:
        """Tests that when the non_const_buffer_sizes contains integers
        overflowing a signed/unsigned 32 bit integer, we can still serialize the
        model and get the same program by deserialization.
        """
        program = get_test_program()
        program.execution_plan[0].non_const_buffer_sizes = [0, 2**48]
        flatbuffer_from_py = bytes(serialize_pte_binary(program))
        self.assert_programs_equal(program, deserialize_pte_binary(flatbuffer_from_py))

    def test_round_trip_no_segments_and_no_header(self) -> None:
        """Tests that a Program serialized with extract_delegate_segments=True
        when there are no segments does not contain an extended header,
        constant segment, or delegate segments. Confirm that a Program remains
        the same after serializing and deserializing.
        """
        program = get_test_program()
        pte_data = bytes(serialize_pte_binary(program, extract_delegate_segments=True))
        self.assertGreater(len(pte_data), 16)

        # File magic should be present at the expected offset.
        self.assert_file_magic_present(pte_data)

        # Extended header should not be present when no segments are created.
        eh = _get_extended_header(pte_data)
        self.assertIsNone(eh)

        # Peek inside the flatbuffer data to confirm that there are no segments.
        program_with_segments = _json_to_program(_program_flatbuffer_to_json(pte_data))
        self.assertEqual(program_with_segments.segments, [])

        # Convert back.
        program2 = deserialize_pte_binary(pte_data)

        # Programs should be the same.
        self.assert_programs_equal(program, program2)

    @staticmethod
    def gen_blob_data(size: int, pattern: bytes) -> bytes:
        """Generates a buffer with special first and last bytes,
        repeating the middle byte of the pattern."""
        assert len(pattern) == 3
        assert size >= 3
        # Stretch out the middle byte to fill the space.
        ret = pattern[0:1] + pattern[1:2] * (size - 2) + pattern[2:3]
        assert len(ret) == size
        return ret

    def test_round_trip_with_segments(self) -> None:
        # Create a program with some delegate data blobs.
        program = get_test_program()
        blobs = (
            self.gen_blob_data(SEGMENT_ALIGNMENT // 5, b"\x10\x11\x01"),
            # Focus on blobs whose sizes fall close to the alignment.
            self.gen_blob_data(SEGMENT_ALIGNMENT - 1, b"\x20\x22\x02"),
            self.gen_blob_data(SEGMENT_ALIGNMENT, b"\x30\x33\x03"),
            self.gen_blob_data(SEGMENT_ALIGNMENT + 1, b"\x40\x44\x04"),
            b"",  # Empty segment.
            self.gen_blob_data(SEGMENT_ALIGNMENT // 10, b"\x50\x55\x05"),
        )
        add_delegate_data(program, program.execution_plan[0], blobs)

        # Extract the blobs into segments during serialization.
        pte_data = bytes(
            serialize_pte_binary(
                program,
                extract_delegate_segments=True,
                segment_alignment=SEGMENT_ALIGNMENT,
            )
        )

        # The input Program should not have been modified.
        self.assertEqual(program.segments, [])
        self.assertEqual(
            program.execution_plan[0].delegates[0].processed.location,
            DataLocation.INLINE,
        )

        # Extended header should be present in the serialized data.
        eh = self.get_and_validate_extended_header(pte_data)
        # Segment offset should be non-zero since there are segments. It
        # should point past the end of the program data, but not beyond
        # the end of the file.
        self.assertGreaterEqual(eh.segment_base_offset, eh.program_size)
        self.assertLess(eh.segment_base_offset, len(pte_data))

        # Peek inside the actual flatbuffer data to see the segments. Note that
        # this also implicity tests the case where we try parsing the entire
        # file with segment data following it, demonstrating that the extra data
        # doesn't upset the flatbuffer parsing path.
        program_with_segments = _json_to_program(_program_flatbuffer_to_json(pte_data))

        # The delegate blobs we added to the program should appear as segments.
        # The one empty blob should have been ignored, hence the `- 1`.
        self.assertEqual(len(program_with_segments.segments), len(blobs) - 1)
        segment_table: List[DataSegment] = program_with_segments.segments

        # Check segment range invariants.
        for i in range(len(segment_table)):
            # All offsets should be a multiple of SEGMENT_ALIGNMENT.
            self.assertTrue(
                segment_table[i].offset % SEGMENT_ALIGNMENT == 0,
                f"Segment {i} offset is not aligned: {segment_table[i]}",
            )
            # There should be no empty segments.
            self.assertGreater(
                segment_table[i].size, 0, f"Segment {i}: {segment_table}"
            )
            if i > 0:
                # Segments should not overlap, and should be sorted from
                # smallest offset to largest.
                self.assertLessEqual(
                    segment_table[i - 1].offset + segment_table[i - 1].size,
                    segment_table[i].offset,
                    f"Segment {i} overlaps or is out of order: {segment_table}",
                )
        # The first segment should begin at zero; i.e., at the segment base
        # offset.
        self.assertEqual(segment_table[0].offset, 0, f"{segment_table}")
        # The final segment should not point past the end of the file.
        self.assertLessEqual(
            segment_table[-1].offset + segment_table[-1].size,
            len(pte_data),
            f"{segment_table}",
        )

        # Check the segment base offset boundary.
        segment_base_offset = eh.segment_base_offset
        self.assertEqual(
            pte_data[segment_base_offset - 2 : segment_base_offset + 3],
            # The padding before the first segment.
            b"\x00\x00"
            # The first few bytes of the first segment.
            + b"\x10\x11\x11",
        )

        # Now that we've shown that the base offset is correct, slice off the
        # front so that all segment offsets are relative to zero.
        segment_data: bytes = pte_data[segment_base_offset:]

        # End of the first segment. It's much smaller than the alignment,
        # so we know that it's followed by zeros.
        self.assertEqual(
            segment_data[segment_table[0].size - 3 : segment_table[0].size + 2],
            # The end of the segment.
            b"\x11\x11\x01"
            # The padding that follows it.
            + b"\x00\x00",
        )

        # Look at the end of segment[2], which is exactly the same size as
        # the alignment. There should be no padding, running right into the
        # next segment.
        self.assertEqual(
            segment_data[segment_table[3].offset - 3 : segment_table[3].offset + 3],
            # The end of segment[2].
            b"\x33\x33\x03"
            # The beginning of segment[3]
            b"\x40\x44\x44",
        )

        # Convert back; the programs should be the same after a round trip,
        # meaning that the segments were moved back to inline. This also
        # demonstrates that the contents of all segments survived, and weren't
        # truncated or corrupted.
        program2 = deserialize_pte_binary(pte_data)
        self.assert_programs_equal(program, program2)

    def test_no_constants(self) -> None:
        program = get_test_program()
        # Insert placeholder for non-const tensors.
        add_constant_data(program, [b""])

        pte_data = bytes(
            serialize_pte_binary(
                program,
                extract_delegate_segments=True,
                segment_alignment=SEGMENT_ALIGNMENT,
                constant_tensor_alignment=CONSTANT_TENSOR_ALIGNMENT,
            )
        )
        # The input Program should not be modified.
        self.assertEqual(program.segments, [])

        # Peek inside the actual flatbuffer data to see the segments.
        flatbuffer_program = _json_to_program(_program_flatbuffer_to_json(pte_data))

        # Constant buffer should be empty.
        self.assertEqual(len(flatbuffer_program.constant_buffer), 0)

        # Constant segment should contain the placeholder.
        self.assertEqual(flatbuffer_program.constant_segment.segment_index, 0)
        self.assertEqual(len(flatbuffer_program.constant_segment.offsets), 1)
        self.assertEqual(flatbuffer_program.constant_segment.offsets[0], 0)

    def test_unused_inline_delegate_blobs_with_segments(self) -> None:
        # Create a program with some delegate data blobs.
        program = get_test_program()
        blobs = (
            self.gen_blob_data(16, b"\x10\x11\x01"),
            self.gen_blob_data(32, b"\x20\x22\x02"),
        )
        add_delegate_data(program, program.execution_plan[0], blobs)

        # Extract the blobs into segments should succeeed.
        pte_data = bytes(
            serialize_pte_binary(
                program,
                extract_delegate_segments=True,
                segment_alignment=SEGMENT_ALIGNMENT,
            )
        )
        self.assertGreater(len(pte_data), 16)

        # Add another inline blob that is not pointed to by a delegate.
        program.backend_delegate_data.append(
            BackendDelegateInlineData(data=self.gen_blob_data(16, b"\x30\x33\x03"))
        )

        # Should cause serialization to fail.
        with self.assertRaises(ValueError):
            serialize_pte_binary(
                program,
                extract_delegate_segments=True,
                segment_alignment=SEGMENT_ALIGNMENT,
            )

    def test_constant_segment_tensor_alignment_16(self) -> None:
        self.constant_segment_with_tensor_alignment(16)

    def test_constant_segment_tensor_alignment_128(self) -> None:
        self.constant_segment_with_tensor_alignment(128)

    def test_constant_segment_tensor_alignment_non_power_of_2_fails(self) -> None:
        # Create a program with some constant tensor data.
        program = get_test_program()
        program.constant_buffer.append(Buffer(storage=b"12345"))

        constant_tensor_alignment: int = 14
        # Extract blobs into constant segment during serialization.
        # Expect failure as tensor alignment 14 is not a power of 2.
        with self.assertRaises(ValueError):
            serialize_pte_binary(
                program,
                segment_alignment=SEGMENT_ALIGNMENT,
                constant_tensor_alignment=constant_tensor_alignment,
            )

    def test_constant_segment_and_delegate_segment(self) -> None:
        # Create a program with some constant tensor data and delegate data blobs.
        program = get_test_program()
        constant_blobs = (
            self.gen_blob_data(CONSTANT_TENSOR_ALIGNMENT // 2, b"\x10\x11\x01"),
            self.gen_blob_data(CONSTANT_TENSOR_ALIGNMENT + 1, b"\x20\x22\x02"),
        )
        delegate_blobs = (
            self.gen_blob_data(SEGMENT_ALIGNMENT // 2, b"\x30\x33\x03"),
            self.gen_blob_data(SEGMENT_ALIGNMENT + 1, b"\x40\x44\x04"),
        )

        add_constant_data(program, constant_blobs)
        add_delegate_data(program, program.execution_plan[0], delegate_blobs)

        # Extract the blobs into segments during serialization.
        pte_data = bytes(
            serialize_pte_binary(
                program,
                extract_delegate_segments=True,
                segment_alignment=SEGMENT_ALIGNMENT,
                constant_tensor_alignment=CONSTANT_TENSOR_ALIGNMENT,
            )
        )

        # The input Program should not be modified.
        self.assertEqual(program.segments, [])
        self.assertEqual(
            program.execution_plan[0].delegates[0].processed.location,
            DataLocation.INLINE,
        )

        # Extended header should be present in the serialized data.
        eh = self.get_and_validate_extended_header(pte_data)

        # Segment offset should be non-zero since there are segments. It
        # should point past the end of the program data, but not beyond
        # the end of the file.
        self.assertGreaterEqual(eh.segment_base_offset, eh.program_size)
        self.assertLess(eh.segment_base_offset, len(pte_data))

        # Peek inside the actual flatbuffer data to see the segments.
        program_with_segments = _json_to_program(_program_flatbuffer_to_json(pte_data))

        # Segment table should contain a constant segment and the delegate blobs.
        segment_table: List[DataSegment] = program_with_segments.segments
        self.assertEqual(len(segment_table), len(delegate_blobs) + 1)
        self.assertEqual(segment_table[0].offset, 0)
        # segment_table[0] is the constant segment, which
        # contains a couple of tensors with sizes:
        # - tensor[0] = CONSTANT_TENSOR_ALIGNMENT
        # - tensor[1] = CONSTANT_TENSOR_ALIGNMENT + 1 (no padding on last tensor)
        self.assertEqual(segment_table[0].size, CONSTANT_TENSOR_ALIGNMENT * 2 + 1)
        self.assertEqual(segment_table[1].offset, SEGMENT_ALIGNMENT)
        self.assertEqual(segment_table[1].size, SEGMENT_ALIGNMENT // 2)
        self.assertEqual(segment_table[2].offset, SEGMENT_ALIGNMENT * 2)
        self.assertEqual(segment_table[2].size, SEGMENT_ALIGNMENT + 1)

        # Check constant_segment index and offsets.
        subsegment_offsets: SubsegmentOffsets = program_with_segments.constant_segment
        self.assertEqual(subsegment_offsets.segment_index, 0)
        self.assertEqual(
            subsegment_offsets.offsets,
            [
                0,  # Start at offset 0.
                16,  # tensor[0] has size CONSTANT_TENSOR_ALIGNMENT. No padding required.
            ],
        )

        # Check constant_buffer is empty, because the data was moved into the segment.
        self.assertEqual(len(program_with_segments.constant_buffer), 0)

        # The first segment should begin at zero; i.e., at the segment base
        # offset.
        self.assertEqual(segment_table[0].offset, 0, f"{segment_table}")
        # The final segment should not point past the end of the file.
        self.assertLessEqual(
            segment_table[-1].offset + segment_table[-1].size,
            len(pte_data),
            f"{segment_table}",
        )

        # Check the segment base offset boundary.
        segment_base_offset = eh.segment_base_offset
        self.assertEqual(
            pte_data[segment_base_offset - 2 : segment_base_offset + 3],
            # Padding before the first segment.
            b"\x00\x00"
            # First few bytes of the first segment.
            + b"\x10\x11\x11",
        )

        # Now that we've shown that the base offset is correct, slice off the
        # front so that all segment offsets are relative to zero.
        segment_data: bytes = pte_data[segment_base_offset:]

        # Check segment[0] for constants.
        offsets = subsegment_offsets.offsets
        # Check tensor[0]: padding at the end.
        self.assertEqual(
            segment_data[0 : offsets[1]],
            # Tensor data.
            b"\x10\x11\x11\x11\x11\x11\x11\x01"
            # Padding.
            + b"\x00\x00\x00\x00\x00\x00\x00\x00",
        )

        # Check tensor[1]: padding at CONSTANT_TENSOR_ALIGNMENT.
        self.assertEqual(
            segment_data[
                offsets[1]
                + CONSTANT_TENSOR_ALIGNMENT
                - 3 : offsets[1]
                + CONSTANT_TENSOR_ALIGNMENT
                + 3
            ],
            # Tensor data.
            b"\x22\x22\x22"
            # Padding.
            + b"\x02\x00\x00",
        )

        # Check segment[0] and segment[1] border.
        self.assertEqual(
            segment_data[segment_table[1].offset - 3 : segment_table[1].offset + 3],
            # Padding for segment[0].
            b"\x00\x00\x00"
            # Start of segment[1].
            + b"\x30\x33\x33",
        )

        # Check segment[1] and segment[2] border.
        self.assertEqual(
            segment_data[segment_table[2].offset - 3 : segment_table[2].offset + 3],
            # Padding for segment[1].
            b"\x00\x00\x00"
            # Start of segment[2].
            + b"\x40\x44\x44",
        )

        # Convert back.
        program2 = deserialize_pte_binary(pte_data)
        # Programs are the same besides constant_buffer, as deserialization
        # does not preserve constant segment; padding may be added
        # during serialization.
        self.assertEqual(program2.execution_plan, program.execution_plan)
        # Number of constant tensors should be the same.
        self.assertEqual(len(program2.constant_buffer), len(program.constant_buffer))


# Common data for extended header tests. The two example values should produce
# the example data.
EXAMPLE_PROGRAM_SIZE: int = 0x1122112233443344
EXAMPLE_SEGMENT_BASE_OFFSET: int = 0x5566556677887788
# This data is intentionally fragile. If the header layout or magic changes,
# this test must change too. The layout of the header is a contract, not an
# implementation detail.
EXAMPLE_HEADER_DATA: bytes = (
    # Magic bytes
    b"eh00"
    # uint32_t header size (little endian)
    + b"\x18\x00\x00\x00"
    # uint64_t program size
    + b"\x44\x33\x44\x33\x22\x11\x22\x11"
    # uint64_t segment base offset
    + b"\x88\x77\x88\x77\x66\x55\x66\x55"
)


class TestExtendedHeader(unittest.TestCase):
    def test_to_bytes(self) -> None:
        eh = _ExtendedHeader(
            program_size=EXAMPLE_PROGRAM_SIZE,
            segment_base_offset=EXAMPLE_SEGMENT_BASE_OFFSET,
        )
        self.assertTrue(eh.is_valid())
        self.assertEqual(eh.to_bytes(), EXAMPLE_HEADER_DATA)

    def test_to_bytes_with_non_defaults(self) -> None:
        eh = _ExtendedHeader(
            program_size=EXAMPLE_PROGRAM_SIZE,
            segment_base_offset=EXAMPLE_SEGMENT_BASE_OFFSET,
            # Override the default magic and length, to demonstrate that this
            # does not affect the serialized header.
            magic=b"ABCD",
            length=0xAABBCCDD,
        )
        # No longer counts as valid.
        self.assertFalse(eh.is_valid())

        # But still produces a valid output header, since to_bytes() ignores
        # magic and length.
        self.assertEqual(eh.to_bytes(), EXAMPLE_HEADER_DATA)

    def test_from_bytes_valid(self) -> None:
        # Parse the serialized extended header.
        eh = _ExtendedHeader.from_bytes(EXAMPLE_HEADER_DATA)

        # This is a valid header: good magic and length.
        self.assertTrue(eh.is_valid())

        self.assertEqual(eh.magic, _ExtendedHeader.EXPECTED_MAGIC)
        self.assertEqual(eh.length, _ExtendedHeader.EXPECTED_LENGTH)
        self.assertEqual(eh.program_size, EXAMPLE_PROGRAM_SIZE)
        self.assertEqual(eh.segment_base_offset, EXAMPLE_SEGMENT_BASE_OFFSET)

    def test_from_bytes_with_more_data_than_necessary(self) -> None:
        # Pass in more data than necessary to parse the header.
        header_data_with_suffix = EXAMPLE_HEADER_DATA + b"\x55" * 16
        eh = _ExtendedHeader.from_bytes(header_data_with_suffix)

        # This is a valid header: good magic and length.
        self.assertTrue(eh.is_valid())

        self.assertEqual(eh.magic, _ExtendedHeader.EXPECTED_MAGIC)
        self.assertEqual(eh.length, _ExtendedHeader.EXPECTED_LENGTH)
        self.assertEqual(eh.program_size, EXAMPLE_PROGRAM_SIZE)
        self.assertEqual(eh.segment_base_offset, EXAMPLE_SEGMENT_BASE_OFFSET)

    def test_from_bytes_larger_than_needed_header_size_field(self) -> None:
        # Simulate a backwards-compatibility situation. Parse a header
        # with a larger-than expected size. This would typically mean that
        # there are additional fields that we don't know about, but we will
        # ignore them.
        input_data: bytes = (
            # Magic bytes
            b"eh00"
            # uint32_t header size (little endian)
            + b"\x1c\x00\x00\x00"  # Longer than expected
            # uint64_t program size
            + b"\x44\x33\x44\x33\x22\x11\x22\x11"
            # uint64_t segment base offset
            + b"\x88\x77\x88\x77\x66\x55\x66\x55"
            # uint32_t new field (ignored)
            + b"\xff\xee\xff\xee"
        )

        # Parse the serialized extended header.
        eh = _ExtendedHeader.from_bytes(input_data)

        # Header is valid despite having a larger than expected size.
        self.assertTrue(eh.is_valid())

        self.assertEqual(eh.magic, _ExtendedHeader.EXPECTED_MAGIC)
        self.assertEqual(eh.length, 28)
        self.assertEqual(eh.program_size, EXAMPLE_PROGRAM_SIZE)
        self.assertEqual(eh.segment_base_offset, EXAMPLE_SEGMENT_BASE_OFFSET)

    def test_from_bytes_not_enough_data_fails(self) -> None:
        # Parsing a truncated prefix should fail.
        with self.assertRaises(ValueError):
            _ExtendedHeader.from_bytes(EXAMPLE_HEADER_DATA[:16])

    def test_from_bytes_invalid_magic(self) -> None:
        # An invalid serialized header
        input_data: bytes = (
            # Magic bytes
            b"ABCD"  # Invalid
            # uint32_t header size (little endian)
            + b"\x18\x00\x00\x00"
            # uint64_t program size
            + b"\x44\x33\x44\x33\x22\x11\x22\x11"
            # uint64_t segment base offset
            + b"\x88\x77\x88\x77\x66\x55\x66\x55"
        )

        # Parse the serialized extended header.
        eh = _ExtendedHeader.from_bytes(input_data)

        # Bad magic makes this invalid
        self.assertFalse(eh.is_valid())

        # But it still parsed out the fields, so that callers can
        # see what went wrong.
        self.assertEqual(eh.magic, b"ABCD")
        self.assertEqual(eh.length, _ExtendedHeader.EXPECTED_LENGTH)
        self.assertEqual(eh.program_size, EXAMPLE_PROGRAM_SIZE)
        self.assertEqual(eh.segment_base_offset, EXAMPLE_SEGMENT_BASE_OFFSET)

    def test_from_bytes_invalid_length(self) -> None:
        # An invalid serialized header
        input_data: bytes = (
            # Magic bytes
            b"eh00"
            # uint32_t header size (little endian)
            + b"\x10\x00\x00\x00"  # Too short
            # uint64_t program size
            + b"\x44\x33\x44\x33\x22\x11\x22\x11"
            # uint64_t segment base offset
            + b"\x88\x77\x88\x77\x66\x55\x66\x55"
        )

        # Parse the serialized extended header.
        eh = _ExtendedHeader.from_bytes(input_data)

        # Bad header size makes this invalid
        self.assertFalse(eh.is_valid())

        # But it still parsed out the fields, so that callers can
        # see what went wrong.
        self.assertEqual(eh.magic, _ExtendedHeader.EXPECTED_MAGIC)
        self.assertEqual(eh.length, 16)
        self.assertEqual(eh.program_size, EXAMPLE_PROGRAM_SIZE)
        self.assertEqual(eh.segment_base_offset, EXAMPLE_SEGMENT_BASE_OFFSET)
