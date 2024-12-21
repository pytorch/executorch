# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import copy
import json
import re

from dataclasses import dataclass
from typing import ClassVar, List, Literal, Optional, Tuple

from executorch.exir._serialize._cord import Cord
from executorch.exir._serialize._dataclass import _DataclassEncoder, _json_to_dataclass
from executorch.exir._serialize._flatbuffer import (
    _FlatbufferResult,
    _program_flatbuffer_to_json,
    _program_json_to_flatbuffer,
)

from executorch.exir._serialize.padding import aligned_size, pad_to, padding_required

from executorch.exir.schema import (
    BackendDelegateDataReference,
    BackendDelegateInlineData,
    Buffer,
    DataLocation,
    DataSegment,
    Program,
    SubsegmentOffsets,
)
from executorch.exir.tensor import ALIGNMENT


# Byte order of numbers written to program headers. Always little-endian
# regardless of the host system, since all commonly-used modern CPUs are little
# endian.
_HEADER_BYTEORDER: Literal["little"] = "little"


def _program_to_json(program: Program) -> str:
    """Returns the JSON representation of the given Program."""
    return json.dumps(program, cls=_DataclassEncoder)


def _json_to_program(program_json: bytes) -> Program:
    """Returns a Program deserialized from the given JSON string."""
    # construct program class recursively from dict
    return _json_to_dataclass(json.loads(program_json), cls=Program)


def _insert_flatbuffer_header(
    flatbuffer_data: bytes, magic_regex: str, header_data: bytes
) -> bytes:
    """Inserts a header just after the magic string of the provided flatbuffer data.

    Args:
        flatbuffer_data: The input data to modify.
        magic_regex: A regex pattern that must match the magic file_identifier
            characters of flatbuffer_data.
        header_data: The data to insert into flatbuffer_data. To ensure that
            flatbuffer internal alignment is preserved, the caller must
            guaranteed that its length is a power of 2 >= the largest
            force_align value in the schema.
    Returns:
        The modified flatbuffer_data with header_data inserted.
    Raises:
        ValueError: If flatbuffer_data is too short to be valid.
        ValueError: If the magic bytes of flatbuffer_data does not match
            magic_regex.
    """
    # The binary flatbuffer file should begin with:
    # - Offset in bytes to root table (4 bytes little endian)
    # - file_identifier string from the schema (4 bytes, string order)
    if len(flatbuffer_data) < 8:
        raise ValueError(f"Flatbuffer data length {len(flatbuffer_data)} < 8")

    # Ensure that the magic matches.
    actual_magic: str = flatbuffer_data[4:8].decode(errors="replace")
    if not re.match(magic_regex, actual_magic):
        raise ValueError(
            f"Flatbuffer data magic bytes {repr(actual_magic)} "
            + f"does not match pattern /{magic_regex}/"
        )

    # Avoid a potentially big allocation/copy if there's nothing to do.
    if len(header_data) == 0:
        return flatbuffer_data

    # We will need to adjust the root object offset after inserting the header.
    root_offset = int.from_bytes(flatbuffer_data[0:4], byteorder=_HEADER_BYTEORDER)

    return (
        # New root offset.
        (root_offset + len(header_data)).to_bytes(4, byteorder=_HEADER_BYTEORDER)
        # Existing magic bytes.
        + flatbuffer_data[4:8]
        # Provided header + padding.
        + header_data
        # Remainder of the file. Note that this can be O(10MB to 100MB), so it
        # can trigger a large allocation + copy.
        + flatbuffer_data[8:]
    )


@dataclass
class _ExtendedHeader:
    # Class constants

    # The magic bytes that should be at the beginning of the header.
    EXPECTED_MAGIC: ClassVar[bytes] = b"eh00"
    # The length of the header in bytes.
    EXPECTED_LENGTH: ClassVar[int] = (
        # Header magic
        4
        # Header length
        + 4
        # Flatbuffer data size
        + 8
        # Segment base offset
        + 8
    )

    # Instance attributes. @dataclass will turn these into ctor args.

    # The size of the serialized program data in bytes.
    program_size: int
    # Offset to the start of the first segment, or zero if there
    # are no segments.
    segment_base_offset: int

    # The magic bytes read from or to be written to the binary header.
    magic: bytes = EXPECTED_MAGIC
    # The header length, in bytes, read from or to be written to the binary
    # header.
    length: int = EXPECTED_LENGTH

    @staticmethod
    def from_bytes(data: bytes) -> "_ExtendedHeader":
        """Tries to read an extended header from the provided data.

        Does not validate that the header is well-formed. Callers should
        use is_valid().

        Args:
            data: The data to read from.
        Returns:
            The contents of the extended header.
        Raises:
            ValueError: If not enough data is provided.
        """
        if len(data) < _ExtendedHeader.EXPECTED_LENGTH:
            raise ValueError(
                f"Not enough data for extended header: {len(data)} "
                + f"< {_ExtendedHeader.EXPECTED_LENGTH}"
            )

        return _ExtendedHeader(
            magic=data[0:4],
            length=int.from_bytes(data[4:8], byteorder=_HEADER_BYTEORDER),
            program_size=int.from_bytes(data[8:16], byteorder=_HEADER_BYTEORDER),
            segment_base_offset=int.from_bytes(
                data[16:24], byteorder=_HEADER_BYTEORDER
            ),
        )

    def is_valid(self) -> bool:
        """Returns true if the extended header appears to be well-formed."""
        return (
            self.magic == _ExtendedHeader.EXPECTED_MAGIC
            and self.length >= _ExtendedHeader.EXPECTED_LENGTH
        )

    def to_bytes(self) -> bytes:
        """Returns the binary representation of the extended header.

        Note that this will ignore self.magic and self.length and will always
        write the proper magic/length.
        """
        data: bytes = (
            # Extended header magic. This lets consumers detect whether the
            # header was inserted or not. Always use the proper magic value
            # (i.e., ignore self.magic) since there's no reason to create an
            # invalid header.
            self.EXPECTED_MAGIC
            # uint32_t: Size of this header. This makes it easier to add new
            # fields to this header in the future. Always use the proper size
            # (i.e., ignore self.length) since there's no reason to create an
            # invalid header.
            + self.EXPECTED_LENGTH.to_bytes(4, byteorder=_HEADER_BYTEORDER)
            # uint64_t: Size of the flatbuffer data, including this header.
            + self.program_size.to_bytes(8, byteorder=_HEADER_BYTEORDER)
            # uint64_t: Offset to the start of the first segment, or zero if
            # there are no segments.
            + self.segment_base_offset.to_bytes(8, byteorder=_HEADER_BYTEORDER)
        )
        return data


def _get_extended_header(program_data: bytes) -> Optional[_ExtendedHeader]:
    """Returns the extended header of the program data, if present and valid."""
    try:
        eh = _ExtendedHeader.from_bytes(program_data[8:])
        if eh.is_valid():
            return eh
    except ValueError:
        pass
    return None


def _extract_delegate_segments(
    program: Program,
    segments: List[Cord],
) -> None:
    """Extracts the delegate segments inlined in the program into a list of buffers.
        The program is modified in-place to remove the delegate data.

    Args:
        program: The program to extract segments from. Modified in-place.
        segments: A list of buffers to append extracted segments to. Modified in-place.
    """
    remaining_inline: List[BackendDelegateInlineData] = []
    inline_indices_seen: set[int] = set()
    for plan in program.execution_plan:
        for delegate in plan.delegates:
            if delegate.processed.location != DataLocation.INLINE:
                raise ValueError(
                    "Program must only contain inline delegate data, "
                    + f"saw {repr(delegate)}"
                )
            # TODO(T144120904): Don't extract small blobs into segments;
            # have a cutoff. Or callers could provide a callback that
            # returns true/false for a given BackendDelegate, letting them
            # use their own logic.
            try:
                inline: BackendDelegateInlineData = program.backend_delegate_data[
                    delegate.processed.index
                ]
            except IndexError:
                raise ValueError(
                    f"Delegate processed index {delegate.processed.index} "
                    + ">= len(Program.backend_delegate_data) "
                    + f"{len(program.backend_delegate_data)} "
                    + f"in {repr(delegate)}"
                )
            inline_indices_seen.add(delegate.processed.index)
            if inline.data:
                # Move the delegate data out of the program.
                segment_index = len(segments)
                segments.append(Cord(inline.data))
                delegate.processed = BackendDelegateDataReference(
                    location=DataLocation.SEGMENT,
                    index=segment_index,
                )
            else:
                # Not moving into a segment. Keep it inline, but update the
                # index.
                new_index = len(remaining_inline)
                remaining_inline.append(inline)
                delegate.processed.index = new_index

    # Make sure we visited all entries in backend_delegate_data, so that it's
    # safe to overwrite it.
    remaining_indices: set[int] = set(
        range(len(program.backend_delegate_data))
    ).difference(inline_indices_seen)
    if remaining_indices:
        raise ValueError(
            "Did not handle all elements of backend_delegate_data; "
            + f"remaining: {remaining_indices}"
        )

    # Preserve any entries that were not moved into segments.
    program.backend_delegate_data = remaining_inline


def _extract_constant_segment(
    constant_buffer: List[Buffer],
    tensor_alignment: Optional[int] = None,
) -> Tuple[Cord, List[int]]:
    """Copies the tensors from the provided list into a Cord and tracks the offsets
        of each tensor.

    Args:
        constant_buffer: list of Buffers from which to extract constants from. Not modified.
        tensor_alignment: Alignment in bytes. Each tensor in the cord will be padded to align
            with this value. Defaults to ALIGNMENT.

    Returns:
        A tuple of (constant segment, list of offsets for each tensor in the segment)
    """
    constant_segment_data: Cord = Cord()
    constant_segment_offsets: List[int] = []
    current_offset: int = 0
    for i in range(len(constant_buffer)):
        buffer = constant_buffer[i]
        constant_segment_data.append(buffer.storage)
        buffer_length = len(buffer.storage)
        pad_length = (
            padding_required(buffer_length, tensor_alignment)
            if tensor_alignment is not None
            else 0
        )
        if i < len(constant_buffer) - 1:
            constant_segment_data.append(b"\x00" * pad_length)
        constant_segment_offsets.append(current_offset)
        current_offset += buffer_length + pad_length

    return constant_segment_data, constant_segment_offsets


def serialize_pte_binary(
    program: Program,
    *,
    mutable_data: Optional[List[Buffer]] = None,
    extract_delegate_segments: bool = False,
    segment_alignment: int = 128,
    constant_tensor_alignment: Optional[int] = None,
    delegate_alignment: Optional[int] = None,
) -> Cord:
    """Returns the runtime binary representation of the given Program.

    Args:
        program: The Program to serialize.
        extract_delegate_segments: Whether to move delegate data blobs from the
            Program into separate segments, rather than encoding those blobs
            in the flatbuffer data. When true, will also:
            - Add an extended header to the output, containing the program size
              and the starting segment offset.
            - Update the Program.segments field with the offsets and lengths
              of each segment.
        segment_alignment: Alignment in bytes. The starting offset of each
            segment will be aligned to this value in the output data.
        constant_tensor_alignment: The minimum alignment of tensor
            buffers in the program. Must be a power of 2. Defaults to ALIGNMENT.
        delegate_alignment: If provided, the minimum alignment of delegate data
            in the program. Must be a power of 2. If not provided, uses the
            value in the schema file.
    Returns:
        The serialized form of the Program, ready for execution by the runtime.
    """
    # Default tensor alignment.
    if constant_tensor_alignment is None:
        constant_tensor_alignment = ALIGNMENT

    # Don't modify the original program.
    # TODO(T144120904): Could avoid yet more huge copies with a more shallow
    # copy, reusing the actual data blobs.
    program = copy.deepcopy(program)

    # Store extracted segment data; this may be constant data or delegate data.
    segments: List[Cord] = []

    constant_segment_data, constant_segment_offsets = _extract_constant_segment(
        program.constant_buffer, tensor_alignment=constant_tensor_alignment
    )

    # If there are no constants, len(constant_segment_data) = 0. However, there may
    # be non-constants, in which case len(constant_segment_offsets) = 1, containing
    # the placeholder value 0. Ensure the placeholder value is put into
    # program.constant_segment.offsets.
    if len(constant_segment_offsets) > 0:
        # Update program.constant_segment with constant subsegment offset information.
        program.constant_segment = SubsegmentOffsets(
            segment_index=len(segments), offsets=constant_segment_offsets
        )
        # Clear the constant buffer, as constant data will be stored in segments.
        program.constant_buffer = []
        # Add to the aggregate segments cord.
        segments.append(constant_segment_data)

    if mutable_data is not None:
        mutable_segment_data, mutable_segment_offsets = _extract_constant_segment(
            mutable_data,
            tensor_alignment=None,  # data is copied at Method load so no need to align.
        )
        if len(mutable_segment_data) > 0:
            # Update program.mutable_segment_data with constant subsegment offset information.
            program.mutable_data_segments = [
                SubsegmentOffsets(
                    segment_index=len(segments), offsets=mutable_segment_offsets
                ),
            ]
            # Add to the aggregate segments cord.
            segments.append(mutable_segment_data)

    if extract_delegate_segments:
        _extract_delegate_segments(program, segments)

    # Append all segments into a single Cord, adding any necessary padding to ensure that
    # each segment begins at the required alignment.
    # Update program.segments with the offsets to each segment.
    segments_data = Cord()
    for data in segments:
        prev_end = (
            (program.segments[-1].offset + program.segments[-1].size)
            if program.segments
            else 0
        )
        program.segments.append(
            DataSegment(
                offset=aligned_size(prev_end, segment_alignment), size=len(data)
            )
        )
        # Add to aggregate segments cord with padding.
        padding_length = padding_required(len(segments_data), segment_alignment)
        if padding_length > 0:
            segments_data.append(b"\x00" * padding_length)
        segments_data.append(data)

    # Convert to a standard flatbuffer binary.
    result: _FlatbufferResult = _program_json_to_flatbuffer(
        _program_to_json(program),
        constant_tensor_alignment=constant_tensor_alignment,
        delegate_alignment=delegate_alignment,
    )

    # If there are no segments present, do not insert the extended header.
    if len(segments_data) == 0:
        return Cord(result.data)

    # Size of the header to insert. Its size is padded to the largest
    # force_align value present in the schema.
    padded_header_length: int = aligned_size(
        input_size=_ExtendedHeader.EXPECTED_LENGTH,
        alignment=result.max_alignment,
    )
    # Size of the program with the header inserted.
    program_size: int = padded_header_length + len(result.data)
    # Offset to the first segment, or zero if there are no segments.
    segment_base_offset: int = (
        aligned_size(input_size=program_size, alignment=segment_alignment)
        if len(segments_data) > 0
        else 0
    )

    # Construct and pad the extended header.
    header_data: bytes = _ExtendedHeader(
        program_size=program_size, segment_base_offset=segment_base_offset
    ).to_bytes()
    header_data = pad_to(header_data, padded_header_length)

    # Insert the header into the flatbuffer data.
    program_data: bytes = _insert_flatbuffer_header(
        flatbuffer_data=result.data,
        magic_regex=r"ET[0-9a-zA-Z][0-9a-zA-Z]",
        header_data=header_data,
    )
    assert len(program_data) == program_size

    # Potentially large. Try to free it as soon as we can.
    del result.data

    # Double-check that the extended header is in the right place and has the
    # right contents.
    eh = _get_extended_header(program_data)
    assert eh is not None
    assert eh.program_size == program_size
    assert eh.segment_base_offset == segment_base_offset

    # Construct the final pte file containing:
    # - program data; written to offset 0.
    # - segments data (optional); aligned to segment_alignment.
    pte_data = Cord(program_data)
    if len(segments_data) > 0:
        padding_length = padding_required(len(pte_data), segment_alignment)
        pte_data.append(b"\x00" * padding_length)
        # The first segment after program data should start at the segment base offset.
        assert (
            len(pte_data) == segment_base_offset
        ), f"Offset of first segment {len(pte_data)} != segment base offset {segment_base_offset}"
        pte_data.append(segments_data)
    return pte_data


def _restore_segments(program: Program, segment_data: bytes) -> Program:
    """Moves segments from `segment_data` into `program`.

    This should recreate the original Program that the segments were extracted
    from.

    Args:
        program: The Program to restore. `program.segments` must describe the
            segment locations.
        segment_data: The data containing the segments. Assumes that this data
            begins at `segment_base_offset` from the extended header: i.e.,
            the preceding data has been stripped off so that the first segment
            begins at offset zero.
    Returns:
        The Program with segments restored.
    """
    # Extract the list of segment data blobs, which parallel program.segments.
    segments: List[bytes] = []
    for i, segment in enumerate(program.segments):
        if segment.offset + segment.size > len(segment_data):
            raise ValueError(
                f"Segment {i} {segment} overflows data length {len(segment_data)}"
            )
        segments.append(segment_data[segment.offset : segment.offset + segment.size])

    # Find and replace the Program's references to these segments, inlining the
    # data.
    for plan_index, plan in enumerate(program.execution_plan):
        for delegate_index, delegate in enumerate(plan.delegates):
            if delegate.processed.location == DataLocation.INLINE:
                continue
            assert delegate.processed.location == DataLocation.SEGMENT
            index = delegate.processed.index
            if index >= len(segments):
                raise ValueError(
                    f"Plan {plan_index} delegate {delegate_index} "
                    + f"segment index {index} >= num segments {len(segments)}"
                )

            data_index: int = len(program.backend_delegate_data)
            program.backend_delegate_data.append(
                BackendDelegateInlineData(data=segments[index])
            )
            delegate.processed = BackendDelegateDataReference(
                location=DataLocation.INLINE, index=data_index
            )

    # Replace constants from constant_segment into constant_buffer.
    if program.constant_segment and len(program.constant_segment.offsets) > 0:
        buffers: List[Buffer] = []
        constant_segment = segments[program.constant_segment.segment_index]
        for i in range(len(program.constant_segment.offsets)):
            start_offset = program.constant_segment.offsets[i]
            # Note: this is the original end offset plus any padding between
            # it and the next start offset.
            end_offset = (
                program.constant_segment.offsets[i + 1]
                if i < len(program.constant_segment.offsets) - 1
                else len(constant_segment)
            )
            buffers.append(Buffer(storage=constant_segment[start_offset:end_offset]))
        program.constant_buffer = buffers
        program.constant_segment.segment_index = 0
        program.constant_segment.offsets = []

    # Clear out the segments list since the original Program didn't have one.
    program.segments = []
    return program


def deserialize_pte_binary(program_data: bytes) -> Program:
    """Returns a Program deserialized from the given runtime binary data."""
    program_size = len(program_data)
    segment_base_offset = 0

    # Look for an extended header to see if segments follow the flatbuffer
    # data.
    eh: Optional[_ExtendedHeader] = _get_extended_header(program_data)
    if eh and eh.is_valid():
        program_size = eh.program_size
        segment_base_offset = eh.segment_base_offset

    # Parse the flatbuffer data.
    program: Program = _json_to_program(
        _program_flatbuffer_to_json(program_data[:program_size])
    )

    if segment_base_offset != 0:
        # Move segment data back into the Program.
        program = _restore_segments(
            program=program, segment_data=program_data[segment_base_offset:]
        )

    return program
