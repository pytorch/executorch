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

from executorch.exir._serialize._dataclass import _DataclassEncoder, _json_to_dataclass
from executorch.exir._serialize._flatbuffer import (
    _FlatbufferResult,
    _program_flatbuffer_to_json,
    _program_json_to_flatbuffer,
)

from executorch.exir.schema import (
    BackendDelegateDataReference,
    BackendDelegateInlineData,
    DataLocation,
    DataSegment,
    Program,
)


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


def _padding_required(offset: int, alignment: int) -> int:
    """Returns the padding required to align `offset` to `alignment`."""
    remainder: int = offset % alignment
    if remainder != 0:
        return alignment - remainder
    return 0


def _aligned_size(input_size: int, alignment: int) -> int:
    """Returns input_size padded up to the next whole multiple of alignment."""
    return input_size + _padding_required(input_size, alignment)


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


def _pad_to(data: bytes, length: int) -> bytes:
    """Returns the input followed by enough zero bytes to become the requested length.

    Args:
        data: The data to pad.
        length: The length of the returned data.
    Returns:
        The padded data.
    Raises:
        ValueError: If the requested length is less than the input length.
    """
    if length < len(data):
        raise ValueError(f"Data length {len(data)} > padded length {length}")
    if length > len(data):
        data = data + b"\x00" * (length - len(data))
    assert len(data) == length
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


def _extract_segments(
    program: Program, segment_alignment: int
) -> Tuple[Program, List[bytes]]:
    """Moves data from the Program into a list of segments.

    The returned program is a copy of `program`. Program.segments parallels the
    returned list of buffers.

    Args:
        program: The program to extract segments from.
        segment_alignment: Alignment in bytes. The starting offset of each
            segment will be aligned to this value.
    Returns:
        A tuple of (modified program, list of segment data).
    """
    if program.segments:
        raise ValueError(
            f"Program already has {len(program.segments)} segments: "
            + f"{repr(program.segments)}"
        )

    # Don't modify the original program.
    # TODO(T144120904): Could avoid yet more huge copies with a more shallow
    # copy, reusing the actual data blobs.
    program = copy.deepcopy(program)

    segments: List[bytes] = []
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
                segments.append(inline.data)
                delegate.processed = BackendDelegateDataReference(
                    location=DataLocation.SEGMENT,
                    index=segment_index,
                )

                # Update the segment list in the root Program object.
                prev_end = (
                    program.segments[-1].offset + program.segments[-1].size
                    if program.segments
                    else 0
                )
                program.segments.append(
                    DataSegment(
                        offset=_aligned_size(prev_end, segment_alignment),
                        size=len(inline.data),
                    ),
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

    return (program, segments)


def _append_segments(
    program_data: bytes,
    segments: List[bytes],
    alignment: int,
    segment_table: List[DataSegment],
    base_offset: int,
) -> bytes:
    """Appends segments to the end of the program data.

    Appends each element of `segments` to `program_data`, with '\0' padding to
    ensure that the offset of each segment is aligned to `alignment`.

    Args:
        program_data: The flatbuffer-serialized Program.
        segments: The list of segments to append to `program_data`.
        alignment: Alignment in bytes. The starting offset of each
            segment will be aligned to this value in the output data.
        segment_table: The expected offsets and sizes of each element in
            `segments`. This is typically `program.segments`. Must have the
            same length as `segments`.
        base_offset: The expected segment base offset from the extended header.
            Should point to the aligned offset following the end of
            `program_data`.
    Returns:
        A copy of `program_data` with the segment data and padding appended.
        If there are no segments, returns `program_data` directly.
    Raises:
        ValueError: If the length of `segments` doesn't match the length of
            `segment_table`.
    """
    if len(segments) != len(segment_table):
        raise ValueError(
            f"Segments length {len(segments)} does not match "
            + f"segment_table length {len(segment_table)}"
        )
    if not segments:
        return program_data

    # The pieces that will be concatenated to create the output data.
    # `program_data` will be its first element.
    padded_segments: List[bytes] = []
    # Length of all elements in padded_segments. Only used for assertions.
    current_offset: int = 0
    for i, segment in enumerate([program_data] + segments):
        # Add padding if necessary to align the start of this segment.
        pad_length: int = _padding_required(current_offset, alignment)
        if pad_length > 0:
            padded_segments.append(b"\x00" * pad_length)
            current_offset += pad_length

        # Make sure that we're about to add this segment to the offset that
        # agrees with program.segments. Skip the first entry, which is the
        # Program itself and isn't included in program.segments.
        if i == 1:
            # The first real segment should start at the base offset.
            assert current_offset == base_offset, (
                f"Offset of first segment {current_offset} "
                + f"!= base_offset {base_offset}"
            )
        if i > 0:
            # Adding a real segment, not `program_data`.
            expected_segment = segment_table[i - 1]
            expected_offset = base_offset + expected_segment.offset
            assert current_offset == expected_offset, (
                f"Segment {i} offset {current_offset} "
                + f"!= expected offset {expected_offset} "
                + f"(base {base_offset} + {expected_segment.offset}) "
            )
            assert expected_segment.size == len(segment), (
                f"Segment {i} size {len(segment)} "
                + f"!= expected size {expected_segment.size}"
            )

        # Add the payload. If this is the final segment, it does not need
        # padding after it.
        padded_segments.append(segment)
        current_offset += len(segment)

    # Use join() instead of appending to avoid O(n) reallocation of these
    # potentially-large buffers.
    return b"".join(padded_segments)


def serialize_pte_binary(
    program: Program,
    *,
    extract_segments: bool = False,
    segment_alignment: int = 4096,
    constant_tensor_alignment: Optional[int] = None,
    delegate_alignment: Optional[int] = None,
) -> bytes:
    """Returns the runtime binary representation of the given Program.

    Args:
        program: The Program to serialize.
        extract_segments: Whether to move certain data blobs from the Program
            into separate segments, rather than encoding those blobs in the
            flatbuffer data. When true, will also:
            - Add an extended header to the output, containing the program size
              and the starting segment offset.
            - Update the Program.segments field with the offsets and lengths
              of each segment.
        segment_alignment: Alignment in bytes. The starting offset of each
            segment will be aligned to this value in the output data.
        constant_tensor_alignment: If provided, the minimum alignment of tensor
            buffers in the program. Must be a power of 2. If not provided, uses
            the value in the schema file.
        delegate_alignment: If provided, the minimum alignment of delegate data
            in the program. Must be a power of 2. If not provided, uses the
            value in the schema file.
    Returns:
        The serialized form of the Program, ready for execution by the runtime.
    """
    # Segment data to be written to the file following the flatbuffer data.
    segments: List[bytes] = []
    if extract_segments:
        # May return a copy of the program to avoid modifying the input.
        program, segments = _extract_segments(
            program=program, segment_alignment=segment_alignment
        )

    # Convert to a standard flatbuffer binary.
    result: _FlatbufferResult = _program_json_to_flatbuffer(
        _program_to_json(program),
        constant_tensor_alignment=constant_tensor_alignment,
        delegate_alignment=delegate_alignment,
    )
    if not extract_segments:
        return result.data

    # Size of the header to insert. Its size is padded to the largest
    # force_align value present in the schema.
    padded_header_length: int = _aligned_size(
        input_size=_ExtendedHeader.EXPECTED_LENGTH,
        alignment=result.max_alignment,
    )
    # Size of the program with the header inserted.
    program_size: int = padded_header_length + len(result.data)
    # Offset to the first segment, or zero if there are no segments.
    segment_base_offset: int = (
        _aligned_size(input_size=program_size, alignment=segment_alignment)
        if segments
        else 0
    )

    # Construct and pad the extended header.
    header_data: bytes = _ExtendedHeader(
        program_size=program_size, segment_base_offset=segment_base_offset
    ).to_bytes()
    header_data = _pad_to(header_data, padded_header_length)

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

    if segments:
        # Add segments to the end of the data, in order, with the appropriate
        # padding.
        program_data = _append_segments(
            program_data=program_data,
            segments=segments,
            alignment=segment_alignment,
            segment_table=program.segments,
            base_offset=segment_base_offset,
        )

    return program_data


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
