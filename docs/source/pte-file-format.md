# `.pte` file format

ExecuTorch `.pte` program files are serialized as modified binary flatbuffer
files with optional data segments appended.

```
             ┌───────────────────────────────────┐
             │Standard flatbuffer header         │
             ├───────────────────────────────────┤
Optional ──> │ExecuTorch extended header         │
             ├───────────────────────────────────┤
             │Flatbuffer-serialized program data │
             │                                   │
             │                                   │
          ┌─ ├───────────────────────────────────┤
          │  │Padding                            │
          │  ├───────────────────────────────────┤
          │  │Segment data                       │
          │  │                                   │
          │  │                                   │
          │  ├───────────────────────────────────┤
          │  │Padding                            │
Optional ─┤  ├───────────────────────────────────┤
          │  │Segment data                       │
          │  │                                   │
          │  │                                   │
          │  ├───────────────────────────────────┤
          │  │Padding                            │
          │  ├───────────────────────────────────┤
          │  │...                                │
          └─ └───────────────────────────────────┘
```

## Compatibility

See the [Runtime Compatibility Policy](
https://github.com/pytorch/executorch/tree/main/runtime/COMPATIBILITY.md) for
details about the compatibility guarantees between the `.pte` format and the
ExecuTorch runtime.

## Headers

Program files can be recognized by the magic string at byte offset 4, beginning
with `ET` and followed by two ASCII decimal digits.

Program files may have an optional extended header at byte offset 8, recognized
by the magic string beginning with `eh` and followed by two ASCII decimal
digits. This header includes the size of the flatbuffer-encoded core program
data, and the starting offset of the segments that may follow the program data.
Note that this header is ExecuTorch-specific, but even when present it does not
upset most flatbuffer-parsing code (apart from the rarely-used
`GetBufferStartFromRootPointer()`).

All numbers are little-endian, regardless of the host system.

Header layout:
```
[0..3] uint32_t byte offset to the beginning of the flatbuffer root table.
[4..7] File magic bytes: "ET" followed by two ASCII decimal digits. The digits
       will change if the binary format of this file is changed in a
       non-backwards-compatible way.
Optional extended header:
|  [8..11] Extended header magic bytes: "eh" followed by two ASCII decimal
|          digits. The digits will change if the binary format of this header is
|          changed in a non-backwards-compatible way.
| [12..15] uint32_t size of this extended header in bytes, including the magic
|          header and this size field. Fields can be added to this header in
|          the future by increasing this size. This size does not include any
|          padding that may follow the header.
| [16..23] uint64_t size of the flatbuffer-encoded program data, starting from
|          byte offset zero above. I.e., it includes these headers.
| [24..31] uint64_t offset (from byte offset zero above) to the start of the
|          first segment, or zero if there are no segments.
|  [31..?] Any zero-padding necessary to preserve the alignment of the data
|          that follows.
End of optional extended header.
```

Example:
```
        Offset to flatbuffer root (0x38)
        |            File magic ("ET??")
        |            |            Extended header magic ("eh??")
        |            |            |            Extended header size (0x18)
        vvvvvvvvvvv  vvvvvvvvvvv  vvvvvvvvvvv  vvvvvvvvvvv
0x0000  38 00 00 00  45 54 3F 3F  65 68 3F 3F  18 00 00 00
0x0010  F0 02 00 00  00 00 00 00  00 10 00 00  00 00 00 00
        ^^^^^^^^^^^^^^^^^^^^^^^^  ^^^^^^^^^^^^^^^^^^^^^^^^
        |                         Offset to segments (0x1000)
        Size of program flatbuffer data (0x2f0)
```

## Program data

See `//executorch/schema/program.fbs` for the Program flatbuffer schema.

The flatbuffer-encoded program data follows the headers. By embedding the size
of this region in the extended header, clients can read only the program data
without reading in segment data. This is useful because program data typically
sticks around for the lifetime of a model, while the large segment data is often
freeable after model initialization.

## Segment data

The first segment starts at the offset embedded in the extended header.
Segments are typically aligned to 4096 or some other power of 2 that matches
the target system's memory page size. This makes it easier to use `mmap()`
if desired.

The `Program.segments` array in the program data contains size/offset
information about the segments that optionally follow. Offsets in this array are
relative to the segment offet in the extended header.
