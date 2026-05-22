# `.ptd` file format

ExecuTorch `.ptd` files are serialized as modified binary flatbuffer
files with data segments appended. They provide a way to store named data using
the FlatTensor format. Named data can be tensors or opaque blob data (usually for backends that do not expose data format).

Code related to the PTD file format is in the `//executorch/extension/flat_tensor/` directory.

```
             ┌───────────────────────────────────┐
             │Standard flatbuffer header         │
             ├───────────────────────────────────┤
             │ExecuTorch extended header         │
             ├───────────────────────────────────┤
             │Flatbuffer-serialized metadata     │
             │(FlatTensor)                       │
             │                                   │
          ┌─ ├───────────────────────────────────┤
          │  │Padding                            │
          │  ├───────────────────────────────────┤
          │  │Data segment                       │
          │  │                                   │
          │  │                                   │
          │  ├───────────────────────────────────┤
          │  │Padding                            │
   Blobs ─┤  ├───────────────────────────────────┤
          │  │Data segment                       │
          │  │                                   │
          │  │                                   │
          │  ├───────────────────────────────────┤
          │  │Padding                            │
          │  ├───────────────────────────────────┤
          │  │...                                │
          └─ └───────────────────────────────────┘
```

## Compatibility

PTD files are designed for storing named data that can be loaded by ExecuTorch
models.

## Headers

PTD files can be recognized by the magic string at byte offset 4, beginning with `FT`
and followed by two ASCII decimal digits (file identifier from the FlatBuffers schema).

PTD files have an extended header at byte offset 8, recognized by the magic string
`FH01`. This header includes the size and offset information for both the
flatbuffer-serialized metadata and the data segments that follow.

Note that this header is ExecuTorch-specific, but even when present it does not
upset most flatbuffer-parsing code (apart from the rarely-used
`GetBufferStartFromRootPointer()`).

All numbers are little-endian, regardless of the host system.

Header layout:
```
[0..3] uint32_t byte offset to the beginning of the flatbuffer root table.
[4..7] File magic bytes: "FT" followed by two ASCII decimal digits. The digits
       correspond to the FlatBuffers file identifier.
Extended header (always present):
|  [8..11] Extended header magic bytes: "FH01" - FlatTensor Header version 01.
| [12..15] uint32_t size of this extended header in bytes, including the magic
|          header and this size field. Currently fixed at 40 bytes.
| [16..23] uint64_t offset (from byte offset zero) to the start of the
|          flatbuffer data.
| [24..31] uint64_t size of the flatbuffer-encoded tensor metadata in bytes.
| [32..39] uint64_t offset (from byte offset zero) to the start of the first
|          data segment.
| [40..47] uint64_t total size of all data segments in bytes.
End of extended header.
```

Example:
```
        Offset to flatbuffer root (0x44)
        |            File magic ("FT01")
        |            |            Extended header magic ("FH01")
        |            |            |            Extended header size (0x28)
        vvvvvvvvvvv  vvvvvvvvvvv  vvvvvvvvvvv  vvvvvvvvvvv
0x0000  44 00 00 00  46 54 30 31  46 48 30 31  28 00 00 00
0x0010  30 00 00 00  00 00 00 00  00 01 00 00  00 00 00 00
0x0020  30 01 00 00  00 00 00 00  20 00 00 00  00 00 00 00
        ^^^^^^^^^^^^^^^^^^^^^^^^  ^^^^^^^^^^^^^^^^^^^^^^^^
        |                         | Flatbuffer size (0x100)
        |                         | Segment data size (0x20)
        Segment base offset (0x130)
```
Note: this example comes from inspecting the ModuleAddMul.ptd file.
```
python -m test.models.export_program --modules "ModuleAddMul" --external-constants --outdir .

xxd -l 64 ModuleAddMulProgram.ptd
```

## FlatTensor

See `//executorch/extension/flat_tensor/serialize/flat_tensor.fbs` for the
FlatTensor flatbuffer schema.

The flatbuffer-encoded metadata follows the headers and contains:

- **Schema version**: Version information for compatibility.
- **Data segments**: List of segment descriptors with offset and size information.
- **Named data**: List of named data entries, each containing:
  - **Key**: String identifier for the data blob.
  - **Segment index**: Reference to the data segment containing the blob.
  - **Tensor layout**: Optional metadata including scalar type, sizes and dim order, if the data segment contains a tensor.

### Tensor Layout

If a data segment contains a canonical tensor, it may have associated layout information:
- **Scalar type**: Data type (float32, int32, etc.) using ExecuTorch scalar types.
- **Sizes**: Dimensions of the tensor.
- **Dim order**: Memory layout order specifying how dimensions are arranged in memory.

## Data segments

The `FlatTensor.segments` list in the metadata contains offset and size
information about each data segment. Offsets in this list are relative to
the segment base offset specified in the extended header.

Each segment contains:
- **Offset**: Relative offset from the segment base offset.
- **Size**: Size of the valid data in bytes (may be followed by padding).

## Named data access

Tensors are accessed by string keys through the `named_data` list. Each entry
maps a string key to:
1. A segment index pointing to the raw data.
2. Optional tensor layout metadata, if the data segment contains a tensor.

This design allows:
- Multiple named data blobs to reference the same data segment.
- Access to tensor layout data without loading the entire blob.

## Usage

PTD files are used to store data outside of the PTE file. Some use-cases:
- On-device training: checkpointing for model weights.
- Deduplication: sharing model weights between multiple executable PTE files.
- Flexible deployment: allow async updates between program and data.
