## FlatTensor

FlatTensor is a flatbuffer-based format for storing and loading data with string-based keys. The format provides efficient serialization and deserialization of data with metadata and supports C++ and Python APIs. FlatTensor files use the `.ptd` extension.

Major usage is to store data outside of the PTE file for clean program-data separation. Stored data may be tensor data or opaque blob data (for backends that do not expose data format).

### Schema

[flat_tensor.fbs](https://github.com/pytorch/executorch/blob/main/extension/flat_tensor/serialize/flat_tensor.fbs) contains the [Flatbuffers](https://google.github.io/flatbuffers/) schema used to serialize ExecuTorch data files.

[flat_tensor_schema.py](https://github.com/pytorch/executorch/blob/main/extension/flat_tensor/serialize/flat_tensor_schema.py) contains the python definition of the schema types.

### C++ APIs

[serialize.h](https://github.com/pytorch/executorch/blob/main/extension/flat_tensor/serialize/serialize.h) contains the APIs to serialize a PTD file.

[flat_tensor_data_map.h](https://github.com/pytorch/executorch/blob/main/extension/flat_tensor/flat_tensor_data_map.h) contains the APIs to deserialize a PTD file and interact with it via the [named_data_map.h](https://github.com/pytorch/executorch/blob/main/runtime/core/named_data_map.h) interface.

### Python APIs

[serialize.py](https://github.com/pytorch/executorch/blob/main/extension/flat_tensor/serialize/serialize.py) contains the Python serialization and deserialization APIs.

### Alignment Considerations

**Segment alignment**: Data segments are aligned to this value. This is usually some multiple of 2. Specified in the [FlatTensorConfig](https://github.com/pytorch/executorch/blob/main/extension/flat_tensor/serialize/serialize.py#L96).

**Tensor alignment**: Tensors are aligned to this value. Specified in the [FlatTensorConfig](https://github.com/pytorch/executorch/blob/main/extension/flat_tensor/serialize/serialize.py#L96).

**Blob alignment**: Blobs (may not be canonical tensors) are aligned to this value. Alignment is specified when blobs are added to the [_named_data_store.py](https://github.com/pytorch/executorch/blob/main/exir/_serialize/_named_data_store.py#L48) and passed to serialize.py.

FlatTensor does not store alignment in the serialized file; the user must ensure the serialized and runtime-expected alignment correspond. The final alignment may be a larger multiple of the specified alignment, as multiple `NamedData` entries can point to a single `DataSegment`. For example:
```
BackendA: {key = key1, data = 0x100, alignment = 4}
BackendB: {key = key2, data = 0x100, alignment = 8}
```
BackendA and BackendB are serializing the same bytes, so the data is deduplicated and the final alignment is the lcm of the two, in this case 8.

### Usage

**AoT**

To export a model as a PTE and PTD pair, see [export_program.py](https://github.com/pytorch/executorch/blob/main/test/models/export_program.py). Use the `--external-constants` argument to move all constants to the separate PTD file.
```
python -m test.models.export_program --modules "ModuleAddMul" --external-constants --outdir .
```

To export a delegated model as PTE and PTD pair, see [export_delegated_program.py](https://github.com/pytorch/executorch/blob/main/test/models/export_delegated_program.py). Use the `--external-constants` argument to move all constants to the separate PTD file. Note, ModuleLinear is used here as linear is consumed by the XNNPACK backend.
```
python -m test.models.export_delegated_program --modules ModuleLinear --backend_id XnnpackBackend --external_constants --outdir .
```

**Runtime**

The `ProgramDataSeparationTest` in [method_test.cpp](https://github.com/pytorch/executorch/blob/main/runtime/executor/test/method_test.cpp) demonstrates how to consume the PTD file at runtime.

For a backend example with XNNPACK, see [test_xnn_data_separation.cpp](https://github.com/pytorch/executorch/blob/main/backends/xnnpack/test/runtime/test_xnn_data_separation.cpp).

### Rules to ensure forward/backward compatibility
See [executorch/schema/README.md](https://github.com/pytorch/executorch/blob/main/schema/README.md).
