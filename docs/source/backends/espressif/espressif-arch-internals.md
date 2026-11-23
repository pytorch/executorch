# Espressif Architecture and Internals

This page covers the backend's implementation and is mainly aimed at contributors. The code is in [`backends/espressif`](https://github.com/pytorch/executorch/tree/main/backends/espressif).

## Overview

```
backends/espressif/
  quantizer/           EspNNQuantizer: PT2E annotations for the ESP-NN int8 scheme
  partition/           EspNNPartitioner: finds the int8 regions that can be delegated
  espnn_preprocess.py  EspNNBackend.preprocess: partition -> XTS3 graph
  serialization/       XTS3 graph builder, operator lowering, writer and parser
  lowering.py          to_espnn_edge(): delegate or operator-library lowering + report
  reference/           bit-exact host model of the device arithmetic
  espnn_runtime/       ESP-IDF component: BackendInterface + int8 interpreter on ESP-NN
  passes/              EspPassManager / AtenToEspPass: Q/DQ patterns -> esp_nn:: operators
  ops/                 esp_nn:: operator schemas (operators.yaml) and their ESP-NN kernels
  utils/               shared quantization helpers
  CMakeLists.txt       ESP-IDF component for the operator library
  test/                pytest suite
```

The ESP-IDF component published as `pytorch/executorch` is generated from the repository by `examples/espressif/build_idf_component.sh`, from the template in `examples/espressif/idf_component_template`. It vendors the runtime, the kernel libraries and the ESP-NN delegate's sources.

## Lowering Flow

1. **Quantization.** `EspNNQuantizer` annotates the model for int8 per-tensor activations and per-channel weights. It fuses ReLU and ReLU6 into their producers, shares quantization parameters through movement, pooling and identity operators, and rewrites splits to slices. See {doc}`espressif-quantization`.
2. **Partitioning.** `EspNNPartitioner` tries to lower every node with the same code that serializes delegates, against structural stand-ins for the constants. A node is supported when its lowering succeeds. The failure message becomes the node's reason in the lowering report. Data movement is kept only when its consumers are delegated. Regions whose boundaries aren't quantize or dequantize nodes are rejected and stay on the CPU.
3. **Preprocessing.** `EspNNBackend.preprocess` builds an `XTS3Graph` from each partition with the real constants, then serializes it. Serialization enforces the device runtime's limits and checks the graph's dataflow (see below), so an invalid graph fails at export time.
4. **Runtime.** On the device, `EspNNBackend::init` parses and validates the graph, and `execute` runs it with ESP-NN.

## The XTS3 Graph Format

Each delegate payload is an XTS3 graph, a small little-endian binary format designed to be read in place, without a FlatBuffers dependency on the device. The current version is 6; the runtime also reads version 5.

- **Header:** magic `XTS3`, the version, and the counts and offsets of the regions.
- **I/O table** (version 6): the tensor IDs of the graph's inputs and outputs, at most 16 of each.
- **Tensors:** one fixed-size record per tensor, with dtype, rank and shape (rank 4 or less), the per-tensor scale and zero point, flags (constant or activation), and the offset of constant data.
- **Operations:** one fixed-size record per operation, with the opcode, input and output tensor IDs (at most 6 and 2), and the offset of its parameter block.
- **Parameters and constants:** per-operation parameters, such as convolution strides, requantization multipliers and lookup tables, followed by constant data. Constants start 16-byte aligned, and int32 constants are 4-byte aligned.

The opcodes are `CONV2D`, `DEPTHWISE_CONV2D`, `TRANSPOSE_CONV2D`, `LINEAR`, `ADD`, `MUL`, `RELU6` (a clamp to any int8 range), `LUT`, `SOFTMAX`, `AVGPOOL2D` (global average), `AVG_POOL2D`, `MAX_POOL2D`, `CONCAT`, `UPSAMPLE`, `COPY` (strided copies, which implement every data movement operator) and `FLATTEN`.

Activations are int8 and stored NHWC, the layout ESP-NN uses. The delegate converts at its boundaries.

### Validation

The host and the device validate graphs independently:

- `check_runtime_limits` on the host enforces the device's limits: at most 1024 tensors and 1024 operations, at most 16 inputs and 16 outputs, the opcodes the graph's version allows, aligned int32 constants, and upsample extents of at most 65535.
- `check_dataflow` on the host checks that every operation reads only graph inputs, constants or tensors written by an earlier operation, and that every graph output is written.
- The device parser (`xts3_graph.cpp`) bounds-checks every region, record and offset before use, including the COPY segment extents and the upsample tables. `test_xts3_parser_host.py` builds the parser for the host with AddressSanitizer and UndefinedBehaviorSanitizer and feeds it corrupted graphs.

## Device Runtime

`espnn_runtime` is an ESP-IDF component. `espnn_backend.cpp` implements `BackendInterface` and registers `EspNNBackend` from a static initializer. `int8_interpreter.cpp` runs the graph with ESP-NN. The component defines no ExecuTorch platform layer; the application provides it.

### init

- The delegate state comes from the runtime allocator.
- The interpreter parses the graph. If the payload is misaligned, or `CONFIG_ESPNN_DELEGATE_CONSTANTS_TO_PSRAM` copies the constants to PSRAM, it keeps its own copy and frees the processed buffer.
- The activation arena, ESP-NN scratch buffers and per-operation state are allocated with ESP-IDF's `heap_caps` allocator. The arena goes to internal RAM when it fits with room to spare, and otherwise to PSRAM.

Nothing is allocated during `execute`.

### execute

1. **Inputs.** Each input must have the graph's shape and a contiguous dim order. Float32 inputs are quantized with the graph's input parameters, using round-half-to-even. Int8 inputs are copied as is. Rank-4 inputs are converted from NCHW to NHWC. Other dtypes are rejected.
2. **Operations.** The interpreter runs the operations in order. Calls are serialized by a process-wide mutex, because ESP-NN's scratch pointers are global. With `CONFIG_ESPNN_DELEGATE_DUAL_CORE`, large convolutions are split across both cores on targets where ESP-NN supports it (the ESP32-S3 and ESP32-P4 with the optimized kernels); on other targets the runtime prints a warning and runs on one core.
3. **Outputs.** Outputs are written back as float32 (dequantized) or int8, matching the output tensor, and converted from NHWC to NCHW for rank 4.

`delegate_stats()` accumulates the number of delegates, the number of calls, the time spent in `execute` (including input and output conversion) and the arena sizes. `CONFIG_ESPNN_DELEGATE_PROFILE` prints a per-operation time table after each delegate's first run.

## Host Reference

`reference/esp_nn_reference.py` reimplements the device arithmetic in NumPy: the ESP-NN kernels, requantization, the lookup tables and the data movement. `ProgramEmulator` runs a lowered ExecuTorch program with each delegate replaced by this reference and the remaining operators in PyTorch. It produces the bit-exact expected outputs used by the tests, by the BundleIO test programs, and by the accuracy reports in the examples.

## Testing

- `pytest backends/espressif/test` runs on the host:
  - operator-by-operator lowering checked against the reference and the quantized PyTorch model;
  - CPU fallback programs;
  - quantizer and partitioner contracts;
  - runtime limits and dataflow checks;
  - the host build of the device parser.
- `.ci/scripts/test_espressif.sh` exports the test models as BundleIO programs with `examples/espressif/export_espnn_test_programs.py`. It runs each one through the ExecuTorch runtime and `examples/espressif/project` in `esp-emu`. Fully delegated programs must match the reference exactly (1e-5); programs with CPU regions are allowed 1%. CI runs it twice:
  - with ESP-NN's optimized ESP32-S3 kernels (`ESPRESSIF_NN_KERNELS=optimized`);
  - with its portable C kernels (`ansi`).

  The optimized run also builds and runs the operator library's tests (`backends/espressif/test/ops`) on the emulator.
- Separate CI jobs build the generated ESP-IDF component and a consumer project against several ESP-IDF versions.

## Operator Library

The operator library is the experimental alternative to the delegate. With `to_espnn_edge(..., use_operator_library=True)`:
- No partitioning. `EspPassManager` runs `AtenToEspPass`, which matches quantized patterns (a dequantize, the operator, a quantize) and replaces them with a call to an `esp_nn::` operator, with the requantization parameters computed on the host.
- Kernel registration. `ops/operators.yaml` declares the operators' schemas and kernels. The operator library's ESP-IDF component generates their registration with ExecuTorch's codegen and links the kernels, which call ESP-NN directly.

So far the library implements `esp_nn::quantized_mul_int8`, from quantized `aten.mul.Tensor`.
