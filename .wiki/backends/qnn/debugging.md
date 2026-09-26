---
title: QNN Debugging Guide
category: DEBUGGING
backends: [QNN]
socs: [SM8450, SM8550, SM8650, SM8750, SA8255, SA8295]
last_validated: 2026-04-15
source_issues: [1176, 1430, 3528, 5120, 5199, 8762, 9084, 10895, 10964, 12537, 15387, 15732, 15985, 16285, 16415, 17755, 18410, 18806]
---

# QNN Debugging Guide

## Enabling Debug Logging

### During Compilation (AOT)

```python
from executorch.backends.qualcomm.utils.utils import generate_qnn_executorch_compiler_spec

compile_spec = generate_qnn_executorch_compiler_spec(
    soc_model=QcomChipset.SM8650,
    backend_options=backend_options,
    debug=True,  # Enable verbose QNN logging
)
```
[Source: #18410]

### During Runtime

Set QNN log level via environment variable:
```bash
export QNN_LOG_LEVEL=5  # Most verbose
```
Or in the compile spec:
```python
backend_options = generate_htp_compiler_spec(use_fp16=False)
# Log level is controlled in compile spec
```
[Source: #16465]

## Step-by-Step Diagnostic Methodology

### 1. Verify Environment Setup First

Run a simple model before debugging complex ones:
```bash
python examples/qualcomm/scripts/export_example.py -m add -g --soc SM8650 -q ptq
# Push to device and run
adb shell "cd /data/local/tmp && \
  export LD_LIBRARY_PATH=/data/local/tmp && \
  export ADSP_LIBRARY_PATH=/data/local/tmp && \
  ./qnn_executor_runner --model_path ./add.pte"
```
If this fails, the issue is environment setup, not the model. [Source: #15387, #16217]

### 2. Check SoC/Library Matching

The `.pte` is compiled for a specific SoC. Running on a different SoC causes:
```
[ERROR] [Qnn ExecuTorch]: Request feature arch with value 75 unsupported
[ERROR] [Qnn ExecuTorch]: Failed to create context from binary with err 0x138d
```
**Fix**: Recompile with the correct `-m` flag matching the target device. [Source: #11100]

Verify SoC detection in logs:
```
[INFO] [Qnn ExecuTorch]: Get soc info for soc model 57.   # Check this matches your SoC
[INFO] [Qnn ExecuTorch]: Get soc info for soc htp arch 75.
```
[Source: #1176]

### 3. Check Library Versions

Ensure correct skel/stub libraries are pushed:
```bash
# Must push the correct version for your arch
adb push $QNN_SDK_ROOT/lib/aarch64-android/libQnnHtp.so /data/local/tmp/
adb push $QNN_SDK_ROOT/lib/hexagon-v75/unsigned/libQnnHtpV75Skel.so /data/local/tmp/
adb push $QNN_SDK_ROOT/lib/aarch64-android/libQnnHtpV75Stub.so /data/local/tmp/
```
Replace `v75` with your target arch (v68, v69, v73, v79, v81). [Source: #1176, #16535]

### 4. Check Partitioning Logs

During compilation, look for op support messages:
```
[QNN Partitioner Op Support]: aten.convolution.default | False  # NOT delegated
[QNN Partitioner Op Support]: aten.linear.default | True        # Delegated
```
Ops marked `False` fall back to CPU. Full delegation is required for good performance. [Source: #5199]

## Common Error Messages

### FastRPC / Skel Loading Failures

```
[ERROR] [Qnn ExecuTorch]: DspTransport.openSession qnn_open failed, 0x80000406
[ERROR] [Qnn ExecuTorch]: Unable to load Skel Library. transportStatus: 9
```
**Cause**: Skel library cannot be loaded on device. [Source: #1176, #1527]

**Fix**:
1. Verify `ADSP_LIBRARY_PATH` is set correctly
2. Ensure correct skel version is pushed (e.g., `libQnnHtpV73Skel.so` for SM8550)
3. Try running `qnn-net-run` from QNN SDK to verify device environment
4. On some devices, use `LD_DEBUG=3` to check linker search paths [Source: #16217]

### HTP PD Memory Exceeded

```
[ERROR] [Qnn ExecuTorch]: fa_alloc.cc:2462::ERROR:graph requires estimated allocation
of 2315388 KB, limit is 2097152 KB
```
**Cause**: Model graph exceeds HTP Process Domain memory. [Source: #15954, #17782]

**Fix**:
- Increase `num_sharding` to split graph across multiple PDs
- Reduce `max_seq_len`
- Use more aggressive quantization (e.g., 4-bit weights)
- For models with encoder (multimodal), shard the encoder separately [Source: #18410]

### Failed to Find Available PD

```
[ERROR] [Qnn ExecuTorch]: Failed to find available PD
```
**Cause**: Too many context binaries (>50 can cause issues). All available PDs are exhausted. [Source: #14985]

**Fix**: Reduce the number of partitions/shards. Custom partitioning with many fallback ops can create too many context binaries.

### SSR Detected (Subsystem Restart)

```
[ERROR] [Qnn ExecuTorch]: SSR Detected - You must free and recreate affected QNN API handles
```
**Cause**: Model exceeds what Hexagon DSP can handle, causing a subsystem restart. Usually from very large models. [Source: #3528]

### DMA-BUF Preregistration Failure (Second Load)

```
PreRegisterMem failed to get file descriptor.
Fail to pre register custom memory handle
```
**Cause**: Legacy code issue when loading a model a second time in the same app session. [Source: #15732]

**Fix**: Fixed in PR #16000 — update to latest ExecuTorch.

### Magic Number Mismatch

```
[INFO] [Qnn ExecuTorch]: QnnContextCustomProtocol expected magic number: 0x5678abcd but get: 0x2000000
```
**Cause**: `.pte` was compiled for a different SoC or QNN SDK version than the runtime. [Source: #11100]

### Overriding Output Data Pointer

```
E method.cpp:939] Overriding output data pointer allocated by memory plan is not allowed.
```
**Cause**: Noisy logging, not an actual error. The output location was memory-planned. Fixed in later releases. [Source: #3528]

## Profiling QNN Models

### Op-Level Profiling with ETDump

```python
from executorch.devtools import generate_etrecord, Inspector
from executorch.devtools.inspector import TimeScale

# After generating PTE with profiling enabled, run on device
# Then analyze ETDump:
inspector = Inspector(etdump_path="etdump.bin", etrecord="etrecord.bin")
for event in inspector.events:
    print(f"{event.name}: {event.perf_data.raw}")
```

Check `is_delegated_op` column to determine if op runs on HTP or CPU. [Source: #12537, #16285]

### Profiling Example (Minimal)

```python
from executorch.backends.qualcomm.utils.utils import (
    generate_htp_compiler_spec,
    generate_qnn_executorch_compiler_spec,
)

# Enable profiling in compile spec
backend_options = generate_htp_compiler_spec(use_fp16=False)
compile_spec = generate_qnn_executorch_compiler_spec(
    soc_model=QcomChipset.SM8650,
    backend_options=backend_options,
    profile=True,  # Enable op profiling
)
```
[Source: #12537]

### Understanding Op Fusion

QNN HTP fuses ops for performance. For example, conv+relu is fused — profiling cycles appear under relu only. Permute nodes from layout transforms (NCHW→NHWC) can be dominant for large inputs. [Source: #12537]

## Memory Profiling

### Getting Activation Memory from PTE

```python
from executorch.exir._serialize._program import deserialize_pte_binary

f = open("model.pte", "rb")
model = deserialize_pte_binary(f.read())
# [0, activation_memory_size, shared_memory_size]
print(model.program.execution_plan[0].non_const_buffer_sizes)
```
[Source: #17755]

### Extracting Context Binary for Analysis

```python
from executorch.backends.qualcomm.utils.utils import dump_context_from_pte
dump_context_from_pte("model.pte")  # Creates forward.bin
```

Then use `qnn-context-binary-utility` to get detailed memory info:
```bash
$QNN_SDK_ROOT/bin/x86_64-linux-clang/qnn-context-binary-utility \
  --context_binary forward.bin --json_file forward.json
```
The JSON output contains graph info including spill bytes and estimated memory usage. [Source: #17755]

## Useful Tools

| Tool | Purpose | Location |
|------|---------|----------|
| `qnn-context-binary-utility` | Analyze context binary metadata | `$QNN_SDK_ROOT/bin/` |
| `qnn-net-run` | Test QNN environment independently | `$QNN_SDK_ROOT/bin/` |
| Model Explorer / Netron | Visualize quantized graph | External tools |
| `dump_context_from_pte()` | Extract context binary from `.pte` | `backends/qualcomm/utils/utils.py` |

## Performance Optimization Tips

### Replace conv1d with conv2d at nn.Module Level

The framework translates conv1d into unsqueeze + conv2d + squeeze. After layout transform, this becomes unsqueeze + permute + conv2d + permute + squeeze. Manually using conv2d in the model minimizes permute overhead. [Source: #12537]

```python
# Replace conv1d with conv2d in model definition
self.conv = torch.nn.Conv2d(
    in_channels=weight.shape[1],
    out_channels=weight.shape[0],
    kernel_size=[weight.shape[2], 1],
    stride=[*stride, 1],
    padding=[*padding, 0],
)
self.conv.weight = torch.nn.Parameter(weight.unsqueeze(-1))
```

### Large Input Permute Optimization

For large inputs, the permute node (from NCHW→NHWC layout transform) can be more expensive than the actual computation. Split the input and permute each chunk separately. [Source: #12537]

### Reducing Compilation Memory

Compilation can use 100GB+ RAM for large models. Options:
- Increase swap space
- Use `num_sharding` to split the model
- A PR reducing lowering memory for Qwen3-1.7B (from ~117GB to ~73GB peak) is in progress [Source: #14402, #17782]

### Failed to Create Transport for Device (Error 4000)

```
[ERROR] [Qnn ExecuTorch]: Failed to create transport for device, error: 4000
[ERROR] [Qnn ExecuTorch]: Failed to load skel, error: 4000
```
**Cause**: The skel library cannot be loaded on the device. This is distinct from the `DspTransport.openSession` error — error 4000 indicates the transport layer itself failed to initialize. [Source: #16415, #10993]

**Fix**:
1. Verify `qnn-net-run` from QNN SDK works independently on the device
2. Ensure correct skel/stub versions match the SoC
3. Check `ADSP_LIBRARY_PATH` contains the skel libraries
4. Try running with `LD_DEBUG=3` to see linker search paths

### libc++ Missing During AOT Compilation

```
Cannot Open QNN library libQnnHtp.so, with error: libc++.so.1: cannot open shared object file
```
**Cause**: QNN SDK requires libc++.so.1 which may not be in the default library path. [Source: #10895, #5120]

**Fix**:
```bash
# Option 1: Install via conda
conda install -c conda-forge libcxx=14.0.0
# Copy libc++.so, libc++.so.1 to LD_LIBRARY_PATH

# Option 2: Install via apt (Ubuntu)
apt install libc++-dev libc++abi-dev
```

### Multi-Graph DMA Execution Error 1100

```
[ERROR] DMA execution error 1100
```
**Cause**: Occurs with multi-graph models (e.g., hybrid LLMs with separate prefill/decode) on second iteration. Fixed in mainline. [Source: #15985]

### Cross-Compilation flatc/flatcc Issues

When cross-compiling for non-Android ARM targets (e.g., Qualcomm Linux boards like RubikPi 3), the flatbuffers compiler may fail because it's compiled for the target architecture instead of the host. [Source: #10964]

**Workaround**: Compile `flatcc` for x86 first and place it in `third-party/flatcc_external_project/bin/` before cross-compiling.

## Verifying HTP Performance Mode

To verify that a runtime performance mode override is taking effect, enable verbose QNN logging and check for voltage corner settings. [Source: #18806]

```bash
./qnn_executor_runner --model_path model.pte --htp_performance_mode 4 --log_level 5
```

**Note**: When using `BackendOptions` in C++, set the template parameter to match the number of options being set:
```cpp
// If setting 2 options (performance_mode + log_level), use BackendOptions<2>
executorch::runtime::BackendOptions<2> backend_options;
```

In the verbose QNN logs, verify the performance mode by checking `coreVoltageCornerMin`:
- **Burst** (mode 2): `coreVoltageCornerMin` will be high (e.g., 128)
- **Power Saver** (mode 4): `coreVoltageCornerMin 64`
- **Default** (mode 0): no performance override applied, device controls behavior

[Source: #18806]

## See Also

- [SoC Compatibility Matrix](soc-compatibility.md) — SoC-to-arch mapping, error signatures per arch
- [QNN Quantization Guide](quantization.md) — Quantization errors and fixes
- [QNN Known Issues](known-issues.md) — Active bugs with workarounds
