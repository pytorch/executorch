---
title: Arm Backend Rules
category: BACKEND_CONSTRAINT
backends: [Arm]
last_validated: 2026-04-05
source_issues: [1004, 1110, 1161, 1163, 1230, 11913, 12237, 12959, 13842, 13901, 16225, 16426, 16541, 16739, 16779, 16784, 16902, 17241, 17397, 17489, 17668, 18306, 18319]
---

# Arm Backend — Critical Tribal Knowledge

1. **No dynamic shapes** — The Arm backend does NOT support models with dynamic shapes.
   SymFloat/SymInt in the graph causes `TypeError: Expected FakeTensor ... got SymFloat`.
   Fix all input dimensions at export time. [Source: #12237]

2. **TOSA output is NOT runnable on hardware** — Setting `output_format=tosa` bypasses
   Vela and produces a .pte that will fail at runtime with `Init failed for backend ArmBackend`.
   You MUST use Vela output for Ethos-U execution. [Source: #1161]

3. **Submodule init blocks on git.mlplatform.org** — Arm submodules hosted on
   git.mlplatform.org have recurring SSL and availability issues. If not using Arm,
   deinit these submodules before running `git submodule update --init`. [Source: #1004, #1163]

4. **Use strict=False for complex models** — Models with attribute mutation (YOLO, etc.)
   require `strict=False` in `torch.export.export_for_training()`. [Source: #12237]

5. **Quantized ops need explicit linking** — Running quantized models without delegation
   requires linking `quantized_ops_lib`. Error: `Missing out variants:
   quantized_decomposed::dequantize_per_tensor`. [Source: #1161]

6. **Non-delegated performance is poor** — Running quantized models on Cortex-M CPU
   without Ethos-U delegation has very low performance. Always use delegation
   for production. [Source: #1161]

7. **NHWC conversion** — TOSA requires channel-last (NHWC). PyTorch is channel-first (NCHW).
   The Permute_Memory_Format_Pass handles this but was historically incomplete. [Source: #1110]

8. **Increase method_allocator_pool for larger models** — The tensor arena size in
   `runner.cpp` defaults to 4KB. Increase it for real models:
   `uint8_t method_allocator_pool[136 * 1024U];` [Source: #1161]

9. **Selective build for baremetal** — Use CMake flags for smaller binaries:
   `-DEXECUTORCH_SELECT_OPS_FROM_MODEL="<model>.pte"`
   `-DEXECUTORCH_DTYPE_SELECTIVE_BUILD=ON` [Source: #11913]

10. **C10_USING_CUSTOM_GENERATED_MACROS** — When building as a separate CMake project,
    define this macro to avoid `c10/macros/cmake_macros.h not found`. [Source: #11999]

11. **Use TOSAQuantizer, not XNNPACKQuantizer** — For Arm targets, use
    `TOSAQuantizer(TosaSpecification.create_from_string("TOSA-0.80+BI+u55"))`.
    XNNPACK quantizer has different numerics. [Source: #1161]

12. **YOLO models work on Ethos-U85** — YOLOv12 tested on Ethos-U85 MAC-256 / Corstone-320
    with fixed input sizes. See `examples/models/yolo12`. [Source: #12237]

13. **Observer sharing bug at residual junctions** — Arm Ethos quantizer incorrectly
    shares observers at Conv-ReLU + residual add nodes. Can cause quantization errors
    in ResNet/MobileNet-like architectures. [Source: #12959]

14. **PReLU not supported on Ethos-U** — `nn.PReLU` decomposes to `torch.where`
    which is not supported. No workaround available. [Source: #16902]

15. **BatchNorm without Conv not delegated** — Standalone `BatchNorm2d` fails
    Ethos-U delegation. Workaround: manually decompose to mul+add. [Source: #17241, #17397]

16. **conv→relu→permute→reshape(5D) crashes** — This graph pattern crashes
    the partitioner during `to_edge_transform_and_lower`. [Source: #16739]

17. **Softmax amax is slow on NPU** — Softmax decomposition uses `aten::amax`
    on the elementwise engine (not MACs). Don't trust Vela cycle estimates —
    profile on FVP or real hardware. [Source: #18319]

18. **LayerNorm quantization is accuracy-sensitive** — Use `--stable_softmax`
    for transformer models. Epsilon value (default 1e-5) can cause accuracy drops
    in int8 quantization of LayerNorm-heavy models. [Source: #16426, #18306]

19. **tosa module requires setup.sh** — `pip install executorch` does NOT install
    tosa dependencies. Run `examples/arm/setup.sh` after pip install, or you'll get
    `No module named 'tosa'`. [Source: #13901]

20. **arm_executor_runner object lifetime bug** — `BufferCleanup` used `free()` on
    non-malloc memory. Fixed in PR #16339. Crashes on real hardware but not FVP. [Source: #16225]

21. **Ethos-U base_addr mismatch on real hardware** — Output buffers may remain
    unchanged on real MCUs despite reported success. Works on FVP. [Source: #16784]

22. **GRU/RNN not supported** — GRU decomposition fails during Ethos-U lowering.
    LSTM CMSIS-NN support planned but not yet implemented. [Source: #12270, #17753]

23. **ConvTranspose2d fallback failure** — Fails to fall back to CPU when NPU
    can't run it, giving "Non-passthrough operation could not run on NPU". [Source: #17668]

24. **setup.sh dependency conflicts are benign** — flatbuffers and numpy version
    conflicts between vela and tosa-tools are known. Backend works despite warnings. [Source: #10899]

25. **SharedQuantizationSpec recursion** — Certain graph topologies cause infinite
    recursion. Fixed in pytorch/ao#3011. [Source: #13842]
