# AMD AOTI pointwise example

Exports a small fused pointwise module through the AOTI delegate on an AMD GPU
and proves that PyTorch Inductor emitted a real Triton kernel for it rather than
falling back to ATen library kernels.

ROCm support is **experimental and off by default**. See
[../../backends/cuda/rocm.md](../../backends/cuda/rocm.md) for what is and is not
covered, including which architectures have execution coverage and which
capabilities are unavailable on ROCm.

Build the runtime pieces on a machine with the ROCm SDK installed; an AMD GPU is
not required for this build:

```bash
cmake -S . -B cmake-out-rocm \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=/opt/rocm \
  -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
  -DEXECUTORCH_BUILD_ROCM=ON
cmake --build cmake-out-rocm --target aoti_cuda_backend aoti_cuda_shims -j
```

Export requires a ROCm-enabled PyTorch build, Triton with its AMD backend, the
ROCm SDK, and a visible AMD GPU:

```bash
python -m examples.cuda.scripts.export_amd_pointwise --output-dir amd-out
```

Inductor targets the visible device, so the example compiles for whatever GPU is
present; there is no cross-compilation knob. It rejects any architecture the
installed PyTorch build does not list in `torch.cuda.get_arch_list()`.

The example emits `amd_triton.pte` and `aoti_cuda_blob.ptd`. It uses a fresh
Inductor cache and fails unless it finds generated Triton source there, and it
checks that the `.pte` embeds a code object for the architecture it compiled for.

## Merge native NVIDIA GPU exports

CUDA AOTI exports record their compiled target SM. Exports of the same program
and weights can be combined so the runtime selects an exactly matching native
AOTI library, or falls back to the PTX carried by the lowest compatible target.

```bash
python -m executorch.backends.cuda.merge_ptes \
  --input-pte a100/model.pte \
  --input-pte rtx5090/model.pte \
  --input-ptd a100/aoti_cuda_blob.ptd \
  --input-ptd rtx5090/aoti_cuda_blob.ptd \
  --output-pte merged/model.pte \
  --output-ptd merged/aoti_cuda_blob.ptd
```

The inputs must come from the same ExecuTorch program and contain identical
weights. The merged PTE stores every architecture-specific shared library. The
output PTD reuses one validated copy of the weights. Each source may contain or
omit PTX. If several sources contain forward-compatible PTX, only the lowest
target SM is advertised as the runtime fallback. If none contains PTX, the
merged PTE supports exact native-SM matches only.

To avoid carrying redundant PTX bytes in higher-SM shared libraries, export the
lowest-SM source with the default behavior and add this compile spec to each
higher-SM `CudaPartitioner`:

```python
CompileSpec("cuda_include_ptx", b"OFF")
```

The merge step cannot safely remove PTX from an already-linked shared library,
so this option must be set while exporting the higher-SM source PTEs.
