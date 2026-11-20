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
AOTI library, or uses one explicitly designated PTX fallback.

```bash
python -m executorch.backends.cuda.scripts.merge_ptes \
  --input-pte a100/model.pte \
  --input-pte rtx5090/model.pte \
  --input-ptd a100/aoti_cuda_blob.ptd \
  --input-ptd rtx5090/aoti_cuda_blob.ptd \
  --fallback-pte portable/model.pte \
  --fallback-ptd portable/aoti_cuda_blob.ptd \
  --output-pte merged/model.pte \
  --output-ptd merged/aoti_cuda_blob.ptd
```

The inputs must come from the same ExecuTorch program and contain identical
weights. A plain `--input-pte` contributes its exact-SM native variant, and any
PTX capability in that input is ignored. A previous merge output may also be
used as an `--input-pte`; its native variants and fallback are carried forward.
Identical inherited fallbacks are combined into one. Different inherited
fallbacks require an explicit replacement or the merge is rejected.

At most one `--fallback-pte` may be provided. It must contain exactly one
variant, which must be PTX-capable, and it replaces any fallback inherited from
regular inputs. Pass a previous merge output as an `--input-pte`, not as the
fallback. The runtime uses the fallback only when no regular input provides a
native cubin for the current SM. The output PTD reuses one validated copy of the
weights. After merging, the tool prints every native SM and PTX fallback
together with its immediate source PTE.

Export every regular input with PTX disabled:

```python
CompileSpec("cuda_include_ptx", b"OFF")
```

Export the fallback with PTX enabled and with any portability constraints, such
as a shared-memory limit, required by its target GPU set. AOTI host code and its
CUDA fatbin are linked into one shared library, so the merge step does not
rewrite ELF sections. Instead, the merged metadata makes regular libraries
native-only and marks the fallback library as PTX-only for runtime selection.

## Run an export on different GPUs than it was made for

The device ids a program records come from its `target_device` compile spec,
which defaults to `cuda:0`; compiling on another GPU does not by itself record
that GPU. But a program exported with an explicit non-zero `target_device`
(`cuda:3`, say) does not load on a machine whose GPUs have other indices.
Rewrite the recorded device ids before deploying:

```bash
python -m executorch.backends.cuda.scripts.remap_pte_devices \
  --input-pte model.pte \
  --output-pte model_dense.pte
```

By default the recorded ids are compressed to a dense range (3, 5, 7 become
0, 1, 2). Pass `--mapping 3=0,5=0,7=0` to choose the targets explicitly,
for example to put everything on one GPU. The `.ptd` needs no changes.
Every remapped device must exist on the target machine; note that the CUDA
runtime currently runs every delegate on the current device (per-device
execution is tracked in pytorch/executorch#24890), so collapsing onto
`cuda:0` is the remap that runs everywhere today. Leaving `target_device`
at its default avoids needing the tool at all. The output is re-serialized
with the default segment/constant/delegate alignments and named-data
alignment is not preserved, so exports made with non-default alignments
may be repacked.
