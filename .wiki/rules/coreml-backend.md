---
title: CoreML Backend Rules
category: BACKEND_CONSTRAINT
backends: [CoreML]
last_validated: 2026-04-05
source_issues: [1020, 10066, 10179, 10451, 11221, 11687, 11714, 11738, 11753, 12059, 13305, 15833, 16484, 16492, 17537]
---

# CoreML Backend — Critical Tribal Knowledge

1. **dim_order breaks CoreML partitioner** — When dim order is enabled (default in v0.6+),
   `_to_dim_order_copy` ops are not recognized by coremltools. The partitioner skips them,
   causing scalar inputs to be passed to the delegate which crashes at runtime.
   Workaround: `EdgeCompileConfig(_skip_dim_order=True)`. [Source: #10451]

2. **No ct.ImageType support** — ET CoreML only supports `ct.TensorType` and `ct.StateType`.
   If your coremltools conversion uses `ct.ImageType` with scale/bias, you must apply
   normalization inside the model's forward method instead. [Source: #10179]

3. **Export path matters** — ET CoreML uses `torch.export.export` path, NOT `torch.jit.trace`.
   Accuracy differences vs direct coremltools may come from this difference. [Source: #10179]

4. **MPS requires Apple Silicon** — MPS backend needs M1+, macOS Sonoma, Xcode 15+.
   x86 Macs fail with `_mtl_device != nil` assertion. [Source: #1020]

5. **pip install works for CoreML (v0.6+)** — `pip install executorch` includes coremltools.
   Building from source is only needed for MPS backend. [Source: #10066]

6. **Use the export script, not raw API** — `python3 -m executorch.examples.apple.coreml.scripts.export`
   includes patches the raw `to_edge_transform_and_lower` + `CoreMLPartitioner` does not. [Source: #10451]

7. **Decomposition warnings are benign** — "ET ignoring decomposition requests from CoreML"
   warnings during export are harmless. They don't mean CoreML rejected ops. [Source: #10179]

8. **iOS linking requires -force_load** — All kernel libraries need `--force_load` linker flag
   because they use static initialization. [Source: #11221, #11753]

9. **Extract mlpackage from pte** — Debug accuracy issues by extracting the `.mlpackage`
   from the `.pte` file. See `docs/source/backends/coreml/coreml-overview.md#extracting-the-mlpackage`. [Source: #10179]

10. **Multi-entry point shared state is WIP** — Shared mutable state across multiple
    entry points is not fully supported. XNNPACK handles constant weight sharing
    via weight cache, but mutable state is an active development area. [Source: #11738]

11. **macOS 26 / iOS 26 ANE regression** — fp16 LLaMA inference produces inf/nan on the
    Apple Neural Engine due to SDPA regression in CoreML. macOS 15.x works fine.
    Workaround: decompose SDPA to avoid the problematic path. [Source: #15833]

12. **PT2E quantization requires iOS 17+** — Setting `minimum_deployment_target=None`
    (iOS 15 default) causes a confusing error. Quantized CoreML models need at
    least `coremltools.target.iOS17`. [Source: #13305, #12059]

13. **CoreML ignores add/sub alpha** — The alpha parameter in aten::add/sub.Tensor
    is silently ignored, producing wrong results. Temp fix: PR #13023.
    Upstream: coremltools#2573. [Source: #11687]

14. **floor_divide crashes** — torch.floor_divide on CoreML crashes the process.
    Fixed by PR #13018. [Source: #11714]

15. **Cached models can produce garbage** — CoreML model cache can corrupt outputs
    for certain models. Clear cache manually and recompile. [Source: #16492]

16. **torchao quantizer migration** — CoreML quantizer moved from
    `torch.ao.quantization.quantizer` to `torchao.quantization.pt2e.quantizer`.
    Update imports if you see module not found errors. [Source: #16484]

17. **aten::where (single-input) segfaults** — Models with `where(x)` or
    `nonzero_numpy` segfault at runtime. This is an underlying CoreML bug
    with dynamic shapes. [Source: #17537]
