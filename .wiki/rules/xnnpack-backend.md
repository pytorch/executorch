---
title: "XNNPACK Backend Critical Rules"
category: BACKEND_CONSTRAINT
backends: [XNNPACK]
last_validated: 2026-04-05
source_issues: [1231, 1263, 1330, 1340, 1350, 3636, 3696, 4005, 7775, 8369, 8539, 8830, 8884, 10297, 11738, 12271, 12804, 14644, 14987, 15914, 17301, 17482, 17669, 18562]
---

# XNNPACK Backend — Critical Tribal Knowledge

1. **Always delegate to XNNPACK for CPU inference.** Portable ops are 10-100x slower. If inference is unexpectedly slow, check that ops are actually delegated and not falling back to portable. [Source: #1231, #3919]

2. **Use `to_edge_transform_and_lower`, not the older `to_edge().to_backend()` flow.** The newer API applies additional optimizations and handles quantized op delegation correctly. [Source: #10297]

3. **Set thread count explicitly.** XNNPACK may default to single-threaded. Call `threadpool::get_threadpool()->set_num_threads(4)` before inference. [Source: #10297]

4. **Build with Release mode.** Debug builds are dramatically slower. Always use `-DCMAKE_BUILD_TYPE=Release` for performance testing. [Source: #4005]

5. **XNNPACK does not support dynamic shapes inside delegated subgraphs.** Tensors with dynamic dimensions cause `Attempted to resize a static tensor` errors. Pad inputs to fixed sizes or let dynamic subgraphs fall back to portable. [Source: #1350, #3636, #8539]

6. **`Missing out variants` for quantized ops means delegation failed.** If you see `Missing out variants: {'quantized_decomposed::...'}`, the quantized graph was not lowered to XNNPACK. Ensure `XnnpackPartitioner()` is used after quantization. [Source: #1263, #7775, #8369]

7. **Batch norm requires a preceding conv for fusion.** Standalone batch_norm is not supported in XNNPACK. The partitioner should automatically skip it, but if you see errors about batch_norm, check that conv+BN patterns are intact. [Source: #1340]

8. **MediaTek Dimensity 6100+ has a known SIGSEGV in XNNWeightsCache.** Crash in `memcmp` during weight cache lookup, specific to this SoC. Other MediaTek chips work fine. [Source: #17669]

9. **On iOS, KleidAI SME kernels may crash on older devices.** If you see crashes at `kai_get_sme_vector_length_u32`, explicitly control KleidAI with `-DENABLE_XNNPACK_KLEIDI` flag. [Source: #17482]

10. **`torch.mm` with two dynamic inputs will NOT be delegated.** XNNPACK requires at least one constant weight tensor for matrix multiply ops. Non-delegated `mm` ops run on portable ops and are slow. [Source: #10297]

11. **Dynamic quantization requires `per_channel=True`.** `get_symmetric_quantization_config(is_dynamic=True)` without `per_channel=True` will crash with `XnnpackBackend init failed`. [Source: #8830]

12. **Always `.contiguous()` input tensors.** `Method.execute()` ignores strides and reads data as contiguous. Non-contiguous inputs silently produce wrong results with no error. [Source: #18562]

13. **`Backend XnnpackBackend is not registered` → linking issue.** When using pre-built libraries or separate build trees, use `--whole-archive` to force static initialization of the backend registration. [Source: #3696]

14. **Computed weights won't be delegated.** If a matmul weight is not recognized as a model parameter (e.g., computed intermediates in Whisper), the op falls back to portable. Check partitioner debug logs if delegation is unexpectedly missing. [Source: #15914]
