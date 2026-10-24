# W4A16 v7 split/reducer retile

Optimized by KernelAgent-Oink(https://github.com/meta-pytorch/KernelAgent).

This bounded Triton package consumes unchanged `CudaCoalescedInt4Tensor` fields
and keeps FP32 affine weight reconstruction with one final BF16 operand cast.
The K256 producer is the validated v5 implementation. V7 varies only split count
and fixed-order reducer tile/warp scheduling across 12 explicit M8/M16 configs.

It does not add runtime buckets, repack weights, cache dense weights, quantize
activations, or use SM100-only instructions. M1/M2/M4 remain on production W4A8.
The public API is:

```python
run(x, qdata, scale, scale_step, zero, zero_step, *, bucket, config)
```

Validation and local SM80 evidence are written under `evidence/w4a16_v7/`.
Validated v5 RTX5090 feedback motivates the schedules but is not v7 performance.
Fresh actual-GGUF SM120, dynamic-M AOTI, server/F1 and memory gates remain
mandatory before integration.
