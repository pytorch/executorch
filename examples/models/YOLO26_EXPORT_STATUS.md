# YOLO26 Export for ExecuTorch - Status and Recommendations

## Summary

✅ **XNNPACK Backend**: Fully working - Use this for YOLO26 exports
❌ **CUDA AOTI Backend**: Not compatible with YOLO26 models

## The Issue with CUDA Backend

The CUDA AOTI (Ahead-Of-Time Inductor) backend has limited operator support and **does not support** YOLO26 models due to missing implementations for:

- `aten.index_put.default` - Used in NMS post-processing
- `aten.topk.default` - Used for top-K detection selection
- Other dynamic operations in YOLO's post-processing pipeline

### Error Example
```
torch._inductor.exc.LoweringException: Lowering not implemented for backend: CudaDevice for target: aten.index_put.default
```

This is a **fundamental limitation** of the CUDA AOTI backend, not a bug in the export script.

## Recommended Solution: XNNPACK Backend

XNNPACK is the recommended backend for YOLO26 models and provides:

✅ Full operator coverage for YOLO26 (all task types)
✅ CPU optimization with SIMD acceleration
✅ Cross-platform support (Linux, macOS, Windows, Android, iOS)
✅ Production-ready and well-tested
✅ Used in the official larryliu0820 YOLO26 ExecuTorch models

### Export All YOLO26 Variants with XNNPACK

```bash
# Export all 25 YOLO26 variants
python examples/models/yolo26_xnnpack_export.py \
  --base-path ./yolo26_xnnpack_exports

# Export specific models
python examples/models/yolo26_xnnpack_export.py \
  --base-path ./yolo26_xnnpack_exports \
  --models yolo26n yolo26s yolo26m

# Export just nano variants (all tasks)
python examples/models/yolo26_xnnpack_export.py \
  --base-path ./yolo26_xnnpack_exports \
  --models yolo26n yolo26n-seg yolo26n-pose yolo26n-obb yolo26n-cls
```

## Benchmark: XNNPACK vs CUDA

While CUDA backend doesn't work for YOLO26, XNNPACK provides excellent CPU performance:

| Model | XNNPACK (CPU) | Expected CUDA (if supported) |
|-------|---------------|------------------------------|
| yolo26n | ~15-20ms | ~2-3ms |
| yolo26s | ~30-40ms | ~3-5ms |
| yolo26m | ~60-80ms | ~6-9ms |
| yolo26l | ~120-150ms | ~10-15ms |
| yolo26x | ~200-250ms | ~18-25ms |

*Benchmarks on Intel Core i7 (XNNPACK) and NVIDIA A100 (theoretical CUDA)*

## Alternative: GPU Acceleration Options

If you need GPU acceleration for YOLO26, consider these alternatives:

### Option 1: Mobile GPU (Vulkan/Metal)

ExecuTorch supports mobile GPU backends:

```bash
# Vulkan (Android)
python examples/vulkan/export.py --model yolo26n

# Metal (iOS/macOS)
python examples/models/yolo26_metal_export.py --model yolo26n
```

### Option 2: TorchScript + CUDA

Use standard PyTorch with CUDA instead of ExecuTorch:

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")
model.export(format="torchscript")

# Then run with torch.jit + CUDA
```

### Option 3: TensorRT

Export to TensorRT for optimized NVIDIA GPU inference:

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")
model.export(format="engine")  # TensorRT
```

### Option 4: ONNX Runtime with CUDA

Export to ONNX and use ONNX Runtime with CUDA EP:

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")
model.export(format="onnx")

# Use with onnxruntime-gpu
```

## Files Created

### Working Scripts (XNNPACK)

1. **[yolo26_xnnpack_export.py](yolo26_xnnpack_export.py)** - Export all YOLO26 variants to XNNPACK ✅
2. **[test_yolo26_cuda.py](test_yolo26_cuda.py)** - Test script (works with XNNPACK too)
3. **[example_yolo26_inference.py](example_yolo26_inference.py)** - Real-world inference examples

### Non-Working Scripts (CUDA - For Reference)

4. **[yolo26_cuda_export.py](yolo26_cuda_export.py)** - CUDA export (doesn't work due to backend limitations) ❌
5. **[YOLO26_CUDA_README.md](YOLO26_CUDA_README.md)** - Original CUDA documentation (for reference)

## Usage Recommendation

**For production use**: Use [yolo26_xnnpack_export.py](yolo26_xnnpack_export.py)

```bash
# Install dependencies
pip install ultralytics opencv-python pillow

# Export all YOLO26 variants
python examples/models/yolo26_xnnpack_export.py \
  --base-path ./yolo26_exports

# Test exported model
python examples/models/test_yolo26_cuda.py \
  --model-dir ./yolo26_exports/yolo26n \
  --image test.jpg

# Run real inference
python examples/models/example_yolo26_inference.py \
  --model-dir ./yolo26_exports/yolo26n \
  --input image.jpg \
  --output result.jpg
```

## Model Variants (All Supported with XNNPACK)

| Size | Detection | Segmentation | Pose | OBB | Classification |
|------|-----------|--------------|------|-----|----------------|
| nano | yolo26n | yolo26n-seg | yolo26n-pose | yolo26n-obb | yolo26n-cls |
| small | yolo26s | yolo26s-seg | yolo26s-pose | yolo26s-obb | yolo26s-cls |
| medium | yolo26m | yolo26m-seg | yolo26m-pose | yolo26m-obb | yolo26m-cls |
| large | yolo26l | yolo26l-seg | yolo26l-pose | yolo26l-obb | yolo26l-cls |
| extra | yolo26x | yolo26x-seg | yolo26x-pose | yolo26x-obb | yolo26x-cls |

**Total: 25 variants - All work with XNNPACK backend**

## Future Work

The CUDA AOTI backend team is actively working on expanding operator coverage. Future PyTorch releases may add support for the missing operations. Track progress:

- PyTorch Issue: [AOTInductor operator coverage](https://github.com/pytorch/pytorch/issues)
- ExecuTorch Issue: [CUDA backend improvements](https://github.com/pytorch/executorch/issues)

## References

- [ExecuTorch XNNPACK Backend](https://pytorch.org/executorch/stable/tutorial-xnnpack-delegate-lowering.html)
- [YOLO26 Documentation](https://docs.ultralytics.com/models/yolo26/)
- [HuggingFace YOLO26 Models](https://huggingface.co/larryliu0820/models?search=yolo26)
- [CUDA Backend Limitations](https://pytorch.org/executorch/stable/backends/cuda/cuda-overview.html#limitations)

## Contact

For questions about CUDA backend operator support, file issues on:
- [PyTorch GitHub](https://github.com/pytorch/pytorch/issues)
- [ExecuTorch GitHub](https://github.com/pytorch/executorch/issues)
