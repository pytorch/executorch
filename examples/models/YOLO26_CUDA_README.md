# YOLO26 CUDA Backend Export

This guide describes how to export all YOLO26 model variants on CUDA backend for ExecuTorch.

## Model Variants

The script exports 25 YOLO26 model variants across 5 sizes (n, s, m, l, x) and 5 task types:

### Detection (5 variants)
- `yolo26n`, `yolo26s`, `yolo26m`, `yolo26l`, `yolo26x`
- 80 COCO object classes
- Output: `(1, 300, 6)` - [x, y, w, h, confidence, class_id]

### Instance Segmentation (5 variants)
- `yolo26n-seg`, `yolo26s-seg`, `yolo26m-seg`, `yolo26l-seg`, `yolo26x-seg`
- Object detection + pixel-level segmentation masks
- Output: Detection boxes + segmentation masks

### Pose Estimation (5 variants)
- `yolo26n-pose`, `yolo26s-pose`, `yolo26m-pose`, `yolo26l-pose`, `yolo26x-pose`
- Human pose keypoint detection (17 keypoints)
- Output: Bounding boxes + keypoint coordinates

### Oriented Object Detection (5 variants)
- `yolo26n-obb`, `yolo26s-obb`, `yolo26m-obb`, `yolo26l-obb`, `yolo26x-obb`
- Rotated bounding box detection
- Output: Oriented boxes (x, y, w, h, angle, confidence, class)

### Classification (5 variants)
- `yolo26n-cls`, `yolo26s-cls`, `yolo26m-cls`, `yolo26l-cls`, `yolo26x-cls`
- Image classification (1000 ImageNet classes)
- Output: Class probabilities

## Prerequisites

### System Requirements
- CUDA-capable GPU (compute capability 7.0+)
- CUDA Toolkit 12.x
- Python 3.10+

### Install Dependencies

```bash
# Install ExecuTorch with CUDA backend
cd /home/dev/executorch
./install_executorch.sh

# Install Ultralytics and dependencies
pip install ultralytics opencv-python pillow

# Install CUDA backend dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## Usage

### Export All Models

```bash
python examples/models/yolo26_cuda_export.py \
  --base-path ./yolo26_cuda_exports
```

This will:
1. Download all 25 YOLO26 model variants from Ultralytics
2. Export each to CUDA backend
3. Test inference on each model
4. Save artifacts to `./yolo26_cuda_exports/<variant_name>/`

### Export Specific Models

```bash
# Export only nano (n) variants
python examples/models/yolo26_cuda_export.py \
  --base-path ./yolo26_cuda_exports \
  --models yolo26n yolo26n-seg yolo26n-pose yolo26n-obb yolo26n-cls

# Export only detection models
python examples/models/yolo26_cuda_export.py \
  --base-path ./yolo26_cuda_exports \
  --models yolo26n yolo26s yolo26m yolo26l yolo26x
```

### Skip Testing

If you want to skip inference testing after export (faster):

```bash
python examples/models/yolo26_cuda_export.py \
  --base-path ./yolo26_cuda_exports \
  --no-test
```

### Skip Existing Models

To resume a partially completed export:

```bash
python examples/models/yolo26_cuda_export.py \
  --base-path ./yolo26_cuda_exports \
  --skip-existing
```

## Output Structure

After export, the directory structure will be:

```
yolo26_cuda_exports/
├── yolo26n/
│   ├── model.pte              # ExecuTorch model file
│   └── aoti_cuda_blob.ptd     # CUDA kernels + weights
├── yolo26n-seg/
│   ├── model.pte
│   └── aoti_cuda_blob.ptd
├── yolo26n-pose/
│   ├── model.pte
│   └── aoti_cuda_blob.ptd
...
└── yolo26x-cls/
    ├── model.pte
    └── aoti_cuda_blob.ptd
```

Each model directory contains:
- **model.pte**: The ExecuTorch model program
- **aoti_cuda_blob.ptd**: CUDA kernel blobs and model weights

## Testing Exported Models

### Python Testing

```python
import torch
from executorch.runtime import Runtime
import numpy as np
from PIL import Image

# Load model
model_path = "yolo26_cuda_exports/yolo26n/model.pte"
with open(model_path, "rb") as f:
    pte_buffer = f.read()

runtime = Runtime.get()
program = runtime.load_program(pte_buffer)
method = program.load_method("forward")

# Prepare input (640x640 RGB image)
img = Image.open("test_image.jpg").resize((640, 640))
img_array = np.array(img).astype(np.float32) / 255.0

# Convert to tensor: HWC -> CHW -> NCHW
x = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
x = x.contiguous()  # IMPORTANT for correct results

# Run inference
outputs = method.execute([x])

# Process outputs based on task type
print(f"Output shape: {outputs[0].shape}")
```

### Model-Specific Testing

#### Detection Models
```python
# Output: (1, 300, 6) - [x, y, w, h, confidence, class_id]
detections = outputs[0]
for det in detections[0]:
    x, y, w, h, conf, cls = det
    if conf > 0.5:  # Confidence threshold
        print(f"Class {int(cls)}: {conf:.2f} at ({x:.0f}, {y:.0f})")
```

#### Segmentation Models
```python
# Output: [detections, masks]
detections, masks = outputs[0], outputs[1]
print(f"Detections: {detections.shape}")
print(f"Masks: {masks.shape}")
```

#### Pose Models
```python
# Output: [detections, keypoints]
detections, keypoints = outputs[0], outputs[1]
print(f"Detections: {detections.shape}")
print(f"Keypoints (17 per person): {keypoints.shape}")
```

#### Classification Models
```python
# Output: (1, 1000) - class probabilities
probs = torch.softmax(outputs[0], dim=1)
top5_prob, top5_idx = torch.topk(probs, 5)
for prob, idx in zip(top5_prob[0], top5_idx[0]):
    print(f"Class {idx}: {prob:.2%}")
```

## Performance Notes

### Model Sizes

Approximate model sizes after export:

| Variant | Detection | Segmentation | Pose | OBB | Classification |
|---------|-----------|--------------|------|-----|----------------|
| nano    | ~10 MB    | ~12 MB       | ~11 MB | ~11 MB | ~5 MB |
| small   | ~28 MB    | ~32 MB       | ~30 MB | ~30 MB | ~12 MB |
| medium  | ~64 MB    | ~72 MB       | ~68 MB | ~68 MB | ~28 MB |
| large   | ~110 MB   | ~125 MB      | ~118 MB | ~118 MB | ~48 MB |
| extra   | ~180 MB   | ~205 MB      | ~195 MB | ~195 MB | ~80 MB |

### Input Specifications

All models support:
- **Input resolution**: 640×640 (default), supports 320-8192 multiples of 32
- **Input format**: FP32 NCHW (batch, channels, height, width)
- **Batch size**: 1 (static)
- **Device**: CUDA

### Expected Latency

Approximate inference times on NVIDIA A100 (640×640 input):

| Variant | Detection | Segmentation | Pose | OBB | Classification |
|---------|-----------|--------------|------|-----|----------------|
| nano    | 1-2 ms    | 2-3 ms       | 2-3 ms | 2-3 ms | 0.5-1 ms |
| small   | 2-3 ms    | 3-5 ms       | 3-4 ms | 3-4 ms | 1-2 ms |
| medium  | 4-6 ms    | 6-9 ms       | 5-7 ms | 5-7 ms | 2-3 ms |
| large   | 7-10 ms   | 10-15 ms     | 9-12 ms | 9-12 ms | 3-5 ms |
| extra   | 12-18 ms  | 18-25 ms     | 15-20 ms | 15-20 ms | 5-8 ms |

*Note: Actual performance varies based on GPU model, CUDA version, and system configuration.*

## Troubleshooting

### CUDA Not Available

```
WARNING: CUDA is not available. Export may fail or produce CPU-only models.
```

**Solution**: Ensure PyTorch with CUDA support is installed:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
python -c "import torch; print(torch.cuda.is_available())"
```

### Model Download Fails

If Ultralytics fails to download models:

```bash
# Pre-download models manually
yolo predict model=yolo26n.pt source=path/to/image.jpg
yolo predict model=yolo26n-seg.pt source=path/to/image.jpg
# ... etc for all variants
```

### Out of Memory During Export

For large models (l, x variants), you may need more GPU memory:

```python
# Export models sequentially with GPU cleanup
for model in ["yolo26n", "yolo26s", ...]:
    export_model_variant(model, ...)
    torch.cuda.empty_cache()
```

### Inference Output Mismatch

Always ensure input tensors are **contiguous**:

```python
x = x.contiguous()  # CRITICAL
outputs = method.execute([x])
```

## References

- [YOLO26 Documentation](https://docs.ultralytics.com/models/yolo26/)
- [Ultralytics YOLO26 Models](https://huggingface.co/larryliu0820/models?search=yolo26)
- [ExecuTorch CUDA Backend](https://pytorch.org/executorch/stable/backends/cuda/cuda-overview.html)
- [ExecuTorch Runtime API](https://pytorch.org/executorch/stable/runtime-python.html)

## Model Cards

All exported models follow the format documented in the HuggingFace model cards:
- Detection: [yolo26n-ExecuTorch-XNNPACK](https://huggingface.co/larryliu0820/yolo26n-ExecuTorch-XNNPACK)
- Segmentation: [yolo26n-seg-ExecuTorch-XNNPACK](https://huggingface.co/larryliu0820/yolo26n-seg-ExecuTorch-XNNPACK)
- Pose: [yolo26n-pose-ExecuTorch-XNNPACK](https://huggingface.co/larryliu0820/yolo26n-pose-ExecuTorch-XNNPACK)
- OBB: [yolo26n-obb-ExecuTorch-XNNPACK](https://huggingface.co/larryliu0820/yolo26n-obb-ExecuTorch-XNNPACK)
- Classification: [yolo26n-cls-ExecuTorch-XNNPACK](https://huggingface.co/larryliu0820/yolo26n-cls-ExecuTorch-XNNPACK)

## License

YOLO26 models are provided by Ultralytics under AGPL-3.0 license.
ExecuTorch is licensed under BSD-3-Clause.
