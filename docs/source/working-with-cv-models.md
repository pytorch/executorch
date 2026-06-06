(working-with-cv-models)=

# Working with Computer Vision Models

Computer vision deployments depend on the boundary between the app and the exported program being precise. Before exporting, write down the tensor contract that your app will satisfy:

- input shape, including whether the model expects `NCHW` (`[batch, channels, height, width]`) or `NHWC` (`[batch, height, width, channels]`)
- input dtype, such as `float32` normalized image values or `uint8` image bytes
- color channel order, such as RGB or BGR
- resize, crop, and normalization rules
- output tensors and the post-processing expected for each task

ExecuTorch runs the graph that you export. It does not infer image layout, resize policy, label mappings, or task-specific post-processing from the model file.

## Choose where preprocessing runs

Treat preprocessing placement as part of the backend contract. Keep platform-dependent image work, such as decoding, orientation handling, resizing, cropping, and UI-driven transforms, in the app. Put tensor-only preprocessing in the exported graph when matching the PyTorch reference exactly is more important and the operations are supported by the target backend.

For Core ML deployments, fixed-shape model inputs are often preferable. The Core ML backend documentation notes that true dynamic shapes use `RangeDim` and fall back to CPU or GPU instead of the Apple Neural Engine (ANE). When targeting the ANE, resize or crop images with iOS image APIs before tensor creation, or use enumerated shapes when the model needs a finite set of input sizes. See {doc}`backends/coreml/coreml-partitioner` for details.

This example accepts already resized and cropped `float32` `NCHW` RGB input in `[0, 1]`, normalizes it, and then calls the image classifier.

```python
import torch
from torch import nn


class ImageClassifierWithNormalization(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        image = (image - self.mean) / self.std
        return self.model(image)


wrapped_model = ImageClassifierWithNormalization(model).eval()
sample_inputs = (torch.zeros(1, 3, 224, 224, dtype=torch.float32),)
exported_program = torch.export.export(wrapped_model, sample_inputs)
```

Validate app-side preprocessing against the same PyTorch preprocessing used during export. For example, if the app resizes and crops before creating the tensor, keep a small reference test that compares the packed tensor or final model output with the PyTorch path.

If the model expects a crop after resizing, keep that policy in exactly one place. A fixed center crop can be implemented in the wrapper for backends where those tensor operations preserve the desired delegation. Camera- or UI-dependent crops, and iOS/Core ML paths that should keep ANE-friendly static shapes, are usually better handled before packing pixels into the input tensor.

## Convert images to tensors in app code

Most mobile image APIs expose decoded pixels as interleaved rows. Most PyTorch vision models expect channels-first tensors. If preprocessing stays in the app, explicitly pack pixels into the model's expected layout.

### Android

For production Android preprocessing, handle decoding, EXIF orientation, and camera-specific transforms before packing pixels into the input tensor. The following Kotlin helper keeps the layout conversion explicit: it resizes a `Bitmap`, reads RGB pixels, applies ImageNet-style normalization, and packs the result as `NCHW` `float32` data for `Tensor.fromBlob`.

```kotlin
import android.graphics.Bitmap
import org.pytorch.executorch.Tensor

fun bitmapToNchwTensor(
    bitmap: Bitmap,
    size: Int,
    mean: FloatArray = floatArrayOf(0.485f, 0.456f, 0.406f),
    std: FloatArray = floatArrayOf(0.229f, 0.224f, 0.225f)
): Tensor {
    val resized = Bitmap.createScaledBitmap(bitmap, size, size, true)
    val pixels = IntArray(size * size)
    resized.getPixels(pixels, 0, size, 0, 0, size, size)

    val input = FloatArray(3 * size * size)
    for (i in pixels.indices) {
        val pixel = pixels[i]
        val r = ((pixel shr 16) and 0xff) / 255.0f
        val g = ((pixel shr 8) and 0xff) / 255.0f
        val b = (pixel and 0xff) / 255.0f

        input[i] = (r - mean[0]) / std[0]
        input[size * size + i] = (g - mean[1]) / std[1]
        input[2 * size * size + i] = (b - mean[2]) / std[2]
    }

    return Tensor.fromBlob(input, longArrayOf(1, 3, size.toLong(), size.toLong()))
}
```

If the exported model accepts `uint8` image bytes instead, use `Tensor.fromBlobUnsigned(...)` and keep dtype conversion inside the exported graph.

```kotlin
val inputBytes = ByteArray(3 * width * height)
// Pack bytes in the same layout expected by the model. For NCHW RGB,
// write all red values, then green values, then blue values.
val inputTensor = Tensor.fromBlobUnsigned(
    inputBytes,
    longArrayOf(1, 3, height.toLong(), width.toLong())
)
```

### iOS

For production iOS preprocessing, prefer platform image APIs and Accelerate, such as vImage for resizing and color conversion and vDSP for normalization, especially for camera frames or other hot paths. The following Swift helper keeps the layout conversion explicit so the tensor contract is easy to inspect: it draws a `UIImage` into a fixed-size RGB buffer, uses vDSP to normalize RGB channels, and creates a channels-first `Tensor<Float>`.

```swift
import Accelerate
import CoreGraphics
import ExecuTorch
import UIKit

func imageToNchwTensor(
  _ image: UIImage,
  size: Int,
  mean: [Float] = [0.485, 0.456, 0.406],
  std: [Float] = [0.229, 0.224, 0.225]
) -> Tensor<Float>? {
  guard size > 0, mean.count == 3, std.count == 3,
        let cgImage = image.cgImage else {
    return nil
  }

  let pixelCount = size * size
  var rgba = [UInt8](repeating: 0, count: pixelCount * 4)
  let colorSpace = CGColorSpaceCreateDeviceRGB()

  let didDraw = rgba.withUnsafeMutableBytes { buffer -> Bool in
    guard let baseAddress = buffer.baseAddress,
          let context = CGContext(
            data: baseAddress,
            width: size,
            height: size,
            bitsPerComponent: 8,
            bytesPerRow: size * 4,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue |
              CGBitmapInfo.byteOrder32Big.rawValue
          ) else {
      return false
    }
    context.draw(cgImage, in: CGRect(x: 0, y: 0, width: size, height: size))
    return true
  }
  guard didDraw else {
    return nil
  }

  let count = vDSP_Length(pixelCount)
  var input = [Float](repeating: 0, count: 3 * pixelCount)
  rgba.withUnsafeBufferPointer { rgbaBuffer in
    input.withUnsafeMutableBufferPointer { inputBuffer in
      guard let rgbaBase = rgbaBuffer.baseAddress,
            let inputBase = inputBuffer.baseAddress else {
        return
      }

      for channel in 0..<3 {
        let source = rgbaBase.advanced(by: channel)
        let destination = inputBase.advanced(by: channel * pixelCount)
        var scale = 1.0 / (255.0 * std[channel])
        var bias = -mean[channel] / std[channel]

        vDSP_vfltu8(source, 4, destination, 1, count)
        vDSP_vsmsa(destination, 1, &scale, &bias, destination, 1, count)
      }
    }
  }

  return Tensor<Float>(input, shape: [1, 3, size, size])
}
```

On either platform, if your model is exported for `NHWC`, keep the same decoded pixels but pack them in row-major `[height, width, channels]` order and use shape `[1, height, width, 3]`.

## Decode common CV outputs

Output tensors are model-specific. Preserve the output schema used during export and keep a small validation test that compares app-side post-processing with PyTorch post-processing.

For TorchVision models, check the [models and pre-trained weights documentation](https://docs.pytorch.org/vision/stable/models.html) for model-specific transforms, categories, and task conventions.

### Image classification

Image classifiers commonly return a logits tensor with shape `[1, num_classes]`. For top-1 classification, find the largest logit and map the index through the same labels file used during training or evaluation.

```kotlin
import org.pytorch.executorch.EValue

val output = module.forward(EValue.from(inputTensor))[0].toTensor()
val logits = output.dataAsFloatArray

var topIndex = 0
for (i in 1 until logits.size) {
    if (logits[i] > logits[topIndex]) {
        topIndex = i
    }
}
val topScore = logits[topIndex]
```

Use `softmax` only when the UI needs probabilities. Ranking classes by logits and by softmax probabilities gives the same order.

### Semantic segmentation

Semantic segmentation models commonly return class scores with shape `[1, classes, height, width]`. For each output pixel, choose the class channel with the largest score, then resize the mask back to the displayed image size if needed.

```kotlin
fun argmaxMask(scores: FloatArray, classes: Int, height: Int, width: Int): IntArray {
    val mask = IntArray(height * width)
    for (y in 0 until height) {
        for (x in 0 until width) {
            val offset = y * width + x
            var bestClass = 0
            var bestScore = scores[offset]
            for (c in 1 until classes) {
                val score = scores[c * height * width + offset]
                if (score > bestScore) {
                    bestScore = score
                    bestClass = c
                }
            }
            mask[offset] = bestClass
        }
    }
    return mask
}
```

See the [DeepLabV3 Android demo](https://github.com/meta-pytorch/executorch-examples/tree/main/dl3/android/DeepLabV3Demo) for an end-to-end ExecuTorch segmentation app that exports a model, runs it on Android, and overlays the predicted mask on an image.

### Object detection and instance segmentation

Detection and instance segmentation models do not have a single universal output format. Common patterns include:

- boxes as `[num_detections, 4]`, usually in `xyxy` or `xywh` coordinates
- labels as `[num_detections]`
- scores as `[num_detections]`
- masks as `[num_detections, height, width]` or `[num_detections, 1, height, width]`

Check whether thresholding, non-maximum suppression, box decoding, and mask resizing are already part of the exported graph. If they are not, keep those steps in the app and document the expected coordinate system. When the model runs on a resized or cropped image, map boxes and masks back to the original image coordinates before rendering overlays.

## Validate the model and app contract

Before shipping a CV model, validate these items:

- The app sends the same dtype, shape, layout, color order, and normalization that the exported graph expects.
- The app uses the same labels, palette, score threshold, and coordinate convention as the PyTorch reference.
- A known image produces matching top classes, masks, or detections in PyTorch and in the ExecuTorch app.
- The preprocessing is applied exactly once. Do not normalize in both the app and the exported model.
- The output code handles model-specific shapes instead of assuming all CV models return classifier logits.

For the basic export and runtime flow, start with {doc}`getting-started`. For mobile runtime integration, see {doc}`using-executorch-android` and {doc}`using-executorch-ios`.
