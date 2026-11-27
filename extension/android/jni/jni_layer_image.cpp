/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <jni.h>

#include <android/bitmap.h>

#include <cstdint>
#include <memory>
#include <string>

#include <executorch/extension/image/image_processor.h>
#include <executorch/extension/tensor/tensor_ptr_maker.h>
#include <executorch/runtime/platform/log.h>

#include <executorch/extension/android/jni/jni_helper.h>

namespace image = ::executorch::extension::image;
using ::executorch::extension::from_blob;
using ::executorch::extension::TensorPtr;
using ::executorch::jni_helper::setExecutorchPendingException;
using ::executorch::runtime::Error;

namespace {

image::ImageProcessor* toProcessor(JNIEnv* env, jlong handle) {
  if (handle == 0) {
    setExecutorchPendingException(
        env,
        static_cast<uint32_t>(Error::InvalidState),
        "ImageProcessor has been closed");
    return nullptr;
  }
  return reinterpret_cast<image::ImageProcessor*>(handle);
}

bool toOrientation(
    JNIEnv* env,
    jint exifCode,
    image::Orientation& orientation) {
  orientation = static_cast<image::Orientation>(exifCode);
  if (!image::is_supported_orientation(orientation)) {
    setExecutorchPendingException(
        env,
        static_cast<uint32_t>(Error::InvalidArgument),
        "Unsupported EXIF orientation: " + std::to_string(exifCode));
    return false;
  }
  return true;
}

// Wraps a caller-supplied direct FloatBuffer as the [1, 3, H, W] output tensor
// the processor writes into. `floatCapacity` is passed from Kotlin rather than
// read via GetDirectBufferCapacity, whose unit is unspecified for view buffers.
TensorPtr outputTensor(
    JNIEnv* env,
    jobject outBuffer,
    jint floatCapacity,
    const image::ImageProcessorConfig& config) {
  void* data = env->GetDirectBufferAddress(outBuffer);
  if (data == nullptr) {
    setExecutorchPendingException(
        env,
        static_cast<uint32_t>(Error::InvalidArgument),
        "Output buffer must be a direct java.nio.FloatBuffer");
    return nullptr;
  }
  // Bound the pixel count before multiplying by the channel count so the
  // guard itself cannot overflow; a jint capacity can never satisfy a
  // requirement past INT32_MAX anyway.
  const int64_t pixels =
      static_cast<int64_t>(config.target_height) * config.target_width;
  if (pixels > INT32_MAX / image::ImageProcessorConfig::kOutputChannels ||
      floatCapacity < image::ImageProcessorConfig::kOutputChannels * pixels) {
    setExecutorchPendingException(
        env,
        static_cast<uint32_t>(Error::InvalidArgument),
        "Output buffer holds " + std::to_string(floatCapacity) +
            " floats, need " +
            std::to_string(
                image::ImageProcessorConfig::kOutputChannels * pixels));
    return nullptr;
  }
  return from_blob(
      static_cast<float*>(data),
      {1,
       image::ImageProcessorConfig::kOutputChannels,
       config.target_height,
       config.target_width},
      ::executorch::aten::ScalarType::Float);
}

// The decoder takes raw pointers, so the plane bounds can only be enforced
// here. `capacity` comes from the Java buffer; GetDirectBufferAddress returns
// the region base, so capacity (not remaining) is the matching bound.
const uint8_t* directBytes(
    JNIEnv* env,
    jobject buffer,
    jint capacity,
    int64_t required,
    const char* name) {
  void* data = env->GetDirectBufferAddress(buffer);
  if (data == nullptr) {
    setExecutorchPendingException(
        env,
        static_cast<uint32_t>(Error::InvalidArgument),
        std::string(name) + " must be a direct java.nio.ByteBuffer");
    return nullptr;
  }
  if (capacity < required) {
    setExecutorchPendingException(
        env,
        static_cast<uint32_t>(Error::InvalidArgument),
        std::string(name) + " holds " + std::to_string(capacity) +
            " bytes, need " + std::to_string(required));
    return nullptr;
  }
  return static_cast<const uint8_t*>(data);
}

// Locks an ARGB_8888 bitmap for the lifetime of the scope. Android stores that
// format as RGBA bytes in memory, so it maps to ColorFormat::RGBA.
class ScopedBitmapLock {
 public:
  ScopedBitmapLock(JNIEnv* env, jobject bitmap) : env_(env), bitmap_(bitmap) {
    if (AndroidBitmap_getInfo(env, bitmap, &info_) !=
        ANDROID_BITMAP_RESULT_SUCCESS) {
      setExecutorchPendingException(
          env,
          static_cast<uint32_t>(Error::InvalidArgument),
          "Failed to read Bitmap info");
      return;
    }
    if (info_.format != ANDROID_BITMAP_FORMAT_RGBA_8888) {
      setExecutorchPendingException(
          env,
          static_cast<uint32_t>(Error::InvalidArgument),
          "Bitmap must be ARGB_8888");
      return;
    }
    if (AndroidBitmap_lockPixels(env, bitmap, &pixels_) !=
        ANDROID_BITMAP_RESULT_SUCCESS) {
      setExecutorchPendingException(
          env,
          static_cast<uint32_t>(Error::AccessFailed),
          "Failed to lock Bitmap pixels");
      pixels_ = nullptr;
      return;
    }
    locked_ = true;
  }

  ~ScopedBitmapLock() {
    if (locked_) {
      AndroidBitmap_unlockPixels(env_, bitmap_);
    }
  }

  ScopedBitmapLock(const ScopedBitmapLock&) = delete;
  ScopedBitmapLock& operator=(const ScopedBitmapLock&) = delete;

  bool ok() const {
    return locked_;
  }
  const uint8_t* pixels() const {
    return static_cast<const uint8_t*>(pixels_);
  }
  int32_t width() const {
    return static_cast<int32_t>(info_.width);
  }
  int32_t height() const {
    return static_cast<int32_t>(info_.height);
  }
  int32_t stride() const {
    return static_cast<int32_t>(info_.stride);
  }

 private:
  JNIEnv* env_;
  jobject bitmap_;
  AndroidBitmapInfo info_{};
  void* pixels_ = nullptr;
  bool locked_ = false;
};

} // namespace

extern "C" {

JNIEXPORT jlong JNICALL
Java_org_pytorch_executorch_extension_image_ImageProcessor_nativeCreate(
    JNIEnv* env,
    jclass /* clazz */,
    jint targetWidth,
    jint targetHeight,
    jint resizeMode,
    jint letterboxAnchor,
    jfloat padValue,
    jfloat scaleFactor,
    jfloatArray mean,
    jfloatArray stdDev) {
  if (targetWidth <= 0 || targetHeight <= 0) {
    setExecutorchPendingException(
        env,
        static_cast<uint32_t>(Error::InvalidArgument),
        "Target dimensions must be positive");
    return 0;
  }
  if (env->GetArrayLength(mean) !=
          image::ImageProcessorConfig::kOutputChannels ||
      env->GetArrayLength(stdDev) !=
          image::ImageProcessorConfig::kOutputChannels) {
    setExecutorchPendingException(
        env,
        static_cast<uint32_t>(Error::InvalidArgument),
        "mean and standardDeviation must each have 3 entries");
    return 0;
  }

  image::ImageProcessorConfig config;
  config.target_width = targetWidth;
  config.target_height = targetHeight;
  config.resize_mode = static_cast<image::ResizeMode>(resizeMode);
  config.letterbox_anchor =
      static_cast<image::LetterboxAnchor>(letterboxAnchor);
  config.pad_value = padValue;
  config.normalization.scale_factor = scaleFactor;
  // The 4th mean/std slot is reserved for a future RGBA output; keep it an
  // identity normalization so it stays divide-safe.
  config.normalization.mean[3] = 0.0f;
  config.normalization.std_dev[3] = 1.0f;
  env->GetFloatArrayRegion(
      mean,
      0,
      image::ImageProcessorConfig::kOutputChannels,
      config.normalization.mean);
  env->GetFloatArrayRegion(
      stdDev,
      0,
      image::ImageProcessorConfig::kOutputChannels,
      config.normalization.std_dev);
  for (int32_t i = 0; i < image::ImageProcessorConfig::kOutputChannels; ++i) {
    if (config.normalization.std_dev[i] == 0.0f) {
      setExecutorchPendingException(
          env,
          static_cast<uint32_t>(Error::InvalidArgument),
          "standardDeviation entries must be nonzero");
      return 0;
    }
  }
  // The portable implementation has no GPU path; keep the CPU sentinel so the
  // config never reports a GPU decision that cannot happen here.
  config.gpu_min_input_pixels = image::ImageProcessorConfig::kGpuNever;

  try {
    return reinterpret_cast<jlong>(
        new image::ImageProcessor(std::move(config)));
  } catch (const std::exception& e) {
    ET_LOG(Error, "Failed to create ImageProcessor: %s", e.what());
    setExecutorchPendingException(
        env,
        static_cast<uint32_t>(Error::Internal),
        "Failed to create ImageProcessor: " + std::string(e.what()));
    return 0;
  }
}

JNIEXPORT void JNICALL
Java_org_pytorch_executorch_extension_image_ImageProcessor_nativeDestroy(
    JNIEnv* /* env */,
    jclass /* clazz */,
    jlong nativeHandle) {
  delete reinterpret_cast<image::ImageProcessor*>(nativeHandle);
}

JNIEXPORT void JNICALL
Java_org_pytorch_executorch_extension_image_ImageProcessor_nativeProcessBitmap(
    JNIEnv* env,
    jclass /* clazz */,
    jlong nativeHandle,
    jobject bitmap,
    jint orientationCode,
    jobject outBuffer,
    jint outCapacity) {
  try {
    auto* processor = toProcessor(env, nativeHandle);
    if (processor == nullptr) {
      return;
    }
    image::Orientation orientation;
    if (!toOrientation(env, orientationCode, orientation)) {
      return;
    }
    auto out = outputTensor(env, outBuffer, outCapacity, processor->config());
    if (out == nullptr) {
      return;
    }

    ScopedBitmapLock lock(env, bitmap);
    if (!lock.ok()) {
      return;
    }

    const Error error = processor->process_into(
        lock.pixels(),
        lock.width(),
        lock.height(),
        lock.stride(),
        image::ColorFormat::RGBA,
        *out,
        orientation);
    if (error != Error::Ok) {
      setExecutorchPendingException(
          env, static_cast<uint32_t>(error), "Failed to process Bitmap");
    }
  } catch (const std::exception& e) {
    setExecutorchPendingException(
        env,
        static_cast<uint32_t>(Error::Internal),
        std::string("Failed to process Bitmap: ") + e.what());
  } catch (...) {
    setExecutorchPendingException(
        env,
        static_cast<uint32_t>(Error::Internal),
        "Failed to process Bitmap: unknown exception");
  }
}

JNIEXPORT void JNICALL
Java_org_pytorch_executorch_extension_image_ImageProcessor_nativeProcessYuv(
    JNIEnv* env,
    jclass /* clazz */,
    jlong nativeHandle,
    jobject yPlane,
    jint yStride,
    jint yCapacity,
    jobject uvPlane,
    jint uvStride,
    jint uvCapacity,
    jint width,
    jint height,
    jint format,
    jint range,
    jint orientationCode,
    jobject outBuffer,
    jint outCapacity) {
  try {
    auto* processor = toProcessor(env, nativeHandle);
    if (processor == nullptr) {
      return;
    }
    image::Orientation orientation;
    if (!toOrientation(env, orientationCode, orientation)) {
      return;
    }
    auto out = outputTensor(env, outBuffer, outCapacity, processor->config());
    if (out == nullptr) {
      return;
    }
    // The last row of each plane is only read up to `width`, not a full
    // stride, and the chroma bound stops one byte short of the last pair,
    // because that is where a CameraX plane view ends. The capacity is passed
    // down as uv_plane_size, so the decode reads the final byte whenever it is
    // actually there and substitutes it only when it is not. Leave the
    // dimension and stride checks themselves to process_yuv_into; clamp to 0
    // here so a bad input cannot produce a negative bound.
    const int64_t yRequired = width > 0 && height > 0
        ? static_cast<int64_t>(yStride) * (height - 1) + width
        : 0;
    const int64_t uvRequiredRaw = width > 0 && height > 0
        ? static_cast<int64_t>(uvStride) * (height / 2 - 1) + width - 1
        : 0;
    const int64_t uvRequired = uvRequiredRaw < 0 ? 0 : uvRequiredRaw;
    const uint8_t* y = directBytes(env, yPlane, yCapacity, yRequired, "yPlane");
    if (y == nullptr) {
      return;
    }
    const uint8_t* uv =
        directBytes(env, uvPlane, uvCapacity, uvRequired, "uvPlane");
    if (uv == nullptr) {
      return;
    }

    const Error error = processor->process_yuv_into(
        y,
        yStride,
        uv,
        uvStride,
        width,
        height,
        static_cast<image::YUVFormat>(format),
        *out,
        orientation,
        image::kFullImage,
        static_cast<image::YUVRange>(range),
        static_cast<int64_t>(uvCapacity));
    if (error != Error::Ok) {
      setExecutorchPendingException(
          env, static_cast<uint32_t>(error), "Failed to process YUV planes");
    }
  } catch (const std::exception& e) {
    setExecutorchPendingException(
        env,
        static_cast<uint32_t>(Error::Internal),
        std::string("Failed to process YUV planes: ") + e.what());
  } catch (...) {
    setExecutorchPendingException(
        env,
        static_cast<uint32_t>(Error::Internal),
        "Failed to process YUV planes: unknown exception");
  }
}

JNIEXPORT jintArray JNICALL
Java_org_pytorch_executorch_extension_image_ImageProcessor_nativeComputeOutputShape(
    JNIEnv* env,
    jclass /* clazz */,
    jlong nativeHandle,
    jint inputWidth,
    jint inputHeight,
    jint orientationCode) {
  try {
    auto* processor = toProcessor(env, nativeHandle);
    if (processor == nullptr) {
      return nullptr;
    }
    image::Orientation orientation;
    if (!toOrientation(env, orientationCode, orientation)) {
      return nullptr;
    }

    const auto shape =
        processor->compute_output_shape(inputWidth, inputHeight, orientation);
    jintArray result = env->NewIntArray(static_cast<jsize>(shape.size()));
    if (result == nullptr) {
      return nullptr;
    }
    env->SetIntArrayRegion(
        result, 0, static_cast<jsize>(shape.size()), shape.data());
    return result;
  } catch (const std::exception& e) {
    setExecutorchPendingException(
        env,
        static_cast<uint32_t>(Error::Internal),
        std::string("Failed to compute output shape: ") + e.what());
    return nullptr;
  } catch (...) {
    setExecutorchPendingException(
        env,
        static_cast<uint32_t>(Error::Internal),
        "Failed to compute output shape: unknown exception");
    return nullptr;
  }
}

JNIEXPORT jintArray JNICALL
Java_org_pytorch_executorch_extension_image_ImageProcessor_nativeComputeLetterboxPadding(
    JNIEnv* env,
    jclass /* clazz */,
    jlong nativeHandle,
    jint inputWidth,
    jint inputHeight,
    jint orientationCode) {
  try {
    auto* processor = toProcessor(env, nativeHandle);
    if (processor == nullptr) {
      return nullptr;
    }
    image::Orientation orientation;
    if (!toOrientation(env, orientationCode, orientation)) {
      return nullptr;
    }

    const auto padding = processor->compute_letterbox_padding(
        inputWidth, inputHeight, orientation);
    const jint values[2] = {padding.first, padding.second};
    jintArray result = env->NewIntArray(2);
    if (result == nullptr) {
      return nullptr;
    }
    env->SetIntArrayRegion(result, 0, 2, values);
    return result;
  } catch (const std::exception& e) {
    setExecutorchPendingException(
        env,
        static_cast<uint32_t>(Error::Internal),
        std::string("Failed to compute letterbox padding: ") + e.what());
    return nullptr;
  } catch (...) {
    setExecutorchPendingException(
        env,
        static_cast<uint32_t>(Error::Internal),
        "Failed to compute letterbox padding: unknown exception");
    return nullptr;
  }
}

} // extern "C"
