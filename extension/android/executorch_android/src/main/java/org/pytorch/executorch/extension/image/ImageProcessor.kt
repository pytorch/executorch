/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.pytorch.executorch.extension.image

import android.graphics.Bitmap
import java.io.Closeable
import java.nio.ByteBuffer
import java.nio.FloatBuffer
import java.util.concurrent.atomic.AtomicLong
import org.pytorch.executorch.DType
import org.pytorch.executorch.ExecutorchRuntimeException
import org.pytorch.executorch.Tensor
import org.pytorch.executorch.annotations.Experimental

/**
 * Converts camera frames and bitmaps into the normalized `[1, 3, H, W]` float tensors vision models
 * expect, replacing the hand-rolled resize/normalize loops apps otherwise write per model.
 *
 * The pipeline rotates the source upright according to the supplied [ImageOrientation], resizes it
 * per [ImageProcessorConfig.resizeMode], and normalizes it per
 * [ImageProcessorConfig.normalization]. The resize and normalize steps use NEON where available.
 *
 * Warning: These APIs are experimental and subject to change without notice
 *
 * Thread-safety: an ImageProcessor is NOT thread-safe. Internal scratch buffers are reused across
 * calls, so concurrent calls on one instance are unsafe. Use one instance per thread; separate
 * instances are independent.
 *
 * @param config The target size, resize mode, and normalization to apply.
 */
@Experimental
class ImageProcessor(val config: ImageProcessorConfig) : Closeable {

  private val nativeHandle = AtomicLong(0L)

  init {
    val handle =
        nativeCreate(
            config.targetWidth,
            config.targetHeight,
            config.resizeMode.ordinal,
            config.letterboxAnchor.ordinal,
            config.padValue,
            config.normalization.scaleFactor,
            config.normalization.mean,
            config.normalization.standardDeviation,
        )
    if (handle == 0L) {
      throw ExecutorchRuntimeException(
          ExecutorchRuntimeException.INTERNAL,
          "Failed to create native ImageProcessor",
      )
    }
    nativeHandle.set(handle)
  }

  companion object {
    init {
      System.loadLibrary("executorch")
    }

    private const val OUTPUT_CHANNELS = 3

    @JvmStatic
    private external fun nativeCreate(
        targetWidth: Int,
        targetHeight: Int,
        resizeMode: Int,
        letterboxAnchor: Int,
        padValue: Float,
        scaleFactor: Float,
        mean: FloatArray,
        standardDeviation: FloatArray,
    ): Long

    @JvmStatic private external fun nativeDestroy(nativeHandle: Long)

    @JvmStatic
    private external fun nativeProcessBitmap(
        nativeHandle: Long,
        bitmap: Bitmap,
        orientationCode: Int,
        outBuffer: FloatBuffer,
        outCapacity: Int,
    )

    @JvmStatic
    private external fun nativeProcessYuv(
        nativeHandle: Long,
        yPlane: ByteBuffer,
        yStride: Int,
        yCapacity: Int,
        uvPlane: ByteBuffer,
        uvStride: Int,
        uvCapacity: Int,
        width: Int,
        height: Int,
        format: Int,
        range: Int,
        orientationCode: Int,
        outBuffer: FloatBuffer,
        outCapacity: Int,
    )

    @JvmStatic
    private external fun nativeComputeOutputShape(
        nativeHandle: Long,
        inputWidth: Int,
        inputHeight: Int,
        orientationCode: Int,
    ): IntArray

    @JvmStatic
    private external fun nativeComputeLetterboxPadding(
        nativeHandle: Long,
        inputWidth: Int,
        inputHeight: Int,
        orientationCode: Int,
    ): IntArray
  }

  /** Check if the native handle is valid (not yet closed). */
  val isValid: Boolean
    get() = nativeHandle.get() != 0L

  /** Releases native resources. Call this when done with the processor. */
  override fun close() {
    val handle = nativeHandle.getAndSet(0L)
    if (handle != 0L) {
      nativeDestroy(handle)
    }
  }

  /**
   * Process an ARGB_8888 bitmap into a normalized float tensor.
   *
   * @param bitmap The source bitmap. Must be [Bitmap.Config.ARGB_8888].
   * @param orientation The EXIF orientation of the bitmap's contents.
   * @return A float Tensor shaped `[1, 3, targetHeight, targetWidth]`.
   * @throws IllegalStateException if the processor has been closed
   * @throws ExecutorchRuntimeException if processing fails
   */
  @JvmOverloads
  fun process(bitmap: Bitmap, orientation: ImageOrientation = ImageOrientation.UP): Tensor {
    val handle = requireHandle()
    val buffer = Tensor.allocateFloatBuffer(outputNumel())
    nativeProcessBitmap(handle, bitmap, orientation.exifCode, buffer, buffer.capacity())
    return Tensor.fromBlob(buffer, outputShape())
  }

  /**
   * Process an ARGB_8888 bitmap into a caller-provided tensor, reusing its storage.
   *
   * Avoids the per-call allocation of [process], which matters for sustained video. `tensor` must
   * be a float Tensor shaped `[1, 3, targetHeight, targetWidth]`; its storage is overwritten, so
   * the caller must finish using the previous contents before calling again.
   *
   * @param bitmap The source bitmap. Must be [Bitmap.Config.ARGB_8888].
   * @param tensor The output tensor to fill.
   * @param orientation The EXIF orientation of the bitmap's contents.
   * @throws IllegalArgumentException if `tensor` has the wrong dtype or shape
   * @throws IllegalStateException if the processor has been closed
   * @throws ExecutorchRuntimeException if processing fails
   */
  @JvmOverloads
  fun processInto(
      bitmap: Bitmap,
      tensor: Tensor,
      orientation: ImageOrientation = ImageOrientation.UP,
  ) {
    val handle = requireHandle()
    val buffer = outputBufferOf(tensor)
    nativeProcessBitmap(handle, bitmap, orientation.exifCode, buffer, buffer.capacity())
  }

  /**
   * Process semi-planar YUV (NV12/NV21) camera planes into a normalized float tensor.
   *
   * For a CameraX `ImageProxy` in `YUV_420_888`, pass `planes[0].buffer` as [yPlane] and the
   * interleaved chroma plane as [uvPlane]: `planes[1].buffer` for [YuvFormat.NV12], or
   * `planes[2].buffer` for [YuvFormat.NV21]. Both buffers must be direct.
   *
   * Only semi-planar chroma is supported. Check `planes[1].pixelStride == 2` before calling;
   * a stride of 1 means fully planar I420, which this path cannot consume.
   *
   * Both buffers are bounds-checked against the strides and dimensions given here. The decode
   * reads the interleaved chroma plane through `uvStride * (height / 2 - 1) + width`, so a plane
   * buffer that a camera HAL trimmed below that is rejected rather than read past its end.
   *
   * @param yPlane Direct buffer holding the luma plane.
   * @param yStride Row stride of the luma plane, in bytes.
   * @param uvPlane Direct buffer holding the interleaved chroma plane.
   * @param uvStride Row stride of the chroma plane, in bytes.
   * @param width Source width in pixels.
   * @param height Source height in pixels.
   * @param format Chroma order of the interleaved plane.
   * @param orientation The EXIF orientation of the frame's contents.
   * @param range Quantization range of the samples.
   * @return A float Tensor shaped `[1, 3, targetHeight, targetWidth]`.
   * @throws IllegalStateException if the processor has been closed
   * @throws ExecutorchRuntimeException if processing fails
   */
  @JvmOverloads
  fun processYuv(
      yPlane: ByteBuffer,
      yStride: Int,
      uvPlane: ByteBuffer,
      uvStride: Int,
      width: Int,
      height: Int,
      format: YuvFormat,
      orientation: ImageOrientation = ImageOrientation.UP,
      range: YuvRange = YuvRange.VIDEO,
  ): Tensor {
    val handle = requireHandle()
    val buffer = Tensor.allocateFloatBuffer(outputNumel())
    nativeProcessYuv(
        handle,
        yPlane,
        yStride,
        yPlane.capacity(),
        uvPlane,
        uvStride,
        uvPlane.capacity(),
        width,
        height,
        format.ordinal,
        range.ordinal,
        orientation.exifCode,
        buffer,
        buffer.capacity(),
    )
    return Tensor.fromBlob(buffer, outputShape())
  }

  /**
   * Process semi-planar YUV camera planes into a caller-provided tensor, reusing its storage.
   *
   * See [processYuv] for the plane contract and [processInto] for the reuse contract.
   *
   * @throws IllegalArgumentException if `tensor` has the wrong dtype or shape
   */
  @JvmOverloads
  fun processYuvInto(
      yPlane: ByteBuffer,
      yStride: Int,
      uvPlane: ByteBuffer,
      uvStride: Int,
      width: Int,
      height: Int,
      format: YuvFormat,
      tensor: Tensor,
      orientation: ImageOrientation = ImageOrientation.UP,
      range: YuvRange = YuvRange.VIDEO,
  ) {
    val handle = requireHandle()
    val buffer = outputBufferOf(tensor)
    nativeProcessYuv(
        handle,
        yPlane,
        yStride,
        yPlane.capacity(),
        uvPlane,
        uvStride,
        uvPlane.capacity(),
        width,
        height,
        format.ordinal,
        range.ordinal,
        orientation.exifCode,
        buffer,
        buffer.capacity(),
    )
  }

  /**
   * Shape of the tensor this processor produces for the given source.
   *
   * @param inputWidth Source width in pixels.
   * @param inputHeight Source height in pixels.
   * @param orientation The EXIF orientation of the source.
   */
  @JvmOverloads
  fun computeOutputShape(
      inputWidth: Int,
      inputHeight: Int,
      orientation: ImageOrientation = ImageOrientation.UP,
  ): LongArray {
    val handle = requireHandle()
    return nativeComputeOutputShape(handle, inputWidth, inputHeight, orientation.exifCode).map {
      it.toLong()
    }
        .toLongArray()
  }

  /**
   * Letterbox padding (per side, in pixels) applied for the given source, letting callers map model
   * output back to source coordinates without replicating the resize geometry. Returns `(0, 0)` for
   * [ResizeMode.STRETCH] or [LetterboxAnchor.TOP_LEFT].
   *
   * @param inputWidth Source width in pixels.
   * @param inputHeight Source height in pixels.
   * @param orientation The EXIF orientation of the source.
   */
  @JvmOverloads
  fun computeLetterboxPadding(
      inputWidth: Int,
      inputHeight: Int,
      orientation: ImageOrientation = ImageOrientation.UP,
  ): LetterboxPadding {
    val handle = requireHandle()
    val padding =
        nativeComputeLetterboxPadding(handle, inputWidth, inputHeight, orientation.exifCode)
    return LetterboxPadding(padding[0], padding[1])
  }

  private fun requireHandle(): Long {
    val handle = nativeHandle.get()
    check(handle != 0L) { "ImageProcessor has been closed" }
    return handle
  }

  private fun outputNumel(): Int = OUTPUT_CHANNELS * config.targetHeight * config.targetWidth

  private fun outputShape(): LongArray =
      longArrayOf(
          1,
          OUTPUT_CHANNELS.toLong(),
          config.targetHeight.toLong(),
          config.targetWidth.toLong(),
      )

  private fun outputBufferOf(tensor: Tensor): FloatBuffer {
    require(tensor.dtype() == DType.FLOAT) { "Output tensor must be float, got ${tensor.dtype()}" }
    require(tensor.shape().contentEquals(outputShape())) {
      "Output tensor must be shaped ${outputShape().contentToString()}, got " +
          tensor.shape().contentToString()
    }
    val buffer = tensor.getRawDataBuffer()
    require(buffer is FloatBuffer && buffer.isDirect) {
      "Output tensor must be backed by a direct FloatBuffer"
    }
    return buffer
  }
}
