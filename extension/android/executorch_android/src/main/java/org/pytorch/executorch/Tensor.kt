/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.pytorch.executorch

import android.util.Log
import com.facebook.jni.HybridData
import com.facebook.jni.annotations.DoNotStrip
import java.nio.Buffer
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.DoubleBuffer
import java.nio.FloatBuffer
import java.nio.IntBuffer
import java.nio.LongBuffer
import java.nio.ShortBuffer
import java.util.Arrays
import java.util.Locale
import org.pytorch.executorch.annotations.Experimental

/**
 * Representation of an ExecuTorch Tensor. Behavior is similar to PyTorch's tensor objects.
 *
 * Most tensors will be constructed as `Tensor.fromBlob(data, shape)`, where `data` can be an array
 * or a direct [Buffer] (of the proper subclass). Helper methods are provided to allocate buffers
 * properly.
 *
 * To access Tensor data, see [dtype], [shape], and various `dataAs*` properties.
 *
 * When constructing `Tensor` objects with `data` as an array, it is not specified whether this data
 * is copied or retained as a reference so it is recommended not to modify it after constructing.
 * `data` passed as a [Buffer] is not copied, so it can be modified between [Module] calls to avoid
 * reallocation. Data retrieved from `Tensor` objects may be copied or may be a reference to the
 * `Tensor`'s internal data buffer. `shape` is always copied.
 *
 * Warning: These APIs are experimental and subject to change without notice
 */
@Experimental
abstract class Tensor internal constructor(shape: LongArray) {

  init {
    for (s in shape) {
      require(s >= 0) { "Shape elements must be non negative" }
    }
  }

  @DoNotStrip @JvmField protected val shape: LongArray = shape.copyOf()

  @DoNotStrip private var mHybridData: HybridData? = null

  /** Returns the number of elements in this tensor. */
  fun numel(): Long = numel(shape)

  /** Returns the shape of this tensor. (The array is a fresh copy.) */
  fun shape(): LongArray = shape.copyOf()

  abstract fun dtype(): DType

  // Called from native via JNI GetMethodID — must not be `internal` (name mangling breaks lookup)
  @DoNotStrip fun dtypeJniCode(): Int = dtype().jniCode

  open val dataAsByteArray: ByteArray
    get() =
        throw IllegalStateException(
            "Tensor of type ${javaClass.simpleName} cannot return data as byte array."
        )

  open val dataAsShortArray: ShortArray
    get() =
        throw IllegalStateException(
            "Tensor of type ${javaClass.simpleName} cannot return data as short array."
        )

  open val dataAsUnsignedByteArray: ByteArray
    get() =
        throw IllegalStateException(
            "Tensor of type ${javaClass.simpleName} cannot return data as unsigned byte array."
        )

  open val dataAsIntArray: IntArray
    get() =
        throw IllegalStateException(
            "Tensor of type ${javaClass.simpleName} cannot return data as int array."
        )

  open val dataAsFloatArray: FloatArray
    get() =
        throw IllegalStateException(
            "Tensor of type ${javaClass.simpleName} cannot return data as float array."
        )

  /**
   * Copies the tensor's data into a caller-provided [FloatBuffer], avoiding the per-call allocation
   * that [dataAsFloatArray] performs.
   *
   * Supported by float32 (zero-copy bulk put) and float16 (per-element half-to-float widening). For
   * raw fp16 bits without widening, use [copyDataInto(ShortBuffer)][copyDataInto].
   */
  open fun copyDataInto(dst: FloatBuffer) {
    throw IllegalStateException(
        "Tensor of type ${javaClass.simpleName} cannot copy data into FloatBuffer."
    )
  }

  open fun copyDataInto(dst: ByteBuffer) {
    throw IllegalStateException(
        "Tensor of type ${javaClass.simpleName} cannot copy data into ByteBuffer."
    )
  }

  open fun copyDataIntoUnsigned(dst: ByteBuffer) {
    throw IllegalStateException(
        "Tensor of type ${javaClass.simpleName} cannot copy data into ByteBuffer (unsigned)."
    )
  }

  open fun copyDataInto(dst: IntBuffer) {
    throw IllegalStateException(
        "Tensor of type ${javaClass.simpleName} cannot copy data into IntBuffer."
    )
  }

  open fun copyDataInto(dst: LongBuffer) {
    throw IllegalStateException(
        "Tensor of type ${javaClass.simpleName} cannot copy data into LongBuffer."
    )
  }

  open fun copyDataInto(dst: DoubleBuffer) {
    throw IllegalStateException(
        "Tensor of type ${javaClass.simpleName} cannot copy data into DoubleBuffer."
    )
  }

  open fun copyDataInto(dst: ShortBuffer) {
    throw IllegalStateException(
        "Tensor of type ${javaClass.simpleName} cannot copy data into ShortBuffer."
    )
  }

  open val dataAsLongArray: LongArray
    get() =
        throw IllegalStateException(
            "Tensor of type ${javaClass.simpleName} cannot return data as long array."
        )

  open val dataAsDoubleArray: DoubleArray
    get() =
        throw IllegalStateException(
            "Tensor of type ${javaClass.simpleName} cannot return data as double array."
        )

  @DoNotStrip
  open fun getRawDataBuffer(): Buffer =
      throw IllegalStateException(
          "Tensor of type ${javaClass.simpleName} cannot return raw data buffer."
      )

  /**
   * Serializes a `Tensor` into a byte array. Note: This method is experimental and subject to
   * change without notice. This does NOT support list type.
   */
  fun toByteArray(): ByteArray {
    var dtypeSize: Int
    val tensorAsByteArray: ByteArray =
        when (dtype()) {
          DType.UINT8 -> {
            dtypeSize = BYTE_SIZE_BYTES
            val arr = ByteArray(numel().toInt())
            ByteBuffer.wrap(arr).put((this as Tensor_uint8).dataAsUnsignedByteArray)
            arr
          }
          DType.INT8 -> {
            dtypeSize = BYTE_SIZE_BYTES
            val arr = ByteArray(numel().toInt())
            ByteBuffer.wrap(arr).put((this as Tensor_int8).dataAsByteArray)
            arr
          }
          DType.HALF -> {
            dtypeSize = HALF_SIZE_BYTES
            val arr = ByteArray(numel().toInt() * HALF_SIZE_BYTES)
            ByteBuffer.wrap(arr).asShortBuffer().put((this as Tensor_float16).dataAsShortArray)
            arr
          }
          DType.INT16 ->
              throw IllegalArgumentException("DType.INT16 is not supported in Java so far")
          DType.INT32 -> {
            dtypeSize = INT_SIZE_BYTES
            val arr = ByteArray(numel().toInt() * INT_SIZE_BYTES)
            ByteBuffer.wrap(arr).asIntBuffer().put((this as Tensor_int32).dataAsIntArray)
            arr
          }
          DType.INT64 -> {
            dtypeSize = LONG_SIZE_BYTES
            val arr = ByteArray(numel().toInt() * LONG_SIZE_BYTES)
            ByteBuffer.wrap(arr).asLongBuffer().put((this as Tensor_int64).dataAsLongArray)
            arr
          }
          DType.FLOAT -> {
            dtypeSize = FLOAT_SIZE_BYTES
            val arr = ByteArray(numel().toInt() * FLOAT_SIZE_BYTES)
            ByteBuffer.wrap(arr).asFloatBuffer().put((this as Tensor_float32).dataAsFloatArray)
            arr
          }
          DType.DOUBLE -> {
            dtypeSize = DOUBLE_SIZE_BYTES
            val arr = ByteArray(numel().toInt() * DOUBLE_SIZE_BYTES)
            ByteBuffer.wrap(arr).asDoubleBuffer().put((this as Tensor_float64).dataAsDoubleArray)
            arr
          }
          else -> throw IllegalArgumentException("Unknown Tensor dtype")
        }
    val byteBuffer = ByteBuffer.allocate(1 + 1 + 4 * shape.size + dtypeSize * numel().toInt())
    byteBuffer.put(dtype().jniCode.toByte())
    byteBuffer.put(shape.size.toByte())
    for (s in shape) {
      byteBuffer.putInt(s.toInt())
    }
    byteBuffer.put(tensorAsByteArray)
    return byteBuffer.array()
  }

  // region nested tensor types

  internal class Tensor_uint8 internal constructor(private val data: ByteBuffer, shape: LongArray) :
      Tensor(shape) {
    override fun dtype(): DType = DType.UINT8

    override fun getRawDataBuffer(): Buffer = data

    override val dataAsUnsignedByteArray: ByteArray
      get() {
        data.rewind()
        val arr = ByteArray(data.remaining())
        data.get(arr)
        return arr
      }

    override fun copyDataIntoUnsigned(dst: ByteBuffer) {
      data.rewind()
      dst.put(data)
    }

    override fun toString(): String = "Tensor(${Arrays.toString(shape)}, dtype=torch.uint8)"
  }

  internal class Tensor_int8 internal constructor(private val data: ByteBuffer, shape: LongArray) :
      Tensor(shape) {
    override fun dtype(): DType = DType.INT8

    override fun getRawDataBuffer(): Buffer = data

    override val dataAsByteArray: ByteArray
      get() {
        data.rewind()
        val arr = ByteArray(data.remaining())
        data.get(arr)
        return arr
      }

    override fun copyDataInto(dst: ByteBuffer) {
      data.rewind()
      dst.put(data)
    }

    override fun toString(): String = "Tensor(${Arrays.toString(shape)}, dtype=torch.int8)"
  }

  internal class Tensor_int32 internal constructor(private val data: IntBuffer, shape: LongArray) :
      Tensor(shape) {
    override fun dtype(): DType = DType.INT32

    override fun getRawDataBuffer(): Buffer = data

    override val dataAsIntArray: IntArray
      get() {
        data.rewind()
        val arr = IntArray(data.remaining())
        data.get(arr)
        return arr
      }

    override fun copyDataInto(dst: IntBuffer) {
      data.rewind()
      dst.put(data)
    }

    override fun toString(): String = "Tensor(${Arrays.toString(shape)}, dtype=torch.int32)"
  }

  internal class Tensor_float32
  internal constructor(private val data: FloatBuffer, shape: LongArray) : Tensor(shape) {
    override fun dtype(): DType = DType.FLOAT

    override fun getRawDataBuffer(): Buffer = data

    override val dataAsFloatArray: FloatArray
      get() {
        data.rewind()
        val arr = FloatArray(data.remaining())
        data.get(arr)
        return arr
      }

    override fun copyDataInto(dst: FloatBuffer) {
      data.rewind()
      dst.put(data)
    }

    override fun toString(): String = "Tensor(${Arrays.toString(shape)}, dtype=torch.float32)"
  }

  internal class Tensor_float16
  internal constructor(private val data: ShortBuffer, shape: LongArray) : Tensor(shape) {
    override fun dtype(): DType = DType.HALF

    override fun getRawDataBuffer(): Buffer = data

    override val dataAsShortArray: ShortArray
      get() {
        data.rewind()
        val arr = ShortArray(data.remaining())
        data.get(arr)
        return arr
      }

    override fun copyDataInto(dst: ShortBuffer) {
      data.rewind()
      dst.put(data)
    }

    override val dataAsFloatArray: FloatArray
      get() {
        data.rewind()
        val remaining = data.remaining()
        val arr = FloatArray(remaining)
        for (i in 0 until remaining) {
          arr[i] = halfBitsToFloat(data.get())
        }
        return arr
      }

    override fun copyDataInto(dst: FloatBuffer) {
      data.rewind()
      val remaining = data.remaining()
      if (dst.remaining() < remaining) {
        throw java.nio.BufferOverflowException()
      }
      for (i in 0 until remaining) {
        dst.put(halfBitsToFloat(data.get()))
      }
    }

    override fun toString(): String = "Tensor(${Arrays.toString(shape)}, dtype=torch.float16)"

    companion object {
      private fun halfBitsToFloat(halfBits: Short): Float {
        val h = halfBits.toInt() and 0xFFFF
        val sign = (h ushr 15) and 0x1
        val exp = (h ushr 10) and 0x1F
        val mant = h and 0x3FF

        if (exp == 0) {
          if (mant == 0) {
            return if (sign == 0) 0.0f else -0.0f
          }
          val result = mant * 5.9604645e-8f // 2^-24
          return if (sign == 0) result else -result
        } else if (exp == 0x1F) {
          if (mant == 0) {
            return if (sign == 0) Float.POSITIVE_INFINITY else Float.NEGATIVE_INFINITY
          }
          val bits = (sign shl 31) or 0x7f800000 or (mant shl 13)
          return Float.fromBits(bits)
        } else {
          val exp32 = exp + 112 // 127 (float bias) - 15 (half bias)
          val bits = (sign shl 31) or (exp32 shl 23) or (mant shl 13)
          return Float.fromBits(bits)
        }
      }
    }
  }

  internal class Tensor_int64 internal constructor(private val data: LongBuffer, shape: LongArray) :
      Tensor(shape) {
    override fun dtype(): DType = DType.INT64

    override fun getRawDataBuffer(): Buffer = data

    override val dataAsLongArray: LongArray
      get() {
        data.rewind()
        val arr = LongArray(data.remaining())
        data.get(arr)
        return arr
      }

    override fun copyDataInto(dst: LongBuffer) {
      data.rewind()
      dst.put(data)
    }

    override fun toString(): String = "Tensor(${Arrays.toString(shape)}, dtype=torch.int64)"
  }

  internal class Tensor_float64
  internal constructor(private val data: DoubleBuffer, shape: LongArray) : Tensor(shape) {
    override fun dtype(): DType = DType.DOUBLE

    override fun getRawDataBuffer(): Buffer = data

    override val dataAsDoubleArray: DoubleArray
      get() {
        data.rewind()
        val arr = DoubleArray(data.remaining())
        data.get(arr)
        return arr
      }

    override fun copyDataInto(dst: DoubleBuffer) {
      data.rewind()
      dst.put(data)
    }

    override fun toString(): String = "Tensor(${Arrays.toString(shape)}, dtype=torch.float64)"
  }

  internal class Tensor_unsupported
  internal constructor(
      private val data: ByteBuffer,
      shape: LongArray,
      private val mDtype: DType,
  ) : Tensor(shape) {
    init {
      Log.e("ExecuTorch", "$this. Please consider re-exporting the model with a proper return type")
    }

    override fun dtype(): DType = mDtype

    override fun toString(): String = "Unsupported tensor(${Arrays.toString(shape)}, dtype=$mDtype)"
  }

  // endregion nested tensor types

  companion object {
    private const val ERROR_MSG_SHAPE_NON_NEGATIVE = "Shape elements must be non negative"
    private const val ERROR_MSG_DATA_BUFFER_MUST_HAVE_NATIVE_BYTE_ORDER =
        "Data buffer must have native byte order (java.nio.ByteOrder#nativeOrder)"
    private const val ERROR_MSG_DATA_BUFFER_MUST_BE_DIRECT =
        "Data buffer must be direct (java.nio.ByteBuffer#allocateDirect)"

    private const val BYTE_SIZE_BYTES = 1
    private const val INT_SIZE_BYTES = 4
    private const val LONG_SIZE_BYTES = 8
    private const val HALF_SIZE_BYTES = 2
    private const val FLOAT_SIZE_BYTES = 4
    private const val DOUBLE_SIZE_BYTES = 8

    @JvmStatic
    fun allocateByteBuffer(numElements: Int): ByteBuffer =
        ByteBuffer.allocateDirect(numElements).order(ByteOrder.nativeOrder())

    @JvmStatic
    fun allocateIntBuffer(numElements: Int): IntBuffer =
        ByteBuffer.allocateDirect(numElements * INT_SIZE_BYTES)
            .order(ByteOrder.nativeOrder())
            .asIntBuffer()

    @JvmStatic
    fun allocateFloatBuffer(numElements: Int): FloatBuffer =
        ByteBuffer.allocateDirect(numElements * FLOAT_SIZE_BYTES)
            .order(ByteOrder.nativeOrder())
            .asFloatBuffer()

    @JvmStatic
    fun allocateLongBuffer(numElements: Int): LongBuffer =
        ByteBuffer.allocateDirect(numElements * LONG_SIZE_BYTES)
            .order(ByteOrder.nativeOrder())
            .asLongBuffer()

    @JvmStatic
    fun allocateHalfBuffer(numElements: Int): ShortBuffer =
        ByteBuffer.allocateDirect(numElements * HALF_SIZE_BYTES)
            .order(ByteOrder.nativeOrder())
            .asShortBuffer()

    @JvmStatic
    fun allocateDoubleBuffer(numElements: Int): DoubleBuffer =
        ByteBuffer.allocateDirect(numElements * DOUBLE_SIZE_BYTES)
            .order(ByteOrder.nativeOrder())
            .asDoubleBuffer()

    // region fromBlob (array)

    @JvmStatic
    fun fromBlobUnsigned(data: ByteArray, shape: LongArray): Tensor {
      checkShape(shape)
      checkShapeAndDataCapacityConsistency(data.size, shape)
      val byteBuffer = allocateByteBuffer(numel(shape).toInt())
      byteBuffer.put(data)
      return Tensor_uint8(byteBuffer, shape)
    }

    @JvmStatic
    fun fromBlob(data: ByteArray, shape: LongArray): Tensor {
      checkShape(shape)
      checkShapeAndDataCapacityConsistency(data.size, shape)
      val byteBuffer = allocateByteBuffer(numel(shape).toInt())
      byteBuffer.put(data)
      return Tensor_int8(byteBuffer, shape)
    }

    @JvmStatic
    fun fromBlob(data: IntArray, shape: LongArray): Tensor {
      checkShape(shape)
      checkShapeAndDataCapacityConsistency(data.size, shape)
      val intBuffer = allocateIntBuffer(numel(shape).toInt())
      intBuffer.put(data)
      return Tensor_int32(intBuffer, shape)
    }

    @JvmStatic
    fun fromBlob(data: FloatArray, shape: LongArray): Tensor {
      checkShape(shape)
      checkShapeAndDataCapacityConsistency(data.size, shape)
      val floatBuffer = allocateFloatBuffer(numel(shape).toInt())
      floatBuffer.put(data)
      return Tensor_float32(floatBuffer, shape)
    }

    @JvmStatic
    fun fromBlob(data: ShortArray, shape: LongArray): Tensor {
      checkShape(shape)
      checkShapeAndDataCapacityConsistency(data.size, shape)
      val shortBuffer = allocateHalfBuffer(numel(shape).toInt())
      shortBuffer.put(data)
      return Tensor_float16(shortBuffer, shape)
    }

    @JvmStatic
    fun fromBlob(data: LongArray, shape: LongArray): Tensor {
      checkShape(shape)
      checkShapeAndDataCapacityConsistency(data.size, shape)
      val longBuffer = allocateLongBuffer(numel(shape).toInt())
      longBuffer.put(data)
      return Tensor_int64(longBuffer, shape)
    }

    @JvmStatic
    fun fromBlob(data: DoubleArray, shape: LongArray): Tensor {
      checkShape(shape)
      checkShapeAndDataCapacityConsistency(data.size, shape)
      val doubleBuffer = allocateDoubleBuffer(numel(shape).toInt())
      doubleBuffer.put(data)
      return Tensor_float64(doubleBuffer, shape)
    }

    // endregion fromBlob (array)

    // region fromBlob (buffer)

    @JvmStatic
    fun fromBlobUnsigned(data: ByteBuffer, shape: LongArray): Tensor {
      checkShape(shape)
      checkShapeAndDataCapacityConsistency(data.capacity(), shape)
      checkArgument(data.isDirect, ERROR_MSG_DATA_BUFFER_MUST_BE_DIRECT)
      checkArgument(
          data.order() == ByteOrder.nativeOrder(),
          ERROR_MSG_DATA_BUFFER_MUST_HAVE_NATIVE_BYTE_ORDER,
      )
      return Tensor_uint8(data, shape)
    }

    @JvmStatic
    fun fromBlob(data: ByteBuffer, shape: LongArray): Tensor {
      checkShape(shape)
      checkShapeAndDataCapacityConsistency(data.capacity(), shape)
      checkArgument(data.isDirect, ERROR_MSG_DATA_BUFFER_MUST_BE_DIRECT)
      checkArgument(
          data.order() == ByteOrder.nativeOrder(),
          ERROR_MSG_DATA_BUFFER_MUST_HAVE_NATIVE_BYTE_ORDER,
      )
      return Tensor_int8(data, shape)
    }

    @JvmStatic
    fun fromBlob(data: IntBuffer, shape: LongArray): Tensor {
      checkShape(shape)
      checkShapeAndDataCapacityConsistency(data.capacity(), shape)
      checkArgument(data.isDirect, ERROR_MSG_DATA_BUFFER_MUST_BE_DIRECT)
      checkArgument(
          data.order() == ByteOrder.nativeOrder(),
          ERROR_MSG_DATA_BUFFER_MUST_HAVE_NATIVE_BYTE_ORDER,
      )
      return Tensor_int32(data, shape)
    }

    @JvmStatic
    fun fromBlob(data: FloatBuffer, shape: LongArray): Tensor {
      checkShape(shape)
      checkShapeAndDataCapacityConsistency(data.capacity(), shape)
      checkArgument(data.isDirect, ERROR_MSG_DATA_BUFFER_MUST_BE_DIRECT)
      checkArgument(
          data.order() == ByteOrder.nativeOrder(),
          ERROR_MSG_DATA_BUFFER_MUST_HAVE_NATIVE_BYTE_ORDER,
      )
      return Tensor_float32(data, shape)
    }

    @JvmStatic
    fun fromBlob(data: ShortBuffer, shape: LongArray): Tensor {
      checkShape(shape)
      checkShapeAndDataCapacityConsistency(data.capacity(), shape)
      checkArgument(data.isDirect, ERROR_MSG_DATA_BUFFER_MUST_BE_DIRECT)
      checkArgument(
          data.order() == ByteOrder.nativeOrder(),
          ERROR_MSG_DATA_BUFFER_MUST_HAVE_NATIVE_BYTE_ORDER,
      )
      return Tensor_float16(data, shape)
    }

    @JvmStatic
    fun fromBlob(data: LongBuffer, shape: LongArray): Tensor {
      checkShape(shape)
      checkShapeAndDataCapacityConsistency(data.capacity(), shape)
      checkArgument(data.isDirect, ERROR_MSG_DATA_BUFFER_MUST_BE_DIRECT)
      checkArgument(
          data.order() == ByteOrder.nativeOrder(),
          ERROR_MSG_DATA_BUFFER_MUST_HAVE_NATIVE_BYTE_ORDER,
      )
      return Tensor_int64(data, shape)
    }

    @JvmStatic
    fun fromBlob(data: DoubleBuffer, shape: LongArray): Tensor {
      checkShape(shape)
      checkShapeAndDataCapacityConsistency(data.capacity(), shape)
      checkArgument(data.isDirect, ERROR_MSG_DATA_BUFFER_MUST_BE_DIRECT)
      checkArgument(
          data.order() == ByteOrder.nativeOrder(),
          ERROR_MSG_DATA_BUFFER_MUST_HAVE_NATIVE_BYTE_ORDER,
      )
      return Tensor_float64(data, shape)
    }

    // endregion fromBlob (buffer)

    @JvmStatic
    fun ones(shape: LongArray, dtype: DType): Tensor {
      checkShape(shape)
      val numElements = numel(shape).toInt()
      return when (dtype) {
        DType.UINT8 -> fromBlobUnsigned(ByteArray(numElements) { 1 }, shape)
        DType.INT8 -> fromBlob(ByteArray(numElements) { 1 }, shape)
        DType.INT32 -> fromBlob(IntArray(numElements) { 1 }, shape)
        DType.FLOAT -> fromBlob(FloatArray(numElements) { 1.0f }, shape)
        DType.INT64 -> fromBlob(LongArray(numElements) { 1L }, shape)
        DType.DOUBLE -> fromBlob(DoubleArray(numElements) { 1.0 }, shape)
        else -> throw IllegalArgumentException("Tensor.ones() cannot be used with DType $dtype")
      }
    }

    @JvmStatic
    fun zeros(shape: LongArray, dtype: DType): Tensor {
      checkShape(shape)
      val numElements = numel(shape).toInt()
      return when (dtype) {
        DType.UINT8 -> fromBlobUnsigned(ByteArray(numElements), shape)
        DType.INT8 -> fromBlob(ByteArray(numElements), shape)
        DType.INT32 -> fromBlob(IntArray(numElements), shape)
        DType.FLOAT -> fromBlob(FloatArray(numElements), shape)
        DType.INT64 -> fromBlob(LongArray(numElements), shape)
        DType.DOUBLE -> fromBlob(DoubleArray(numElements), shape)
        else -> throw IllegalArgumentException("Tensor.zeros() cannot be used with DType $dtype")
      }
    }

    /** Calculates the number of elements in a tensor with the specified shape. */
    @JvmStatic
    fun numel(shape: LongArray): Long {
      checkShape(shape)
      var result = 1L
      for (s in shape) {
        result *= s
      }
      return result
    }

    // Called from native
    @DoNotStrip
    @JvmStatic
    private fun nativeNewTensor(
        data: ByteBuffer,
        shape: LongArray,
        dtype: Int,
        hybridData: HybridData,
    ): Tensor {
      val tensor =
          when {
            DType.FLOAT.jniCode == dtype -> Tensor_float32(data.asFloatBuffer(), shape)
            DType.HALF.jniCode == dtype -> Tensor_float16(data.asShortBuffer(), shape)
            DType.INT32.jniCode == dtype -> Tensor_int32(data.asIntBuffer(), shape)
            DType.INT64.jniCode == dtype -> Tensor_int64(data.asLongBuffer(), shape)
            DType.DOUBLE.jniCode == dtype -> Tensor_float64(data.asDoubleBuffer(), shape)
            DType.UINT8.jniCode == dtype -> Tensor_uint8(data, shape)
            DType.INT8.jniCode == dtype -> Tensor_int8(data, shape)
            else -> Tensor_unsupported(data, shape, DType.fromJniCode(dtype))
          }
      tensor.mHybridData = hybridData
      return tensor
    }

    /**
     * Deserializes a `Tensor` from a byte array. Note: This method is experimental and subject to
     * change without notice. This does NOT support list type.
     */
    @JvmStatic
    fun fromByteArray(bytes: ByteArray): Tensor {
      val buffer = ByteBuffer.wrap(bytes)
      require(buffer.hasRemaining()) { "invalid buffer" }
      val dtype = buffer.get()
      val shapeLength = buffer.get()
      val shape = LongArray(shapeLength.toInt())
      for (i in shape.indices) {
        val dim = buffer.getInt()
        require(dim >= 0) { "invalid shape" }
        shape[i] = dim.toLong()
      }
      return when (dtype.toInt()) {
        DType.UINT8.jniCode -> Tensor_uint8(buffer, shape)
        DType.INT8.jniCode -> Tensor_int8(buffer, shape)
        DType.HALF.jniCode -> Tensor_float16(buffer.asShortBuffer(), shape)
        DType.INT32.jniCode -> Tensor_int32(buffer.asIntBuffer(), shape)
        DType.INT64.jniCode -> Tensor_int64(buffer.asLongBuffer(), shape)
        DType.FLOAT.jniCode -> Tensor_float32(buffer.asFloatBuffer(), shape)
        DType.DOUBLE.jniCode -> Tensor_float64(buffer.asDoubleBuffer(), shape)
        else -> throw IllegalArgumentException("Unknown Tensor dtype")
      }
    }

    // region checks
    private fun checkArgument(expression: Boolean, errorMessage: String, vararg args: Any) {
      if (!expression) {
        throw IllegalArgumentException(String.format(Locale.US, errorMessage, *args))
      }
    }

    private fun checkShape(shape: LongArray) {
      for (s in shape) {
        checkArgument(s >= 0, ERROR_MSG_SHAPE_NON_NEGATIVE)
      }
    }

    private fun checkShapeAndDataCapacityConsistency(dataCapacity: Int, shape: LongArray) {
      val numel = numel(shape)
      checkArgument(
          numel == dataCapacity.toLong(),
          "Inconsistent data capacity:%d and shape number elements:%d shape:%s",
          dataCapacity,
          numel,
          Arrays.toString(shape),
      )
    }
    // endregion checks
  }
}
