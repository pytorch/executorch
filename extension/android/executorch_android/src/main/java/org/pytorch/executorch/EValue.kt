/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.pytorch.executorch

import com.facebook.jni.annotations.DoNotStrip
import java.nio.ByteBuffer
import java.nio.charset.StandardCharsets
import java.util.Arrays
import java.util.Locale
import org.pytorch.executorch.annotations.Experimental

/**
 * Java representation of an ExecuTorch value, which is implemented as tagged union that can be one
 * of the supported types: https://pytorch.org/docs/stable/jit.html#types .
 *
 * Calling `toX` methods for inappropriate types will throw [IllegalStateException].
 *
 * `EValue` objects are constructed with `EValue.from(value)`, depending on the value type.
 *
 * Data is retrieved from `EValue` objects with the `toX()` methods. Note that `str`-type EValues
 * must be extracted with [toStr], rather than [toString].
 *
 * `EValue` objects may retain references to objects passed into their constructors, and may return
 * references to their internal state from `toX()`.
 *
 * Warning: These APIs are experimental and subject to change without notice
 */
@Experimental
@DoNotStrip
class EValue
@DoNotStrip
private constructor(
    // JNI reads this field by name via GetFieldID("mTypeCode")
    @JvmField @DoNotStrip val mTypeCode: Int
) {

  // JNI accesses this field by name via GetFieldID("mData"), requires @JvmField for direct field
  // access
  @JvmField @DoNotStrip var mData: Any? = null

  private val typeNames = arrayOf("None", "Tensor", "String", "Double", "Int", "Bool")

  val isNone: Boolean
    @DoNotStrip get() = TYPE_CODE_NONE == mTypeCode

  val isTensor: Boolean
    @DoNotStrip get() = TYPE_CODE_TENSOR == mTypeCode

  val isBool: Boolean
    @DoNotStrip get() = TYPE_CODE_BOOL == mTypeCode

  val isInt: Boolean
    @DoNotStrip get() = TYPE_CODE_INT == mTypeCode

  val isDouble: Boolean
    @DoNotStrip get() = TYPE_CODE_DOUBLE == mTypeCode

  val isString: Boolean
    @DoNotStrip get() = TYPE_CODE_STRING == mTypeCode

  @DoNotStrip
  fun toTensor(): Tensor {
    preconditionType(TYPE_CODE_TENSOR, mTypeCode)
    return mData as? Tensor ?: throw IllegalStateException("EValue data is null or not a Tensor")
  }

  @DoNotStrip
  fun toBool(): Boolean {
    preconditionType(TYPE_CODE_BOOL, mTypeCode)
    return mData as? Boolean ?: throw IllegalStateException("EValue data is null or not a Boolean")
  }

  @DoNotStrip
  fun toInt(): Long {
    preconditionType(TYPE_CODE_INT, mTypeCode)
    return mData as? Long ?: throw IllegalStateException("EValue data is null or not a Long")
  }

  @DoNotStrip
  fun toDouble(): Double {
    preconditionType(TYPE_CODE_DOUBLE, mTypeCode)
    return mData as? Double ?: throw IllegalStateException("EValue data is null or not a Double")
  }

  @DoNotStrip
  fun toStr(): String {
    preconditionType(TYPE_CODE_STRING, mTypeCode)
    return mData as? String ?: throw IllegalStateException("EValue data is null or not a String")
  }

  private fun preconditionType(typeCodeExpected: Int, typeCode: Int) {
    if (typeCode != typeCodeExpected) {
      throw IllegalStateException(
          String.format(
              Locale.US,
              "Expected EValue type %s, actual type %s",
              getTypeName(typeCodeExpected),
              getTypeName(typeCode),
          )
      )
    }
  }

  private fun getTypeName(typeCode: Int): String =
      if (typeCode in typeNames.indices) typeNames[typeCode] else "Unknown"

  /**
   * Serializes an `EValue` into a byte array. Note: This method is experimental and subject to
   * change without notice.
   */
  fun toByteArray(): ByteArray =
      when {
        isNone -> ByteBuffer.allocate(1).put(TYPE_CODE_NONE.toByte()).array()
        isTensor -> {
          val tByteArray = toTensor().toByteArray()
          ByteBuffer.allocate(1 + tByteArray.size)
              .put(TYPE_CODE_TENSOR.toByte())
              .put(tByteArray)
              .array()
        }
        isBool ->
            ByteBuffer.allocate(2)
                .put(TYPE_CODE_BOOL.toByte())
                .put(if (toBool()) 1.toByte() else 0.toByte())
                .array()
        isInt -> ByteBuffer.allocate(9).put(TYPE_CODE_INT.toByte()).putLong(toInt()).array()
        isDouble ->
            ByteBuffer.allocate(9).put(TYPE_CODE_DOUBLE.toByte()).putDouble(toDouble()).array()
        isString -> {
          val strBytes = toStr().toByteArray(StandardCharsets.UTF_8)
          ByteBuffer.allocate(1 + 4 + strBytes.size)
              .put(TYPE_CODE_STRING.toByte())
              .putInt(strBytes.size)
              .put(strBytes)
              .array()
        }
        else -> throw IllegalArgumentException("Unknown EValue type code: $mTypeCode")
      }

  companion object {
    private const val TYPE_CODE_NONE = 0
    private const val TYPE_CODE_TENSOR = 1
    private const val TYPE_CODE_STRING = 2
    private const val TYPE_CODE_DOUBLE = 3
    private const val TYPE_CODE_INT = 4
    private const val TYPE_CODE_BOOL = 5

    /** Creates a new `EValue` of type `Optional` that contains no value. */
    @DoNotStrip @JvmStatic fun optionalNone(): EValue = EValue(TYPE_CODE_NONE)

    /** Creates a new `EValue` of type `Tensor`. */
    @DoNotStrip
    @JvmStatic
    fun from(tensor: Tensor): EValue = EValue(TYPE_CODE_TENSOR).also { it.mData = tensor }

    /** Creates a new `EValue` of type `bool`. */
    @DoNotStrip
    @JvmStatic
    fun from(value: Boolean): EValue = EValue(TYPE_CODE_BOOL).also { it.mData = value }

    /** Creates a new `EValue` of type `int`. */
    @DoNotStrip
    @JvmStatic
    fun from(value: Long): EValue = EValue(TYPE_CODE_INT).also { it.mData = value }

    /** Creates a new `EValue` of type `double`. */
    @DoNotStrip
    @JvmStatic
    fun from(value: Double): EValue = EValue(TYPE_CODE_DOUBLE).also { it.mData = value }

    /** Creates a new `EValue` of type `str`. */
    @DoNotStrip
    @JvmStatic
    fun from(value: String): EValue = EValue(TYPE_CODE_STRING).also { it.mData = value }

    /**
     * Deserializes an `EValue` from a byte[]. Note: This method is experimental and subject to
     * change without notice.
     */
    @JvmStatic
    fun fromByteArray(bytes: ByteArray): EValue {
      val buffer = ByteBuffer.wrap(bytes)
      require(buffer.hasRemaining()) { "invalid buffer" }
      return when (val typeCode = buffer.get().toInt()) {
        TYPE_CODE_NONE -> EValue(TYPE_CODE_NONE)
        TYPE_CODE_TENSOR -> {
          val bufferArray = buffer.array()
          from(Tensor.fromByteArray(Arrays.copyOfRange(bufferArray, 1, bufferArray.size)))
        }
        TYPE_CODE_STRING -> {
          val strLen = buffer.getInt()
          val strBytes = ByteArray(strLen)
          buffer.get(strBytes)
          from(String(strBytes, StandardCharsets.UTF_8))
        }
        TYPE_CODE_DOUBLE -> from(buffer.getDouble())
        TYPE_CODE_INT -> from(buffer.getLong())
        TYPE_CODE_BOOL -> from(buffer.get().toInt() != 0)
        else -> throw IllegalArgumentException("invalid type code: $typeCode")
      }
    }
  }
}
