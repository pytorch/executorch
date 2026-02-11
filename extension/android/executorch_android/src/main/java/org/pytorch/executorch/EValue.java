/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.pytorch.executorch;

import java.nio.ByteBuffer;
import java.util.Arrays;
import java.util.Locale;
import org.pytorch.executorch.annotations.Experimental;

/**
 * Java representation of an ExecuTorch value, which is implemented as tagged union that can be one
 * of the supported types: https://pytorch.org/docs/stable/jit.html#types .
 *
 * <p>Calling {@code toX} methods for inappropriate types will throw {@link IllegalStateException}.
 *
 * <p>{@code EValue} objects are constructed with {@code EValue.from(value)}, {@code
 * EValue.tupleFrom(value1, value2, ...)}, {@code EValue.listFrom(value1, value2, ...)}, or one of
 * the {@code dict} methods, depending on the key type.
 *
 * <p>Data is retrieved from {@code EValue} objects with the {@code toX()} methods. Note that {@code
 * str}-type EValues must be extracted with {@link #toStr()}, rather than {@link #toString()}.
 *
 * <p>{@code EValue} objects may retain references to objects passed into their constructors, and
 * may return references to their internal state from {@code toX()}.
 *
 * <p>Warning: These APIs are experimental and subject to change without notice
 */
@Experimental
public class EValue {
  private static final int TYPE_CODE_NONE = 0;

  private static final int TYPE_CODE_TENSOR = 1;
  private static final int TYPE_CODE_STRING = 2;
  private static final int TYPE_CODE_DOUBLE = 3;
  private static final int TYPE_CODE_INT = 4;
  private static final int TYPE_CODE_BOOL = 5;

  private String[] TYPE_NAMES = {
    "None", "Tensor", "String", "Double", "Int", "Bool",
  };

  final int mTypeCode;
  Object mData;

  private EValue(int typeCode) {
    this.mTypeCode = typeCode;
  }

  public boolean isNone() {
    return TYPE_CODE_NONE == this.mTypeCode;
  }

  
  public boolean isTensor() {
    return TYPE_CODE_TENSOR == this.mTypeCode;
  }

  
  public boolean isBool() {
    return TYPE_CODE_BOOL == this.mTypeCode;
  }

  
  public boolean isInt() {
    return TYPE_CODE_INT == this.mTypeCode;
  }

  
  public boolean isDouble() {
    return TYPE_CODE_DOUBLE == this.mTypeCode;
  }

  
  public boolean isString() {
    return TYPE_CODE_STRING == this.mTypeCode;
  }

  /** Creates a new {@code EValue} of type {@code Optional} that contains no value. */
  
  public static EValue optionalNone() {
    return new EValue(TYPE_CODE_NONE);
  }

  /** Creates a new {@code EValue} of type {@code Tensor}. */
  
  public static EValue from(Tensor tensor) {
    final EValue iv = new EValue(TYPE_CODE_TENSOR);
    iv.mData = tensor;
    return iv;
  }

  /** Creates a new {@code EValue} of type {@code bool}. */
  
  public static EValue from(boolean value) {
    final EValue iv = new EValue(TYPE_CODE_BOOL);
    iv.mData = value;
    return iv;
  }

  /** Creates a new {@code EValue} of type {@code int}. */
  
  public static EValue from(long value) {
    final EValue iv = new EValue(TYPE_CODE_INT);
    iv.mData = value;
    return iv;
  }

  /** Creates a new {@code EValue} of type {@code double}. */
  
  public static EValue from(double value) {
    final EValue iv = new EValue(TYPE_CODE_DOUBLE);
    iv.mData = value;
    return iv;
  }

  /** Creates a new {@code EValue} of type {@code str}. */
  
  public static EValue from(String value) {
    final EValue iv = new EValue(TYPE_CODE_STRING);
    iv.mData = value;
    return iv;
  }

  
  public Tensor toTensor() {
    preconditionType(TYPE_CODE_TENSOR, mTypeCode);
    return (Tensor) mData;
  }

  
  public boolean toBool() {
    preconditionType(TYPE_CODE_BOOL, mTypeCode);
    return (boolean) mData;
  }

  
  public long toInt() {
    preconditionType(TYPE_CODE_INT, mTypeCode);
    return (long) mData;
  }

  
  public double toDouble() {
    preconditionType(TYPE_CODE_DOUBLE, mTypeCode);
    return (double) mData;
  }

  
  public String toStr() {
    preconditionType(TYPE_CODE_STRING, mTypeCode);
    return (String) mData;
  }

  private void preconditionType(int typeCodeExpected, int typeCode) {
    if (typeCode != typeCodeExpected) {
      throw new IllegalStateException(
          String.format(
              Locale.US,
              "Expected EValue type %s, actual type %s",
              getTypeName(typeCodeExpected),
              getTypeName(typeCode)));
    }
  }

  private String getTypeName(int typeCode) {
    return typeCode >= 0 && typeCode < TYPE_NAMES.length ? TYPE_NAMES[typeCode] : "Unknown";
  }

  /**
   * Serializes an {@code EValue} into a byte array. Note: This method is experimental and subject
   * to change without notice.
   *
   * @return The serialized byte array.
   */
  public byte[] toByteArray() {
    if (isNone()) {
      return ByteBuffer.allocate(1).put((byte) TYPE_CODE_NONE).array();
    } else if (isTensor()) {
      Tensor t = toTensor();
      byte[] tByteArray = t.toByteArray();
      return ByteBuffer.allocate(1 + tByteArray.length)
          .put((byte) TYPE_CODE_TENSOR)
          .put(tByteArray)
          .array();
    } else if (isBool()) {
      return ByteBuffer.allocate(2)
          .put((byte) TYPE_CODE_BOOL)
          .put((byte) (toBool() ? 1 : 0))
          .array();
    } else if (isInt()) {
      return ByteBuffer.allocate(9).put((byte) TYPE_CODE_INT).putLong(toInt()).array();
    } else if (isDouble()) {
      return ByteBuffer.allocate(9).put((byte) TYPE_CODE_DOUBLE).putDouble(toDouble()).array();
    } else if (isString()) {
      return ByteBuffer.allocate(1 + toString().length())
          .put((byte) TYPE_CODE_STRING)
          .put(toString().getBytes())
          .array();
    } else {
      throw new IllegalArgumentException("Unknown Tensor dtype");
    }
  }

  /**
   * Deserializes an {@code EValue} from a byte[]. Note: This method is experimental and subject to
   * change without notice.
   *
   * @param bytes The byte array to deserialize from.
   * @return The deserialized {@code EValue}.
   */
  public static EValue fromByteArray(byte[] bytes) {
    ByteBuffer buffer = ByteBuffer.wrap(bytes);
    if (buffer == null) {
      throw new IllegalArgumentException("buffer cannot be null");
    }
    if (!buffer.hasRemaining()) {
      throw new IllegalArgumentException("invalid buffer");
    }
    int typeCode = buffer.get();
    switch (typeCode) {
      case TYPE_CODE_NONE:
        return new EValue(TYPE_CODE_NONE);
      case TYPE_CODE_TENSOR:
        byte[] bufferArray = buffer.array();
        return from(Tensor.fromByteArray(Arrays.copyOfRange(bufferArray, 1, bufferArray.length)));
      case TYPE_CODE_STRING:
        throw new IllegalArgumentException("TYPE_CODE_STRING is not supported");
      case TYPE_CODE_DOUBLE:
        return from(buffer.getDouble());
      case TYPE_CODE_INT:
        return from(buffer.getLong());
      case TYPE_CODE_BOOL:
        return from(buffer.get() != 0);
    }
    throw new IllegalArgumentException("invalid type code: " + typeCode);
  }
}
