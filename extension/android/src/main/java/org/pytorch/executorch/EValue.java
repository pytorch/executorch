/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.pytorch.executorch;

import com.facebook.jni.annotations.DoNotStrip;
import java.util.Locale;
import java.util.Optional;
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
@DoNotStrip
public class EValue {
  private static final int TYPE_CODE_NONE = 0;

  private static final int TYPE_CODE_TENSOR = 1;
  private static final int TYPE_CODE_STRING = 2;
  private static final int TYPE_CODE_DOUBLE = 3;
  private static final int TYPE_CODE_INT = 4;
  private static final int TYPE_CODE_BOOL = 5;

  private static final int TYPE_CODE_LIST_BOOL = 6;
  private static final int TYPE_CODE_LIST_DOUBLE = 7;
  private static final int TYPE_CODE_LIST_INT = 8;
  private static final int TYPE_CODE_LIST_TENSOR = 9;
  private static final int TYPE_CODE_LIST_SCALAR = 10;
  private static final int TYPE_CODE_LIST_OPTIONAL_TENSOR = 11;

  private String[] TYPE_NAMES = {
    "None",
    "Tensor",
    "String",
    "Double",
    "Int",
    "Bool",
    "ListBool",
    "ListDouble",
    "ListInt",
    "ListTensor",
    "ListScalar",
    "ListOptionalScalar",
  };

  @DoNotStrip private final int mTypeCode;
  @DoNotStrip private Object mData;

  @DoNotStrip
  private EValue(int typeCode) {
    this.mTypeCode = typeCode;
  }

  @DoNotStrip
  public boolean isNone() {
    return TYPE_CODE_NONE == this.mTypeCode;
  }

  @DoNotStrip
  public boolean isTensor() {
    return TYPE_CODE_TENSOR == this.mTypeCode;
  }

  @DoNotStrip
  public boolean isBool() {
    return TYPE_CODE_BOOL == this.mTypeCode;
  }

  @DoNotStrip
  public boolean isInt() {
    return TYPE_CODE_INT == this.mTypeCode;
  }

  @DoNotStrip
  public boolean isDouble() {
    return TYPE_CODE_DOUBLE == this.mTypeCode;
  }

  @DoNotStrip
  public boolean isString() {
    return TYPE_CODE_STRING == this.mTypeCode;
  }

  @DoNotStrip
  public boolean isBoolList() {
    return TYPE_CODE_LIST_BOOL == this.mTypeCode;
  }

  @DoNotStrip
  public boolean isIntList() {
    return TYPE_CODE_LIST_INT == this.mTypeCode;
  }

  @DoNotStrip
  public boolean isDoubleList() {
    return TYPE_CODE_LIST_DOUBLE == this.mTypeCode;
  }

  @DoNotStrip
  public boolean isTensorList() {
    return TYPE_CODE_LIST_TENSOR == this.mTypeCode;
  }

  @DoNotStrip
  public boolean isOptionalTensorList() {
    return TYPE_CODE_LIST_OPTIONAL_TENSOR == this.mTypeCode;
  }

  /** Creates a new {@code EValue} of type {@code Optional} that contains no value. */
  @DoNotStrip
  public static EValue optionalNone() {
    return new EValue(TYPE_CODE_NONE);
  }

  /** Creates a new {@code EValue} of type {@code Tensor}. */
  @DoNotStrip
  public static EValue from(Tensor tensor) {
    final EValue iv = new EValue(TYPE_CODE_TENSOR);
    iv.mData = tensor;
    return iv;
  }

  /** Creates a new {@code EValue} of type {@code bool}. */
  @DoNotStrip
  public static EValue from(boolean value) {
    final EValue iv = new EValue(TYPE_CODE_BOOL);
    iv.mData = value;
    return iv;
  }

  /** Creates a new {@code EValue} of type {@code int}. */
  @DoNotStrip
  public static EValue from(long value) {
    final EValue iv = new EValue(TYPE_CODE_INT);
    iv.mData = value;
    return iv;
  }

  /** Creates a new {@code EValue} of type {@code double}. */
  @DoNotStrip
  public static EValue from(double value) {
    final EValue iv = new EValue(TYPE_CODE_DOUBLE);
    iv.mData = value;
    return iv;
  }

  /** Creates a new {@code EValue} of type {@code str}. */
  @DoNotStrip
  public static EValue from(String value) {
    final EValue iv = new EValue(TYPE_CODE_STRING);
    iv.mData = value;
    return iv;
  }

  /** Creates a new {@code EValue} of type {@code List[bool]}. */
  @DoNotStrip
  public static EValue listFrom(boolean... list) {
    final EValue iv = new EValue(TYPE_CODE_LIST_BOOL);
    iv.mData = list;
    return iv;
  }

  /** Creates a new {@code EValue} of type {@code List[int]}. */
  @DoNotStrip
  public static EValue listFrom(long... list) {
    final EValue iv = new EValue(TYPE_CODE_LIST_INT);
    iv.mData = list;
    return iv;
  }

  /** Creates a new {@code EValue} of type {@code List[double]}. */
  @DoNotStrip
  public static EValue listFrom(double... list) {
    final EValue iv = new EValue(TYPE_CODE_LIST_DOUBLE);
    iv.mData = list;
    return iv;
  }

  /** Creates a new {@code EValue} of type {@code List[Tensor]}. */
  @DoNotStrip
  public static EValue listFrom(Tensor... list) {
    final EValue iv = new EValue(TYPE_CODE_LIST_TENSOR);
    iv.mData = list;
    return iv;
  }

  /** Creates a new {@code EValue} of type {@code List[Optional[Tensor]]}. */
  @DoNotStrip
  public static EValue listFrom(Optional<Tensor>... list) {
    final EValue iv = new EValue(TYPE_CODE_LIST_OPTIONAL_TENSOR);
    iv.mData = list;
    return iv;
  }

  @DoNotStrip
  public Tensor toTensor() {
    preconditionType(TYPE_CODE_TENSOR, mTypeCode);
    return (Tensor) mData;
  }

  @DoNotStrip
  public boolean toBool() {
    preconditionType(TYPE_CODE_BOOL, mTypeCode);
    return (boolean) mData;
  }

  @DoNotStrip
  public long toInt() {
    preconditionType(TYPE_CODE_INT, mTypeCode);
    return (long) mData;
  }

  @DoNotStrip
  public double toDouble() {
    preconditionType(TYPE_CODE_DOUBLE, mTypeCode);
    return (double) mData;
  }

  @DoNotStrip
  public String toStr() {
    preconditionType(TYPE_CODE_STRING, mTypeCode);
    return (String) mData;
  }

  @DoNotStrip
  public boolean[] toBoolList() {
    preconditionType(TYPE_CODE_LIST_BOOL, mTypeCode);
    return (boolean[]) mData;
  }

  @DoNotStrip
  public long[] toIntList() {
    preconditionType(TYPE_CODE_LIST_INT, mTypeCode);
    return (long[]) mData;
  }

  @DoNotStrip
  public double[] toDoubleList() {
    preconditionType(TYPE_CODE_LIST_DOUBLE, mTypeCode);
    return (double[]) mData;
  }

  @DoNotStrip
  public Tensor[] toTensorList() {
    preconditionType(TYPE_CODE_LIST_TENSOR, mTypeCode);
    return (Tensor[]) mData;
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
}
