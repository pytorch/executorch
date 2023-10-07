/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

package com.example.executorchdemo.executor;

import com.facebook.jni.annotations.DoNotStrip;
import java.util.Locale;
import java.util.Map;

/**
 * Java representation of a TorchScript value, which is implemented as tagged union that can be one
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
 */
@DoNotStrip
public class EValue {
  private static final int TYPE_CODE_NULL = 1;

  private static final int TYPE_CODE_TENSOR = 2;
  private static final int TYPE_CODE_BOOL = 3;
  private static final int TYPE_CODE_LONG = 4;
  private static final int TYPE_CODE_DOUBLE = 5;
  private static final int TYPE_CODE_STRING = 6;

  private static final int TYPE_CODE_TUPLE = 7;
  private static final int TYPE_CODE_BOOL_LIST = 8;
  private static final int TYPE_CODE_LONG_LIST = 9;
  private static final int TYPE_CODE_DOUBLE_LIST = 10;
  private static final int TYPE_CODE_TENSOR_LIST = 11;
  private static final int TYPE_CODE_LIST = 12;

  private static final int TYPE_CODE_DICT_STRING_KEY = 13;
  private static final int TYPE_CODE_DICT_LONG_KEY = 14;

  private String[] TYPE_NAMES = {
    "Unknown",
    "Null",
    "Tensor",
    "Bool",
    "Long",
    "Double",
    "String",
    "Tuple",
    "BoolList",
    "LongList",
    "DoubleList",
    "TensorList",
    "GenericList",
    "DictStringKey",
    "DictLongKey",
  };

  @DoNotStrip private final int mTypeCode;
  @DoNotStrip private Object mData;

  @DoNotStrip
  private EValue(int typeCode) {
    this.mTypeCode = typeCode;
  }

  @DoNotStrip
  public boolean isNull() {
    return TYPE_CODE_NULL == this.mTypeCode;
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
  public boolean isLong() {
    return TYPE_CODE_LONG == this.mTypeCode;
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
  public boolean isTuple() {
    return TYPE_CODE_TUPLE == this.mTypeCode;
  }

  @DoNotStrip
  public boolean isBoolList() {
    return TYPE_CODE_BOOL_LIST == this.mTypeCode;
  }

  @DoNotStrip
  public boolean isLongList() {
    return TYPE_CODE_LONG_LIST == this.mTypeCode;
  }

  @DoNotStrip
  public boolean isDoubleList() {
    return TYPE_CODE_DOUBLE_LIST == this.mTypeCode;
  }

  @DoNotStrip
  public boolean isTensorList() {
    return TYPE_CODE_TENSOR_LIST == this.mTypeCode;
  }

  @DoNotStrip
  public boolean isList() {
    return TYPE_CODE_LIST == this.mTypeCode;
  }

  @DoNotStrip
  public boolean isDictStringKey() {
    return TYPE_CODE_DICT_STRING_KEY == this.mTypeCode;
  }

  @DoNotStrip
  public boolean isDictLongKey() {
    return TYPE_CODE_DICT_LONG_KEY == this.mTypeCode;
  }

  /** Creates a new {@code EValue} of type {@code Optional} that contains no value. */
  @DoNotStrip
  public static EValue optionalNull() {
    return new EValue(TYPE_CODE_NULL);
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
    final EValue iv = new EValue(TYPE_CODE_LONG);
    iv.mData = value;
    return iv;
  }
  /** Creates a new {@code EValue} of type {@code float}. */
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
    final EValue iv = new EValue(TYPE_CODE_BOOL_LIST);
    iv.mData = list;
    return iv;
  }
  /** Creates a new {@code EValue} of type {@code List[int]}. */
  @DoNotStrip
  public static EValue listFrom(long... list) {
    final EValue iv = new EValue(TYPE_CODE_LONG_LIST);
    iv.mData = list;
    return iv;
  }
  /** Creates a new {@code EValue} of type {@code List[float]}. */
  @DoNotStrip
  public static EValue listFrom(double... list) {
    final EValue iv = new EValue(TYPE_CODE_DOUBLE_LIST);
    iv.mData = list;
    return iv;
  }

  /** Creates a new {@code EValue} of type {@code List[Tensor]}. */
  @DoNotStrip
  public static EValue listFrom(Tensor... list) {
    final EValue iv = new EValue(TYPE_CODE_TENSOR_LIST);
    iv.mData = list;
    return iv;
  }

  /** Creates a new {@code EValue} of type {@code List[T]}. All elements must have the same type. */
  @DoNotStrip
  public static EValue listFrom(EValue... array) {
    final int size = array.length;
    if (size > 0) {
      final int typeCode0 = array[0].mTypeCode;
      for (int i = 1; i < size; i++) {
        if (typeCode0 != array[i].mTypeCode) {
          throw new IllegalArgumentException("List must contain items of the same type");
        }
      }
    }

    final EValue iv = new EValue(TYPE_CODE_LIST);
    iv.mData = array;
    return iv;
  }
  /** Creates a new {@code EValue} of type {@code Tuple[T0, T1, ...]}. */
  @DoNotStrip
  public static EValue tupleFrom(EValue... array) {
    final EValue iv = new EValue(TYPE_CODE_TUPLE);
    iv.mData = array;
    return iv;
  }

  /** Creates a new {@code EValue} of type {@code Dict[str, V]}. */
  @DoNotStrip
  public static EValue dictStringKeyFrom(Map<String, EValue> map) {
    final EValue iv = new EValue(TYPE_CODE_DICT_STRING_KEY);
    iv.mData = map;
    return iv;
  }
  /** Creates a new {@code EValue} of type {@code Dict[int, V]}. */
  @DoNotStrip
  public static EValue dictLongKeyFrom(Map<Long, EValue> map) {
    final EValue iv = new EValue(TYPE_CODE_DICT_LONG_KEY);
    iv.mData = map;
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
  public long toLong() {
    preconditionType(TYPE_CODE_LONG, mTypeCode);
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
    preconditionType(TYPE_CODE_BOOL_LIST, mTypeCode);
    return (boolean[]) mData;
  }

  @DoNotStrip
  public long[] toLongList() {
    preconditionType(TYPE_CODE_LONG_LIST, mTypeCode);
    return (long[]) mData;
  }

  @DoNotStrip
  public double[] toDoubleList() {
    preconditionType(TYPE_CODE_DOUBLE_LIST, mTypeCode);
    return (double[]) mData;
  }

  @DoNotStrip
  public Tensor[] toTensorList() {
    preconditionType(TYPE_CODE_TENSOR_LIST, mTypeCode);
    return (Tensor[]) mData;
  }

  @DoNotStrip
  public EValue[] toList() {
    preconditionType(TYPE_CODE_LIST, mTypeCode);
    return (EValue[]) mData;
  }

  @DoNotStrip
  public EValue[] toTuple() {
    preconditionType(TYPE_CODE_TUPLE, mTypeCode);
    return (EValue[]) mData;
  }

  @DoNotStrip
  public Map<String, EValue> toDictStringKey() {
    preconditionType(TYPE_CODE_DICT_STRING_KEY, mTypeCode);
    return (Map<String, EValue>) mData;
  }

  @DoNotStrip
  public Map<Long, EValue> toDictLongKey() {
    preconditionType(TYPE_CODE_DICT_LONG_KEY, mTypeCode);
    return (Map<Long, EValue>) mData;
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
