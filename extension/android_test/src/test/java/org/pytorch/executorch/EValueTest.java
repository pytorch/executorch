/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.pytorch.executorch;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.fail;

import com.facebook.jni.annotations.DoNotStrip;

import java.util.List;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Locale;
import java.util.Optional;

import org.pytorch.executorch.Tensor.Tensor_int64;
import org.pytorch.executorch.annotations.Experimental;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@link EValue}. */
@RunWith(JUnit4.class)
public class EValueTest {

  @Test
  public void testNone() {
    EValue evalue = EValue.optionalNone();
    assertTrue(evalue.isNone());
  }

  @Test
  public void testTensorValue() {
    long[] data = {1, 2, 3};
    long[] shape = {1, 3};
    EValue evalue = EValue.from(Tensor.fromBlob(data, shape));
    assertTrue(evalue.isTensor());
    assertTrue(Arrays.equals(evalue.toTensor().shape, shape));
    assertTrue(Arrays.equals(evalue.toTensor().getDataAsLongArray(), data));
  }

  @Test
  public void testBoolValue() {
    EValue evalue = EValue.from(true);
    assertTrue(evalue.isBool());
    assertTrue(evalue.toBool());
  }

  @Test
  public void testIntValue() {
    EValue evalue = EValue.from(1);
    assertTrue(evalue.isInt());
    assertEquals(evalue.toInt(), 1);
  }

  @Test
  public void testDoubleValue() {
    EValue evalue = EValue.from(0.1d);
    assertTrue(evalue.isDouble());
    assertEquals(evalue.toDouble(), 0.1d, 0.0001d);
  }

  @Test
  public void testStringValue() {
    EValue evalue = EValue.from("a");
    assertTrue(evalue.isString());
    assertEquals(evalue.toStr(), "a");
  }

  @Test
  public void testBoolListValue() {
    boolean[] value = {true, false, true};
    EValue evalue = EValue.listFrom(value);
    assertTrue(evalue.isBoolList());
    assertTrue(Arrays.equals(value, evalue.toBoolList()));
  }

  @Test
  public void testIntListValue() {
    long[] value = {Long.MIN_VALUE, 0, Long.MAX_VALUE};
    EValue evalue = EValue.listFrom(value);
    assertTrue(evalue.isIntList());
    assertTrue(Arrays.equals(value, evalue.toIntList()));
  }

  @Test
  public void testDoubleListValue() {
    double[] value = {Double.MIN_VALUE,0.1d, 0.01d, 0.001d, Double.MAX_VALUE};
    EValue evalue = EValue.listFrom(value);
    assertTrue(evalue.isDoubleList());
    assertTrue(Arrays.equals(value, evalue.toDoubleList()));
  }

  @Test
  public void testTensorListValue() {
    long[][] data = {{1, 2, 3}, {1, 2, 3, 4, 5, 6}};
    long[][] shape = {{1, 3}, {2, 3}};
    Tensor[] tensors = {Tensor.fromBlob(data[0], shape[0]), Tensor.fromBlob(data[1], shape[1])};

    EValue evalue = EValue.listFrom(tensors);
    assertTrue(evalue.isTensorList());

    assertTrue(Arrays.equals(evalue.toTensorList()[0].shape, shape[0]));
    assertTrue(Arrays.equals(evalue.toTensorList()[0].getDataAsLongArray(), data[0]));

    assertTrue(Arrays.equals(evalue.toTensorList()[1].shape, shape[1]));
    assertTrue(Arrays.equals(evalue.toTensorList()[1].getDataAsLongArray(), data[1]));
  }

  @Test
  @SuppressWarnings("unchecked")
  public void testOptionalTensorListValue() {
    long[][] data = {{1, 2, 3}, {1, 2, 3, 4, 5, 6}};
    long[][] shape = {{1, 3}, {2, 3}};

    EValue evalue = EValue.listFrom(
        Optional.<Tensor>empty(), 
        Optional.of(Tensor.fromBlob(data[0], shape[0])), 
        Optional.of(Tensor.fromBlob(data[1], shape[1])));
    assertTrue(evalue.isOptionalTensorList());

    assertTrue(!evalue.toOptionalTensorList()[0].isPresent());

    assertTrue(evalue.toOptionalTensorList()[1].isPresent());
    assertTrue(Arrays.equals(evalue.toOptionalTensorList()[1].get().shape, shape[0]));
    assertTrue(Arrays.equals(evalue.toOptionalTensorList()[1].get().getDataAsLongArray(), data[0]));

    assertTrue(evalue.toOptionalTensorList()[2].isPresent());
    assertTrue(Arrays.equals(evalue.toOptionalTensorList()[2].get().shape, shape[1]));
    assertTrue(Arrays.equals(evalue.toOptionalTensorList()[2].get().getDataAsLongArray(), data[1]));
  }

  @Test
  public void testAllIllegalCast() {
    EValue evalue = EValue.optionalNone();
    assertTrue(evalue.isNone());
    
    // try Tensor
    assertFalse(evalue.isTensor());
    try {
        evalue.toTensor();
        fail("Should have thrown an exception");
    } catch (IllegalStateException e) {}
    
    // try bool
    assertFalse(evalue.isBool());
    try {
        evalue.toBool();
        fail("Should have thrown an exception");
    } catch (IllegalStateException e) {}
    
    // try int
    assertFalse(evalue.isInt());
    try {
        evalue.toInt();
        fail("Should have thrown an exception");
    } catch (IllegalStateException e) {}
    
    // try double
    assertFalse(evalue.isDouble());
    try {
        evalue.toDouble();
        fail("Should have thrown an exception");
    } catch (IllegalStateException e) {}
    
    // try string
    assertFalse(evalue.isString());
    try {
        evalue.toStr();
        fail("Should have thrown an exception");
    } catch (IllegalStateException e) {}
    
    // try bool list
    assertFalse(evalue.isBoolList());
    try {
        evalue.toBoolList();
        fail("Should have thrown an exception");
    } catch (IllegalStateException e) {}
    
    // try int list
    assertFalse(evalue.isIntList());
    try {
        evalue.toIntList();
        fail("Should have thrown an exception");
    } catch (IllegalStateException e) {}
    
    // try double list
    assertFalse(evalue.isDoubleList());
    try {
        evalue.toBool();
        fail("Should have thrown an exception");
    } catch (IllegalStateException e) {}
    
    // try Tensor list
    assertFalse(evalue.isTensorList());
    try {
        evalue.toTensorList();
        fail("Should have thrown an exception");
    } catch (IllegalStateException e) {}
    
    // try optional Tensor list
    assertFalse(evalue.isOptionalTensorList());
    try {
        evalue.toOptionalTensorList();
        fail("Should have thrown an exception");
    } catch (IllegalStateException e) {}
  }
}
