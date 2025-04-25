/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
package org.pytorch.executorch

import org.junit.Assert
import org.junit.Test
import org.junit.runner.RunWith
import org.junit.runners.JUnit4

/** Unit tests for [EValue].  */
@RunWith(JUnit4::class)
class EValueTest {
    @Test
    fun testNone() {
        val evalue = EValue.optionalNone()
        Assert.assertTrue(evalue.isNone)
    }

    @Test
    fun testTensorValue() {
        val data = longArrayOf(1, 2, 3)
        val shape = longArrayOf(1, 3)
        val evalue = EValue.from(Tensor.fromBlob(data, shape))
        Assert.assertTrue(evalue.isTensor)
        Assert.assertTrue(evalue.toTensor().shape.contentEquals(shape))
        Assert.assertTrue(evalue.toTensor().dataAsLongArray.contentEquals(data))
    }

    @Test
    fun testBoolValue() {
        val evalue = EValue.from(true)
        Assert.assertTrue(evalue.isBool)
        Assert.assertTrue(evalue.toBool())
    }

    @Test
    fun testIntValue() {
        val evalue = EValue.from(1)
        Assert.assertTrue(evalue.isInt)
        Assert.assertEquals(evalue.toInt(), 1)
    }

    @Test
    fun testDoubleValue() {
        val evalue = EValue.from(0.1)
        Assert.assertTrue(evalue.isDouble)
        Assert.assertEquals(evalue.toDouble(), 0.1, 0.0001)
    }

    @Test
    fun testStringValue() {
        val evalue = EValue.from("a")
        Assert.assertTrue(evalue.isString)
        Assert.assertEquals(evalue.toStr(), "a")
    }

    @Test
    fun testAllIllegalCast() {
        val evalue = EValue.optionalNone()
        Assert.assertTrue(evalue.isNone)

        // try Tensor
        Assert.assertFalse(evalue.isTensor)
        try {
            evalue.toTensor()
            Assert.fail("Should have thrown an exception")
        } catch (e: IllegalStateException) {
        }

        // try bool
        Assert.assertFalse(evalue.isBool)
        try {
            evalue.toBool()
            Assert.fail("Should have thrown an exception")
        } catch (e: IllegalStateException) {
        }

        // try int
        Assert.assertFalse(evalue.isInt)
        try {
            evalue.toInt()
            Assert.fail("Should have thrown an exception")
        } catch (e: IllegalStateException) {
        }

        // try double
        Assert.assertFalse(evalue.isDouble)
        try {
            evalue.toDouble()
            Assert.fail("Should have thrown an exception")
        } catch (e: IllegalStateException) {
        }

        // try string
        Assert.assertFalse(evalue.isString)
        try {
            evalue.toStr()
            Assert.fail("Should have thrown an exception")
        } catch (e: IllegalStateException) {
        }
    }

    @Test
    fun testNoneSerde() {
        val evalue = EValue.optionalNone()
        val bytes = evalue.toByteArray()

        val deser = EValue.fromByteArray(bytes)
        Assert.assertEquals(deser.isNone, true)
    }

    @Test
    fun testBoolSerde() {
        val evalue = EValue.from(true)
        val bytes = evalue.toByteArray()
        Assert.assertEquals(1, bytes[1].toLong())

        val deser = EValue.fromByteArray(bytes)
        Assert.assertEquals(deser.isBool, true)
        Assert.assertEquals(deser.toBool(), true)
    }

    @Test
    fun testBoolSerde2() {
        val evalue = EValue.from(false)
        val bytes = evalue.toByteArray()
        Assert.assertEquals(0, bytes[1].toLong())

        val deser = EValue.fromByteArray(bytes)
        Assert.assertEquals(deser.isBool, true)
        Assert.assertEquals(deser.toBool(), false)
    }

    @Test
    fun testIntSerde() {
        val evalue = EValue.from(1)
        val bytes = evalue.toByteArray()
        Assert.assertEquals(0, bytes[1].toLong())
        Assert.assertEquals(0, bytes[2].toLong())
        Assert.assertEquals(0, bytes[3].toLong())
        Assert.assertEquals(0, bytes[4].toLong())
        Assert.assertEquals(0, bytes[5].toLong())
        Assert.assertEquals(0, bytes[6].toLong())
        Assert.assertEquals(0, bytes[7].toLong())
        Assert.assertEquals(1, bytes[8].toLong())

        val deser = EValue.fromByteArray(bytes)
        Assert.assertEquals(deser.isInt, true)
        Assert.assertEquals(deser.toInt(), 1)
    }

    @Test
    fun testLargeIntSerde() {
        val evalue = EValue.from(256000)
        val bytes = evalue.toByteArray()

        val deser = EValue.fromByteArray(bytes)
        Assert.assertEquals(deser.isInt, true)
        Assert.assertEquals(deser.toInt(), 256000)
    }

    @Test
    fun testDoubleSerde() {
        val evalue = EValue.from(1.345e-2)
        val bytes = evalue.toByteArray()

        val deser = EValue.fromByteArray(bytes)
        Assert.assertEquals(deser.isDouble, true)
        Assert.assertEquals(1.345e-2, deser.toDouble(), 1e-6)
    }

    @Test
    fun testLongTensorSerde() {
        val data = longArrayOf(1, 2, 3, 4)
        val shape = longArrayOf(2, 2)
        val tensor = Tensor.fromBlob(data, shape)

        val evalue = EValue.from(tensor)
        val bytes = evalue.toByteArray()

        val deser = EValue.fromByteArray(bytes)
        Assert.assertEquals(deser.isTensor, true)
        val deserTensor = deser.toTensor()
        val deserShape = deserTensor.shape()
        val deserData = deserTensor.dataAsLongArray

        for (i in data.indices) {
            Assert.assertEquals(data[i], deserData[i])
        }

        for (i in shape.indices) {
            Assert.assertEquals(shape[i], deserShape[i])
        }
    }

    @Test
    fun testFloatTensorSerde() {
        val data = floatArrayOf(Float.MIN_VALUE, 0f, 0.1f, Float.MAX_VALUE)
        val shape = longArrayOf(2, 2)
        val tensor = Tensor.fromBlob(data, shape)

        val evalue = EValue.from(tensor)
        val bytes = evalue.toByteArray()

        val deser = EValue.fromByteArray(bytes)
        Assert.assertEquals(deser.isTensor, true)
        val deserTensor = deser.toTensor()
        val deserShape = deserTensor.shape()
        val deserData = deserTensor.dataAsFloatArray

        for (i in data.indices) {
            Assert.assertEquals(data[i].toDouble(), deserData[i].toDouble(), 1e-5)
        }

        for (i in shape.indices) {
            Assert.assertEquals(shape[i], deserShape[i])
        }
    }
}
