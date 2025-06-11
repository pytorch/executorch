/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
package org.pytorch.executorch

import org.assertj.core.api.Assertions.assertThatThrownBy
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.junit.runners.JUnit4

/** Unit tests for [EValue]. */
@RunWith(JUnit4::class)
class EValueTest {
    @Test
    fun testNone() {
        val evalue = EValue.optionalNone()
        assertTrue(evalue.isNone)
    }

    @Test
    fun testTensorValue() {
        val data = longArrayOf(1, 2, 3)
        val shape = longArrayOf(1, 3)
        val evalue = EValue.from(Tensor.fromBlob(data, shape))
        assertTrue(evalue.isTensor)
        assertTrue(evalue.toTensor().shape.contentEquals(shape))
        assertTrue(evalue.toTensor().dataAsLongArray.contentEquals(data))
    }

    @Test
    fun testBoolValue() {
        val evalue = EValue.from(true)
        assertTrue(evalue.isBool)
        assertTrue(evalue.toBool())
    }

    @Test
    fun testIntValue() {
        val evalue = EValue.from(1)
        assertTrue(evalue.isInt)
        assertEquals(evalue.toInt(), 1)
    }

    @Test
    fun testDoubleValue() {
        val evalue = EValue.from(0.1)
        assertTrue(evalue.isDouble)
        assertEquals(evalue.toDouble(), 0.1, 0.0001)
    }

    @Test
    fun testStringValue() {
        val evalue = EValue.from("a")
        assertTrue(evalue.isString)
        assertEquals(evalue.toStr(), "a")
    }

    @Test
    fun testAllIllegalCast() {
        val evalue = EValue.optionalNone()
        assertTrue(evalue.isNone)

        // try Tensor
        assertFalse(evalue.isTensor)
        assertThatThrownBy { evalue.toTensor() }
            .isInstanceOf(IllegalStateException::class.java)
            .hasMessage("Expected EValue type Tensor, actual type None")

        // try bool
        assertFalse(evalue.isBool)
        assertThatThrownBy { evalue.toBool() }
            .isInstanceOf(IllegalStateException::class.java)
            .hasMessage("Expected EValue type Bool, actual type None")

        // try int
        assertFalse(evalue.isInt)
        assertThatThrownBy { evalue.toInt() }
            .isInstanceOf(IllegalStateException::class.java)
            .hasMessage("Expected EValue type Int, actual type None")

        // try double
        assertFalse(evalue.isDouble)
        assertThatThrownBy { evalue.toDouble() }
            .isInstanceOf(IllegalStateException::class.java)
            .hasMessage("Expected EValue type Double, actual type None")

        // try string
        assertFalse(evalue.isString)
        assertThatThrownBy { evalue.toStr() }
            .isInstanceOf(IllegalStateException::class.java)
            .hasMessage("Expected EValue type String, actual type None")
    }

    @Test
    fun testNoneSerde() {
        val evalue = EValue.optionalNone()
        val bytes = evalue.toByteArray()

        val deser = EValue.fromByteArray(bytes)
        assertEquals(deser.isNone, true)
    }

    @Test
    fun testBoolSerde() {
        val evalue = EValue.from(true)
        val bytes = evalue.toByteArray()
        assertEquals(1, bytes[1].toLong())

        val deser = EValue.fromByteArray(bytes)
        assertEquals(deser.isBool, true)
        assertEquals(deser.toBool(), true)
    }

    @Test
    fun testBoolSerde2() {
        val evalue = EValue.from(false)
        val bytes = evalue.toByteArray()
        assertEquals(0, bytes[1].toLong())

        val deser = EValue.fromByteArray(bytes)
        assertEquals(deser.isBool, true)
        assertEquals(deser.toBool(), false)
    }

    @Test
    fun testIntSerde() {
        val evalue = EValue.from(1)
        val bytes = evalue.toByteArray()
        assertEquals(0, bytes[1].toLong())
        assertEquals(0, bytes[2].toLong())
        assertEquals(0, bytes[3].toLong())
        assertEquals(0, bytes[4].toLong())
        assertEquals(0, bytes[5].toLong())
        assertEquals(0, bytes[6].toLong())
        assertEquals(0, bytes[7].toLong())
        assertEquals(1, bytes[8].toLong())

        val deser = EValue.fromByteArray(bytes)
        assertEquals(deser.isInt, true)
        assertEquals(deser.toInt(), 1)
    }

    @Test
    fun testLargeIntSerde() {
        val evalue = EValue.from(256000)
        val bytes = evalue.toByteArray()

        val deser = EValue.fromByteArray(bytes)
        assertEquals(deser.isInt, true)
        assertEquals(deser.toInt(), 256000)
    }

    @Test
    fun testDoubleSerde() {
        val evalue = EValue.from(1.345e-2)
        val bytes = evalue.toByteArray()

        val deser = EValue.fromByteArray(bytes)
        assertEquals(deser.isDouble, true)
        assertEquals(1.345e-2, deser.toDouble(), 1e-6)
    }

    @Test
    fun testLongTensorSerde() {
        val data = longArrayOf(1, 2, 3, 4)
        val shape = longArrayOf(2, 2)
        val tensor = Tensor.fromBlob(data, shape)

        val evalue = EValue.from(tensor)
        val bytes = evalue.toByteArray()

        val deser = EValue.fromByteArray(bytes)
        assertEquals(deser.isTensor, true)
        val deserTensor = deser.toTensor()
        val deserShape = deserTensor.shape()
        val deserData = deserTensor.dataAsLongArray

        for (i in data.indices) {
            assertEquals(data[i], deserData[i])
        }

        for (i in shape.indices) {
            assertEquals(shape[i], deserShape[i])
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
        assertEquals(deser.isTensor, true)
        val deserTensor = deser.toTensor()
        val deserShape = deserTensor.shape()
        val deserData = deserTensor.dataAsFloatArray

        for (i in data.indices) {
            assertEquals(data[i].toDouble(), deserData[i].toDouble(), 1e-5)
        }

        for (i in shape.indices) {
            assertEquals(shape[i], deserShape[i])
        }
    }
}
