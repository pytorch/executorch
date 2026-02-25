/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
package org.pytorch.executorch

import androidx.test.ext.junit.runners.AndroidJUnit4
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.abs
import org.apache.commons.io.FileUtils
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.pytorch.executorch.TestFileUtils.getTestFilePath

/** End-to-end tests that verify JNI inference against nightly golden artifacts. */
@RunWith(AndroidJUnit4::class)
class ModuleE2ETest {

  private fun loadFloatArrayFromResource(path: String): FloatArray {
    val bytes = javaClass.getResourceAsStream(path)!!.use { it.readBytes() }
    val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
    return FloatArray(bytes.size / 4).also { buffer.asFloatBuffer().get(it) }
  }

  private fun assertOutputsClose(actual: FloatArray, expected: FloatArray, atol: Float = 1e-3f) {
    assertEquals("Output size mismatch", expected.size, actual.size)
    for (i in actual.indices) {
      assertTrue(
          "Output[$i]: expected=${expected[i]}, actual=${actual[i]}, diff=${abs(actual[i] - expected[i])}",
          abs(actual[i] - expected[i]) <= atol,
      )
    }
  }

  private fun testGoldenModel(modelName: String, inputShape: LongArray) {
    val inputData = loadFloatArrayFromResource("/${modelName}_input.bin")
    val expectedOutput = loadFloatArrayFromResource("/${modelName}_expected_output.bin")
    val inputTensor = Tensor.fromBlob(inputData, inputShape)

    val pteStream = javaClass.getResourceAsStream("/${modelName}.pte")!!
    val pteFile = File(getTestFilePath("/${modelName}.pte"))
    FileUtils.copyInputStreamToFile(pteStream, pteFile)

    val module = Module.load(pteFile.absolutePath)
    val results = module.forward(EValue.from(inputTensor))
    val actualOutput = results[0].toTensor().dataAsFloatArray

    assertOutputsClose(actualOutput, expectedOutput)
    module.destroy()
  }

  @Test
  fun testXnnpackBackendRequired() {
    val pteFile = File(getTestFilePath("/mobilenet_v2.pte"))
    val inputStream = javaClass.getResourceAsStream("/mobilenet_v2.pte")
    FileUtils.copyInputStreamToFile(inputStream, pteFile)
    inputStream.close()

    val module = Module.load(pteFile.absolutePath)
    val expectedBackends = arrayOf("XnnpackBackend")
    assertArrayEquals(expectedBackends, module.getMethodMetadata("forward").backends)
  }

  @Test
  fun testMobilenetV2() {
    testGoldenModel("mobilenet_v2", longArrayOf(1, 3, 224, 224))
  }

  @Test
  fun testVitB16() {
    testGoldenModel("vit_b_16", longArrayOf(1, 3, 224, 224))
  }
}
