/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.pytorch.executorch.training

import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import java.io.File
import java.io.IOException
import java.net.URISyntaxException
import kotlin.random.Random
import kotlin.test.assertContains
import org.apache.commons.io.FileUtils
import org.junit.Assert
import org.junit.Test
import org.junit.runner.RunWith
import org.pytorch.executorch.EValue
import org.pytorch.executorch.Tensor
import org.pytorch.executorch.TestFileUtils

/** Unit tests for [TrainingModule]. */
@RunWith(AndroidJUnit4::class)
class TrainingModuleE2ETest {

  @Test
  @Throws(IOException::class, URISyntaxException::class)
  fun testTrainXOR() {
    val pteFilePath = "/xor.pte"
    val ptdFilePath = "/xor.ptd"

    val pteFile = File(TestFileUtils.getTestFilePath(pteFilePath))
    val pteInputStream = javaClass.getResourceAsStream(pteFilePath)
    FileUtils.copyInputStreamToFile(pteInputStream, pteFile)
    pteInputStream.close()

    val ptdFile = File(TestFileUtils.getTestFilePath(ptdFilePath))
    val ptdInputStream = javaClass.getResourceAsStream(ptdFilePath)
    FileUtils.copyInputStreamToFile(ptdInputStream, ptdFile)
    ptdInputStream.close()

    val module =
        TrainingModule.load(
            TestFileUtils.getTestFilePath(pteFilePath),
            TestFileUtils.getTestFilePath(ptdFilePath),
        )
    val params = module.namedParameters("forward")

    Assert.assertEquals(4, params.size)
    assertContains(params, LIN_WEIGHT)
    assertContains(params, LIN_BIAS)
    assertContains(params, LIN2_WEIGHT)
    assertContains(params, LIN2_BIAS)

    val sgd = SGD.create(params, 0.5)
    val dataset =
        listOf<Tensor>(
            Tensor.fromBlob(floatArrayOf(1.0f, 1.0f), longArrayOf(1, 2)),
            Tensor.fromBlob(longArrayOf(0), longArrayOf(1)),
            Tensor.fromBlob(floatArrayOf(0.0f, 0.0f), longArrayOf(1, 2)),
            Tensor.fromBlob(longArrayOf(0), longArrayOf(1)),
            Tensor.fromBlob(floatArrayOf(1.0f, 0.0f), longArrayOf(1, 2)),
            Tensor.fromBlob(longArrayOf(1), longArrayOf(1)),
            Tensor.fromBlob(floatArrayOf(0.0f, 1.0f), longArrayOf(1, 2)),
            Tensor.fromBlob(longArrayOf(1), longArrayOf(1)),
        )

    val numEpochs = 5000
    var finalLoss = Float.MAX_VALUE

    for (i in 0 until numEpochs) {
      val inputDex = 2 * Random.nextInt(dataset.size / 2)
      val targetDex = inputDex + 1
      val input = dataset.get(inputDex)
      val target = dataset.get(targetDex)
      val out = module.executeForwardBackward("forward", EValue.from(input), EValue.from(target))
      val gradients = module.namedGradients("forward")

      if (i == 0) {
        Assert.assertEquals(4, gradients.size)
        assertContains(gradients, LIN_WEIGHT)
        assertContains(gradients, LIN_BIAS)
        assertContains(gradients, LIN2_WEIGHT)
        assertContains(gradients, LIN2_BIAS)
      }

      if (i % 500 == 0 || i == numEpochs - 1) {
        Log.i(
            "testTrainXOR",
            String.format(
                "Step %d, Loss %f, Input [%.0f, %.0f], Prediction %d, Label %d",
                i,
                out[0].toTensor().getDataAsFloatArray()[0],
                input.getDataAsFloatArray()[0],
                input.getDataAsFloatArray()[1],
                out[1].toTensor().getDataAsLongArray()[0],
                target.getDataAsLongArray()[0],
            ),
        )
      }

      sgd.step(gradients)

      if (i == numEpochs - 1) {
        finalLoss = out[0].toTensor().dataAsFloatArray[0]
      }
    }
    Assert.assertTrue(finalLoss < 0.1f)
  }

  @Test
  @Throws(IOException::class, URISyntaxException::class)
  fun testTrainXOR_PTEOnly() {
    val pteFilePath = "/xor_full.pte"

    val pteFile = File(TestFileUtils.getTestFilePath(pteFilePath))
    val pteInputStream = javaClass.getResourceAsStream(pteFilePath)
    FileUtils.copyInputStreamToFile(pteInputStream, pteFile)
    pteInputStream.close()

    val module = TrainingModule.load(TestFileUtils.getTestFilePath(pteFilePath))
    val params = module.namedParameters("forward")

    Assert.assertEquals(4, params.size)
    assertContains(params, LIN_WEIGHT)
    assertContains(params, LIN_BIAS)
    assertContains(params, LIN2_WEIGHT)
    assertContains(params, LIN2_BIAS)

    val sgd = SGD.create(params, 0.5)
    val dataset =
        listOf<Tensor>(
            Tensor.fromBlob(floatArrayOf(1.0f, 1.0f), longArrayOf(1, 2)),
            Tensor.fromBlob(longArrayOf(0), longArrayOf(1)),
            Tensor.fromBlob(floatArrayOf(0.0f, 0.0f), longArrayOf(1, 2)),
            Tensor.fromBlob(longArrayOf(0), longArrayOf(1)),
            Tensor.fromBlob(floatArrayOf(1.0f, 0.0f), longArrayOf(1, 2)),
            Tensor.fromBlob(longArrayOf(1), longArrayOf(1)),
            Tensor.fromBlob(floatArrayOf(0.0f, 1.0f), longArrayOf(1, 2)),
            Tensor.fromBlob(longArrayOf(1), longArrayOf(1)),
        )

    val numEpochs = 5000
    var finalLoss = Float.MAX_VALUE

    for (i in 0 until numEpochs) {
      val inputDex = 2 * Random.nextInt(dataset.size / 2)
      val targetDex = inputDex + 1
      val input = dataset.get(inputDex)
      val target = dataset.get(targetDex)
      val out = module.executeForwardBackward("forward", EValue.from(input), EValue.from(target))
      val gradients = module.namedGradients("forward")

      if (i == 0) {
        Assert.assertEquals(4, gradients.size)
        assertContains(gradients, LIN_WEIGHT)
        assertContains(gradients, LIN_BIAS)
        assertContains(gradients, LIN2_WEIGHT)
        assertContains(gradients, LIN2_BIAS)
      }

      if (i % 500 == 0 || i == numEpochs - 1) {
        Log.i(
            "testTrainXOR_PTEOnly",
            String.format(
                "Step %d, Loss %f, Input [%.0f, %.0f], Prediction %d, Label %d",
                i,
                out[0].toTensor().getDataAsFloatArray()[0],
                input.getDataAsFloatArray()[0],
                input.getDataAsFloatArray()[1],
                out[1].toTensor().getDataAsLongArray()[0],
                target.getDataAsLongArray()[0],
            ),
        )
      }

      sgd.step(gradients)

      if (i == numEpochs - 1) {
        finalLoss = out[0].toTensor().dataAsFloatArray[0]
      }
    }
    Assert.assertTrue(finalLoss < 0.1f)
  }

  @Test
  @Throws(IOException::class)
  fun testMissingPteFile() {
    val exception =
        Assert.assertThrows(RuntimeException::class.java) {
          TrainingModule.load(TestFileUtils.getTestFilePath(MISSING_PTE_NAME))
        }
    Assert.assertEquals(
        exception.message,
        "Cannot load model path!! " + TestFileUtils.getTestFilePath(MISSING_PTE_NAME),
    )
  }

  @Test
  @Throws(IOException::class)
  fun testMissingPtdFile() {
    val exception =
        Assert.assertThrows(RuntimeException::class.java) {
          val pteFilePath = "/xor.pte"
          val pteFile = File(TestFileUtils.getTestFilePath(pteFilePath))
          val pteInputStream = javaClass.getResourceAsStream(pteFilePath)
          FileUtils.copyInputStreamToFile(pteInputStream, pteFile)
          pteInputStream.close()

          TrainingModule.load(
              TestFileUtils.getTestFilePath(pteFilePath),
              TestFileUtils.getTestFilePath(MISSING_PTD_NAME),
          )
        }
    Assert.assertEquals(
        exception.message,
        "Cannot load data path!! " + TestFileUtils.getTestFilePath(MISSING_PTD_NAME),
    )
  }

  companion object {
    private const val LIN_WEIGHT = "net.linear.weight"
    private const val LIN_BIAS = "net.linear.bias"
    private const val LIN2_WEIGHT = "net.linear2.weight"
    private const val LIN2_BIAS = "net.linear2.bias"
    private const val MISSING_PTE_NAME = "/missing.pte"
    private const val MISSING_PTD_NAME = "/missing.ptd"
  }
}
