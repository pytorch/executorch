/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
package org.pytorch.executorch

import android.Manifest
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import androidx.test.InstrumentationRegistry
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.rule.GrantPermissionRule
import java.io.File
import java.io.IOException
import java.net.URISyntaxException
import org.apache.commons.io.FileUtils
import org.junit.Assert
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import org.pytorch.executorch.TensorImageUtils.bitmapToFloat32Tensor
import org.pytorch.executorch.TestFileUtils.getTestFilePath

/** Unit tests for [Module]. */
@RunWith(AndroidJUnit4::class)
class ModuleE2ETest {
    @get:Rule
    var runtimePermissionRule: GrantPermissionRule =
        GrantPermissionRule.grant(Manifest.permission.READ_EXTERNAL_STORAGE)

    @Throws(IOException::class, URISyntaxException::class)
    fun testClassification(filePath: String) {
        val pteFile = File(getTestFilePath(filePath))
        val inputStream = javaClass.getResourceAsStream(filePath)
        FileUtils.copyInputStreamToFile(inputStream, pteFile)
        inputStream.close()

        val imgInputStream = javaClass.getResourceAsStream("/banana.jpeg")
        var bitmap = BitmapFactory.decodeStream(imgInputStream)
        bitmap = Bitmap.createScaledBitmap(bitmap!!, 224, 224, true)
        imgInputStream.close()

        val inputTensor =
            bitmapToFloat32Tensor(
                bitmap,
                TensorImageUtils.TORCHVISION_NORM_MEAN_RGB,
                TensorImageUtils.TORCHVISION_NORM_STD_RGB,
            )

        val module = Module.load(getTestFilePath(filePath))

        val results = module.forward(EValue.from(inputTensor))
        Assert.assertTrue(results[0].isTensor)
        val scores = results[0].toTensor().dataAsFloatArray

        val bananaClass = 954 // From ImageNet 1K
        Assert.assertEquals(bananaClass.toLong(), argmax(scores).toLong())
    }

    @Test
    @Throws(IOException::class, URISyntaxException::class)
    fun testXnnpackBackendRequired() {
        val pteFile = File(getTestFilePath("/mv3_xnnpack_fp32.pte"))
        val inputStream = javaClass.getResourceAsStream("/mv3_xnnpack_fp32.pte")
        FileUtils.copyInputStreamToFile(inputStream, pteFile)
        inputStream.close()

        val module = Module.load(getTestFilePath("/mv3_xnnpack_fp32.pte"))
        val expectedBackends = arrayOf("XnnpackBackend")
    }

    @Test
    @Throws(IOException::class, URISyntaxException::class)
    fun testMv2Fp32() {
        testClassification("/mv2_xnnpack_fp32.pte")
    }

    @Test
    @Throws(IOException::class, URISyntaxException::class)
    fun testMv3Fp32() {
        testClassification("/mv3_xnnpack_fp32.pte")
    }

    @Test
    @Throws(IOException::class, URISyntaxException::class)
    fun testResnet50() {
        testClassification("/resnet50_xnnpack_q8.pte")
    }

    companion object {

        fun argmax(array: FloatArray): Int {
            require(array.isNotEmpty()) { "Array cannot be empty" }
            var maxIndex = 0
            var maxValue = array[0]
            for (i in 1 until array.size) {
                if (array[i] > maxValue) {
                    maxValue = array[i]
                    maxIndex = i
                }
            }
            return maxIndex
        }
    }
}
