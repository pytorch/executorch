/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
package org.pytorch.executorch

import android.Manifest
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.rule.GrantPermissionRule
import java.io.File
import java.io.IOException
import java.net.URISyntaxException
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger
import org.apache.commons.io.FileUtils
import org.junit.Assert
import org.junit.Before
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import org.pytorch.executorch.TestFileUtils.getTestFilePath

/** Unit tests for [Module]. */
@RunWith(AndroidJUnit4::class)
class ModuleInstrumentationTest {
  @Before
  @Throws(IOException::class)
  fun setUp() {
    // copy zipped test resources to local device
    val addPteFile = File(getTestFilePath(TEST_FILE_NAME))
    var inputStream = javaClass.getResourceAsStream(TEST_FILE_NAME)
    FileUtils.copyInputStreamToFile(inputStream, addPteFile)
    inputStream.close()

    val nonPteFile = File(getTestFilePath(NON_PTE_FILE_NAME))
    inputStream = javaClass.getResourceAsStream(NON_PTE_FILE_NAME)
    FileUtils.copyInputStreamToFile(inputStream, nonPteFile)
    inputStream.close()
  }

  @get:Rule
  var runtimePermissionRule: GrantPermissionRule =
      GrantPermissionRule.grant(Manifest.permission.READ_EXTERNAL_STORAGE)

  @Test
  @Throws(IOException::class, URISyntaxException::class)
  fun testModuleLoadAndForward() {
    val module = Module.load(getTestFilePath(TEST_FILE_NAME))

    val results = module.forward()
    Assert.assertTrue(results[0].isTensor)
  }

  @Test
  @Throws(IOException::class, URISyntaxException::class)
  fun testMethodMetadata() {
    val module = Module.load(getTestFilePath(TEST_FILE_NAME))
  }

  @Test
  @Throws(IOException::class)
  fun testModuleLoadMethodAndForward() {
    val module = Module.load(getTestFilePath(TEST_FILE_NAME))

    val loadMethod = module.loadMethod(FORWARD_METHOD)
    Assert.assertEquals(loadMethod.toLong(), OK.toLong())

    val results = module.forward()
    Assert.assertTrue(results[0].isTensor)
  }

  @Test
  @Throws(IOException::class)
  fun testModuleLoadForwardExplicit() {
    val module = Module.load(getTestFilePath(TEST_FILE_NAME))

    val results = module.execute(FORWARD_METHOD)
    Assert.assertTrue(results[0].isTensor)
  }

  @Test(expected = RuntimeException::class)
  @Throws(IOException::class)
  fun testModuleLoadNonExistantFile() {
    val module = Module.load(getTestFilePath(MISSING_FILE_NAME))
  }

  @Test
  @Throws(IOException::class)
  fun testModuleLoadMethodNonExistantMethod() {
    val module = Module.load(getTestFilePath(TEST_FILE_NAME))

    val loadMethod = module.loadMethod(NONE_METHOD)
    Assert.assertEquals(loadMethod.toLong(), INVALID_ARGUMENT.toLong())
  }

  @Test(expected = RuntimeException::class)
  @Throws(IOException::class)
  fun testNonPteFile() {
    val module = Module.load(getTestFilePath(NON_PTE_FILE_NAME))

    val loadMethod = module.loadMethod(FORWARD_METHOD)
    Assert.assertEquals(loadMethod.toLong(), INVALID_ARGUMENT.toLong())
  }

  @Test
  @Throws(IOException::class)
  fun testLoadOnDestroyedModule() {
    val module = Module.load(getTestFilePath(TEST_FILE_NAME))

    module.destroy()

    val loadMethod = module.loadMethod(FORWARD_METHOD)
    Assert.assertEquals(loadMethod.toLong(), INVALID_STATE.toLong())
  }

  @Test
  @Throws(IOException::class)
  fun testForwardOnDestroyedModule() {
    val module = Module.load(getTestFilePath(TEST_FILE_NAME))

    val loadMethod = module.loadMethod(FORWARD_METHOD)
    Assert.assertEquals(loadMethod.toLong(), OK.toLong())

    module.destroy()

    val results = module.forward()
    Assert.assertEquals(0, results.size.toLong())
  }

  @Test
  @Throws(InterruptedException::class, IOException::class)
  fun testForwardFromMultipleThreads() {
    val module = Module.load(getTestFilePath(TEST_FILE_NAME))

    val numThreads = 100
    val latch = CountDownLatch(numThreads)
    val completed = AtomicInteger(0)

    val runnable = Runnable {
      try {
        latch.countDown()
        latch.await(5000, TimeUnit.MILLISECONDS)
        val results = module.forward()
        Assert.assertTrue(results[0].isTensor)
        completed.incrementAndGet()
      } catch (_: InterruptedException) {}
    }

    val threads = arrayOfNulls<Thread>(numThreads)
    for (i in 0 until numThreads) {
      threads[i] = Thread(runnable)
      threads[i]!!.start()
    }

    for (i in 0 until numThreads) {
      threads[i]!!.join()
    }

    Assert.assertEquals(numThreads.toLong(), completed.get().toLong())
  }

  companion object {
    private const val TEST_FILE_NAME = "/ModuleAdd.pte"
    private const val MISSING_FILE_NAME = "/missing.pte"
    private const val NON_PTE_FILE_NAME = "/test.txt"
    private const val FORWARD_METHOD = "forward"
    private const val NONE_METHOD = "none"
    private const val OK = 0x00
    private const val INVALID_STATE = 0x2
    private const val INVALID_ARGUMENT = 0x12
    private const val ACCESS_FAILED = 0x22
  }
}
