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
import java.io.IOException
import java.net.URISyntaxException
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger
import org.apache.commons.io.FileUtils
import org.junit.Assert
import org.junit.Before
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

  @Test
  @Throws(IOException::class, URISyntaxException::class)
  fun testModuleLoadAndForward() {
    val module = Module.load(getTestFilePath(TEST_FILE_NAME))
    try {
      val results = module.forward(EValue.from(dummyInput()))
      Assert.assertTrue(results[0].isTensor)
    } finally {
      module.destroy()
    }
  }

  @Test
  @Throws(IOException::class)
  fun testModuleLoadMethodAndForward() {
    val module = Module.load(getTestFilePath(TEST_FILE_NAME))
    try {
      module.loadMethod(FORWARD_METHOD)

      val results = module.forward(EValue.from(dummyInput()))
      Assert.assertTrue(results[0].isTensor)
    } finally {
      module.destroy()
    }
  }

  @Test
  @Throws(IOException::class)
  fun testModuleLoadForwardExplicit() {
    val module = Module.load(getTestFilePath(TEST_FILE_NAME))
    try {
      val results = module.execute(FORWARD_METHOD, EValue.from(dummyInput()))
      Assert.assertTrue(results[0].isTensor)
    } finally {
      module.destroy()
    }
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
    try {
      val exception =
          Assert.assertThrows(ExecutorchRuntimeException::class.java) {
            module.loadMethod(NONE_METHOD)
          }
      Assert.assertEquals(
          ExecutorchRuntimeException.INVALID_ARGUMENT,
          exception.getErrorCode(),
      )
    } finally {
      module.destroy()
    }
  }

  @Test
  @Throws(IOException::class)
  fun testNonPteFile() {
    Assert.assertThrows(RuntimeException::class.java) {
      val module = Module.load(getTestFilePath(NON_PTE_FILE_NAME))
      try {
        module.loadMethod(FORWARD_METHOD)
      } finally {
        module.destroy()
      }
    }
  }

  @Test
  @Throws(IOException::class)
  fun testLoadOnDestroyedModule() {
    val module = Module.load(getTestFilePath(TEST_FILE_NAME))

    module.destroy()

    Assert.assertThrows(IllegalStateException::class.java) { module.loadMethod(FORWARD_METHOD) }
  }

  @Test
  @Throws(IOException::class)
  fun testForwardOnDestroyedModule() {
    val module = Module.load(getTestFilePath(TEST_FILE_NAME))

    module.loadMethod(FORWARD_METHOD)

    module.destroy()

    Assert.assertThrows(IllegalStateException::class.java) { module.forward() }
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
        val results = module.forward(EValue.from(dummyInput()))
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
    module.destroy()
  }

  // --- Load mode tests ---

  @Test
  @Throws(IOException::class)
  fun testLoadWithMmapMode() {
    val module = Module.load(getTestFilePath(TEST_FILE_NAME), Module.LOAD_MODE_MMAP)
    try {
      val results = module.forward(EValue.from(dummyInput()))
      Assert.assertTrue(results[0].isTensor)
    } finally {
      module.destroy()
    }
  }

  @Test
  @Throws(IOException::class)
  fun testLoadWithFileMode() {
    val module = Module.load(getTestFilePath(TEST_FILE_NAME), Module.LOAD_MODE_FILE)
    try {
      val results = module.forward(EValue.from(dummyInput()))
      Assert.assertTrue(results[0].isTensor)
    } finally {
      module.destroy()
    }
  }

  // --- getMethods / getMethodMetadata tests ---

  @Test
  @Throws(IOException::class)
  fun testGetMethods() {
    val module = Module.load(getTestFilePath(TEST_FILE_NAME))
    try {
      val methods = module.getMethods()
      Assert.assertNotNull(methods)
      Assert.assertTrue(methods.contains(FORWARD_METHOD))
    } finally {
      module.destroy()
    }
  }

  @Test
  @Throws(IOException::class)
  fun testGetMethodMetadata() {
    val module = Module.load(getTestFilePath(TEST_FILE_NAME))
    try {
      val metadata = module.getMethodMetadata(FORWARD_METHOD)
      Assert.assertNotNull(metadata)
      Assert.assertEquals(FORWARD_METHOD, metadata.name)
      Assert.assertNotNull(metadata.backends)
    } finally {
      module.destroy()
    }
  }

  // --- Log buffer tests ---

  @Test
  @Throws(IOException::class)
  fun testReadLogBuffer() {
    val module = Module.load(getTestFilePath(TEST_FILE_NAME))
    try {
      val logs = module.readLogBuffer()
      Assert.assertNotNull(logs)
    } finally {
      module.destroy()
    }
  }

  @Test
  fun testReadLogBufferStatic() {
    val logs = Module.readLogBufferStatic()
    Assert.assertNotNull(logs)
  }

  // --- etdump test ---

  @Test
  @Throws(IOException::class)
  fun testEtdump() {
    val module = Module.load(getTestFilePath(TEST_FILE_NAME))
    try {
      module.etdump()
    } finally {
      module.destroy()
    }
  }

  // --- Destroyed-state tests for remaining methods ---

  @Test
  @Throws(IOException::class)
  fun testGetMethodsOnDestroyedModule() {
    val module = Module.load(getTestFilePath(TEST_FILE_NAME))
    module.destroy()
    Assert.assertThrows(IllegalStateException::class.java) { module.getMethods() }
  }

  @Test
  @Throws(IOException::class)
  fun testGetMethodMetadataOnDestroyedModule() {
    val module = Module.load(getTestFilePath(TEST_FILE_NAME))
    module.destroy()
    Assert.assertThrows(IllegalStateException::class.java) {
      module.getMethodMetadata(FORWARD_METHOD)
    }
  }

  @Test
  @Throws(IOException::class)
  fun testReadLogBufferOnDestroyedModule() {
    val module = Module.load(getTestFilePath(TEST_FILE_NAME))
    module.destroy()
    Assert.assertThrows(IllegalStateException::class.java) { module.readLogBuffer() }
  }

  @Test
  @Throws(IOException::class)
  fun testEtdumpOnDestroyedModule() {
    val module = Module.load(getTestFilePath(TEST_FILE_NAME))
    module.destroy()
    Assert.assertThrows(IllegalStateException::class.java) { module.etdump() }
  }

  @Test
  @Throws(IOException::class)
  fun testDoubleDestroyIsSafe() {
    val module = Module.load(getTestFilePath(TEST_FILE_NAME))
    module.destroy()
    module.destroy()
  }

  companion object {
    private const val TEST_FILE_NAME = "/mobilenet_v2.pte"
    private const val MISSING_FILE_NAME = "/missing.pte"
    private const val NON_PTE_FILE_NAME = "/test.txt"
    private const val FORWARD_METHOD = "forward"
    private const val NONE_METHOD = "none"
    private val inputShape = longArrayOf(1, 3, 224, 224)

    private fun dummyInput(): Tensor = Tensor.ones(inputShape, DType.FLOAT)
  }
}
