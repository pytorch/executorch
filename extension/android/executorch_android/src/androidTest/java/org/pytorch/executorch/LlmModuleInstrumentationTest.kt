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
import java.nio.ByteBuffer
import java.nio.ByteOrder
import org.apache.commons.io.FileUtils
import org.json.JSONException
import org.json.JSONObject
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.pytorch.executorch.TestFileUtils.getTestFilePath
import org.pytorch.executorch.extension.llm.LlmCallback
import org.pytorch.executorch.extension.llm.LlmModule

/** Unit tests for [org.pytorch.executorch.extension.llm.LlmModule]. */
@RunWith(AndroidJUnit4::class)
class LlmModuleInstrumentationTest : LlmCallback {
  private val results: MutableList<String> = ArrayList()
  private val tokensPerSecond: MutableList<Float> = ArrayList()
  private lateinit var llmModule: LlmModule

  @Before
  @Throws(IOException::class)
  fun setUp() {
    // copy zipped test resources to local device
    val addPteFile = File(getTestFilePath(TEST_FILE_NAME))
    var inputStream = javaClass.getResourceAsStream(TEST_FILE_NAME)
    FileUtils.copyInputStreamToFile(inputStream, addPteFile)
    inputStream.close()

    val tokenizerFile = File(getTestFilePath(TOKENIZER_FILE_NAME))
    inputStream = javaClass.getResourceAsStream(TOKENIZER_FILE_NAME)
    FileUtils.copyInputStreamToFile(inputStream, tokenizerFile)
    inputStream.close()

    llmModule =
        LlmModule(getTestFilePath(TEST_FILE_NAME), getTestFilePath(TOKENIZER_FILE_NAME), 0.0f)
  }

  @After
  fun tearDown() {
    if (::llmModule.isInitialized) {
      llmModule.close()
    }
  }

  @Test
  @Throws(IOException::class, URISyntaxException::class)
  fun testGenerate() {
    llmModule.load()

    llmModule.generate(TEST_PROMPT, SEQ_LEN, this@LlmModuleInstrumentationTest)
    assertEquals(results.size.toLong(), SEQ_LEN.toLong())
    assertTrue(tokensPerSecond[tokensPerSecond.size - 1] > 0)
  }

  @Test
  @Throws(IOException::class, URISyntaxException::class)
  fun testGenerateAndStop() {
    llmModule.generate(
        TEST_PROMPT,
        SEQ_LEN,
        object : LlmCallback {
          override fun onResult(result: String) {
            this@LlmModuleInstrumentationTest.onResult(result)
            llmModule.stop()
          }

          override fun onStats(stats: String) {
            this@LlmModuleInstrumentationTest.onStats(stats)
          }
        },
    )

    val stoppedResultSize = results.size
    assertTrue(stoppedResultSize < SEQ_LEN)
  }

  override fun onResult(result: String) {
    results.add(result)
  }

  override fun onStats(stats: String) {
    var tps = 0f
    try {
      val jsonObject = JSONObject(stats)
      val numGeneratedTokens = jsonObject.getInt("generated_tokens")
      val inferenceEndMs = jsonObject.getInt("inference_end_ms")
      val promptEvalEndMs = jsonObject.getInt("prompt_eval_end_ms")
      tps = numGeneratedTokens.toFloat() / (inferenceEndMs - promptEvalEndMs) * 1000
      tokensPerSecond.add(tps)
    } catch (_: JSONException) {}
  }

  // --- prefillImages(ByteBuffer) validation tests ---

  @Test
  fun testPrefillImagesByteBuffer_nonDirectThrows() {
    val heapBuffer = ByteBuffer.allocate(2 * 2 * 3)
    assertThrows(IllegalArgumentException::class.java) {
      llmModule.prefillImages(heapBuffer, 2, 2, 3)
    }
  }

  @Test
  fun testPrefillImagesByteBuffer_insufficientRemainingThrows() {
    val buffer = ByteBuffer.allocateDirect(10)
    assertThrows(IllegalArgumentException::class.java) { llmModule.prefillImages(buffer, 2, 2, 3) }
  }

  @Test
  fun testPrefillImagesByteBuffer_zeroWidthThrows() {
    val buffer = ByteBuffer.allocateDirect(12)
    assertThrows(IllegalArgumentException::class.java) { llmModule.prefillImages(buffer, 0, 2, 3) }
  }

  @Test
  fun testPrefillImagesByteBuffer_zeroHeightThrows() {
    val buffer = ByteBuffer.allocateDirect(12)
    assertThrows(IllegalArgumentException::class.java) { llmModule.prefillImages(buffer, 2, 0, 3) }
  }

  @Test
  fun testPrefillImagesByteBuffer_zeroChannelsThrows() {
    val buffer = ByteBuffer.allocateDirect(12)
    assertThrows(IllegalArgumentException::class.java) { llmModule.prefillImages(buffer, 2, 2, 0) }
  }

  @Test
  fun testPrefillImagesByteBuffer_negativeWidthThrows() {
    val buffer = ByteBuffer.allocateDirect(12)
    assertThrows(IllegalArgumentException::class.java) { llmModule.prefillImages(buffer, -1, 2, 3) }
  }

  @Test
  fun testPrefillImagesByteBuffer_negativeHeightThrows() {
    val buffer = ByteBuffer.allocateDirect(12)
    assertThrows(IllegalArgumentException::class.java) { llmModule.prefillImages(buffer, 2, -1, 3) }
  }

  @Test
  fun testPrefillImagesByteBuffer_negativeChannelsThrows() {
    val buffer = ByteBuffer.allocateDirect(12)
    assertThrows(IllegalArgumentException::class.java) { llmModule.prefillImages(buffer, 2, 2, -1) }
  }

  @Test
  fun testPrefillImagesByteBuffer_validBufferPassesValidation() {
    val buffer = ByteBuffer.allocateDirect(2 * 2 * 3)
    try {
      llmModule.prefillImages(buffer, 2, 2, 3)
    } catch (e: IllegalArgumentException) {
      throw AssertionError("Validation should not reject a correctly sized direct buffer", e)
    } catch (_: RuntimeException) {
      // Expected: native call may fail since this is a text-only model
    }
  }

  // --- prefillNormalizedImage(ByteBuffer) validation tests ---

  @Test
  fun testPrefillNormalizedImage_nonDirectThrows() {
    val heapBuffer = ByteBuffer.allocate(2 * 2 * 3 * 4)
    assertThrows(IllegalArgumentException::class.java) {
      llmModule.prefillNormalizedImage(heapBuffer, 2, 2, 3)
    }
  }

  @Test
  fun testPrefillNormalizedImage_insufficientRemainingThrows() {
    val buffer = ByteBuffer.allocateDirect(10)
    buffer.order(ByteOrder.nativeOrder())
    assertThrows(IllegalArgumentException::class.java) {
      llmModule.prefillNormalizedImage(buffer, 2, 2, 3)
    }
  }

  @Test
  fun testPrefillNormalizedImage_zeroWidthThrows() {
    val buffer = ByteBuffer.allocateDirect(2 * 2 * 3 * 4)
    buffer.order(ByteOrder.nativeOrder())
    assertThrows(IllegalArgumentException::class.java) {
      llmModule.prefillNormalizedImage(buffer, 0, 2, 3)
    }
  }

  @Test
  fun testPrefillNormalizedImage_zeroHeightThrows() {
    val buffer = ByteBuffer.allocateDirect(2 * 2 * 3 * 4)
    buffer.order(ByteOrder.nativeOrder())
    assertThrows(IllegalArgumentException::class.java) {
      llmModule.prefillNormalizedImage(buffer, 2, 0, 3)
    }
  }

  @Test
  fun testPrefillNormalizedImage_zeroChannelsThrows() {
    val buffer = ByteBuffer.allocateDirect(2 * 2 * 3 * 4)
    buffer.order(ByteOrder.nativeOrder())
    assertThrows(IllegalArgumentException::class.java) {
      llmModule.prefillNormalizedImage(buffer, 2, 2, 0)
    }
  }

  @Test
  fun testPrefillNormalizedImage_negativeWidthThrows() {
    val buffer = ByteBuffer.allocateDirect(2 * 2 * 3 * 4)
    buffer.order(ByteOrder.nativeOrder())
    assertThrows(IllegalArgumentException::class.java) {
      llmModule.prefillNormalizedImage(buffer, -1, 2, 3)
    }
  }

  @Test
  fun testPrefillNormalizedImage_negativeHeightThrows() {
    val buffer = ByteBuffer.allocateDirect(2 * 2 * 3 * 4)
    buffer.order(ByteOrder.nativeOrder())
    assertThrows(IllegalArgumentException::class.java) {
      llmModule.prefillNormalizedImage(buffer, 2, -1, 3)
    }
  }

  @Test
  fun testPrefillNormalizedImage_negativeChannelsThrows() {
    val buffer = ByteBuffer.allocateDirect(2 * 2 * 3 * 4)
    buffer.order(ByteOrder.nativeOrder())
    assertThrows(IllegalArgumentException::class.java) {
      llmModule.prefillNormalizedImage(buffer, 2, 2, -1)
    }
  }

  @Test
  fun testPrefillNormalizedImage_nonNativeByteOrderThrows() {
    val buffer = ByteBuffer.allocateDirect(2 * 2 * 3 * 4)
    val nonNativeOrder =
        if (ByteOrder.nativeOrder() == ByteOrder.LITTLE_ENDIAN) ByteOrder.BIG_ENDIAN
        else ByteOrder.LITTLE_ENDIAN
    buffer.order(nonNativeOrder)
    assertThrows(IllegalArgumentException::class.java) {
      llmModule.prefillNormalizedImage(buffer, 2, 2, 3)
    }
  }

  @Test
  fun testPrefillNormalizedImage_misalignedPositionThrows() {
    val buffer = ByteBuffer.allocateDirect(2 * 2 * 3 * 4 + 1)
    buffer.order(ByteOrder.nativeOrder())
    buffer.position(1)
    assertThrows(IllegalArgumentException::class.java) {
      llmModule.prefillNormalizedImage(buffer, 2, 2, 3)
    }
  }

  @Test
  fun testPrefillNormalizedImage_validBufferPassesValidation() {
    val buffer = ByteBuffer.allocateDirect(2 * 2 * 3 * 4)
    buffer.order(ByteOrder.nativeOrder())
    try {
      llmModule.prefillNormalizedImage(buffer, 2, 2, 3)
    } catch (e: IllegalArgumentException) {
      throw AssertionError("Validation should not reject a correctly sized direct buffer", e)
    } catch (_: RuntimeException) {
      // Expected: native call may fail since this is a text-only model
    }
  }

  // --- Lifecycle tests ---

  @Test
  fun testUseAfterCloseThrows() {
    llmModule.close()
    assertThrows(IllegalStateException::class.java) {
      llmModule.generate(TEST_PROMPT, SEQ_LEN, this@LlmModuleInstrumentationTest)
    }
  }

  @Test
  fun testCloseIsIdempotent() {
    llmModule.close()
    llmModule.close()
  }

  companion object {
    private const val TEST_FILE_NAME = "/stories.pte"
    private const val TOKENIZER_FILE_NAME = "/tokenizer.bin"
    private const val TEST_PROMPT = "Hello"
    private const val SEQ_LEN = 32
  }
}
