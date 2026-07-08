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
import org.apache.commons.io.FileUtils
import org.json.JSONObject
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.pytorch.executorch.TestFileUtils.getTestFilePath
import org.pytorch.executorch.extension.llm.LlmCallback
import org.pytorch.executorch.extension.llm.LlmGenerationConfig
import org.pytorch.executorch.extension.llm.LlmModule

/** Tests for [LlmGenerationConfig] API, error handling, and callback contract. */
@RunWith(AndroidJUnit4::class)
class LlmGenerationConfigTest {
  private lateinit var llmModule: LlmModule

  @Before
  fun setUp() {
    val pteFile = File(getTestFilePath(TEST_FILE_NAME))
    requireNotNull(javaClass.getResourceAsStream(TEST_FILE_NAME)) {
          "Test resource $TEST_FILE_NAME not found; did android_test_setup.sh run?"
        }
        .use { FileUtils.copyInputStreamToFile(it, pteFile) }

    val tokenizerFile = File(getTestFilePath(TOKENIZER_FILE_NAME))
    requireNotNull(javaClass.getResourceAsStream(TOKENIZER_FILE_NAME)) {
          "Test resource $TOKENIZER_FILE_NAME not found; did android_test_setup.sh run?"
        }
        .use { FileUtils.copyInputStreamToFile(it, tokenizerFile) }

    llmModule =
        LlmModule(getTestFilePath(TEST_FILE_NAME), getTestFilePath(TOKENIZER_FILE_NAME), 0.0f)
  }

  @After
  fun tearDown() {
    if (::llmModule.isInitialized) {
      llmModule.close()
    }
  }

  @Test(timeout = MAX_TEST_TIMEOUT_MS)
  fun testGenerateWithConfig() {
    val config = buildConfig(echo = false)
    val callback = CollectingCallback()
    llmModule.generate(TEST_PROMPT, config, callback)
    assertTrue("Expected non-empty output from generate with config", callback.results.isNotEmpty())
  }

  @Test(timeout = MAX_TEST_TIMEOUT_MS)
  fun testDefaultConfigWorks() {
    val config = LlmGenerationConfig.create().build()
    val callback = CollectingCallback()
    llmModule.generate(TEST_PROMPT, config, callback)
    assertTrue("Default config should produce output", callback.results.isNotEmpty())
  }

  @Test(timeout = MAX_TEST_TIMEOUT_MS)
  fun testEchoModeTrue() {
    val config = buildConfig(echo = true)
    val callback = CollectingCallback()
    llmModule.generate(TEST_PROMPT, config, callback)
    val output = callback.results.joinToString("")
    assertTrue(
        "Echo mode should include prompt tokens",
        output.contains("Hello") || output.contains("hello"),
    )
  }

  @Test(timeout = MAX_TEST_TIMEOUT_MS)
  fun testEchoModeFalse() {
    val config = buildConfig(echo = false)
    val callback = CollectingCallback()
    llmModule.generate(TEST_PROMPT, config, callback)
    assertTrue("Should produce output", callback.results.isNotEmpty())
    val output = callback.results.joinToString("")
    assertFalse(
        "With echo=false, output should NOT start with prompt text",
        output.startsWith(TEST_PROMPT),
    )
  }

  @Test(timeout = MAX_TEST_TIMEOUT_MS)
  fun testSeqLenRespected() {
    val shortSeqLen = 10
    val config = buildConfig(echo = false, seqLen = shortSeqLen)
    val callback = CollectingCallback()
    llmModule.generate(TEST_PROMPT, config, callback)
    // Note: results.size counts onResult callbacks, which is 1:1 with tokens for LlmModule
    assertTrue("Should produce at least 1 token", callback.results.isNotEmpty())
    assertTrue(
        "Token count (${callback.results.size}) should be <= seqLen ($shortSeqLen)",
        callback.results.size <= shortSeqLen,
    )
  }

  @Test(timeout = MAX_TEST_TIMEOUT_MS)
  fun testEmptyPromptHandled() {
    val config = buildConfig(echo = false)
    val callback = CollectingCallback()
    try {
      llmModule.generate("", config, callback)
      assertTrue(
          "Empty prompt should still produce at least one token",
          callback.results.isNotEmpty(),
      )
    } catch (e: RuntimeException) {
      assertTrue(
          "Exception for empty prompt should have a descriptive message",
          e.message != null && e.message!!.isNotEmpty(),
      )
    }
  }

  @Test(timeout = 30_000)
  fun testInvalidModelPath() {
    val noOpCallback =
        object : LlmCallback {
          override fun onResult(result: String) {}
        }
    try {
      val badModule = LlmModule("/nonexistent/path.pte", "/nonexistent/tok.bin", 0.0f)
      badModule.load()
      badModule.generate("test", 10, noOpCallback)
      fail("Should have thrown for invalid model path")
    } catch (_: RuntimeException) {
      // Expected — invalid path detected at construction, load, or generate
    }
  }

  @Test(timeout = MAX_TEST_TIMEOUT_MS)
  fun testGenerateAfterClose() {
    llmModule.close()
    val config = buildConfig(echo = false)
    assertThrows(IllegalStateException::class.java) {
      llmModule.generate(
          TEST_PROMPT,
          config,
          object : LlmCallback {
            override fun onResult(result: String) {}
          },
      )
    }
  }

  @Test(timeout = MAX_TEST_TIMEOUT_MS)
  fun testCallbackContractOnResult() {
    val config = buildConfig(echo = false)
    val callback = CollectingCallback()
    llmModule.generate(TEST_PROMPT, config, callback)
    assertTrue("onResult should be called at least once", callback.results.size >= 1)
    assertTrue(
        "onResult count (${callback.results.size}) should be at most seqLen ($SEQ_LEN)",
        callback.results.size <= SEQ_LEN,
    )
  }

  @Test(timeout = MAX_TEST_TIMEOUT_MS)
  fun testCallbackContractOnStats() {
    val config = buildConfig(echo = false)
    val statsResults = mutableListOf<String>()
    llmModule.generate(
        TEST_PROMPT,
        config,
        object : LlmCallback {
          override fun onResult(result: String) {}

          override fun onStats(stats: String) {
            statsResults.add(stats)
          }
        },
    )
    assertEquals("onStats should be called exactly once per generate()", 1, statsResults.size)
    try {
      val json = JSONObject(statsResults[0])
      assertTrue("Stats JSON must have 'generated_tokens'", json.has("generated_tokens"))
      assertTrue("Stats JSON must have 'inference_end_ms'", json.has("inference_end_ms"))
      assertTrue("Stats JSON must have 'prompt_eval_end_ms'", json.has("prompt_eval_end_ms"))
    } catch (e: Exception) {
      fail("onStats JSON parsing failed: ${e.message}\nRaw stats: ${statsResults[0]}")
    }
  }

  @Test(timeout = MAX_TEST_TIMEOUT_MS)
  fun testCallbackOrdering() {
    val config = buildConfig(echo = false)
    val callOrder = mutableListOf<String>()
    llmModule.generate(
        TEST_PROMPT,
        config,
        object : LlmCallback {
          override fun onResult(result: String) {
            callOrder.add("onResult")
          }

          override fun onStats(stats: String) {
            callOrder.add("onStats")
          }
        },
    )
    assertTrue("Should have received callbacks", callOrder.isNotEmpty())
    val statsIndex = callOrder.indexOfFirst { it == "onStats" }
    if (statsIndex >= 0) {
      val anyResultAfterStats =
          callOrder.subList(statsIndex + 1, callOrder.size).any { it == "onResult" }
      assertFalse(
          "All onResult calls must happen BEFORE onStats (never interleaved)",
          anyResultAfterStats,
      )
    }
  }

  private fun buildConfig(echo: Boolean = false, seqLen: Int = SEQ_LEN) =
      LlmGenerationConfig.create().seqLen(seqLen).temperature(0.0f).echo(echo).build()

  private fun assertThrows(exClass: Class<out Throwable>, block: () -> Unit) {
    try {
      block()
      fail("Expected ${exClass.simpleName} but no exception was thrown")
    } catch (e: Throwable) {
      assertTrue(
          "Expected ${exClass.simpleName} but got ${e.javaClass.simpleName}: ${e.message}",
          exClass.isInstance(e),
      )
    }
  }

  /** Simple callback that collects onResult tokens. */
  private class CollectingCallback : LlmCallback {
    val results = mutableListOf<String>()

    override fun onResult(result: String) {
      results.add(result)
    }
  }

  companion object {
    private const val TEST_FILE_NAME = "/stories.pte"
    private const val TOKENIZER_FILE_NAME = "/tokenizer.bin"
    private const val TEST_PROMPT = "Hello"
    private const val SEQ_LEN = 32
    private const val MAX_TEST_TIMEOUT_MS = 60_000L
  }
}
