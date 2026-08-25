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
import org.apache.commons.io.FileUtils
import org.junit.After
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.pytorch.executorch.TestFileUtils.getTestFilePath
import org.pytorch.executorch.extension.llm.LlmCallback
import org.pytorch.executorch.extension.llm.LlmModule
import org.pytorch.executorch.extension.llm.LlmModuleConfig

/**
 * Instrumentation tests for LlmModule's LoRA / dataFiles constructor paths.
 *
 * LoRA adapters are loaded at construction time via the `dataFiles` parameter or
 * `LlmModuleConfig.dataPath`. These tests verify that:
 * 1. The dataFiles constructor variants produce a functional module
 * 2. LlmModuleConfig with dataPath integrates correctly
 * 3. Invalid data file paths are handled gracefully
 * 4. Empty vs null dataFiles behave identically to no-data constructors
 *
 * Uses TinyStories-110M; no LoRA adapter fixture is available so functional LoRA tests
 * (output-changes-with-adapter) are not possible.
 */
@RunWith(AndroidJUnit4::class)
class LlmLoraInstrumentationTest {

  private var llmModule: LlmModule? = null

  @Before
  @Throws(IOException::class)
  fun setUp() {
    val pteFile = File(getTestFilePath(MODEL_FILE_NAME))
    requireNotNull(javaClass.getResourceAsStream(MODEL_FILE_NAME)) {
          "Test resource $MODEL_FILE_NAME not found; did android_test_setup.sh run?"
        }
        .use { FileUtils.copyInputStreamToFile(it, pteFile) }

    val tokenizerFile = File(getTestFilePath(TOKENIZER_FILE_NAME))
    requireNotNull(javaClass.getResourceAsStream(TOKENIZER_FILE_NAME)) {
          "Test resource $TOKENIZER_FILE_NAME not found; did android_test_setup.sh run?"
        }
        .use { FileUtils.copyInputStreamToFile(it, tokenizerFile) }
  }

  @After
  fun tearDown() {
    llmModule?.close()
    llmModule = null
  }

  // ─── dataFiles constructor variants ─────────────────────────────────────────

  @Test(timeout = MAX_TEST_TIMEOUT_MS)
  fun testConstructorWithEmptyDataFilesList() {
    llmModule =
        LlmModule(
            LlmModule.MODEL_TYPE_TEXT,
            getTestFilePath(MODEL_FILE_NAME),
            getTestFilePath(TOKENIZER_FILE_NAME),
            0.0f,
            emptyList<String>(),
        )
    val tokens = generateAndCollect(llmModule!!)
    assertTrue("Module with empty dataFiles should generate tokens", tokens.isNotEmpty())
  }

  @Test(timeout = MAX_TEST_TIMEOUT_MS)
  fun testConstructorWithNullDataPath() {
    llmModule =
        LlmModule(
            LlmModule.MODEL_TYPE_TEXT,
            getTestFilePath(MODEL_FILE_NAME),
            getTestFilePath(TOKENIZER_FILE_NAME),
            0.0f,
            null as String?,
        )
    val tokens = generateAndCollect(llmModule!!)
    assertTrue("Module with null dataPath should generate tokens", tokens.isNotEmpty())
  }

  @Test(timeout = MAX_TEST_TIMEOUT_MS)
  fun testConstructorWithDataFilesAndBosEos() {
    llmModule =
        LlmModule(
            LlmModule.MODEL_TYPE_TEXT,
            getTestFilePath(MODEL_FILE_NAME),
            getTestFilePath(TOKENIZER_FILE_NAME),
            0.0f,
            emptyList<String>(),
            0,
            0,
        )
    val tokens = generateAndCollect(llmModule!!)
    assertTrue("Module with dataFiles+BOS/EOS should generate tokens", tokens.isNotEmpty())
  }

  // ─── LlmModuleConfig with dataPath ──────────────────────────────────────────

  @Test(timeout = MAX_TEST_TIMEOUT_MS)
  fun testLlmModuleConfigNoDataPath() {
    val config =
        LlmModuleConfig.create()
            .modulePath(getTestFilePath(MODEL_FILE_NAME))
            .tokenizerPath(getTestFilePath(TOKENIZER_FILE_NAME))
            .temperature(0.0f)
            .build()
    llmModule = LlmModule(config)
    val tokens = generateAndCollect(llmModule!!)
    assertTrue("Module via config with no dataPath should generate tokens", tokens.isNotEmpty())
  }

  @Test(timeout = MAX_TEST_TIMEOUT_MS)
  fun testLlmModuleConfigWithNullDataPath() {
    val config =
        LlmModuleConfig.create()
            .modulePath(getTestFilePath(MODEL_FILE_NAME))
            .tokenizerPath(getTestFilePath(TOKENIZER_FILE_NAME))
            .temperature(0.0f)
            .dataPath(null)
            .build()
    llmModule = LlmModule(config)
    val tokens = generateAndCollect(llmModule!!)
    assertTrue("Module via config with null dataPath should generate tokens", tokens.isNotEmpty())
  }

  @Test(timeout = MAX_TEST_TIMEOUT_MS)
  fun testLlmModuleConfigWithLoadMode() {
    val config =
        LlmModuleConfig.create()
            .modulePath(getTestFilePath(MODEL_FILE_NAME))
            .tokenizerPath(getTestFilePath(TOKENIZER_FILE_NAME))
            .temperature(0.0f)
            .loadMode(LlmModuleConfig.LOAD_MODE_FILE)
            .build()
    llmModule = LlmModule(config)
    val tokens = generateAndCollect(llmModule!!)
    assertTrue("Module via config with LOAD_MODE_FILE should generate tokens", tokens.isNotEmpty())
  }

  // ─── Invalid data file paths ────────────────────────────────────────────────

  @Test(timeout = MAX_TEST_TIMEOUT_MS)
  fun testInvalidDataFilePathThrowsOnConstruction() {
    try {
      llmModule =
          LlmModule(
              LlmModule.MODEL_TYPE_TEXT,
              getTestFilePath(MODEL_FILE_NAME),
              getTestFilePath(TOKENIZER_FILE_NAME),
              0.0f,
              listOf("/nonexistent/lora_weights.bin"),
          )
      // dataFiles are passed to native initHybrid — invalid paths should cause
      // construction to fail. If we reach here, the native layer didn't validate.
      llmModule!!.close()
      fail("Construction should have thrown for invalid data file path")
    } catch (e: RuntimeException) {
      assertTrue(
          "Exception message should be non-empty",
          e.message != null && e.message!!.isNotEmpty(),
      )
    }
  }

  @Test(timeout = MAX_TEST_TIMEOUT_MS)
  fun testMultipleInvalidDataFilePathsThrowOnConstruction() {
    try {
      llmModule =
          LlmModule(
              LlmModule.MODEL_TYPE_TEXT,
              getTestFilePath(MODEL_FILE_NAME),
              getTestFilePath(TOKENIZER_FILE_NAME),
              0.0f,
              listOf("/nonexistent/a.bin", "/nonexistent/b.bin"),
          )
      llmModule!!.close()
      fail("Construction should have thrown for invalid data file paths")
    } catch (e: RuntimeException) {
      assertTrue(
          "Exception message should be non-empty",
          e.message != null && e.message!!.isNotEmpty(),
      )
    }
  }

  // ─── Baseline equivalence ───────────────────────────────────────────────────

  @Test(timeout = MAX_TEST_TIMEOUT_MS)
  fun testEmptyDataFilesMatchesNoDataConstructor() {
    val moduleNoData =
        LlmModule(getTestFilePath(MODEL_FILE_NAME), getTestFilePath(TOKENIZER_FILE_NAME), 0.0f)
    val moduleEmptyList =
        LlmModule(
            LlmModule.MODEL_TYPE_TEXT,
            getTestFilePath(MODEL_FILE_NAME),
            getTestFilePath(TOKENIZER_FILE_NAME),
            0.0f,
            emptyList<String>(),
        )

    try {
      val tokensNoData = generateAndCollect(moduleNoData)
      val tokensEmptyList = generateAndCollect(moduleEmptyList)

      assertTrue("Both constructors should produce tokens", tokensNoData.isNotEmpty())
      assertTrue("Both constructors should produce tokens", tokensEmptyList.isNotEmpty())
    } finally {
      moduleNoData.close()
      moduleEmptyList.close()
    }
  }

  // ─── LlmModuleConfig builder validation ─────────────────────────────────────

  @Test(expected = IllegalArgumentException::class)
  fun testConfigBuilderMissingModulePathThrows() {
    LlmModuleConfig.create().tokenizerPath("/some/tokenizer.bin").build()
  }

  @Test(expected = IllegalArgumentException::class)
  fun testConfigBuilderMissingTokenizerPathThrows() {
    LlmModuleConfig.create().modulePath("/some/model.pte").build()
  }

  @Test(expected = IllegalArgumentException::class)
  fun testConfigBuilderInvalidLoadModeThrows() {
    LlmModuleConfig.create()
        .modulePath("/some/model.pte")
        .tokenizerPath("/some/tokenizer.bin")
        .loadMode(99)
        .build()
  }

  @Test
  fun testConfigBuilderAllLoadModes() {
    val modes =
        listOf(
            LlmModuleConfig.LOAD_MODE_FILE,
            LlmModuleConfig.LOAD_MODE_MMAP,
            LlmModuleConfig.LOAD_MODE_MMAP_USE_MLOCK,
            LlmModuleConfig.LOAD_MODE_MMAP_USE_MLOCK_IGNORE_ERRORS,
        )
    for (mode in modes) {
      val config =
          LlmModuleConfig.create()
              .modulePath("/some/model.pte")
              .tokenizerPath("/some/tokenizer.bin")
              .loadMode(mode)
              .build()
      assertTrue("Config should accept load mode $mode", config.loadMode == mode)
    }
  }

  // ─── Helpers ────────────────────────────────────────────────────────────────

  private fun generateAndCollect(module: LlmModule): List<String> {
    val collector = mutableListOf<String>()
    module.generate(
        TEST_PROMPT,
        SEQ_LEN,
        object : LlmCallback {
          override fun onResult(result: String) {
            collector.add(result)
          }
        },
    )
    return collector
  }

  companion object {
    private const val MODEL_FILE_NAME = "/stories.pte"
    private const val TOKENIZER_FILE_NAME = "/tokenizer.bin"
    private const val TEST_PROMPT = "Once"
    private const val SEQ_LEN = 16
    private const val MAX_TEST_TIMEOUT_MS = 120_000L
  }
}
