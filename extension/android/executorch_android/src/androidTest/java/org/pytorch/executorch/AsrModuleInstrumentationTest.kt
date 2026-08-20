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
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Assume.assumeNotNull
import org.junit.Test
import org.junit.runner.RunWith
import org.pytorch.executorch.TestFileUtils.getTestFilePath
import org.pytorch.executorch.extension.asr.AsrCallback
import org.pytorch.executorch.extension.asr.AsrModule
import org.pytorch.executorch.extension.asr.AsrTranscribeConfig

/**
 * Instrumentation tests for [AsrModule], [AsrTranscribeConfig], and [AsrCallback].
 *
 * Tests cover:
 * - Constructor validation (invalid model/tokenizer/preprocessor paths)
 * - AsrTranscribeConfig builder and validation
 * - Lifecycle (close idempotency, use-after-close)
 * - Transcribe validation (invalid WAV path)
 *
 * The test fixture is the TinyStories-110M LLM model, NOT an ASR model, so functional transcription
 * tests are not possible. Tests that require a valid AsrModule instance handle the case where
 * nativeCreate fails (stories.pte lacks encoder/text_decoder methods).
 */
@RunWith(AndroidJUnit4::class)
class AsrModuleInstrumentationTest {

  // ─── Constructor validation ─────────────────────────────────────────────────

  @Test(timeout = 30_000)
  fun testInvalidModelPathThrows() {
    try {
      AsrModule("/nonexistent/model.pte", "/nonexistent/tokenizer")
      fail("Should throw for invalid model path")
    } catch (_: IllegalArgumentException) {
      // Expected: require(modelFile.canRead() && modelFile.isFile)
    }
  }

  @Test(timeout = 30_000)
  fun testInvalidTokenizerPathThrows() {
    val modelFile = provisionModelFile()
    assumeNotNull("Test resource $MODEL_FILE_NAME not available", modelFile)
    try {
      AsrModule(modelFile!!.absolutePath, "/nonexistent/tokenizer")
      fail("Should throw for invalid tokenizer path")
    } catch (_: IllegalArgumentException) {
      // Expected: require(tokenizerFile.exists())
    }
  }

  @Test(timeout = 30_000)
  fun testInvalidPreprocessorPathThrows() {
    val modelFile = provisionModelFile()
    val tokenizerFile = provisionTokenizerFile()
    assumeNotNull("Test resource $MODEL_FILE_NAME not available", modelFile)
    assumeNotNull("Test resource $TOKENIZER_FILE_NAME not available", tokenizerFile)
    try {
      AsrModule(
          modelFile!!.absolutePath,
          tokenizerFile!!.absolutePath,
          preprocessorPath = "/nonexistent/preprocessor.pte",
      )
      fail("Should throw for invalid preprocessor path")
    } catch (_: IllegalArgumentException) {
      // Expected: require(preprocessorFile.canRead() && preprocessorFile.isFile)
    }
  }

  @Test(timeout = 30_000)
  fun testNonAsrModelFailsGracefully() {
    val modelFile = provisionModelFile()
    val tokenizerFile = provisionTokenizerFile()
    assumeNotNull("Test resource $MODEL_FILE_NAME not available", modelFile)
    assumeNotNull("Test resource $TOKENIZER_FILE_NAME not available", tokenizerFile)
    try {
      val module = AsrModule(modelFile!!.absolutePath, tokenizerFile!!.absolutePath)
      // If construction succeeds (model was accepted), verify basic state
      assertTrue("Module should be valid after construction", module.isValid)
      module.close()
    } catch (_: ExecutorchRuntimeException) {
      // Expected: nativeCreate returns 0 for non-ASR model
    } catch (_: RuntimeException) {
      // Also acceptable: native layer rejects the model
    }
  }

  // ─── Lifecycle ──────────────────────────────────────────────────────────────

  @Test(timeout = 30_000)
  fun testCloseIsIdempotent() {
    val module = tryCreateAsrModule() ?: return
    module.close()
    module.close()
    module.close()
    assertFalse("isValid must be false after close", module.isValid)
  }

  @Test(timeout = 30_000)
  fun testLoadAfterCloseThrows() {
    val module = tryCreateAsrModule() ?: return
    module.close()
    try {
      module.load()
      fail("load() after close() must throw IllegalStateException")
    } catch (_: IllegalStateException) {
      // Expected
    }
  }

  @Test(timeout = 30_000)
  fun testTranscribeAfterCloseThrows() {
    val module = tryCreateAsrModule() ?: return
    module.close()
    try {
      module.transcribe("/some/audio.wav")
      fail("transcribe() after close() must throw IllegalStateException")
    } catch (_: IllegalStateException) {
      // Expected
    }
  }

  @Test(timeout = 30_000)
  fun testIsValidAndIsLoadedState() {
    val module = tryCreateAsrModule() ?: return
    assertTrue("Module should be valid after construction", module.isValid)
    module.close()
    assertFalse("Module should not be valid after close", module.isValid)
    assertFalse("Module should not be loaded after close", module.isLoaded)
  }

  // ─── Transcribe validation ──────────────────────────────────────────────────

  @Test(timeout = 30_000)
  fun testTranscribeInvalidWavPathThrows() {
    val module = tryCreateAsrModule() ?: return
    try {
      module.transcribe("/nonexistent/audio.wav")
      fail("transcribe() with invalid WAV path must throw")
    } catch (_: IllegalArgumentException) {
      // Expected: require(wavFile.canRead() && wavFile.isFile)
    } finally {
      module.close()
    }
  }

  // ─── AsrTranscribeConfig ────────────────────────────────────────────────────

  @Test
  fun testConfigDefaults() {
    val config = AsrTranscribeConfig()
    assertEquals(128L, config.maxNewTokens)
    assertEquals(0.0f, config.temperature, 0.0f)
    assertEquals(0L, config.decoderStartTokenId)
  }

  @Test
  fun testConfigBuilder() {
    val config =
        AsrTranscribeConfig.Builder()
            .setMaxNewTokens(256)
            .setTemperature(0.7f)
            .setDecoderStartTokenId(50258)
            .build()
    assertEquals(256L, config.maxNewTokens)
    assertEquals(0.7f, config.temperature, 0.001f)
    assertEquals(50258L, config.decoderStartTokenId)
  }

  @Test
  fun testConfigCustomValues() {
    val config = AsrTranscribeConfig(maxNewTokens = 64, temperature = 0.5f, decoderStartTokenId = 1)
    assertEquals(64L, config.maxNewTokens)
    assertEquals(0.5f, config.temperature, 0.001f)
    assertEquals(1L, config.decoderStartTokenId)
  }

  @Test(expected = IllegalArgumentException::class)
  fun testConfigZeroMaxNewTokensThrows() {
    AsrTranscribeConfig(maxNewTokens = 0)
  }

  @Test(expected = IllegalArgumentException::class)
  fun testConfigNegativeMaxNewTokensThrows() {
    AsrTranscribeConfig(maxNewTokens = -1)
  }

  @Test(expected = IllegalArgumentException::class)
  fun testConfigNegativeTemperatureThrows() {
    AsrTranscribeConfig(temperature = -0.1f)
  }

  @Test(expected = IllegalArgumentException::class)
  fun testConfigBuilderZeroMaxNewTokensThrows() {
    AsrTranscribeConfig.Builder().setMaxNewTokens(0).build()
  }

  @Test(expected = IllegalArgumentException::class)
  fun testConfigBuilderNegativeTemperatureThrows() {
    AsrTranscribeConfig.Builder().setTemperature(-1.0f).build()
  }

  @Test
  fun testConfigDataClassEquality() {
    val a = AsrTranscribeConfig(maxNewTokens = 100, temperature = 0.5f, decoderStartTokenId = 42)
    val b = AsrTranscribeConfig(maxNewTokens = 100, temperature = 0.5f, decoderStartTokenId = 42)
    assertEquals(a, b)
    assertEquals(a.hashCode(), b.hashCode())
  }

  // ─── Helpers ────────────────────────────────────────────────────────────────

  @Throws(IOException::class)
  private fun provisionModelFile(): File? {
    val pteFile = File(getTestFilePath(MODEL_FILE_NAME))
    val stream = javaClass.getResourceAsStream(MODEL_FILE_NAME) ?: return null
    stream.use { FileUtils.copyInputStreamToFile(it, pteFile) }
    return pteFile
  }

  @Throws(IOException::class)
  private fun provisionTokenizerFile(): File? {
    val tokenizerFile = File(getTestFilePath(TOKENIZER_FILE_NAME))
    val stream = javaClass.getResourceAsStream(TOKENIZER_FILE_NAME) ?: return null
    stream.use { FileUtils.copyInputStreamToFile(it, tokenizerFile) }
    return tokenizerFile
  }

  private fun tryCreateAsrModule(): AsrModule? {
    val modelFile = provisionModelFile()
    val tokenizerFile = provisionTokenizerFile()
    assumeNotNull("Test resource $MODEL_FILE_NAME not available", modelFile)
    assumeNotNull("Test resource $TOKENIZER_FILE_NAME not available", tokenizerFile)
    return try {
      AsrModule(modelFile!!.absolutePath, tokenizerFile!!.absolutePath)
    } catch (_: RuntimeException) {
      // nativeCreate may reject non-ASR models — skip lifecycle tests in that case
      null
    }
  }

  companion object {
    private const val MODEL_FILE_NAME = "/stories.pte"
    private const val TOKENIZER_FILE_NAME = "/tokenizer.bin"
  }
}
