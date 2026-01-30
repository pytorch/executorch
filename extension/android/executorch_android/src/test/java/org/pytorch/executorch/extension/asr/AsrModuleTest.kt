/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.pytorch.executorch.extension.asr

import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.junit.runners.JUnit4

/**
 * Unit tests for [AsrModule].
 *
 * Note: These are behavioral tests that verify the API contract. Full integration tests
 * with actual models should be done in androidTest.
 */
@RunWith(JUnit4::class)
class AsrModuleTest {

  @Test
  fun testTranscribeWithInvalidModelPathThrowsException() {
    // Verify that AsrModule constructor throws when model file doesn't exist
    assertThrows(IllegalArgumentException::class.java) {
      AsrModule(
          modelPath = "/nonexistent/model.pte",
          tokenizerPath = "/tmp",
      )
    }
  }

  @Test
  fun testTranscribeWithInvalidTokenizerPathThrowsException() {
    // Create a temporary model file
    val tempModelFile = java.io.File.createTempFile("model", ".pte")
    try {
      tempModelFile.writeText("dummy content")

      // Verify that AsrModule constructor throws when tokenizer doesn't exist
      assertThrows(IllegalArgumentException::class.java) {
        AsrModule(
            modelPath = tempModelFile.absolutePath,
            tokenizerPath = "/nonexistent/tokenizer",
        )
      }
    } finally {
      tempModelFile.delete()
    }
  }

  @Test
  fun testAsrTranscribeConfigValidation() {
    // Test that maxNewTokens must be positive
    assertThrows(IllegalArgumentException::class.java) {
      AsrTranscribeConfig(maxNewTokens = 0)
    }

    assertThrows(IllegalArgumentException::class.java) {
      AsrTranscribeConfig(maxNewTokens = -1)
    }

    // Test that temperature must be non-negative
    assertThrows(IllegalArgumentException::class.java) {
      AsrTranscribeConfig(temperature = -0.1f)
    }
  }

  @Test
  fun testAsrTranscribeConfigBuilder() {
    // Test builder pattern for Java interoperability
    val config = AsrTranscribeConfig.Builder()
        .setMaxNewTokens(256)
        .setTemperature(0.5f)
        .setDecoderStartTokenId(123)
        .build()

    assertTrue(config.maxNewTokens == 256L)
    assertTrue(config.temperature == 0.5f)
    assertTrue(config.decoderStartTokenId == 123L)
  }

  @Test
  fun testAsrTranscribeConfigBuilderValidation() {
    // Test that builder validates maxNewTokens
    assertThrows(IllegalArgumentException::class.java) {
      AsrTranscribeConfig.Builder()
          .setMaxNewTokens(0)
          .build()
    }

    // Test that builder validates temperature
    assertThrows(IllegalArgumentException::class.java) {
      AsrTranscribeConfig.Builder()
          .setTemperature(-1.0f)
          .build()
    }
  }
}

