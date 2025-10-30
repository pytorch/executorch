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
import org.apache.commons.io.FileUtils
import org.json.JSONException
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Rule
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

  @get:Rule
  var runtimePermissionRule: GrantPermissionRule =
      GrantPermissionRule.grant(Manifest.permission.READ_EXTERNAL_STORAGE)

  @Test
  @Throws(IOException::class, URISyntaxException::class)
  fun testGenerate() {
    val loadResult = llmModule.load()
    // Check that the model can be load successfully
    assertEquals(OK.toLong(), loadResult.toLong())

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

  companion object {
    private const val TEST_FILE_NAME = "/stories.pte"
    private const val TOKENIZER_FILE_NAME = "/tokenizer.bin"
    private const val TEST_PROMPT = "Hello"
    private const val OK = 0x00
    private const val SEQ_LEN = 32
  }
}
