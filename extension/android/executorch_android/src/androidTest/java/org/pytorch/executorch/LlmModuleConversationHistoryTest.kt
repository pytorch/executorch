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
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.pytorch.executorch.TestFileUtils.getTestFilePath
import org.pytorch.executorch.extension.llm.LlmCallback
import org.pytorch.executorch.extension.llm.LlmModule

/**
 * Behavioral tests for multi-turn / conversation-history semantics on [LlmModule].
 *
 * These tests run on the TinyStories-110M fixture pulled by `android_test_setup.sh`, which is too
 * small and not instruction-tuned, so we cannot assert anything about the *content* of generated
 * text (e.g. "did the model recall the user's name"). Instead, we assert structural invariants of
 * the KV-cache + reset plumbing that any conversation-history feature depends on:
 * 1. Determinism after [LlmModule.resetContext] at temperature=0 (greedy decode).
 * 2. State preservation across successive [LlmModule.generate] calls (no reset → output diverges).
 * 3. [LlmModule.prefillPrompt] influences the next [LlmModule.generate] call.
 * 4. [LlmModule.resetContext] fully clears prefilled state.
 *
 * All tests run on both internal (fbsource Sandcastle) and OSS (GitHub Actions) Android CI because
 * the fixture is fetched from the public `ossci-android` S3 bucket by `android_test_setup.sh` and
 * the test only depends on the public `LlmModule` API.
 */
@RunWith(AndroidJUnit4::class)
class LlmModuleConversationHistoryTest {

  private lateinit var llmModule: LlmModule

  @Before
  @Throws(IOException::class)
  fun setUp() {
    val pteFile = File(getTestFilePath(TEST_FILE_NAME))
    requireNotNull(javaClass.getResourceAsStream(TEST_FILE_NAME)) {
          "Test resource $TEST_FILE_NAME not found; did android_test_setup.sh run?"
        }
        .use { pteStream -> FileUtils.copyInputStreamToFile(pteStream, pteFile) }

    val tokenizerFile = File(getTestFilePath(TOKENIZER_FILE_NAME))
    requireNotNull(javaClass.getResourceAsStream(TOKENIZER_FILE_NAME)) {
          "Test resource $TOKENIZER_FILE_NAME not found; did android_test_setup.sh run?"
        }
        .use { tokenizerStream -> FileUtils.copyInputStreamToFile(tokenizerStream, tokenizerFile) }

    llmModule =
        LlmModule(getTestFilePath(TEST_FILE_NAME), getTestFilePath(TOKENIZER_FILE_NAME), 0.0f)
    llmModule.load()
  }

  @After
  fun tearDown() {
    if (::llmModule.isInitialized) {
      llmModule.close()
    }
  }

  /**
   * resetContext() + greedy decode (temperature=0) must produce identical output across two runs
   * with the same prompt. This is the foundational invariant any conversation-history feature
   * relies on: clearing the KV cache truly returns the model to a clean state.
   */
  @Test
  @Throws(IOException::class)
  fun testResetContextProducesDeterministicOutput() {
    val firstRun = generateAndCollect(PROMPT_A)
    llmModule.resetContext()
    val secondRun = generateAndCollect(PROMPT_A)

    assertTrue("Expected non-empty generation on first run", firstRun.isNotEmpty())
    assertTrue("Expected non-empty generation on second run", secondRun.isNotEmpty())
    assertEquals(
        "Greedy generation after resetContext() must be deterministic for the same prompt.",
        firstRun,
        secondRun,
    )
  }

  /**
   * Without resetContext() between calls, KV-cache state persists and influences subsequent
   * generation. Generating the same prompt twice in a row should produce different output the
   * second time (because the KV cache is no longer empty and start position is non-zero), or the
   * second call may throw because the runtime detects the stale KV state.
   *
   * Either outcome proves state persistence. If this test ever starts failing (i.e. both calls
   * succeed with equal output), the runtime is silently dropping state between generate() calls —
   * that would break multi-turn conversations.
   */
  @Test
  @Throws(IOException::class)
  fun testKvCacheStatePersistsAcrossGenerateCalls() {
    val firstRun = generateAndCollect(PROMPT_A)
    assertTrue("Expected non-empty generation on first run", firstRun.isNotEmpty())

    try {
      val secondRun = generateAndCollect(PROMPT_A)
      assertNotEquals(
          "Without resetContext(), repeated generate() calls must reflect persisted KV state.",
          firstRun,
          secondRun,
      )
    } catch (_: ExecutorchRuntimeException) {
      // The second generate() threw because KV-cache state from the first call
      // affected execution — this also proves state persistence.
    }
  }

  /**
   * prefillPrompt() must influence the next generate() — i.e. prefilled tokens are part of the
   * conversation history. If prefilling has no effect, multi-turn flows that rely on injecting
   * prior turns via prefill are broken.
   */
  @Test
  @Throws(IOException::class)
  fun testPrefillPromptInfluencesNextGeneration() {
    val baselineRun = generateAndCollect(PROMPT_A)

    llmModule.resetContext()
    llmModule.prefillPrompt(PREFILL_HISTORY)
    val withHistoryRun = generateAndCollect(PROMPT_A)

    assertTrue("Expected non-empty baseline generation", baselineRun.isNotEmpty())
    assertTrue("Expected non-empty post-prefill generation", withHistoryRun.isNotEmpty())
    assertNotEquals(
        "prefillPrompt() must alter the KV state seen by the next generate() call.",
        baselineRun,
        withHistoryRun,
    )
  }

  /**
   * resetContext() must fully clear prefilled state — running prefill then resetting then
   * generating should match a clean-slate generation of the same prompt.
   */
  @Test
  @Throws(IOException::class)
  fun testResetContextClearsPrefilledHistory() {
    val cleanRun = generateAndCollect(PROMPT_A)

    llmModule.resetContext()
    llmModule.prefillPrompt(PREFILL_HISTORY)
    llmModule.resetContext()
    val postResetRun = generateAndCollect(PROMPT_A)

    assertTrue("Expected non-empty clean run", cleanRun.isNotEmpty())
    assertTrue("Expected non-empty post-reset run", postResetRun.isNotEmpty())
    assertEquals(
        "resetContext() after a prefillPrompt() must fully clear KV state.",
        cleanRun,
        postResetRun,
    )
  }

  private fun generateAndCollect(prompt: String): List<String> {
    val collector = CollectingCallback()
    llmModule.generate(prompt, SEQ_LEN, collector)
    return collector.tokens()
  }

  private class CollectingCallback : LlmCallback {
    private val tokens: MutableList<String> = ArrayList()

    override fun onResult(result: String) {
      tokens.add(result)
    }

    override fun onStats(stats: String) = Unit

    fun tokens(): List<String> = tokens.toList()
  }

  companion object {
    private const val TEST_FILE_NAME = "/stories.pte"
    private const val TOKENIZER_FILE_NAME = "/tokenizer.bin"

    /** Short prompt; SEQ_LEN kept small to keep the test fast on CI emulators/devices. */
    private const val PROMPT_A = "Once"
    private const val PREFILL_HISTORY = "Long ago, in a small village by the sea, "
    private const val SEQ_LEN = 24
  }
}
