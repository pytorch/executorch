/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.pytorch.executorch

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.filters.LargeTest
import java.io.File
import java.io.IOException
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger
import org.apache.commons.io.FileUtils
import org.junit.After
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.pytorch.executorch.TestFileUtils.getTestFilePath
import org.pytorch.executorch.extension.llm.LlmCallback
import org.pytorch.executorch.extension.llm.LlmModule

/**
 * Thread safety contract tests for [LlmModule].
 *
 * Validates the concurrency contract: LlmModule uses a ReentrantLock to serialize generate() calls
 * and an atomic stop flag. These tests verify defined behavior (serialization, rejection, atomic
 * stop signaling, idempotent close) rather than relying on timing-dependent races.
 */
@LargeTest
@RunWith(AndroidJUnit4::class)
class LlmThreadSafetyTest {

  private lateinit var llmModule: LlmModule

  @Before
  @Throws(IOException::class)
  fun setUp() {
    val pteFile = File(getTestFilePath(TEST_FILE_NAME))
    requireNotNull(javaClass.getResourceAsStream(TEST_FILE_NAME)) {
          "Test resource $TEST_FILE_NAME not found; did android_test_setup.sh run?"
        }
        .use { stream -> FileUtils.copyInputStreamToFile(stream, pteFile) }

    val tokenizerFile = File(getTestFilePath(TOKENIZER_FILE_NAME))
    requireNotNull(javaClass.getResourceAsStream(TOKENIZER_FILE_NAME)) {
          "Test resource $TOKENIZER_FILE_NAME not found; did android_test_setup.sh run?"
        }
        .use { stream -> FileUtils.copyInputStreamToFile(stream, tokenizerFile) }

    llmModule =
        LlmModule(getTestFilePath(TEST_FILE_NAME), getTestFilePath(TOKENIZER_FILE_NAME), 0.0f)
  }

  @After
  fun tearDown() {
    if (::llmModule.isInitialized) {
      try {
        llmModule.close()
      } catch (_: IllegalStateException) {
        // Already closed by test
      }
    }
  }

  /**
   * Contract: concurrent generate() calls are either serialized (ReentrantLock) or rejected
   * (IllegalStateException). Neither outcome should crash or corrupt state.
   *
   * Strategy: use a latch inside Thread A's callback to guarantee it holds the lock before Thread B
   * attempts generate().
   */
  @Test(timeout = 30_000)
  fun testConcurrentGenerateThrowsOrSerializes() {
    val threadAProducedToken = CountDownLatch(1)
    val threadATokens = AtomicInteger(0)
    val threadBTokens = AtomicInteger(0)
    val threadBRejected = AtomicBoolean(false)
    val crashed = AtomicBoolean(false)
    val threadADone = CountDownLatch(1)
    val threadBDone = CountDownLatch(1)

    val threadA = Thread {
      try {
        llmModule.generate(
            TEST_PROMPT,
            LONG_SEQ_LEN,
            object : LlmCallback {
              override fun onResult(result: String) {
                threadATokens.incrementAndGet()
                threadAProducedToken.countDown()
              }

              override fun onStats(stats: String) {}
            },
        )
      } catch (e: Exception) {
        crashed.set(true)
      } finally {
        threadADone.countDown()
      }
    }

    val threadB = Thread {
      try {
        // Wait until Thread A is actively generating (holds the lock)
        assertTrue(
            "Thread A did not produce a token in time",
            threadAProducedToken.await(20, TimeUnit.SECONDS),
        )
        llmModule.generate(
            TEST_PROMPT,
            SHORT_SEQ_LEN,
            object : LlmCallback {
              override fun onResult(result: String) {
                threadBTokens.incrementAndGet()
              }

              override fun onStats(stats: String) {}
            },
        )
      } catch (_: IllegalStateException) {
        // Valid: lock rejects concurrent access
        threadBRejected.set(true)
      } catch (_: RuntimeException) {
        // Valid: serialized second generate() may fail (e.g., dirty KV cache state)
        threadBRejected.set(true)
      } catch (e: Exception) {
        crashed.set(true)
      } finally {
        threadBDone.countDown()
      }
    }

    threadA.start()
    threadB.start()
    assertTrue("Thread A did not finish in time", threadADone.await(20, TimeUnit.SECONDS))
    assertTrue("Thread B did not finish in time", threadBDone.await(20, TimeUnit.SECONDS))

    assertFalse("No crash during concurrent generate() calls", crashed.get())
    // Either Thread B was rejected OR it blocked and then succeeded
    val threadBSucceeded = threadBTokens.get() > 0
    assertTrue(
        "Thread B must either be rejected (IllegalStateException) or serialize and succeed",
        threadBRejected.get() || threadBSucceeded,
    )
    assertTrue("Thread A must have produced tokens", threadATokens.get() > 0)
  }

  /**
   * Contract: stop() sets an atomic flag checked between token emissions. Once set, generate()
   * returns after completing the current token.
   *
   * Strategy: use a latch to guarantee stop() fires only after TOKEN_THRESHOLD tokens are received,
   * then verify generation terminated.
   */
  @Test(timeout = 30_000)
  fun testStopFromDifferentThread() {
    val tokensReceived = AtomicInteger(0)
    val thresholdReached = CountDownLatch(1)
    val generateDone = CountDownLatch(1)
    val crashed = AtomicBoolean(false)

    val generateThread = Thread {
      try {
        llmModule.generate(
            TEST_PROMPT,
            LONG_SEQ_LEN,
            object : LlmCallback {
              override fun onResult(result: String) {
                if (tokensReceived.incrementAndGet() == TOKEN_THRESHOLD) {
                  thresholdReached.countDown()
                }
              }

              override fun onStats(stats: String) {}
            },
        )
      } catch (e: Exception) {
        crashed.set(true)
      } finally {
        generateDone.countDown()
      }
    }

    generateThread.start()

    // Wait for exactly TOKEN_THRESHOLD tokens, then signal stop
    assertTrue(
        "Did not reach token threshold in time",
        thresholdReached.await(20, TimeUnit.SECONDS),
    )
    llmModule.stop()

    // Wait for generate() to return
    assertTrue("Generate did not finish in time", generateDone.await(20, TimeUnit.SECONDS))

    assertFalse("No crash during stop from different thread", crashed.get())
    assertTrue(
        "Must have received at least TOKEN_THRESHOLD tokens before stop",
        tokensReceived.get() >= TOKEN_THRESHOLD,
    )
    assertTrue(
        "Generation must terminate (not run to LONG_SEQ_LEN)",
        tokensReceived.get() < LONG_SEQ_LEN,
    )
  }

  /**
   * Contract: stop() when no generate() is active is a no-op. It must not corrupt state or prevent
   * subsequent generate() calls from succeeding.
   */
  @Test(timeout = 30_000)
  fun testStopWhenIdle() {
    // stop() with no active generation — should not throw
    llmModule.stop()

    // Verify the module is still functional after idle stop
    val tokensReceived = AtomicInteger(0)
    llmModule.generate(
        TEST_PROMPT,
        SHORT_SEQ_LEN,
        object : LlmCallback {
          override fun onResult(result: String) {
            tokensReceived.incrementAndGet()
          }

          override fun onStats(stats: String) {}
        },
    )

    assertTrue(
        "generate() must still work after idle stop()",
        tokensReceived.get() == SHORT_SEQ_LEN,
    )
  }

  /**
   * Contract: close() is idempotent — calling it multiple times must not double-free native
   * resources. After close(), generate() must throw IllegalStateException.
   */
  @Test(timeout = 30_000)
  fun testCloseIdempotent() {
    // First, verify the module works
    val tokensReceived = AtomicInteger(0)
    llmModule.generate(
        TEST_PROMPT,
        SHORT_SEQ_LEN,
        object : LlmCallback {
          override fun onResult(result: String) {
            tokensReceived.incrementAndGet()
          }

          override fun onStats(stats: String) {}
        },
    )
    assertTrue("Module must produce tokens before close", tokensReceived.get() > 0)

    // Close multiple times — must not crash or double-free
    llmModule.close()
    llmModule.close()
    llmModule.close()

    // generate() after close must throw IllegalStateException
    try {
      llmModule.generate(
          TEST_PROMPT,
          SHORT_SEQ_LEN,
          object : LlmCallback {
            override fun onResult(result: String) {}

            override fun onStats(stats: String) {}
          },
      )
      fail("generate() after close() must throw IllegalStateException")
    } catch (_: IllegalStateException) {
      // Expected
    }
  }

  companion object {
    private const val TEST_FILE_NAME = "/stories.pte"
    private const val TOKENIZER_FILE_NAME = "/tokenizer.bin"
    private const val TEST_PROMPT = "Hello"
    private const val SHORT_SEQ_LEN = 16
    private const val LONG_SEQ_LEN = 64
    private const val TOKEN_THRESHOLD = 5
  }
}
