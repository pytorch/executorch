/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
package org.pytorch.executorch

import android.os.Bundle
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import java.io.File
import java.io.IOException
import java.util.Collections
import org.apache.commons.io.FileUtils
import org.json.JSONException
import org.json.JSONObject
import org.junit.After
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.pytorch.executorch.TestFileUtils.getTestFilePath
import org.pytorch.executorch.extension.llm.LlmCallback
import org.pytorch.executorch.extension.llm.LlmModule

/**
 * Performance regression tests for LLM inference on ExecuTorch Android.
 *
 * Measures tokens-per-second (TPS), TPS stability, and time-to-first-token (TTFT). Results are
 * reported via [InstrumentationRegistry] so CI systems can capture and trend metrics over time.
 *
 * Uses the same TinyStories-110M fixture as [LlmModuleConversationHistoryTest], so no additional
 * test infrastructure is needed. Works on both OSS (GitHub Actions) and internal (Sandcastle) CI.
 *
 * To run locally:
 * ```
 * ./gradlew :executorch_android:connectedAndroidTest \
 *     -Pandroid.testInstrumentationRunnerArguments.class=org.pytorch.executorch.LlmPerformanceTest
 * ```
 *
 * To override the TPS threshold for physical devices:
 * ```
 * -Pandroid.testInstrumentationRunnerArguments.minTps=10.0
 * ```
 */
@RunWith(AndroidJUnit4::class)
class LlmPerformanceTest : LlmCallback {

  private lateinit var llmModule: LlmModule
  private val generatedTokens: MutableList<String> =
      Collections.synchronizedList(mutableListOf<String>())
  private val tpsResults: MutableList<Float> = Collections.synchronizedList(mutableListOf<Float>())
  @Volatile private var lastStatsJson: String? = null

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
  }

  @After
  fun tearDown() {
    if (::llmModule.isInitialized) {
      llmModule.close()
    }
  }

  /**
   * Measures TPS after a warm-up run and asserts it exceeds a minimum threshold.
   *
   * The warm-up is necessary because the first inference includes one-time costs (memory
   * allocation, kernel compilation on some backends) that would unfairly penalize the measurement.
   *
   * Default threshold is conservative (1.0 TPS) for emulator CI. Override with the `minTps`
   * instrumentation argument for physical device runs where 10-30+ TPS is expected.
   */
  @Test(timeout = MAX_TEST_TIMEOUT_MS)
  fun testTpsAboveThreshold() {
    llmModule.load()

    // Warm-up: first inference includes one-time overhead
    resetState()
    llmModule.generate(TEST_PROMPT, SEQ_LEN, this)
    assertTrue("Warm-up produced no tokens — model may be broken", generatedTokens.isNotEmpty())
    val warmupTps = tpsResults.lastOrNull() ?: 0f
    reportMetric("warmup_tps", warmupTps)

    // Measured run
    resetState()
    llmModule.generate(TEST_PROMPT, SEQ_LEN, this)

    assertTrue("Measured run produced no tokens", generatedTokens.isNotEmpty())
    assertTrue("No TPS stats received from onStats callback", tpsResults.isNotEmpty())

    val measuredTps = tpsResults.last()
    val minTps = getMinTpsThreshold()
    val statsTokenCount =
        try {
          JSONObject(lastStatsJson!!).getInt("generated_tokens")
        } catch (_: Exception) {
          -1
        }

    reportMetric("measured_tps", measuredTps)
    reportMetric("measured_tokens", statsTokenCount.toFloat())
    reportMetric("min_tps_threshold", minTps)

    assertTrue(
        "TPS regression detected! measured=${"%.2f".format(measuredTps)} " +
            "< threshold=${"%.2f".format(minTps)}. Raw stats: $lastStatsJson",
        measuredTps >= minTps,
    )
  }

  /**
   * Validates that TPS is stable across multiple consecutive runs.
   *
   * Large variance in TPS (high coefficient of variation) may indicate thread contention, GC
   * pressure, thermal throttling, or non-deterministic scheduling — all of which degrade the user
   * experience even if average TPS is acceptable.
   */
  @Test(timeout = MAX_TEST_TIMEOUT_MS)
  fun testTpsStability() {
    llmModule.load()

    // Warm-up
    resetState()
    llmModule.generate(TEST_PROMPT, SEQ_LEN, this)

    // Collect TPS over multiple runs
    val measurements = mutableListOf<Float>()
    for (i in 1..STABILITY_ITERATIONS) {
      resetState()
      llmModule.generate(TEST_PROMPT, SEQ_LEN, this)
      if (tpsResults.isNotEmpty()) {
        measurements.add(tpsResults.last())
      }
    }

    assertTrue(
        "Not enough TPS measurements (${measurements.size}/$STABILITY_ITERATIONS)",
        measurements.size >= STABILITY_ITERATIONS,
    )

    val mean = measurements.average().toFloat()
    val variance = measurements.map { (it - mean) * (it - mean) }.average().toFloat()
    val stddev = Math.sqrt(variance.toDouble()).toFloat()
    val cv = if (mean > 0f) stddev / mean else Float.MAX_VALUE

    reportMetric("stability_mean_tps", mean)
    reportMetric("stability_stddev", stddev)
    reportMetric("stability_cv", cv)
    reportMetric("stability_min", measurements.minOrNull()!!)
    reportMetric("stability_max", measurements.maxOrNull()!!)

    assertTrue(
        "TPS too unstable! CV=${"%.3f".format(cv)} exceeds max $MAX_CV. " +
            "Measurements: $measurements",
        cv <= MAX_CV,
    )
  }

  /**
   * Measures time-to-first-token (TTFT) — the delay from calling generate() until the first token
   * is produced (i.e., prompt evaluation / prefill time).
   *
   * High TTFT directly impacts perceived responsiveness: the user types a message and sees nothing
   * happen until prefill completes.
   */
  @Test(timeout = MAX_TEST_TIMEOUT_MS)
  fun testTimeToFirstToken() {
    llmModule.load()

    // Warm-up
    resetState()
    llmModule.generate(TEST_PROMPT, SEQ_LEN, this)

    // Measured TTFT
    resetState()
    llmModule.generate(TEST_PROMPT, SEQ_LEN, this)

    val statsJson = lastStatsJson
    assertTrue("No stats JSON received from onStats callback", statsJson != null)

    try {
      val json = JSONObject(statsJson!!)
      val inferenceStartMs = json.getLong("inference_start_ms")
      val firstTokenMs = json.getLong("first_token_ms")
      val ttftMs = firstTokenMs - inferenceStartMs

      reportMetric("ttft_ms", ttftMs.toFloat())

      assertTrue(
          "TTFT too slow: ${ttftMs}ms exceeds max ${MAX_TTFT_MS}ms. " +
              "First token latency is too high.",
          ttftMs <= MAX_TTFT_MS,
      )
    } catch (e: JSONException) {
      fail("Failed to parse onStats JSON for TTFT: $statsJson. Error: ${e.message}")
    }
  }

  // ─── LlmCallback ──────────────────────────────────────────────────────────────────

  override fun onResult(result: String) {
    generatedTokens.add(result)
  }

  override fun onStats(stats: String) {
    lastStatsJson = stats
    try {
      val json = JSONObject(stats)
      val numTokens = json.getInt("generated_tokens")
      val inferenceEndMs = json.getLong("inference_end_ms")
      val promptEvalEndMs = json.getLong("prompt_eval_end_ms")
      val decodeTimeMs = inferenceEndMs - promptEvalEndMs
      if (decodeTimeMs > 0) {
        tpsResults.add(numTokens.toFloat() / decodeTimeMs.toFloat() * 1000f)
      }
    } catch (_: JSONException) {
      // Parsing failure — test will fail on assertion
    }
  }

  // ─── Helpers ─────────────────────────────────────────────────────────────────────

  private fun resetState() {
    generatedTokens.clear()
    tpsResults.clear()
    lastStatsJson = null
    llmModule.resetContext()
  }

  /**
   * Returns the minimum TPS threshold. Overridable via instrumentation arg `minTps` so the same
   * test binary can gate at different levels for emulator vs physical device CI.
   */
  private fun getMinTpsThreshold(): Float {
    val override =
        InstrumentationRegistry.getArguments().getString("minTps") ?: return DEFAULT_MIN_TPS
    val parsed = override.toFloatOrNull()
    require(parsed != null && parsed.isFinite() && parsed > 0f) {
      "Invalid instrumentation arg minTps='$override'. Expected a finite, positive float."
    }
    return parsed
  }

  private fun reportMetric(key: String, value: Float) {
    val bundle = Bundle().apply { putFloat(key, value) }
    InstrumentationRegistry.getInstrumentation().sendStatus(0, bundle)
  }

  companion object {
    private const val TEST_FILE_NAME = "/stories.pte"
    private const val TOKENIZER_FILE_NAME = "/tokenizer.bin"

    /** Prompt for inference. Kept short to minimize test wall-time. */
    private const val TEST_PROMPT = "Once upon a time"
    private const val SEQ_LEN = 64

    /**
     * Minimum TPS for the test to pass. Conservative for x86_64 emulator (API 34). For physical
     * devices, override via: -Pandroid.testInstrumentationRunnerArguments.minTps=10.0
     */
    private const val DEFAULT_MIN_TPS = 1.0f

    /** Maximum time-to-first-token in milliseconds. 30s is generous for emulator. */
    private const val MAX_TTFT_MS = 30_000

    /**
     * Maximum coefficient of variation (stddev/mean) for TPS across runs. 0.5 = up to 50% relative
     * variance, which is generous for noisy emulator environments. Tighten for dedicated devices.
     */
    private const val MAX_CV = 0.5f

    /** Number of runs for the stability test. */
    private const val STABILITY_ITERATIONS = 3

    /** Per-test timeout: 5 minutes to accommodate slow emulator environments. */
    private const val MAX_TEST_TIMEOUT_MS = 300_000L
  }
}
