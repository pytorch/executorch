/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.pytorch.minibench

import android.util.Log
import org.json.JSONException
import org.json.JSONObject

class LlmBenchmark(
    private val activity: BenchmarkActivity,
    modelFile: String,
    tokenizerPath: String,
    private val prompt: String,
    temperature: Float,
    private val results: MutableList<BenchmarkMetric>,
) : LlmModelRunnerCallback {

  private val runner: LlmModelRunner
  private val statsInfo = StatsInfo()

  init {
    statsInfo.modelName = modelFile.substringAfterLast('/').removeSuffix(".pte")
    runner = LlmModelRunner(modelFile, tokenizerPath, temperature, this)
    statsInfo.loadStart = System.nanoTime()
  }

  override fun onModelLoaded(status: Int) {
    statsInfo.loadEnd = System.nanoTime()
    statsInfo.loadStatus = status
    if (status != 0) {
      Log.e("LlmBenchmarkRunner", "Loaded failed: $status")
      onGenerationStopped()
      return
    }
    statsInfo.generateStart = System.nanoTime()
    runner.generate(prompt)
  }

  override fun onTokenGenerated(token: String) {}

  override fun onStats(stats: String) {
    try {
      val json = JSONObject(stats)
      val numGeneratedTokens = json.getInt("generated_tokens")
      val inferenceEndMs = json.getInt("inference_end_ms")
      val promptEvalEndMs = json.getInt("prompt_eval_end_ms")
      statsInfo.tps = numGeneratedTokens.toFloat() / (inferenceEndMs - promptEvalEndMs) * 1000
    } catch (e: JSONException) {
      Log.e("LLM", "Error parsing JSON: ${e.message}")
    }
  }

  override fun onGenerationStopped() {
    statsInfo.generateEnd = System.nanoTime()

    val benchmarkModel = BenchmarkMetric.extractBackendAndQuantization(statsInfo.modelName)
    results.add(BenchmarkMetric(benchmarkModel, "load_status", statsInfo.loadStatus.toDouble(), 0.0))
    results.add(
        BenchmarkMetric(
            benchmarkModel,
            "llm_model_load_time(ms)",
            (statsInfo.loadEnd - statsInfo.loadStart) * 1e-6,
            0.0,
        ))
    results.add(
        BenchmarkMetric(
            benchmarkModel,
            "generate_time(ms)",
            (statsInfo.generateEnd - statsInfo.generateStart) * 1e-6,
            0.0,
        ))
    results.add(BenchmarkMetric(benchmarkModel, "token_per_sec", statsInfo.tps.toDouble(), 0.0))
    activity.writeResult()
  }
}

private class StatsInfo {
  var loadStatus: Int = 0
  var loadStart: Long = 0
  var loadEnd: Long = 0
  var generateStart: Long = 0
  var generateEnd: Long = 0
  var tps: Float = 0f
  var modelName: String = ""
}
