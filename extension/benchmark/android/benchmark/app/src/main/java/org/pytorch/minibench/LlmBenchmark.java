/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.pytorch.minibench;

import android.util.Log;
import java.util.List;
import org.json.JSONException;
import org.json.JSONObject;

public class LlmBenchmark implements LlmModelRunnerCallback {
  LlmModelRunner mLlmModelRunner;

  String mPrompt;
  StatsInfo mStatsInfo;

  List<BenchmarkMetric> mResults;
  BenchmarkActivity mActivity;

  LlmBenchmark(
      BenchmarkActivity activity,
      String modelFile,
      String tokenizerPath,
      String prompt,
      float temperature,
      List<BenchmarkMetric> results) {
    mResults = results;
    mActivity = activity;
    mStatsInfo = new StatsInfo();
    mStatsInfo.modelName = modelFile.substring(modelFile.lastIndexOf('/') + 1).replace(".pte", "");
    mPrompt = prompt;
    mLlmModelRunner = new LlmModelRunner(modelFile, tokenizerPath, temperature, this);
    mStatsInfo.loadStart = System.nanoTime();
  }

  @Override
  public void onModelLoaded(int status) {
    mStatsInfo.loadEnd = System.nanoTime();
    mStatsInfo.loadStatus = status;
    if (status != 0) {
      Log.e("LlmBenchmarkRunner", "Loaded failed: " + status);
      onGenerationStopped();
      return;
    }
    mStatsInfo.generateStart = System.nanoTime();
    mLlmModelRunner.generate(mPrompt);
  }

  @Override
  public void onTokenGenerated(String token) {}

  @Override
  public void onStats(String stats) {
    float tps = 0;
    try {
      JSONObject jsonObject = new JSONObject(stats);
      int numGeneratedTokens = jsonObject.getInt("generated_tokens");
      int inferenceEndMs = jsonObject.getInt("inference_end_ms");
      int promptEvalEndMs = jsonObject.getInt("prompt_eval_end_ms");
      tps = (float) numGeneratedTokens / (inferenceEndMs - promptEvalEndMs) * 1000;
      mStatsInfo.tps = tps;
    } catch (JSONException e) {
      Log.e("LLM", "Error parsing JSON: " + e.getMessage());
    }
  }

  @Override
  public void onGenerationStopped() {
    mStatsInfo.generateEnd = System.nanoTime();

    final BenchmarkMetric.BenchmarkModel benchmarkModel =
        BenchmarkMetric.extractBackendAndQuantization(mStatsInfo.modelName);
    // The list of metrics we have atm includes:
    // Load status
    mResults.add(new BenchmarkMetric(benchmarkModel, "load_status", mStatsInfo.loadStatus, 0));
    // Model load time
    mResults.add(
        new BenchmarkMetric(
            benchmarkModel,
            "llm_model_load_time(ms)",
            (mStatsInfo.loadEnd - mStatsInfo.loadStart) * 1e-6,
            0.0f));
    // LLM generate time
    mResults.add(
        new BenchmarkMetric(
            benchmarkModel,
            "generate_time(ms)",
            (mStatsInfo.generateEnd - mStatsInfo.generateStart) * 1e-6,
            0.0f));
    // Token per second
    mResults.add(new BenchmarkMetric(benchmarkModel, "token_per_sec", mStatsInfo.tps, 0.0f));
    mActivity.writeResult();
  }
}

class StatsInfo {
  int loadStatus;
  long loadStart;
  long loadEnd;
  long generateStart;
  long generateEnd;
  float tps;
  String modelName;

  @Override
  public String toString() {
    return "loadStart: "
        + loadStart
        + "\nloadEnd: "
        + loadEnd
        + "\ngenerateStart: "
        + generateStart
        + "\ngenerateEnd: "
        + generateEnd
        + "\n"
        + tps;
  }
}
