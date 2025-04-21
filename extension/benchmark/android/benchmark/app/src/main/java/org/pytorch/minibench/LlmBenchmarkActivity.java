/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.pytorch.minibench;

import android.app.Activity;
import android.content.Intent;
import android.os.Bundle;
import android.system.ErrnoException;
import android.system.Os;
import android.util.Log;
import com.google.gson.Gson;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.json.JSONException;
import org.json.JSONObject;

public class LlmBenchmarkActivity extends Activity implements ModelRunnerCallback {
  ModelRunner mModelRunner;

  String mPrompt;
  StatsInfo mStatsInfo;

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);

    Intent intent = getIntent();

    File modelDir = new File(intent.getStringExtra("model_dir"));
    File model =
        Arrays.stream(modelDir.listFiles())
            .filter(file -> file.getName().endsWith(".pte"))
            .findFirst()
            .get();
    String tokenizerPath = intent.getStringExtra("tokenizer_path");

    float temperature = intent.getFloatExtra("temperature", 0.8f);
    mPrompt = intent.getStringExtra("prompt");
    if (mPrompt == null) {
      mPrompt = "The ultimate answer";
    }

    try {
      Os.setenv("ADSP_LIBRARY_PATH", getApplicationInfo().nativeLibraryDir, true);
    } catch (ErrnoException e) {
      finish();
    }

    mStatsInfo = new StatsInfo();
    mStatsInfo.modelName = model.getName().replace(".pte", "");
    mModelRunner = new ModelRunner(model.getPath(), tokenizerPath, temperature, this);
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
    mModelRunner.generate(mPrompt);
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
    final List<BenchmarkMetric> results = new ArrayList<>();
    // The list of metrics we have atm includes:
    // Load status
    results.add(new BenchmarkMetric(benchmarkModel, "load_status", mStatsInfo.loadStatus, 0));
    // Model load time
    results.add(
        new BenchmarkMetric(
            benchmarkModel,
            "model_load_time(ms)",
            (mStatsInfo.loadEnd - mStatsInfo.loadStart) * 1e-6,
            0.0f));
    // LLM generate time
    results.add(
        new BenchmarkMetric(
            benchmarkModel,
            "generate_time(ms)",
            (mStatsInfo.generateEnd - mStatsInfo.generateStart) * 1e-6,
            0.0f));
    // Token per second
    results.add(new BenchmarkMetric(benchmarkModel, "token_per_sec", mStatsInfo.tps, 0.0f));

    try (FileWriter writer = new FileWriter(getFilesDir() + "/benchmark_results.json")) {
      Gson gson = new Gson();
      writer.write(gson.toJson(results));
    } catch (IOException e) {
      e.printStackTrace();
    }
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
