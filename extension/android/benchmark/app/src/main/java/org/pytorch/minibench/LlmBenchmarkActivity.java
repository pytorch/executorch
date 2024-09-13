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
import java.util.Arrays;

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
    mModelRunner = new ModelRunner(model.getPath(), tokenizerPath, temperature, this);
    mStatsInfo.loadStart = System.currentTimeMillis();
  }

  @Override
  public void onModelLoaded(int status) {
    mStatsInfo.loadEnd = System.currentTimeMillis();
    if (status != 0) {
      Log.e("LlmBenchmarkRunner", "Loaded failed: " + status);
      onGenerationStopped();
      return;
    }
    mStatsInfo.generateStart = System.currentTimeMillis();
    mModelRunner.generate(mPrompt);
  }

  @Override
  public void onTokenGenerated(String token) {}

  @Override
  public void onStats(String stats) {
    mStatsInfo.tokens = stats;
  }

  @Override
  public void onGenerationStopped() {
    mStatsInfo.generateEnd = System.currentTimeMillis();

    // TODO (huydhn): Remove txt files here once the JSON format is ready
    try (FileWriter writer = new FileWriter(getFilesDir() + "/benchmark_results.txt")) {
      writer.write(mStatsInfo.toString());
    } catch (IOException e) {
      e.printStackTrace();
    }

    // TODO (huydhn): Figure out on what the final JSON results looks like, we need something
    // with the same number of fields as https://github.com/pytorch/pytorch/pull/135042
    try (FileWriter writer = new FileWriter(getFilesDir() + "/benchmark_results.json")) {
      Gson gson = new Gson();
      writer.write(gson.toJson(mStatsInfo));
    } catch (IOException e) {
      e.printStackTrace();
    }
  }
}

class StatsInfo {
  long loadStart;
  long loadEnd;
  long generateStart;
  long generateEnd;
  String tokens;

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
        + tokens;
  }
}
