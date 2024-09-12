/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

package com.example.executorchllamademo;

import android.app.Activity;
import android.content.Intent;
import android.os.Build;
import android.os.Bundle;
import android.util.Log;
import android.widget.TextView;
import androidx.annotation.NonNull;
import com.google.gson.Gson;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class LlmBenchmarkRunner extends Activity implements ModelRunnerCallback {
  ModelRunner mModelRunner;

  String mPrompt;
  TextView mTextView;
  StatsDump mStatsDump;

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    setContentView(R.layout.activity_benchmarking);
    mTextView = findViewById(R.id.log_view);

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

    mStatsDump = new StatsDump();
    mStatsDump.name = model.getName().replace(".pte", "");
    mModelRunner = new ModelRunner(model.getPath(), tokenizerPath, temperature, this);
    mStatsDump.loadStart = System.currentTimeMillis();
  }

  @Override
  public void onModelLoaded(int status) {
    mStatsDump.loadEnd = System.currentTimeMillis();
    if (status != 0) {
      Log.e("LlmBenchmarkRunner", "Loaded failed: " + status);
      onGenerationStopped();
      return;
    }
    mStatsDump.generateStart = System.currentTimeMillis();
    mModelRunner.generate(mPrompt);
  }

  @Override
  public void onTokenGenerated(String token) {
    runOnUiThread(
        () -> {
          mTextView.append(token);
        });
  }

  @Override
  public void onStats(String stats) {
    mStatsDump.tokens = stats;
  }

  @Override
  public void onGenerationStopped() {
    mStatsDump.generateEnd = System.currentTimeMillis();
    runOnUiThread(
        () -> {
          mTextView.append(mStatsDump.toString());
        });

    List<BenchmarkMetric> results = new ArrayList<>();
    // The list of metrics we have atm includes:
    // - Model load time
    BenchmarkMetric modelLoadTime = new BenchmarkMetric();
    modelLoadTime.name = mStatsDump.name;
    modelLoadTime.metric = "model_load_time(ms)";
    modelLoadTime.actual = mStatsDump.loadEnd - mStatsDump.loadStart;
    results.add(modelLoadTime);

    // - LLM generate time
    BenchmarkMetric generateTime = new BenchmarkMetric();
    generateTime.name = mStatsDump.name;
    generateTime.metric = "generate_time(ms)";
    generateTime.actual = mStatsDump.generateEnd - mStatsDump.generateStart

    // - Token per second


    // TODO (huydhn): Figure out on what the final JSON results looks like, we need something
    // with the same number of fields as https://github.com/pytorch/pytorch/pull/135042
    try (FileWriter writer = new FileWriter(getFilesDir() + "/benchmark_results.json")) {
      Gson gson = new Gson();
      writer.write(gson.toJson(mStatsDump));
    } catch (IOException e) {
      e.printStackTrace();
    }
  }
}

class BenchmarkMetric {
  // The model name, i.e. stories110M
  String name;

  // The metric name, i.e. TPS
  String metric;

  // The actual value and the option target value
  float actual;
  float target = 0.0f;

  // TODO (huydhn): Is there a way to get this information from the export model itself?
  String dtype = "float32";

  // Let's see which information we want to include here
  String device = android.os.Build.BRAND;
  // DEBUG DEBUG
  String arch = android.os.Build.DEVICE + " / " + android.os.Build.MODEL;
}
class StatsDump {
  long loadStart;
  long loadEnd;
  long generateStart;
  long generateEnd;
  String tokens;
  String name;

  @NonNull
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
