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
import java.util.regex.Matcher;
import java.util.regex.Pattern;

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

    final List<BenchmarkMetric> results = new ArrayList<>();
    // The list of metrics we have atm includes:
    // Model load time
    results.add(
        new BenchmarkMetric(
            mStatsDump.name,
            "model_load_time(ms)",
            mStatsDump.loadEnd - mStatsDump.loadStart,
            0.0f));
    // LLM generate time
    results.add(
        new BenchmarkMetric(
            mStatsDump.name,
            "generate_time(ms)",
            mStatsDump.generateEnd - mStatsDump.generateStart,
            0.0f));
    // Token per second
    results.add(
        new BenchmarkMetric(mStatsDump.name, "token_per_sec", extractTPS(mStatsDump.tokens), 0.0f));

    try (FileWriter writer = new FileWriter(getFilesDir() + "/benchmark_results.json")) {
      Gson gson = new Gson();
      writer.write(gson.toJson(results));
    } catch (IOException e) {
      e.printStackTrace();
    }
  }

  private double extractTPS(final String tokens) {
    final Matcher m = Pattern.compile("\\d+\\.?\\d*").matcher(tokens);
    if (m.find()) {
      return Double.parseDouble(m.group());
    } else {
      return 0.0f;
    }
  }
}

class BenchmarkMetric {
  // The model name, i.e. stories110M
  String name;

  // The metric name, i.e. TPS
  String metric;

  // The actual value and the option target value
  double actual;
  double target;

  // TODO (huydhn): Is there a way to get this information from the export model itself?
  final String dtype = "float32";

  // Let's see which information we want to include here
  final String device = android.os.Build.BRAND;
  // DEBUG DEBUG
  final String arch = android.os.Build.DEVICE + " / " + android.os.Build.MODEL;

  public BenchmarkMetric(
      final String name, final String metric, final double actual, final double target) {
    this.name = name;
    this.metric = metric;
    this.actual = actual;
    this.target = target;
  }
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
