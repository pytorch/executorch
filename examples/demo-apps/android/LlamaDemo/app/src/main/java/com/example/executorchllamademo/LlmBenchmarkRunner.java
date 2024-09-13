/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

package com.example.executorchllamademo;

import android.app.Activity;
import android.app.ActivityManager;
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
    mStatsDump.modelName = model.getName().replace(".pte", "");
    mModelRunner = new ModelRunner(model.getPath(), tokenizerPath, temperature, this);
    mStatsDump.loadStart = System.nanoTime();
  }

  @Override
  public void onModelLoaded(int status) {
    mStatsDump.loadEnd = System.nanoTime();
    mStatsDump.loadStatus = status;
    if (status != 0) {
      Log.e("LlmBenchmarkRunner", "Loaded failed: " + status);
      onGenerationStopped();
      return;
    }
    mStatsDump.generateStart = System.nanoTime();
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
    mStatsDump.generateEnd = System.nanoTime();
    runOnUiThread(
        () -> {
          mTextView.append(mStatsDump.toString());
        });

    final BenchmarkMetric.BenchmarkModel benchmarkModel =
        BenchmarkMetric.extractBackendAndQuantization(mStatsDump.modelName);
    final List<BenchmarkMetric> results = new ArrayList<>();
    // The list of metrics we have atm includes:
    // Load status
    results.add(new BenchmarkMetric(benchmarkModel, "load_status", mStatsDump.loadStatus, 0));
    // Model load time
    results.add(
        new BenchmarkMetric(
            benchmarkModel,
            "model_load_time(ms)",
            (mStatsDump.loadEnd - mStatsDump.loadStart) * 1e-6,
            0.0f));
    // LLM generate time
    results.add(
        new BenchmarkMetric(
            benchmarkModel,
            "generate_time(ms)",
            (mStatsDump.generateEnd - mStatsDump.generateStart) * 1e-6,
            0.0f));
    // Token per second
    results.add(
        new BenchmarkMetric(benchmarkModel, "token_per_sec", extractTPS(mStatsDump.tokens), 0.0f));

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
  public static class BenchmarkModel {
    // The model name, i.e. stories110M
    String name;
    String backend;
    String quantization;

    public BenchmarkModel(final String name, final String backend, final String quantization) {
      this.name = name;
      this.backend = backend;
      this.quantization = quantization;
    }
  }

  BenchmarkModel benchmarkModel;

  // The metric name, i.e. TPS
  String metric;

  // The actual value and the option target value
  double actualValue;
  double targetValue;

  public static class DeviceInfo {
    // Let's see which information we want to include here
    final String device = Build.BRAND;
    // The phone model and Android release version
    final String arch = Build.MODEL;
    final String os = "Android " + Build.VERSION.RELEASE;
    final long totalMem = new ActivityManager.MemoryInfo().totalMem;
    final long availMem = new ActivityManager.MemoryInfo().availMem;
  }

  DeviceInfo deviceInfo = new DeviceInfo();

  public BenchmarkMetric(
      final BenchmarkModel benchmarkModel,
      final String metric,
      final double actualValue,
      final double targetValue) {
    this.benchmarkModel = benchmarkModel;
    this.metric = metric;
    this.actualValue = actualValue;
    this.targetValue = targetValue;
  }

  // TODO (huydhn): Figure out a way to extract the backend and quantization information from
  // the .pte model itself instead of parsing its name
  public static BenchmarkMetric.BenchmarkModel extractBackendAndQuantization(final String model) {
    final Matcher m =
        Pattern.compile("(?<name>\\w+)_(?<backend>\\w+)_(?<quantization>\\w+)").matcher(model);
    if (m.matches()) {
      return new BenchmarkMetric.BenchmarkModel(
          m.group("name"), m.group("backend"), m.group("quantization"));
    } else {
      return new BenchmarkMetric.BenchmarkModel(model, "", "");
    }
  }
}

class StatsDump {
  int loadStatus;
  long loadStart;
  long loadEnd;
  long generateStart;
  long generateEnd;
  String tokens;
  String modelName;

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
