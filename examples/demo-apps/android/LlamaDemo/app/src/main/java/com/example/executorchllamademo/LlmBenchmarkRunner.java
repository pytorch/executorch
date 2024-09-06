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
import java.util.Arrays;

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

    // TODO (huydhn): Remove txt files here once the JSON format is ready
    try (FileWriter writer = new FileWriter(getFilesDir() + "/benchmark_results.txt")) {
      writer.write(mStatsDump.toString());
    } catch (IOException e) {
      e.printStackTrace();
    }

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

class StatsDump {
  long loadStart;
  long loadEnd;
  long generateStart;
  long generateEnd;
  String tokens;

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
