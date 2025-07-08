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
import android.os.Handler;
import android.os.HandlerThread;
import android.os.Looper;
import android.system.ErrnoException;
import android.system.Os;
import com.google.gson.Gson;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class BenchmarkActivity extends Activity {

  File mModel;
  int mNumIter;
  int mNumWarmupIter;
  String mTokenizerPath;
  float mTemperature;
  String mPrompt;

  HandlerThread mHandlerThread;
  BenchmarkHandler mHandler;

  List<BenchmarkMetric> mResult;

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);

    try {
      Os.setenv("ADSP_LIBRARY_PATH", getApplicationInfo().nativeLibraryDir, true);
    } catch (ErrnoException e) {
      finish();
    }

    Intent intent = getIntent();
    File modelDir = new File(intent.getStringExtra("model_dir"));
    File model =
        Arrays.stream(modelDir.listFiles())
            .filter(file -> file.getName().endsWith(".pte"))
            .findFirst()
            .get();

    int numIter = intent.getIntExtra("num_iter", 50);
    int numWarmupIter = intent.getIntExtra("num_warm_up_iter", 10);
    String tokenizerPath = intent.getStringExtra("tokenizer_path");
    float temperature = intent.getFloatExtra("temperature", 0.8f);
    String prompt = intent.getStringExtra("prompt");

    mModel = model;
    mNumIter = numIter;
    mNumWarmupIter = numWarmupIter;
    mTokenizerPath = tokenizerPath;
    mTemperature = temperature;
    mPrompt = prompt;
    if (mPrompt == null) {
      mPrompt = "The ultimate answer";
    }
    mResult = new ArrayList<>();

    mHandlerThread = new HandlerThread("ModelRunner");
    mHandlerThread.start();
    mHandler = new BenchmarkHandler(mHandlerThread.getLooper(), this);

    mHandler.sendEmptyMessage(BenchmarkHandler.MESSAGE_RUN_BENCHMARK);
  }

  void writeResult() {
    try (FileWriter writer = new FileWriter(getFilesDir() + "/benchmark_results.json")) {
      Gson gson = new Gson();
      writer.write(gson.toJson(mResult));
    } catch (IOException e) {
      e.printStackTrace();
    } finally {
      finish();
    }
  }
}

class BenchmarkHandler extends Handler {
  public static int MESSAGE_RUN_BENCHMARK = 1;
  public static int MESSAGE_LLM_RUN_BENCHMARK = 2;

  ModelRunner mModelRunner;
  BenchmarkActivity mBenchmarkActivity;

  LlmModelRunner mLlmModelRunner;
  LlmBenchmark mLlmBenchmark;

  public BenchmarkHandler(Looper looper, BenchmarkActivity benchmarkActivity) {
    super(looper);
    mModelRunner = new ModelRunner();
    mBenchmarkActivity = benchmarkActivity;
  }

  @Override
  public void handleMessage(android.os.Message msg) {
    if (msg.what == MESSAGE_RUN_BENCHMARK) {
      mModelRunner.runBenchmark(
          mBenchmarkActivity.mModel,
          mBenchmarkActivity.mNumWarmupIter,
          mBenchmarkActivity.mNumIter,
          mBenchmarkActivity.mResult);

      if (mBenchmarkActivity.mTokenizerPath == null) {
        mBenchmarkActivity.writeResult();
      } else {
        this.sendEmptyMessage(MESSAGE_LLM_RUN_BENCHMARK);
      }
    } else if (msg.what == MESSAGE_LLM_RUN_BENCHMARK) {
      mLlmBenchmark =
          new LlmBenchmark(
              mBenchmarkActivity,
              mBenchmarkActivity.mModel.getPath(),
              mBenchmarkActivity.mTokenizerPath,
              mBenchmarkActivity.mPrompt,
              mBenchmarkActivity.mTemperature,
              mBenchmarkActivity.mResult);
    }
  }
}
