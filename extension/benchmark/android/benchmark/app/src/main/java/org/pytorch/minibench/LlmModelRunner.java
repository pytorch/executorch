/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.pytorch.minibench;

import android.os.Handler;
import android.os.HandlerThread;
import android.os.Looper;
import android.os.Message;
import org.pytorch.executorch.extension.llm.LlmCallback;
import org.pytorch.executorch.extension.llm.LlmModule;

/** A helper class to handle all model running logic within this class. */
public class LlmModelRunner implements LlmCallback {
  LlmModule mModule = null;

  String mModelFilePath = "";
  String mTokenizerFilePath = "";

  LlmModelRunnerCallback mCallback = null;

  HandlerThread mHandlerThread = null;
  Handler mHandler = null;

  /**
   * ] Helper class to separate between UI logic and model runner logic. Automatically handle
   * generate() request on worker thread.
   *
   * @param modelFilePath
   * @param tokenizerFilePath
   * @param callback
   */
  LlmModelRunner(
      String modelFilePath,
      String tokenizerFilePath,
      float temperature,
      LlmModelRunnerCallback callback) {
    mModelFilePath = modelFilePath;
    mTokenizerFilePath = tokenizerFilePath;
    mCallback = callback;

    mModule = new LlmModule(mModelFilePath, mTokenizerFilePath, 0.8f);
    mHandlerThread = new HandlerThread("LlmModelRunner");
    mHandlerThread.start();
    mHandler = new LlmModelRunnerHandler(mHandlerThread.getLooper(), this);

    mHandler.sendEmptyMessage(LlmModelRunnerHandler.MESSAGE_LOAD_MODEL);
  }

  int generate(String prompt) {
    Message msg = Message.obtain(mHandler, LlmModelRunnerHandler.MESSAGE_GENERATE, prompt);
    msg.sendToTarget();
    return 0;
  }

  void stop() {
    mModule.stop();
  }

  @Override
  public void onResult(String result) {
    mCallback.onTokenGenerated(result);
  }

  @Override
  public void onStats(String result) {
    mCallback.onStats(result);
  }
}

class LlmModelRunnerHandler extends Handler {
  public static int MESSAGE_LOAD_MODEL = 1;
  public static int MESSAGE_GENERATE = 2;

  private final LlmModelRunner mLlmModelRunner;

  public LlmModelRunnerHandler(Looper looper, LlmModelRunner llmModelRunner) {
    super(looper);
    mLlmModelRunner = llmModelRunner;
  }

  @Override
  public void handleMessage(android.os.Message msg) {
    if (msg.what == MESSAGE_LOAD_MODEL) {
      int status = mLlmModelRunner.mModule.load();
      mLlmModelRunner.mCallback.onModelLoaded(status);
    } else if (msg.what == MESSAGE_GENERATE) {
      mLlmModelRunner.mModule.generate((String) msg.obj, mLlmModelRunner);
      mLlmModelRunner.mCallback.onGenerationStopped();
    }
  }
}
