/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

package com.example.executorchllamademo;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;

import android.os.Bundle;
import androidx.test.ext.junit.runners.AndroidJUnit4;
import androidx.test.platform.app.InstrumentationRegistry;
import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.pytorch.executorch.LlamaCallback;
import org.pytorch.executorch.LlamaModule;

@RunWith(AndroidJUnit4.class)
public class PerfTest implements LlamaCallback {

  private static final String RESOURCE_PATH = "/data/local/tmp/llama/";
  private static final String TOKENIZER_BIN = "tokenizer.bin";

  private final List<String> results = new ArrayList<>();
  private final List<Float> tokensPerSecond = new ArrayList<>();

  @Test
  public void testTokensPerSecond() {
    String tokenizerPath = RESOURCE_PATH + TOKENIZER_BIN;
    // Find out the model name
    File directory = new File(RESOURCE_PATH);
    Arrays.stream(directory.listFiles())
        .filter(file -> file.getName().endsWith(".pte"))
        .forEach(
            model -> {
              LlamaModule mModule = new LlamaModule(model.getPath(), tokenizerPath, 0.8f);
              // Print the model name because there might be more than one of them
              report("ModelName", model.getName());

              int loadResult = mModule.load();
              // Check that the model can be load successfully
              assertEquals(0, loadResult);

              // Run a testing prompt
              mModule.generate("How do you do! I'm testing llama2 on mobile device", PerfTest.this);
              assertFalse(tokensPerSecond.isEmpty());

              final Float tps = tokensPerSecond.get(tokensPerSecond.size() - 1);
              report("TPS", tps);
            });
  }

  @Override
  public void onResult(String result) {
    results.add(result);
  }

  @Override
  public void onStats(float tps) {
    tokensPerSecond.add(tps);
  }

  private void report(final String metric, final Float value) {
    Bundle bundle = new Bundle();
    bundle.putFloat(metric, value);
    InstrumentationRegistry.getInstrumentation().sendStatus(0, bundle);
  }

  private void report(final String key, final String value) {
    Bundle bundle = new Bundle();
    bundle.putString(key, value);
    InstrumentationRegistry.getInstrumentation().sendStatus(0, bundle);
  }
}
