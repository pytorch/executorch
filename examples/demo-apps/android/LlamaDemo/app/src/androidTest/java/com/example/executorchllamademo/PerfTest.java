/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

package com.example.executorchllamademo;

import static junit.framework.TestCase.assertTrue;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;

import androidx.test.ext.junit.runners.AndroidJUnit4;
import java.util.ArrayList;
import java.util.List;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.pytorch.executorch.LlamaCallback;
import org.pytorch.executorch.LlamaModule;

@RunWith(AndroidJUnit4.class)
public class PerfTest implements LlamaCallback {

  private static final String RESOURCE_PATH = "/data/local/tmp/llama/";
  private static final String MODEL_NAME = "xnnpack_llama2.pte";
  private static final String TOKENIZER_BIN = "tokenizer.bin";

  // From https://github.com/pytorch/executorch/blob/main/examples/models/llama2/README.md
  private static final Float EXPECTED_TPS = 10.0F;

  private final List<String> results = new ArrayList<>();
  private final List<Float> tokensPerSecond = new ArrayList<>();

  @Test
  public void testTokensPerSecond() {
    String modelPath = RESOURCE_PATH + MODEL_NAME;
    String tokenizerPath = RESOURCE_PATH + TOKENIZER_BIN;
    LlamaModule mModule = new LlamaModule(modelPath, tokenizerPath, 0.8f);

    int loadResult = mModule.load();
    // Check that the model can be load successfully
    assertEquals(0, loadResult);

    // Run a testing prompt
    mModule.generate("How do you do! I'm testing llama2 on mobile device", PerfTest.this);
    assertFalse(tokensPerSecond.isEmpty());

    final Float tps = tokensPerSecond.get(tokensPerSecond.size() - 1);
    assertTrue(
        "The observed TPS " + tps + " is less than the expected TPS " + EXPECTED_TPS,
        tps >= EXPECTED_TPS);
  }

  @Override
  public void onResult(String result) {
    results.add(result);
  }

  @Override
  public void onStats(float tps) {
    tokensPerSecond.add(tps);
  }
}
