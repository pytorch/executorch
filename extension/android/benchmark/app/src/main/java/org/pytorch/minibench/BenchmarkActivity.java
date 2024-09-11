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
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;
import org.pytorch.executorch.Module;

public class BenchmarkActivity extends Activity {
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

    int numIter = intent.getIntExtra("num_iter", 10);

    // TODO: Format the string with a parsable format
    StringBuilder resultText = new StringBuilder();

    Module module = Module.load(model.getPath());
    for (int i = 0; i < numIter; i++) {
      long start = System.currentTimeMillis();
      module.forward();
      long forwardMs = System.currentTimeMillis() - start;
      resultText.append(forwardMs).append(";");
    }

    try (FileWriter writer = new FileWriter(getFilesDir() + "/benchmark_results.txt")) {
      writer.write(resultText.toString());
    } catch (IOException e) {
      e.printStackTrace();
    }
  }
}
