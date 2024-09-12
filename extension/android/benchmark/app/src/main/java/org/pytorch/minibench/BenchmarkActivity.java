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
import com.google.gson.Gson;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
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
    Stats stats = new Stats();

    Module module = Module.load(model.getPath());
    for (int i = 0; i < numIter; i++) {
      long start = System.currentTimeMillis();
      module.forward();
      long forwardMs = System.currentTimeMillis() - start;
      stats.latency.add(forwardMs);
    }

    // TODO (huydhn): Remove txt files here once the JSON format is ready
    try (FileWriter writer = new FileWriter(getFilesDir() + "/benchmark_results.txt")) {
      writer.write(stats.toString());
    } catch (IOException e) {
      e.printStackTrace();
    }

    // TODO (huydhn): Figure out on what the final JSON results looks like, we need something
    // with the same number of fields as https://github.com/pytorch/pytorch/pull/135042
    try (FileWriter writer = new FileWriter(getFilesDir() + "/benchmark_results.json")) {
      Gson gson = new Gson();
      writer.write(gson.toJson(stats));
    } catch (IOException e) {
      e.printStackTrace();
    }
  }
}

class Stats {
  List<Long> latency = new ArrayList<>();

  @Override
  public String toString() {
    return "latency: " + latency.stream().map(Object::toString).collect(Collectors.joining(""));
  }
}
