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
import android.system.ErrnoException;
import android.system.Os;
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

    int numIter = intent.getIntExtra("num_iter", 10);

    // TODO: Format the string with a parsable format
    Stats stats = new Stats();

    // Record the time it takes to load the model and the forward method
    stats.loadStart = System.nanoTime();
    Module module = Module.load(model.getPath());
    stats.errorCode = module.loadMethod("forward");
    stats.loadEnd = System.nanoTime();

    for (int i = 0; i < numIter; i++) {
      long start = System.nanoTime();
      module.forward();
      double forwardMs = (System.nanoTime() - start) * 1e-6;
      stats.latency.add(forwardMs);
    }

    final BenchmarkMetric.BenchmarkModel benchmarkModel =
        BenchmarkMetric.extractBackendAndQuantization(model.getName().replace(".pte", ""));
    final List<BenchmarkMetric> results = new ArrayList<>();
    // The list of metrics we have atm includes:
    // Avg inference latency after N iterations
    results.add(
        new BenchmarkMetric(
            benchmarkModel,
            "avg_inference_latency(ms)",
            stats.latency.stream().mapToDouble(l -> l).average().orElse(0.0f),
            0.0f));
    // Model load time
    results.add(
        new BenchmarkMetric(
            benchmarkModel, "model_load_time(ms)", (stats.loadEnd - stats.loadStart) * 1e-6, 0.0f));
    // Load status
    results.add(new BenchmarkMetric(benchmarkModel, "load_status", stats.errorCode, 0));

    try (FileWriter writer = new FileWriter(getFilesDir() + "/benchmark_results.json")) {
      Gson gson = new Gson();
      writer.write(gson.toJson(results));
    } catch (IOException e) {
      e.printStackTrace();
    }
  }
}

class Stats {
  long loadStart;
  long loadEnd;
  List<Double> latency = new ArrayList<>();
  int errorCode = 0;

  @Override
  public String toString() {
    return "latency: " + latency.stream().map(Object::toString).collect(Collectors.joining(""));
  }
}
