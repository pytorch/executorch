/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.pytorch.minibench;

import android.os.Debug;
import java.io.File;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import org.pytorch.executorch.Module;

public class ModelRunner {
  /**
   * @return list of #BenchmarkMetric
   */
  public void runBenchmark(
      File model, int numWarmupIter, int numIter, List<BenchmarkMetric> results) {
    long pssIdle = Debug.getPss();

    List<Double> latency = new ArrayList<>();

    long loadStart = System.nanoTime();
    Module module = Module.load(model.getPath());
    int errorCode = module.loadMethod("forward");
    long loadEnd = System.nanoTime();

    for (int i = 0; i < numWarmupIter; i++) {
      module.forward();
    }

    for (int i = 0; i < numIter; i++) {
      long start = System.nanoTime();
      module.forward();
      double forwardMs = (System.nanoTime() - start) * 1e-6;
      latency.add(forwardMs);
    }

    module.etdump();

    final BenchmarkMetric.BenchmarkModel benchmarkModel =
        BenchmarkMetric.extractBackendAndQuantization(model.getName().replace(".pte", ""));
    // The list of metrics we have atm includes:
    // Avg inference latency after N iterations
    // Currently the result has large variance from outliers, so only use
    // 80% samples in the middle (trimmean 0.2)
    Collections.sort(latency);
    int resultSize = latency.size();
    List<Double> usedLatencyResults = latency.subList(resultSize / 10, resultSize * 9 / 10);

    results.add(
        new BenchmarkMetric(
            benchmarkModel,
            "avg_inference_latency(ms)",
            latency.stream().mapToDouble(l -> l).average().orElse(0.0f),
            0.0f));
    results.add(
        new BenchmarkMetric(
            benchmarkModel,
            "trimmean_inference_latency(ms)",
            usedLatencyResults.stream().mapToDouble(l -> l).average().orElse(0.0f),
            0.0f));
    // Model load time
    results.add(
        new BenchmarkMetric(
            benchmarkModel, "model_load_time(ms)", (loadEnd - loadStart) * 1e-6, 0.0f));
    // Load status
    results.add(new BenchmarkMetric(benchmarkModel, "load_status", errorCode, 0));
    // RAM PSS usage
    results.add(
        new BenchmarkMetric(
            benchmarkModel, "ram_pss_usage(mb)", (Debug.getPss() - pssIdle) / 1024, 0));
  }
}
