/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.pytorch.minibench

import android.os.Debug
import java.io.File
import org.pytorch.executorch.ExecutorchRuntimeException
import org.pytorch.executorch.Module

class ModelRunner {

  fun runBenchmark(
      model: File,
      numWarmupIter: Int,
      numIter: Int,
      results: MutableList<BenchmarkMetric>,
  ) {
    val pssIdle = Debug.getPss()
    val latency = mutableListOf<Double>()

    val loadStart = System.nanoTime()
    val module = Module.load(model.path)
    var errorCode = 0
    try {
      module.loadMethod("forward")
    } catch (e: ExecutorchRuntimeException) {
      errorCode = e.errorCode
    } catch (e: Exception) {
      errorCode = -1
    }
    val loadEnd = System.nanoTime()

    val benchmarkModel =
        BenchmarkMetric.extractBackendAndQuantization(model.name.removeSuffix(".pte"))

    if (errorCode != 0) {
      results.add(
          BenchmarkMetric(benchmarkModel, "model_load_time(ms)", (loadEnd - loadStart) * 1e-6, 0.0))
      results.add(BenchmarkMetric(benchmarkModel, "load_status", errorCode.toDouble(), 0.0))
      module.destroy()
      return
    }

    try {
      repeat(numWarmupIter) { module.forward() }

      repeat(numIter) {
        val start = System.nanoTime()
        module.forward()
        latency.add((System.nanoTime() - start) * 1e-6)
      }

      module.etdump()

      // Currently the result has large variance from outliers, so only use
      // 80% samples in the middle (trimmean 0.2)
      latency.sort()
      val trimmed = latency.subList(latency.size / 10, latency.size * 9 / 10)

      results.add(
          BenchmarkMetric(
              benchmarkModel,
              "avg_inference_latency(ms)",
              latency.average(),
              0.0,
          ))
      results.add(
          BenchmarkMetric(
              benchmarkModel,
              "trimmean_inference_latency(ms)",
              trimmed.average(),
              0.0,
          ))
      results.add(
          BenchmarkMetric(benchmarkModel, "model_load_time(ms)", (loadEnd - loadStart) * 1e-6, 0.0))
      results.add(BenchmarkMetric(benchmarkModel, "load_status", errorCode.toDouble(), 0.0))
      results.add(
          BenchmarkMetric(
              benchmarkModel, "ram_pss_usage(mb)", (Debug.getPss() - pssIdle) / 1024.0, 0.0))
    } finally {
      module.destroy()
    }
  }
}
