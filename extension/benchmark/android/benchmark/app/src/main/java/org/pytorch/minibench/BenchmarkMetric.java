/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.pytorch.minibench;

import android.app.ActivityManager;
import android.os.Build;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

class BenchmarkMetric {
  public static class BenchmarkModel {
    // The model name, i.e. stories110M
    String name;
    String backend;
    String quantization;

    public BenchmarkModel(final String name, final String backend, final String quantization) {
      this.name = name;
      this.backend = backend;
      this.quantization = quantization;
    }
  }

  BenchmarkModel benchmarkModel;

  // The metric name, i.e. TPS
  String metric;

  // The actual value and the option target value
  double actualValue;
  double targetValue;

  public static class DeviceInfo {
    // Let's see which information we want to include here
    final String device = Build.BRAND;
    // The phone model and Android release version
    final String arch = Build.MODEL;
    final String os = "Android " + Build.VERSION.RELEASE;
    final long totalMem = new ActivityManager.MemoryInfo().totalMem;
    final long availMem = new ActivityManager.MemoryInfo().availMem;
  }

  DeviceInfo deviceInfo = new DeviceInfo();

  public BenchmarkMetric(
      final BenchmarkModel benchmarkModel,
      final String metric,
      final double actualValue,
      final double targetValue) {
    this.benchmarkModel = benchmarkModel;
    this.metric = metric;
    this.actualValue = actualValue;
    this.targetValue = targetValue;
  }

  // TODO (huydhn): Figure out a way to extract the backend and quantization information from
  // the .pte model itself instead of parsing its name
  public static BenchmarkMetric.BenchmarkModel extractBackendAndQuantization(final String model) {
    final Matcher m =
        Pattern.compile("(?<name>\\w+)_(?<backend>\\w+)_(?<quantization>\\w+)").matcher(model);
    if (m.matches()) {
      return new BenchmarkMetric.BenchmarkModel(
          m.group("name"), m.group("backend"), m.group("quantization"));
    } else {
      return new BenchmarkMetric.BenchmarkModel(model, "", "");
    }
  }
}
