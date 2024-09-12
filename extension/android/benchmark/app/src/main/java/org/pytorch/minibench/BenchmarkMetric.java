/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.pytorch.minibench;

class BenchmarkMetric {
  // The model name, i.e. stories110M
  String name;

  // The metric name, i.e. TPS
  String metric;

  // The actual value and the option target value
  double actual;
  double target;

  // TODO (huydhn): Is there a way to get this information from the export model itself?
  final String dtype = "float32";

  // Let's see which information we want to include here
  final String device = android.os.Build.BRAND;
  // DEBUG DEBUG
  final String arch = android.os.Build.DEVICE + " / " + android.os.Build.MODEL;

  public BenchmarkMetric(
      final String name, final String metric, final double actual, final double target) {
    this.name = name;
    this.metric = metric;
    this.actual = actual;
    this.target = target;
  }
}
