/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.pytorch.minibench

import android.app.ActivityManager
import android.os.Build

class BenchmarkMetric(
    val benchmarkModel: BenchmarkModel,
    val metric: String,
    val actualValue: Double,
    val targetValue: Double,
) {
  data class BenchmarkModel(
      val name: String,
      val backend: String,
      val quantization: String,
  )

  class DeviceInfo {
    val device: String = Build.BRAND
    val arch: String = Build.MODEL
    val os: String = "Android ${Build.VERSION.RELEASE}"
    val totalMem: Long = ActivityManager.MemoryInfo().totalMem
    val availMem: Long = ActivityManager.MemoryInfo().availMem
  }

  val deviceInfo: DeviceInfo = DeviceInfo()

  companion object {
    // TODO (huydhn): Figure out a way to extract the backend and quantization information from
    // the .pte model itself instead of parsing its name
    @JvmStatic
    fun extractBackendAndQuantization(model: String): BenchmarkModel {
      val pattern = Regex("(?<name>\\w+)_(?<backend>[\\w+]+)_(?<quantization>\\w+)")
      val match = pattern.matchEntire(model)
      return if (match != null) {
        BenchmarkModel(
            match.groups["name"]!!.value,
            match.groups["backend"]!!.value,
            match.groups["quantization"]!!.value,
        )
      } else {
        BenchmarkModel(model, "", "")
      }
    }
  }
}
