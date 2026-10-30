/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.pytorch.executorch

import android.util.Log as AndroidLog

/**
 * Android [Logger] implementation that forwards ExecuTorch logs to `android.util.Log` (logcat).
 * Installed automatically at process start by [ExecuTorchInitProvider].
 */
class AndroidLogger : Logger {
  override fun e(tag: String, msg: String) {
    AndroidLog.e(tag, msg)
  }

  override fun w(tag: String, msg: String) {
    AndroidLog.w(tag, msg)
  }

  override fun i(tag: String, msg: String) {
    AndroidLog.i(tag, msg)
  }

  override fun d(tag: String, msg: String) {
    AndroidLog.d(tag, msg)
  }
}
