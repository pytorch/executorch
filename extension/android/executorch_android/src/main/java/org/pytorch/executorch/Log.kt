/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.pytorch.executorch

import android.util.Log as AndroidLog

/** Android logging helper that forwards logs directly to android.util.Log. */
internal object Log {
  fun e(tag: String, msg: String) {
    AndroidLog.e(tag, msg)
  }

  fun w(tag: String, msg: String) {
    AndroidLog.w(tag, msg)
  }

  fun i(tag: String, msg: String) {
    AndroidLog.i(tag, msg)
  }

  fun d(tag: String, msg: String) {
    AndroidLog.d(tag, msg)
  }
}
