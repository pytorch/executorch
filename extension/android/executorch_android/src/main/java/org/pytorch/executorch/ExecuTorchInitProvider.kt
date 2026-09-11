/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.pytorch.executorch

import android.content.ContentProvider
import android.content.ContentValues
import android.database.Cursor
import android.net.Uri

/**
 * Initializes the Android-specific behavior of the ExecuTorch Java API at process start, before
 * any application or library code runs (same mechanism as androidx Startup / Firebase init).
 *
 * Currently this installs [AndroidLogger] so ExecuTorch logs continue to go to logcat exactly as
 * before. No reflection or ServiceLoader is involved; the wiring is compile-checked.
 */
class ExecuTorchInitProvider : ContentProvider() {
  override fun onCreate(): Boolean {
    Log.install(AndroidLogger())
    return true
  }

  override fun query(
      uri: Uri,
      projection: Array<out String>?,
      selection: String?,
      selectionArgs: Array<out String>?,
      sortOrder: String?,
  ): Cursor? = null

  override fun getType(uri: Uri): String? = null

  override fun insert(uri: Uri, values: ContentValues?): Uri? = null

  override fun delete(uri: Uri, selection: String?, selectionArgs: Array<out String>?): Int = 0

  override fun update(
      uri: Uri,
      values: ContentValues?,
      selection: String?,
      selectionArgs: Array<out String>?,
  ): Int = 0
}
