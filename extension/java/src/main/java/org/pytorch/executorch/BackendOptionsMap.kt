/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.pytorch.executorch

import org.pytorch.executorch.annotations.Experimental

/** A single backend option: an integer [value] for [key]. */
@Experimental class BackendOption(val key: String, val value: Int)

/**
 * A map of backend name -> backend options, passed to [Module.load] to configure ExecuTorch
 * backends at model-load (delegate-init) time. Mirrors the iOS `BackendOptionsMap`.
 *
 * Because the options travel with the load, this is per-model and ordering-safe (no process-global
 * setter to sequence before the load). Example:
 * ```
 * val options = BackendOptionsMap().setInt("XnnpackBackend", "workspace_sharing_mode", 2)
 * val module = Module.load(path, options)
 * ```
 *
 * Warning: These APIs are experimental and subject to change without notice.
 */
@Experimental
class BackendOptionsMap {

  private val options = mutableMapOf<String, MutableList<BackendOption>>()

  /** Set an integer-valued option [key] on [backendName]. */
  fun setInt(backendName: String, key: String, value: Int): BackendOptionsMap {
    options.getOrPut(backendName) { mutableListOf() }.add(BackendOption(key, value))
    return this
  }

  /** True if no options have been set. */
  fun isEmpty(): Boolean = options.isEmpty()

  // Flatten to parallel (backend, key, value) arrays for the JNI boundary in a single pass, so the
  // three arrays are element-aligned by construction (the native side regroups them by backend).
  internal fun toJniArrays(): Triple<Array<String>, Array<String>, IntArray> {
    val backends = mutableListOf<String>()
    val keys = mutableListOf<String>()
    val values = mutableListOf<Int>()
    for ((backend, opts) in options) {
      for (opt in opts) {
        backends.add(backend)
        keys.add(opt.key)
        values.add(opt.value)
      }
    }
    return Triple(backends.toTypedArray(), keys.toTypedArray(), values.toIntArray())
  }
}
