/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.pytorch.minibench

/**
 * A helper interface within the app for MainActivity and Benchmarking to handle callback from
 * ModelRunner.
 */
interface LlmModelRunnerCallback {

    fun onModelLoaded(status: Int)

    fun onTokenGenerated(token: String)

    fun onStats(result: String)

    fun onGenerationStopped()
}
