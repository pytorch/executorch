/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

package com.example.executorchllamademo;

/**
 * A helper interface within the app for MainActivity and Benchmarking to handle callback from
 * ModelRunner.
 */
public interface ModelRunnerCallback {

  void onModelLoaded(int status);

  void onTokenGenerated(String token);

  void onStats(String stats);

  void onGenerationStopped();
}
