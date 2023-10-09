/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

package com.example.executorchdemo.executor;

import com.facebook.soloader.nativeloader.NativeLoader;

public class LoadQnnLibrary {
  static {
    // Loads libqnndelegate.so from jniLibs
    NativeLoader.loadLibrary("libqnndelegate");
  }
}
