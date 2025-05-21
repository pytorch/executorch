/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

package org.pytorch.executorch;

import org.pytorch.executorch.annotations.Experimental;
import com.facebook.jni.annotations.DoNotStrip;
import com.facebook.soloader.nativeloader.NativeLoader;
import com.facebook.soloader.nativeloader.SystemDelegate;

/**
 * Java API to access native runtime functionality on Android.
 *
 * <p>Warning: These APIs are experimental and subject to change without notice.
 */
@Experimental
@DoNotStrip
public class Runtime {
    private static boolean initialized = false;

    static {
        if (!NativeLoader.isInitialized()) {
            NativeLoader.init(new SystemDelegate());
        }
        // Loads libexecutorch.so from jniLibs
        NativeLoader.loadLibrary("executorch");
        initialized = true;
    }

    public static boolean isInitialized() {
        return initialized;
    }

    @DoNotStrip
    public static native String[] getRegisteredOps();

    @DoNotStrip
    public static native String[] getRegisteredBackends();
}
