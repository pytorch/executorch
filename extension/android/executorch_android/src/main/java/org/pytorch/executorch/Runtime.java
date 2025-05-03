package org.pytorch.executorch;

import com.facebook.soloader.nativeloader.NativeLoader;
import com.facebook.soloader.nativeloader.SystemDelegate;
import com.facebook.jni.annotations.DoNotStrip;

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