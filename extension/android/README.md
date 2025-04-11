# ExecuTorch Android

This directory contains the Android Java/Kotlin binding. The final product is an AAR,
which contains the `.so` libraries for c++ runtime, and `.jar` for Java API, and required
metadata `AndroidManifest.xml`.

## Core contents

Under `extension/android/`,

- `executorch_android/` is the root for the Java `org.pytorch.executorch` package
  - `src/`
    - `androidTest/` contains the android instrumentation test source
    - `main/` contains the Java source
    - `test/` contains the Java unit test source
  - `build.gradle` is the rule to build the Java package.
- `jni/` contains the JNI layer code, which depends on the ExecuTorch c++ runtime library.
- `CMakeLists.txt` is the rule for building the JNI library.

## Build

`scripts/build_android_library.sh` is a helper script to build the Java library (into .jar), native library (into .so), and the packaged AAR file.

The usage is:
```sh
export ANDROID_HOME=/path/to/sdk
export ANDROID_NDK=/path/to/ndk
sh scripts/build_android_library.sh
```

Please see [Android building from source](https://pytorch.org/executorch/main/using-executorch-android.html#building-from-source) for details

## Test

After the library is built,

```sh
# Set up models for testing
sh executorch_android/android_test_setup.sh

# Run unit test
./gradlew :executorch_android:testDebugUnitTest

# Run instrumentation test
./gradlew :executorch_android:connectedAndroidTest
```
