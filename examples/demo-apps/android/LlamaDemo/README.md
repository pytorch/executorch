# Building ExecuTorch LLaMA Android Demo App

This app demonstrates the use of the LLaMA chat app demonstrating local inference use case with ExecuTorch.

## Prerequisites
* Set up your ExecuTorch repo and environment if you haven’t done so by following the [Setting up ExecuTorch](https://pytorch.org/executorch/stable/getting-started-setup) to set up the repo and dev environment.
* Install [Java 17 JDK](https://www.oracle.com/java/technologies/javase/jdk17-archive-downloads.html).
* Install the [Android SDK API Level 34](https://developer.android.com/about/versions/14/setup-sdk) and
  [Android NDK 25.0.8775105](https://developer.android.com/studio/projects/install-ndk).
 * If you have Android Studio set up, you can install them with
   * Android Studio Settings -> Language & Frameworks -> Android SDK -> SDK Platforms -> Check the row with API Level 34.
   * Android Studio Settings -> Language & Frameworks -> Android SDK -> SDK Tools -> Check NDK (Side by side) row.
 * Alternatively, you can follow [this guide](https://github.com/pytorch/executorch/blob/856e085b9344c8b0bf220a97976140a5b76356aa/examples/demo-apps/android/LlamaDemo/SDK.md) to set up Java/SDK/NDK with CLI.
* Supported Host OS: CentOS, macOS Sonoma on Apple Silicon.

Note: This demo app and tutorial has only been validated with arm64-v8a [ABI](https://developer.android.com/ndk/guides/abis), with NDK 25.0.8775105.

## Getting models
Please refer to the [ExecuTorch Llama2 docs](https://github.com/pytorch/executorch/blob/main/examples/models/llama2/README.md) to export the model.

After you export the model and generate tokenizer.bin, push them device:
```bash
adb shell mkdir -p /data/local/tmp/llama
adb push llama2.pte /data/local/tmp/llama
adb push tokenizer.bin /data/local/tmp/llama
```

Note: The demo app searches in `/data/local/tmp/llama` for .pte and .bin files as LLAMA model and tokenizer.

## Build library
For the demo app to build, we need to build the ExecuTorch AAR library first.

The AAR library contains the required Java package and the corresponding JNI
library for using ExecuTorch in your Android app.

### Alternative 1: Use prebuilt AAR library (recommended)

1. Open a terminal window and navigate to the root directory of the `executorch`.
2. Run the following command to download the prebuilt library:
```bash
bash examples/demo-apps/android/LlamaDemo/download_prebuilt_lib.sh
```

The prebuilt AAR library contains the Java library and the JNI binding for
NativePeer.java and ExecuTorch native library, including core ExecuTorch
runtime libraries, XNNPACK backend, Portable kernels, Optimized kernels,
and Quantized kernels. It comes with two ABI variants, arm64-v8a and x86_64.

If you want to use the prebuilt library for your own app, please refer to
[Using Android prebuilt libraries (AAR)](./android-prebuilt-library.md) for
tutorial.

If you need to use other dependencies (like tokenizer), please refer to
Alternative 2: Build from local machine option.

### Alternative 2: Build from local machine
1. Open a terminal window and navigate to the root directory of the `executorch`.
2. Set the following environment variables:
```bash
export ANDROID_NDK=<path_to_android_ndk>
export ANDROID_ABI=arm64-v8a
```
Note: `<path_to_android_ndk>` is the root for the NDK, which is usually under
`~/Library/Android/sdk/ndk/XX.Y.ZZZZZ` for macOS, and contains NOTICE and README.md.
We use `<path_to_android_ndk>/build/cmake/android.toolchain.cmake` for CMake to cross-compile.

3. (Optional) If you need to use tiktoken as the tokenizer (for LLaMA3), set
`EXECUTORCH_USE_TIKTOKEN=ON` and later CMake will use it as the tokenizer.
If you need to run other models like LLaMA2, skip this skip.

```bash
export EXECUTORCH_USE_TIKTOKEN=ON # Only for LLaMA3
```

4. Build the Android Java extension code:
```bash
pushd extension/android
./gradlew build
popd
```

5. Run the following command set up the required JNI library:
```bash
pushd examples/demo-apps/android/LlamaDemo
./gradlew :app:setup
popd
```
This is running the shell script [setup.sh](./setup.sh) which configures the required core ExecuTorch, LLAMA2, and Android libraries, builds them, and copy to jniLibs.

## Build APK
### Alternative 1: Android Studio (Recommended)
1. Open Android Studio and select "Open an existing Android Studio project" to open examples/demo-apps/android/LlamaDemo.
2. Run the app (^R). This builds and launches the app on the phone.

### Alternative 2: Command line
Without Android Studio UI, we can run gradle directly to build the app. We need to set up the Android SDK path and invoke gradle.
```bash
export ANDROID_HOME=<path_to_android_sdk_home>
pushd examples/demo-apps/android/LlamaDemo
./gradlew :app:installDebug
popd
```

On the phone or emulator, you can try running the model:
<img src="../_static/img/android_llama_app.png" alt="Android LLaMA App" /><br>

## Takeaways
Through this tutorial we've learnt how to build the ExecuTorch LLAMA library, and expose it to JNI layer to build the Android app.

## Reporting Issues
If you encountered any bugs or issues following this tutorial please file a bug/issue here on [Github](https://github.com/pytorch/executorch/issues/new).
