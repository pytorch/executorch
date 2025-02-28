# Using ExecuTorch on Android

To use from Android, ExecuTorch provides Java/Kotlin API bindings and Android platform integration, available as an AAR file.

Note: This page covers Android app integration through the AAR library. The ExecuTorch C++ APIs can also be used from Android native, and the documentation can be found on [this page about cross compilation](https://pytorch.org/executorch/main/using-executorch-building-from-source.html#cross-compilation).

## Installation

We package all ExecuTorch Android libraries into an Android library (AAR), `executorch.aar` for both generic (image/audio processing) and LLM (LLaMA) use case. In each release, we will upload the prebuilt AAR artifacts. Users can also build the AAR from source.

### Contents of library

The AAR artifact contains the Java library for users to integrate with their Java/Kotlin application code, as well as the corresponding JNI library (.so file), which is loaded by the Java code during initialization.

- [Java library](https://github.com/pytorch/executorch/tree/main/extension/android/src/main/java/org/pytorch/executorch)
- JNI contains the JNI binding for the corresponding Java code, and ExecuTorch native library, including
  - core ExecuTorch runtime libraries
  - XNNPACK backend
  - Portable kernels
  - Optimized kernels
  - Quantized kernels
  - LLaMa-specific Custom ops library.
- Comes with two ABI variants, arm64-v8a and x86\_64.

## Downloading AAR

### Released versions (recommended)

| Version | AAR | SHASUMS |
| ------- | --- | ------- |
| [v0.5.0](https://github.com/pytorch/executorch/releases/tag/v0.5.0) | [executorch.aar](https://ossci-android.s3.amazonaws.com/executorch/release/v0.5.0-rc3/executorch.aar) | [executorch.aar.sha256sums](https://ossci-android.s3.amazonaws.com/executorch/release/v0.5.0-rc3/executorch.aar) |

### Snapshots from main branch

| Date | AAR | SHASUMS |
| ------- | --- | ------- |
| 2025-02-27 | [executorch.aar](https://ossci-android.s3.amazonaws.com/executorch/release/executorch-20250227/executorch.aar) | [executorch.aar.sha256sums](https://ossci-android.s3.amazonaws.com/executorch/release/executorch-20250227/executorch.aar.sha256sums) |

## Using prebuilt libraries

To add the Java library to your app:
1. Download the AAR.
2. Add it to your gradle build rule as a file path.

The Java package requires `fbjni` and `soloader`, and currently requires users to explicitly declare the dependency. Therefore, two more `dependencies` in gradle rule is required:
```
implementation("com.facebook.soloader:soloader:0.10.5")
implementation("com.facebook.fbjni:fbjni:0.5.1")
```

### Example usage

In your app working directory, such as executorch/examples/demo-apps/android/LlamaDemo,
```
mkdir -p app/libs
curl https://ossci-android.s3.amazonaws.com/executorch/release/v0.5.0-rc3/executorch.aar -o app/libs/executorch.aar
```

And include it in gradle:
```
# app/build.grardle.kts
dependencies {
    implementation(files("libs/executorch.aar"))
    implementation("com.facebook.soloader:soloader:0.10.5")
    implementation("com.facebook.fbjni:fbjni:0.5.1")
}
```

Now you can compile your app with the ExecuTorch Android library.

## Building from Source

`build/build_android_library.sh` is a helper script to build the Java library (into .jar), native library (into .so), and the packaged AAR file. It can also build
demo apps to showcase the AAR is integrated into a user app correctly.

You need Android [SDK](https://developer.android.com/studio) and [NDK](https://developer.android.com/ndk/downloads) to use it.

Current NDK version used in ExecuTorch CI: r27b.

You need to set `ANDROID_NDK` to the correct NDK root (containing NOTICE file).

```
export ANDROID_NDK=/path/to/ndk
sh build/build_android_library.sh
```

### Optional environment variables

Optionally, set these environment variables before running `build_android_library.sh`.

#### ANDROID_ABIS
Set environment variable `ANDROID_ABIS` to either `arm64-v8a` or `x86_64` if you only need to build the native library for one ABI only.
```
export ANDROID_ABIS=arm64-v8a
# or
# export ANDROID_ABIS=x86_64
sh build/build_android_library.sh
```

#### EXECUTORCH_CMAKE_BUILD_TYPE
Set environment variable `EXECUTORCH_CMAKE_BUILD_TYPE` to `Release` or `Debug` based on your needs.

#### Using MediaTek backend

To use [MediaTek backend](https://pytorch.org/executorch/main/backends-mediatek.html),
after installing and setting up the SDK, set `NEURON_BUFFER_ALLOCATOR_LIB` and `NEURON_USDK_ADAPTER_LIB` to the corresponding path.

#### Using Qualcomm AI Engine Backend

To use [Qualcomm AI Engine Backend](https://pytorch.org/executorch/main/backends-qualcomm.html#qualcomm-ai-engine-backend),
after installing and setting up the SDK, set `QNN_SDK_ROOT` to the corresponding path

## Android Backends

TODO Describe commonly used backends, including XNN, Vulkan, and NPUs.

## Runtime Integration

TODO Code sample in Java

Here is an example code sample in Java that demonstrates how to integrate ExecuTorch into an Android app:

```
import org.pytorch.executorch.EValue;
import org.pytorch.executorch.Module;
import org.pytorch.executorch.Tensor;

public class MainActivity extends Activity {
    private Module module;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        // Load the ExecuTorch module
        module = Module.load("/path/to/module.pte");
    }
    public void runInference(View view) {
        // Prepare input data
        Tensor input = Tensor.fromBlob(getInputData());
        // Run inference
        Tensor output = module.forward(EValue.from(input))[0].toTensor();
        // Process output data
        processOutput(output);
    }
}
```
This example loads an ExecuTorch module, prepares input data, runs inference, and processes the output data.


## Next Steps

TODO Link to Java API reference and other relevant material
