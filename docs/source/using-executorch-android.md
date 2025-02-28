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

### Building from Source

TODO Instructions on re-creating and customizing the Android AAR.

## Android Backends

TODO Describe commonly used backends, including XNN, Vulkan, and NPUs.

## Runtime Integration

TODO Code sample in Java

## Next Steps

TODO Link to Java API reference and other relevant material
