# Using ExecuTorch on Android

To use from Android, ExecuTorch provides Java API bindings and Android platform integration, available as a AAR file. The ExecuTorch C++ APIs can also be used from Android native.

## Installation

We provide a prebuilt Android library (AAR), `executorch.aar` for both generic (image/audio processing) and LLAMA use case.

## Contents of library
- `executorch.aar`
  - [Java library](https://github.com/pytorch/executorch/tree/main/extension/android/src/main/java/org/pytorch/executorch)
  - JNI contains the JNI binding for the corresponding Java code, and ExecuTorch native library, including core ExecuTorch runtime libraries, XNNPACK backend, Portable kernels, Optimized kernels, Quantized kernels, and LLAMA-specific Custom ops library.
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

To add the Java library to your app, simply download the AAR, and add it to your gradle build rule.

In your app working directory, such as example executorch/examples/demo-apps/android/LlamaDemo,
```
mkdir -p app/libs
curl https://ossci-android.s3.amazonaws.com/executorch/release/executorch-241002/executorch.aar -o app/libs/executorch.aar
```

And include it in gradle:
```
# app/build.grardle.kts
dependencies {
    implementation(files("libs/executorch.aar"))
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
