# Integrating and Running ExecuTorch on Apple Platforms

**Author:** [Anthony Shoumikhin](https://github.com/shoumikhin)

The ExecuTorch Runtime for iOS and macOS is distributed as a collection of prebuilt [.xcframework](https://developer.apple.com/documentation/xcode/creating-a-multi-platform-binary-framework-bundle) binary targets. These targets are compatible with both iOS and macOS devices and simulators and are available in both release and debug modes:

* `executorch` - Main Runtime components
* `backend_coreml` - Core ML backend
* `backend_mps` - MPS backend
* `backend_xnnpack` - XNNPACK backend
* `kernels_custom` - Custom kernels for LLMs
* `kernels_optimized` - Optimized kernels
* `kernels_portable` - Portable kernels (naive implementation used as a reference)
* `kernels_quantized` - Quantized kernels

Link your binary with the ExecuTorch runtime and any backends or kernels used by the exported ML model. It is recommended to link the core runtime to the components that use ExecuTorch directly, and link kernels and backends against the main app target.

**Note:** To access logs, link against the Debug build of the ExecuTorch runtime, i.e., the `executorch_debug` framework. For optimal performance, always link against the Release version of the deliverables (those without the `_debug` suffix), which have all logging overhead removed.

## Integration

### Swift Package Manager

The prebuilt ExecuTorch runtime, backend, and kernels are available as a [Swift PM](https://www.swift.org/documentation/package-manager/) package.

#### Xcode

In Xcode, go to `File > Add Package Dependencies`. Paste the URL of the [ExecuTorch repo](https://github.com/pytorch/executorch) into the search bar and select it. Make sure to change the branch name to the desired ExecuTorch version in format "swiftpm-<version>", (e.g. "swiftpm-0.4.0"), or a branch name in format "swiftpm-<version>.<year_month_date>" (e.g. "swiftpm-0.4.0-20241114") for a nightly build on a specific date.

![](_static/img/swiftpm_xcode1.png)

Then select which ExecuTorch framework should link against which target.

![](_static/img/swiftpm_xcode2.png)

Click the screenshot below to watch the *demo video* on how to add the package and run a simple ExecuTorch model on iOS.

<a href="https://pytorch.org/executorch/main/_static/img/swiftpm_xcode.mp4">
  <img src="https://pytorch.org/executorch/main/_static/img/swiftpm_xcode.png" width="800" alt="Integrating and Running ExecuTorch on Apple Platforms">
</a>

#### CLI

Add a package and target dependencies on ExecuTorch to your package file like this:

```swift
// swift-tools-version:5.0
import PackageDescription

let package = Package(
  name: "YourPackageName",
  products: [
    .library(name: "YourPackageName", targets: ["YourTargetName"]),
  ],
  dependencies: [
    // Use "swiftpm-0.4.0.<year_month_day>" branch name for a nightly build.
    .package(url: "https://github.com/pytorch/executorch.git", .branch("swiftpm-0.4.0"))
  ],
  targets: [
    .target(
      name: "YourTargetName",
      dependencies: [
        .product(name: "executorch", package: "executorch"),
        .product(name: "xnnpack_backend", package: "executorch")
      ]),
  ]
)
```

Then check if everything works correctly:

```bash
cd path/to/your/package

swift package resolve

# or just build it
swift build
```

### Local Build

Another way to integrate the ExecuTorch runtime is to build the necessary components from sources locally and link against them. This route is more involved but certainly doable.

1. Install [Xcode](https://developer.apple.com/xcode/resources/) 15+ and Command Line Tools:

```bash
xcode-select --install
```

2. Clone ExecuTorch:

```bash
git clone https://github.com/pytorch/executorch.git --depth 1 --recurse-submodules --shallow-submodules && cd executorch
```

3. Set up [Python](https://www.python.org/downloads/macos/) 3.10+ and activate a virtual environment:

```bash
python3 -m venv .venv && source .venv/bin/activate && pip install --upgrade pip
```

4. Install the required dependencies, including those needed for the backends like [Core ML](build-run-coreml.md) or [MPS](build-run-mps.md), if you plan to build them as well:

```bash
./install_requirements.sh --pybind coreml mps xnnpack

# Optional dependencies for Core ML backend.
./backends/apple/coreml/scripts/install_requirements.sh

# And MPS backend.
./backends/apple/mps/install_requirements.sh
```

5. Install [CMake](https://cmake.org):

Download the macOS binary distribution from the [CMake website](https://cmake.org/download), open the `.dmg` file, move `CMake.app` to the `/Applications` directory, and then run the following command to install the CMake command-line tools:

```bash
sudo /Applications/CMake.app/Contents/bin/cmake-gui --install
```

6. Use the provided script to build .xcframeworks:

```bash
./build/build_apple_frameworks.sh --help
```

For example, the following invocation will build the ExecuTorch Runtime and all currently available kernels and backends for the Apple platform:

```bash
./build/build_apple_frameworks.sh --coreml --mps --xnnpack --custom --optimized --portable --quantized
```

Append a `--Debug` flag to the above command to build the binaries with debug symbols if needed.

After the build finishes successfully, the resulting frameworks can be found in the `cmake-out` directory.
Copy them to your project and link them against your targets.

## Linkage

ExecuTorch initializes its backends and kernels (operators) during app startup by registering them in a static dictionary. If you encounter errors like "unregistered kernel" or "unregistered backend" at runtime, you may need to explicitly force-load certain components. Use the `-all_load` or `-force_load` linker flags in your Xcode build configuration to ensure components are registered early.

Here's an example of a Xcode configuration file (`.xcconfig`):

```
ET_PLATFORM[sdk=iphonesimulator*] = simulator
ET_PLATFORM[sdk=iphoneos*] = ios
ET_PLATFORM[sdk=macos*] = macos

OTHER_LDFLAGS = $(inherited) \
    -force_load $(BUILT_PRODUCTS_DIR)/libexecutorch-$(ET_PLATFORM)-release.a \
    -force_load $(BUILT_PRODUCTS_DIR)/libbackend_coreml-$(ET_PLATFORM)-release.a \
    -force_load $(BUILT_PRODUCTS_DIR)/libbackend_mps-$(ET_PLATFORM)-release.a \
    -force_load $(BUILT_PRODUCTS_DIR)/libbackend_xnnpack-$(ET_PLATFORM)-release.a \
    -force_load $(BUILT_PRODUCTS_DIR)/libkernels_optimized-$(ET_PLATFORM)-release.a \
    -force_load $(BUILT_PRODUCTS_DIR)/libkernels_quantized-$(ET_PLATFORM)-release.a
```

For a Debug build configuration, replace `release` with `debug` in the library file names. Remember to link against the ExecuTorch runtime (`libexecutorch`) in Debug mode even if other components are built for Release to preserve logs if needed.

You can assign such a config file to your target in Xcode:

1.	Add the `.xcconfig` file to your project.
2.	Navigate to the project’s Info tab.
3.	Select the configuration file in the build configurations for Release (or Debug) mode.

## Runtime API

Check out the [C++ Runtime API](extension-module.md) and [Tensors](extension-tensor.md) tutorials to learn more about how to load and run an exported model. It is recommended to use the C++ API for macOS or iOS, wrapped with Objective-C++ and Swift code if needed to expose it for other components. Please refer to the [Demo App](demo-apps-ios.md) as an example of such a setup.

Once linked against the `executorch` runtime framework, the target can now import all ExecuTorch public headers. For example, in Objective-C++:

```objectivecpp
#import <ExecuTorch/ExecuTorch.h>
#import <executorch/extension/module/module.h>
#import <executorch/extension/tensor/tensor.h>
```

Or in Swift:

```swift
import ExecuTorch
```

**Note:** Importing the ExecuTorch umbrella header (or ExecuTorch module in Swift) provides access to the logging API only. You still need to import the other runtime headers explicitly as needed, e.g., `module.h`. There is no support for other runtime APIs in Objective-C or Swift beyond logging described below.

**Note:** Logs are stripped in the release builds of ExecuTorch frameworks. To preserve logging, use debug builds during development.

### Logging

We provide extra APIs for logging in Objective-C and Swift as a lightweight wrapper of the internal ExecuTorch machinery. To use it, just import the main framework header in Objective-C. Then use the `ExecuTorchLog` interface (or the `Log` class in Swift) to subscribe your own implementation of the `ExecuTorchLogSink` protocol (or `LogSink` in Swift) to listen to log events.

```objectivec
#import <ExecuTorch/ExecuTorch.h>
#import <os/log.h>

@interface MyClass : NSObject<ExecuTorchLogSink>
@end

@implementation MyClass

- (instancetype)init {
  self = [super init];
  if (self) {
#if DEBUG
    [ExecuTorchLog.sharedLog addSink:self];
#endif
  }
  return self;
}

- (void)dealloc {
#if DEBUG
  [ExecuTorchLog.sharedLog removeSink:self];
#endif
}

#if DEBUG
- (void)logWithLevel:(ExecuTorchLogLevel)level
           timestamp:(NSTimeInterval)timestamp
            filename:(NSString *)filename
                line:(NSUInteger)line
             message:(NSString *)message {
  NSString *logMessage = [NSString stringWithFormat:@"%@:%lu %@", filename, (unsigned long)line, message];
  switch (level) {
    case ExecuTorchLogLevelDebug:
      os_log_with_type(OS_LOG_DEFAULT, OS_LOG_TYPE_DEBUG, "%{public}@", logMessage);
      break;
    case ExecuTorchLogLevelInfo:
      os_log_with_type(OS_LOG_DEFAULT, OS_LOG_TYPE_INFO, "%{public}@", logMessage);
      break;
    case ExecuTorchLogLevelError:
      os_log_with_type(OS_LOG_DEFAULT, OS_LOG_TYPE_ERROR, "%{public}@", logMessage);
      break;
    case ExecuTorchLogLevelFatal:
      os_log_with_type(OS_LOG_DEFAULT, OS_LOG_TYPE_FAULT, "%{public}@", logMessage);
      break;
    default:
      os_log(OS_LOG_DEFAULT, "%{public}@", logMessage);
      break;
  }
}
#endif

@end
```

Swift version:

```swift
import ExecuTorch
import os.log

public class MyClass {
  public init() {
    #if DEBUG
    Log.shared.add(sink: self)
    #endif
  }
  deinit {
    #if DEBUG
    Log.shared.remove(sink: self)
    #endif
  }
}

#if DEBUG
extension MyClass: LogSink {
  public func log(level: LogLevel, timestamp: TimeInterval, filename: String, line: UInt, message: String) {
    let logMessage = "\(filename):\(line) \(message)"
    switch level {
    case .debug:
      os_log(.debug, "%{public}@", logMessage)
    case .info:
      os_log(.info, "%{public}@", logMessage)
    case .error:
      os_log(.error, "%{public}@", logMessage)
    case .fatal:
      os_log(.fault, "%{public}@", logMessage)
    default:
      os_log("%{public}@", logMessage)
    }
  }
}
#endif
```

**Note:** In the example, the logs are intentionally stripped out when the code is not built for Debug mode, i.e., the `DEBUG` macro is not defined or equals zero.

## Debugging

If you are linking against a Debug build of the ExecuTorch frameworks, configure your debugger to map the source code correctly by using the following LLDB command in the debug session:

```
settings append target.source-map /executorch <path_to_executorch_source_code>
```

## Troubleshooting

### Slow execution

Ensure the exported model is using an appropriate backend, such as XNNPACK, Core ML, or MPS. If the correct backend is invoked but performance issues persist, confirm that you are linking against the Release build of the backend runtime.

For optimal performance, link the ExecuTorch runtime in Release mode too. If debugging is needed, you can keep the ExecuTorch runtime in Debug mode with minimal impact on performance, but preserve logging and debug symbols.

### Swift PM

If you encounter a checksum mismatch error with Swift PM, clear the package cache using the Xcode menu (`File > Packages > Reset Package Caches`) or the following command:

```bash
rm -rf <YouProjectName>.xcodeproj/project.xcworkspace/xcshareddata/swiftpm \
  ~/Library/org.swift.swiftpm \
  ~/Library/Caches/org.swift.swiftpm \
  ~/Library/Caches/com.apple.dt.Xcode \
  ~/Library/Developer/Xcode/DerivedData
```
**Note:** Ensure Xcode is fully quit before running the terminal command to avoid conflicts with active processes.
