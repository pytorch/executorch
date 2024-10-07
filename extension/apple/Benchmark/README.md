# ExecuTorch Benchmark App for Apple Platforms

## Introduction

The **Benchmark App** is a tool designed to help developers measure the performance of PyTorch models on Apple devices using the ExecuTorch runtime.
It provides a flexible framework for dynamically generating and running performance tests on your models, allowing you to assess metrics such as load times, inference speeds, memory usage, and more.

<p align="center">
<img src="https://raw.githubusercontent.com/pytorch/executorch/refs/heads/main/docs/source/_static/img/ios_benchmark_app.png" alt="Benchmark App" style="width:800px">
</p>

## Prerequisites

- [Xcode](https://apps.apple.com/us/app/xcode/id497799835?mt=12/) 15.0 or later with command-line tools if not already installed (`xcode-select --install`).
- [CMake](https://cmake.org/download/) 3.19 or later
  - Download and open the macOS `.dmg` installer and move the CMake app to `/Applications` folder.
  - Install CMake command line tools: `sudo /Applications/CMake.app/Contents/bin/cmake-gui --install`
- A development provisioning profile with the [`increased-memory-limit`](https://developer.apple.com/documentation/bundleresources/entitlements/com_apple_developer_kernel_increased-memory-limit) entitlement if targeting iOS devices.

## Setting Up the App

### Get the Code

To get started, clone the ExecuTorch repository and cd into the source code directory:

```bash
git clone https://github.com/pytorch/executorch.git --depth 1 --recurse-submodules --shallow-submodules
cd executorch
```

This command performs a shallow clone to speed up the process.

### Set Up the Frameworks

The Benchmark App relies on prebuilt ExecuTorch frameworks.
You have two options:

<details>
<summary>Option 1: Download Prebuilt Frameworks</summary>
<br/>

Run the provided script to download the prebuilt frameworks:

```bash
./extension/apple/Benchmark/Frameworks/download_frameworks.sh
```
</details>

<details>
<summary>Option 2: Build Frameworks Locally</summary>
<br/>

Alternatively, you can build the frameworks yourself by following the [guide](https://pytorch.org/executorch/main/apple-runtime.html#local-build).
</details>

Once the frameworks are downloaded or built, verify that the `Frameworks` directory contains the necessary `.xcframework` files:

```bash
ls extension/apple/Benchmark/Frameworks
```

You should see:

```
backend_coreml.xcframework
backend_mps.xcframework
backend_xnnpack.xcframework
executorch.xcframework
kernels_custom.xcframework
kernels_optimized.xcframework
kernels_portable.xcframework
kernels_quantized.xcframework
```

## Adding Models and Resources

Place your exported model files (`.pte`) and any other resources (e.g., `tokenizer.bin`) into the `extension/apple/Benchmark/Resources` directory:

```bash
cp <path/to/my_model.pte> <path/to/llama3.pte> <path/to/tokenizer.bin> extension/apple/Benchmark/Resources
```

Optionally, check that the files are there:

```bash
ls extension/apple/Benchmark/Resources
```

For this example you should see:

```
llama3.pte
my_model.pte
tokenizer.bin
```

The app automatically bundles these resources and makes them available to the test suite.

## Running the Tests

### Build and Run the Tests

Open the Benchmark Xcode project:

```bash
open extension/apple/Benchmark/Benchmark.xcodeproj
```

Select the destination device or simulator and press `Command+U`, or click `Product` > `Test` in the menu to run the test suite.

<p align="center">
<img src="https://raw.githubusercontent.com/pytorch/executorch/refs/heads/main/docs/source/_static/img/ios_benchmark_app_tests.png" alt="Benchmark App Tests" style="width:800px">
</p>

### Configure Signing (if necessary)

If you plan to run the app on a physical device, you may need to set up code signing:

1. Open the **Project Navigator** by pressing `Command+1` and click on the `Benchmark` root of the file tree.
2. Under Targets section go to the **Signing & Capabilities** tab of both the `App` and `Tests` targets.
3. Select your development team. Alternatively, manually pick a provisioning profile that supports the increased memory limit entitlement and modify the bundle identifier if needed.

<p align="center">
<img src="https://raw.githubusercontent.com/pytorch/executorch/refs/heads/main/docs/source/_static/img/ios_benchmark_app_signing.png" alt="Benchmark App Signing" style="width:800px">
</p>

## Viewing Test Results and Metrics

After running the tests, you can view the results in Xcode:

1. Open the **Test Report Navigator** by pressing `Command+9`.
2. Select the most recent test run.
3. You'll see a list of tests that ran, along with their status (passed or failed).
4. To view metrics for a specific test:
   - Double-click on the test in the list.
   - Switch to the **Metrics** tab to see detailed performance data.

**Note**: The tests use `XCTMeasureOptions` to run each test multiple times (usually five) to obtain average performance metrics.

<p align="center">
<img src="https://raw.githubusercontent.com/pytorch/executorch/refs/heads/main/docs/source/_static/img/ios_benchmark_app_test_load.png" alt="Benchmark App Test Load" style="width:800px">
</p>
<p align="center">
<img src="https://raw.githubusercontent.com/pytorch/executorch/refs/heads/main/docs/source/_static/img/ios_benchmark_app_test_forward.png" alt="Benchmark App Test Forward" style="width:800px">
</p>
<p align="center">
<img src="https://raw.githubusercontent.com/pytorch/executorch/refs/heads/main/docs/source/_static/img/ios_benchmark_app_test_generate.png" alt="Benchmark App Test Generate" style="width:800px">
</p>

## Understanding the Test Suite

The Benchmark App uses a dynamic test generation framework to create tests based on the resources you provide.

### Dynamic Test Generation

The key components are:

- **`DynamicTestCase`**: A subclass of `XCTestCase` that allows for the dynamic creation of test methods.
- **`ResourceTestCase`**: Builds upon `DynamicTestCase` to generate tests based on resources that match specified criteria.

### How It Works

1. **Define Directories and Predicates**: Override the `directories` and `predicates` methods to specify where to look for resources and how to match them.

2. **Generate Resource Combinations**: The framework searches the specified `directories` for files matching the `predicates`, generating all possible combinations.

3. **Create Dynamic Tests**: For each combination of resources, it calls `dynamicTestsForResources`, where you define the tests to run.

4. **Test Naming**: Test names are dynamically formed using the format:

   ```
   test_<TestName>_<Resource1>_<Resource2>_..._<OS>_<Version>_<DeviceModel>
   ```

   This ensures that each test is uniquely identifiable based on the resources and device.

### Example: Generic Model Tests

Here's how you might create a test to measure model load and inference times:

```objective-c
@interface GenericTests : ResourceTestCase
@end

@implementation GenericTests

+ (NSArray<NSString *> *)directories {
  return @[@"Resources"];
}

+ (NSDictionary<NSString *, BOOL (^)(NSString *)> *)predicates {
  return @{
    @"model" : ^BOOL(NSString *filename) {
      return [filename hasSuffix:@".pte"];
    },
  };
}

+ (NSDictionary<NSString *, void (^)(XCTestCase *)> *)dynamicTestsForResources:(NSDictionary<NSString *, NSString *> *)resources {
  NSString *modelPath = resources[@"model"];
  return @{
    @"load" : ^(XCTestCase *testCase) {
      [testCase measureWithMetrics:@[[XCTClockMetric new], [XCTMemoryMetric new]] block:^{
        XCTAssertEqual(Module(modelPath.UTF8String).load_forward(), Error::Ok);
      }];
    },
    @"forward" : ^(XCTestCase *testCase) {
      // Set up and measure the forward pass...
    },
  };
}

@end
```

In this example:

- We look for `.pte` files in the `Resources` directory.
- For each model found, we create two tests: `load` and `forward`.
- The tests measure the time and memory usage of loading and running the model.

## Extending the Test Suite

You can create custom tests by subclassing `ResourceTestCase` and overriding the necessary methods.

### Steps to Create Custom Tests

1. **Subclass `ResourceTestCase`**:

   ```objective-c
   @interface MyCustomTests : ResourceTestCase
   @end
   ```

2. **Override `directories` and `predicates`**:

   Specify where to look for resources and how to match them.

   ```objective-c
   + (NSArray<NSString *> *)directories {
     return @[@"Resources"];
   }

   + (NSDictionary<NSString *, BOOL (^)(NSString *)> *)predicates {
     return @{
       @"model" : ^BOOL(NSString *filename) {
         return [filename hasSuffix:@".pte"];
       },
       @"config" : ^BOOL(NSString *filename) {
         return [filename isEqualToString:@"config.json"];
       },
     };
   }
   ```

3. **Implement `dynamicTestsForResources`**:

   Define the tests to run for each combination of resources.

   ```objective-c
   + (NSDictionary<NSString *, void (^)(XCTestCase *)> *)dynamicTestsForResources:(NSDictionary<NSString *, NSString *> *)resources {
     NSString *modelPath = resources[@"model"];
     NSString *configPath = resources[@"config"];
     return @{
       @"customTest" : ^(XCTestCase *testCase) {
         // Implement your test logic here.
       },
     };
   }
   ```

4. **Add the Test Class to the Test Target**:

   Ensure your new test class is included in the test target in Xcode.

### Example: LLaMA Token Generation Test

An example of a more advanced test is measuring the tokens per second during text generation with the LLaMA model.

```objective-c
@interface LLaMATests : ResourceTestCase
@end

@implementation LLaMATests

+ (NSArray<NSString *> *)directories {
  return @[@"Resources"];
}

+ (NSDictionary<NSString *, BOOL (^)(NSString *)> *)predicates {
  return @{
    @"model" : ^BOOL(NSString *filename) {
      return [filename hasSuffix:@".pte"] && [filename containsString:@"llama"];
    },
    @"tokenizer" : ^BOOL(NSString *filename) {
      return [filename isEqualToString:@"tokenizer.bin"];
    },
  };
}

+ (NSDictionary<NSString *, void (^)(XCTestCase *)> *)dynamicTestsForResources:(NSDictionary<NSString *, NSString *> *)resources {
  NSString *modelPath = resources[@"model"];
  NSString *tokenizerPath = resources[@"tokenizer"];
  return @{
    @"generate" : ^(XCTestCase *testCase) {
      // Implement the token generation test...
    },
  };
}

@end
```

In this test:

- We look for LLaMA model files and a `tokenizer.bin`.
- We measure tokens per second and memory usage during text generation.

## Measuring Performance

The Benchmark App leverages Apple's performance testing APIs to measure metrics such as execution time and memory usage.

- **Measurement Options**: By default, each test is run five times to calculate average metrics.
- **Custom Metrics**: You can define custom metrics by implementing the `XCTMetric` protocol.
- **Available Metrics**:
  - `XCTClockMetric`: Measures wall-clock time.
  - `XCTMemoryMetric`: Measures memory usage.
  - **Custom Metrics**: For example, the LLaMA test includes a `TokensPerSecondMetric`.

## Running Tests from the Command Line

You can also run the tests using `xcodebuild`:

```bash
# Run on an iOS Simulator
xcodebuild test -project extension/apple/Benchmark/Benchmark.xcodeproj \
-scheme Benchmark \
-destination 'platform=iOS Simulator,name=<SimulatorName>' \
-testPlan Tests

# Run on a physical iOS device
xcodebuild test -project extension/apple/Benchmark/Benchmark.xcodeproj \
-scheme Benchmark \
-destination 'platform=iOS,name=<DeviceName>' \
-testPlan Tests \
-allowProvisioningUpdates DEVELOPMENT_TEAM=<YourTeamID>
```

Replace `<SimulatorName>`, `<DeviceName>`, and `<YourTeamID>` with your simulator/device name and Apple development team ID.

## macOS

The app can be built and run on macOS, just add it as the destination platform.

<p align="center">
<img src="https://raw.githubusercontent.com/pytorch/executorch/refs/heads/main/docs/source/_static/img/ios_benchmark_app_macos.png" alt="Benchmark App macOS" style="width:700px">
</p>

Also, set up app signing to run locally.

<p align="center">
<img src="https://raw.githubusercontent.com/pytorch/executorch/refs/heads/main/docs/source/_static/img/ios_benchmark_app_macos_signing.png" alt="Benchmark App macOS Signing" style="width:800px">
</p>

## Conclusion

The ExecuTorch Benchmark App provides a flexible and powerful framework for testing and measuring the performance of PyTorch models on Apple devices. By leveraging dynamic test generation, you can easily add your models and resources to assess their performance metrics. Whether you're optimizing existing models or developing new ones, this tool can help you gain valuable insights into their runtime behavior.
