# ExecuTorch: iOS Demo App Setup

This README guides you through the setup of ExecuTorch on iOS using a demo app.
This app utilizes the MobileNet v3 model to process live camera images.

## Prerequisites

1. **Install Xcode 15 and Command Line Tools**

   ```bash
   xcode-select --install
   ```

2. **Python 3.10+ and `pip`** (Pre-installed from MacOS 13.5+)

   [Download](https://www.python.org/downloads/macos/) and install Python 3.10
   or 3.11, if needed, and verify the versions:

   ```bash
   which python3 pip
   python3 --version
   pip --version
   ```

3. **Follow the [Getting Started](../../../docs/source/getting-started-setup.md)
   Tutorial**

4. **Backend Dependency Installation**

   Install additional dependencies for **CoreML**:

   ```bash
   ./backends/apple/coreml/scripts/install_requirements.sh
   ```

   And **Metal Performance Shaders**:

   ```bash
   ./backends/apple/mps/install_requirements.sh
   ```

## Model Export & Bundling

1. **Export MobileNet v3 model with CoreML, MPS and XNNPACK delegates**

   ```bash
   python3 -m examples.portable.scripts.export --model_name="mv3"
   python3 -m examples.xnnpack.aot_compiler --delegate --model_name="mv3"
   python3 -m examples.apple.coreml.scripts.export_and_delegate --model_name="mv3"
   python3 -m examples.apple.mps.scripts.mps_example --model_name="mv3"

   mkdir -p examples/demo-apps/apple_ios/ExecuTorchDemo/ExecuTorchDemo/Resources/Models/MobileNet/
   mv mv3*.pte examples/demo-apps/apple_ios/ExecuTorchDemo/ExecuTorchDemo/Resources/Models/MobileNet/
   ```

2. **Download MobileNet model labels**

   ```bash
   curl https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt \
     -o examples/demo-apps/apple_ios/ExecuTorchDemo/ExecuTorchDemo/Resources/Models/MobileNet/imagenet_classes.txt
   ```

## ExecuTorch & Backend Building

1. **Build frameworks**

   ```bash
   ./build/build_apple_frameworks.sh --Release --coreml --mps --xnnpack
   ```

2. **Move frameworks for app linking**

   ```bash
   mv cmake-out examples/demo-apps/apple_ios/ExecuTorchDemo/ExecuTorchDemo/Frameworks
   ```

## Final Steps

1. **Open project in Xcode**

   ```bash
   open examples/demo-apps/apple_ios/ExecuTorchDemo/ExecuTorchDemo.xcodeproj
   ```

2. **Run tests in Xcode** (Cmd + U) or command line:

   ```bash
   xcrun simctl create executorch "iPhone 15"
   xcodebuild clean test \
        -project examples/demo-apps/apple_ios/ExecuTorchDemo/ExecuTorchDemo.xcodeproj \
        -scheme App \
        -destination name=executorch
   xcrun simctl delete executorch
   ```

3. **Setup Code Signing and run app** (Cmd + R).
