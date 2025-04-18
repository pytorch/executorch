# ExecuTorch Llama iOS Demo App

Get hands-on with running LLaMA and LLaVA models — exported via ExecuTorch — natively on your iOS device!

*Click the image below to see it in action!*

<p align="center">
<a href="https://pytorch.org/executorch/main/_static/img/llama_ios_app.mp4">
  <img src="https://pytorch.org/executorch/main/_static/img/llama_ios_app.png" width="600" alt="iOS app running a LlaMA model">
</a>
</p>

## Requirements
- [Xcode](https://apps.apple.com/us/app/xcode/id497799835?mt=12/) 15.0 or later
- [Cmake](https://cmake.org/download/) 3.19 or later
  - Download and open the macOS `.dmg` installer and move the Cmake app to `/Applications` folder.
  - Install Cmake command line tools: `sudo /Applications/CMake.app/Contents/bin/cmake-gui --install`
- A development provisioning profile with the [`increased-memory-limit`](https://developer.apple.com/documentation/bundleresources/entitlements/com_apple_developer_kernel_increased-memory-limit) entitlement.

## Models

Download already exported LLaMA/LLaVA models along with tokenizers from [HuggingFace](https://huggingface.co/executorch-community) or export your own empowered by [XNNPACK](docs/delegates/xnnpack_README.md) or [MPS](docs/delegates/mps_README.md) backends.

## Build and Run

1. Make sure git submodules are up-to-date:
   ```bash
   git submodule update --init --recursive
   ```

2. Open the Xcode project:
    ```bash
    open examples/demo-apps/apple_ios/LLaMA/LLaMA.xcodeproj
    ```
    
3. Click the Play button to launch the app in the Simulator.

4. To run on a device, ensure you have it set up for development and a provisioning profile with the `increased-memory-limit` entitlement. Update the app's bundle identifier to match your provisioning profile with the required capability.

5. After successfully launching the app, copy the exported ExecuTorch model (`.pte`) and tokenizer (`.model`) files to the iLLaMA folder.

    - **For the Simulator:** Drag and drop both files onto the Simulator window and save them in the `On My iPhone > iLLaMA` folder.
    - **For a Device:** Open a separate Finder window, navigate to the Files tab, drag and drop both files into the iLLaMA folder, and wait for the copying to finish.

6. Follow the app's UI guidelines to select the model and tokenizer files from the local filesystem and issue a prompt.

For more details check out the [Using ExecuTorch on iOS](../../../../docs/source/using-executorch-ios.md) page.
