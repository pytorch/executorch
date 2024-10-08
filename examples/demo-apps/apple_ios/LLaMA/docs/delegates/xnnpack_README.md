# Building Llama iOS Demo for XNNPACK Backend

**[UPDATE - 09/25]** We have added support for running [Llama 3.2 models](#for-llama-32-1b-and-3b-models) on the XNNPACK backend. We currently support inference on their original data type (BFloat16).

This tutorial covers the end to end workflow for building an iOS demo app using XNNPACK backend on device. More specifically, it covers:
1. Export and quantization of Llama models against the XNNPACK backend.
2. Building and linking libraries that are required to inference on-device for iOS platform using XNNPACK.
3. Building the iOS demo app itself.

## Prerequisites
* [Xcode 15](https://developer.apple.com/xcode)
* [iOS 17 SDK](https://developer.apple.com/ios)

## Setup ExecuTorch
In this section, we will need to set up the ExecuTorch repo first with Conda environment management. Make sure you have Conda available in your system (or follow the instructions to install it [here](https://conda.io/projects/conda/en/latest/user-guide/install/index.html)). The commands below are running on Linux (CentOS).

Create a Conda environment

```
conda create -n et_xnnpack python=3.10.0
conda activate et_xnnpack
```

Checkout ExecuTorch repo and sync submodules

```
git clone https://github.com/pytorch/executorch.git
cd executorch
git submodule sync
git submodule update --init
```

Install dependencies

```
./install_requirements.sh
```

## Prepare Models
In this demo app, we support text-only inference with up-to-date Llama models.

Install the required packages to export the model

```
sh examples/models/llama2/install_requirements.sh
```

### For Llama 3.2 1B and 3B models
We have supported BFloat16 as a data type on the XNNPACK backend for Llama 3.2 1B/3B models.
* You can download original model weights for Llama through Meta official [website](https://llama.meta.com/).
* For chat use-cases, download the instruct models instead of pretrained.
* Run “examples/models/llama2/install_requirements.sh” to install dependencies.
* The 1B model in BFloat16 format can run on mobile devices with 8GB RAM (iPhone 15 Pro and later). The 3B model will require 12GB+ RAM and hence will not fit on 8GB RAM phones.
* Export Llama model and generate .pte file as below:

```
python -m examples.models.llama2.export_llama --checkpoint <checkpoint.pth> --params <params.json> -kv -X -d bf16 --metadata '{"get_bos_id":128000, "get_eos_ids":[128009, 128001]}' --output_name="llama3_2.pte"
```

For more detail using Llama 3.2 lightweight models including prompt template, please go to our official [website](https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_2#-llama-3.2-lightweight-models-(1b/3b)-).

### For Llama 3.1 and Llama 2 models

Export the model
```
python -m examples.models.llama2.export_llama --checkpoint <consolidated.00.pth> -p <params.json> -kv --use_sdpa_with_kv_cache -X -qmode 8da4w  --group_size 128 -d fp32 --metadata '{"get_bos_id":128000, "get_eos_ids":[128009, 128001]}' --embedding-quantize 4,32 --output_name="llama3_kv_sdpa_xnn_qe_4_32.pte"
```

### For LLaVA model
* For the Llava 1.5 model, you can get it from Huggingface [here](https://huggingface.co/llava-hf/llava-1.5-7b-hf).
* Run `examples/models/llava/install_requirements.sh` to install dependencies.
* Run the following command to generate llava.pte, tokenizer.bin and an image tensor (serialized in TorchScript) image.pt.

```
python -m executorch.examples.models.llava.export_llava --pte-name llava.pte --with-artifacts
```
* You can find more information [here](https://github.com/pytorch/executorch/tree/main/examples/models/llava).


## Configure the XCode Project

### 1. Install CMake
Download and open the macOS .dmg installer at https://cmake.org/download and move the Cmake app to /Applications folder.
Install Cmake command line tools:

```
sudo /Applications/CMake.app/Contents/bin/cmake-gui --install
```

### 2. Add ExecuTorch Runtime Package

There are two options to add ExecuTorch runtime package into your XCode project:

- [Recommended] Prebuilt package (via Swift Package Manager)
- Manually build the package locally and link them


### 2.1 [Recommended] Prebuilt package (via Swift Package Manager)

The current XCode project is pre-configured to automatically download and link the latest prebuilt package via Swift Package Manager.

If you have an old ExecuTorch package cached before in XCode, or are running into any package dependencies issues (incorrect checksum hash, missing package, outdated package), close XCode and run the following command in terminal inside your ExecuTorch directory

```
rm -rf \
  ~/Library/org.swift.swiftpm \
  ~/Library/Caches/org.swift.swiftpm \
  ~/Library/Caches/com.apple.dt.Xcode \
  ~/Library/Developer/Xcode/DerivedData \
  examples/demo-apps/apple_ios/LLaMA/LLaMA.xcodeproj/project.xcworkspace/xcshareddata/swiftpm
```

The command above will clear all the package cache, and when you re-open the XCode project, it should re-download the latest package and link them correctly.

#### (Optional) Changing the prebuilt package version
While we recommended using the latest prebuilt package pre-configured with the XCode project, you can also change the package version manually to your desired version.

Go to Project Navigator, click on LLaMA. `Project --> LLaMA --> Package Dependencies`, and update the package dependencies to any of the available options below:

- Branch --> latest
- Branch --> 0.4.0
- Branch --> 0.3.0
- Commit --> (Specify the commit hash, for example: `bdf3f5a1047c73ef61bb3e956d1d4528de743077`. Full list [here](https://github.com/pytorch/executorch/commits/latest/))


### 2.2 Manually build the package locally and link them

Note: You should only use this step if the prebuilt package doesn't work for your usecase (For example, you require the latest PRs from main, where there are no pre-built package yet)

If you need to manually build the package, run the following command in your terminal
```
# Install a compatible version of Buck2
BUCK2_RELEASE_DATE="2024-05-15"
BUCK2_ARCHIVE="buck2-aarch64-apple-darwin.zst"
BUCK2=".venv/bin/buck2"

curl -LO "https://github.com/facebook/buck2/releases/download/$BUCK2_RELEASE_DATE/$BUCK2_ARCHIVE"
zstd -cdq "$BUCK2_ARCHIVE" > "$BUCK2" && chmod +x "$BUCK2"
rm "$BUCK2_ARCHIVE"

./build/build_apple_frameworks.sh --buck2="$(realpath $BUCK2)" --coreml --custom --mps --optimized --portable --quantized --xnnpack
```

 After the build finishes successfully, the resulting frameworks can be found in the `cmake-out` directory. Copy them to your project and link them against your targets.

The following packages should be linked in your app target `LLaMA` (left side, LLaMA --> General --> select LLaMA under "TARGETS" --> scroll down to "Frameworks, Libraries, and Embedded Content")
- backend_coreml
- backend_mps
- backend_xnnpack
- kernels_custom
- kernels_optimized
- kernels_portable
- kernels_quantized

The following package should be linked in your target app `LLaMARunner` (left side, LLaMA --> General --> select LLaMARunner under "TARGETS" --> scroll down to "Frameworks and Libraries")
- executorch

<p align="center">
<img src="https://raw.githubusercontent.com/pytorch/executorch/refs/heads/main/docs/source/_static/img/ios_demo_app_choosing_package.png" alt="iOS LLaMA App Choosing package" style="width:600px">
</p>

If you cannot add the package into your app target (it's greyed out), it might have been linked before. You can verify it by checking it from your target app `(LLaMA / LLaMARunner) --> Build Phases --> Link Binary With Libraries`.



 More details on integrating and Running ExecuTorch on Apple Platforms, check out the detailed guide [here](https://pytorch.org/executorch/main/apple-runtime.html#local-build).

### 3. Configure Build Schemes

The project has two build configurations:
- Debug
- [Recommended] Release

Navigate to `Product --> Scheme --> Edit Scheme --> Info --> Build Configuration` and update the configuration to "Release".

We recommend that you only use the Debug build scheme during development, where you might need to access additional logs. Debug build has logging overhead and will impact inferencing performance, while release build has compiler optimizations enabled and all logging overhead removed.

For more details integrating and Running ExecuTorch on Apple Platforms or building the package locally, checkout this [link](https://pytorch.org/executorch/main/apple-runtime.html).

### 4. Build and Run the project

Click the "play" button on top right of your XCode app, or navigate to `Product --> Run` to build and run the app on your phone.

### 5. Pushing Model and Tokenizer

There are two options to copy the model (.pte) and tokenizer files (.model) to your app, depending on whether you are running it on a simulator or device.

#### 5.1 Copy the model and tokenizer to Simulator
* Drag&drop the model and tokenizer files onto the Simulator window and save them somewhere inside the iLLaMA folder.
* Pick the files in the app dialog, type a prompt and click the arrow-up button.

#### 5.2 Copy the model and tokenizer to Device
* Plug the device into your Mac and open the contents in Finder.
* Navigate to the Files tab and drag & drop the model and tokenizer files onto the iLLaMA folder.
* Wait until the files are copied.

### 6. Try out the app
Open the iLLaMA app, click the settings button at the top left of the app to select the model and tokenizer files. When the app successfully runs on your device, you should see something like below:

<p align="center">
<img src="https://raw.githubusercontent.com/pytorch/executorch/refs/heads/main/docs/source/_static/img/ios_demo_app.jpg" alt="iOS LLaMA App" style="width:300px">
</p>


For Llava 1.5 models, you can select and image (via image/camera selector button) before typing prompt and send button.

<p align="center">
<img src="https://raw.githubusercontent.com/pytorch/executorch/refs/heads/main/docs/source/_static/img/ios_demo_app_llava.jpg" alt="iOS LLaMA App" style="width:300px">
</p>

## Reporting Issues
If you encountered any bugs or issues following this tutorial please file a bug/issue here on [Github](https://github.com/pytorch/executorch/issues/new).
