# Building Llama iOS Demo for XNNPack Backend

**[UPDATE - 09/25]** We have added support for running [Llama 3.2 models](#for-llama-32-1b-and-3b-models) on the XNNPack backend. We currently support inference on their original data type (BFloat16).

This tutorial covers the end to end workflow for building an iOS demo app using XNNPack backend on device.
More specifically, it covers:
1. Export and quantization of Llama models against the XNNPack backend.
2. Building and linking libraries that are required to inference on-device for iOS platform using XNNPack.
3. Building the iOS demo app itself.

## Prerequisites
* [Xcode 15](https://developer.apple.com/xcode)
* [iOS 17 SDK](https://developer.apple.com/ios)
* Set up your ExecuTorch repo and environment if you haven’t done so by following the [Setting up ExecuTorch](https://pytorch.org/executorch/stable/getting-started-setup) to set up the repo and dev environment:

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
We have supported BFloat16 as a data type on the XNNPack backend for Llama 3.2 1B/3B models.
* You can download original model weights for Llama through Meta official [website](https://llama.meta.com/).
* For chat use-cases, download the instruct models instead of pretrained.
* Run “examples/models/llama2/install_requirements.sh” to install dependencies.
* The 1B model in BFloat16 format can run on mobile devices with 8GB RAM (iPhone 15 Pro and later). The 3B model will require 12GB+ RAM and hence will not fit on 8GB RAM phones.
* Export Llama model and generate .pte file as below:

```
python -m examples.models.llama2.export_llama --checkpoint <checkpoint.pth> --params <params.json> -kv -X -d bf16 --metadata '{"get_bos_id":128000, "get_eos_ids":[128009, 128001]}' --output_name="llama3_2.pte"
```

* Convert tokenizer for Llama 3.2 - Rename 'tokenizer.model' to 'tokenizer.bin'.

For more detail using Llama 3.2 lightweight models including prompt template, please go to our official [website](https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_2#-llama-3.2-lightweight-models-(1b/3b)-).

### For Llama 3.1 and Llama 2 models

Export the model
```
python -m examples.models.llama2.export_llama --checkpoint <consolidated.00.pth> -p <params.json> -kv --use_sdpa_with_kv_cache -X -qmode 8da4w  --group_size 128 -d fp32 --metadata '{"get_bos_id":128000, "get_eos_ids":[128009, 128001]}' --embedding-quantize 4,32 --output_name="llama3_kv_sdpa_xnn_qe_4_32.pte"
```

## Pushing Model and Tokenizer

### Copy the model to Simulator
* Drag&drop the model and tokenizer files onto the Simulator window and save them somewhere inside the iLLaMA folder.
* Pick the files in the app dialog, type a prompt and click the arrow-up button.

### Copy the model to Device
* Wire-connect the device and open the contents in Finder.
* Navigate to the Files tab and drag & drop the model and tokenizer files onto the iLLaMA folder.
* Wait until the files are copied.

## Configure the XCode Project

### Install CMake
Download and open the macOS .dmg installer at https://cmake.org/download and move the Cmake app to /Applications folder.
Install Cmake command line tools:

```
sudo /Applications/CMake.app/Contents/bin/cmake-gui --install
```


### Swift Package Manager
The prebuilt ExecuTorch runtime, backend, and kernels are available as a Swift PM package.

### Xcode
Open the project in Xcode.In Xcode, go to `File > Add Package Dependencies`. Paste the URL of the ExecuTorch repo into the search bar and select it. Make sure to change the branch name to the desired ExecuTorch version, e.g., “0.3.0”, or just use the “latest” branch name for the latest stable build.

Note: If you're running into any issues related to package dependencies, quit Xcode entirely, delete the whole executorch repo, clean the caches by running the command below in terminal and clone the repo again.

```
rm -rf \
  ~/Library/org.swift.swiftpm \
  ~/Library/Caches/org.swift.swiftpm \
  ~/Library/Caches/com.apple.dt.Xcode \
  ~/Library/Developer/Xcode/DerivedData
```

Link your binary with the ExecuTorch runtime and any backends or kernels used by the exported ML model. It is recommended to link the core runtime to the components that use ExecuTorch directly, and link kernels and backends against the main app target.

Note: To access logs, link against the Debug build of the ExecuTorch runtime, i.e., the executorch_debug framework. For optimal performance, always link against the Release version of the deliverables (those without the _debug suffix), which have all logging overhead removed.

For more details integrating and Running ExecuTorch on Apple Platforms, checkout this [link](https://pytorch.org/executorch/main/apple-runtime.html).

<p align="center">
<img src="https://raw.githubusercontent.com/pytorch/executorch/refs/heads/main/docs/source/_static/img/ios_demo_app_swift_pm.png" alt="iOS LLaMA App Swift PM" style="width:600px">
</p>

Then select which ExecuTorch framework should link against which target.

<p align="center">
<img src="https://raw.githubusercontent.com/pytorch/executorch/refs/heads/main/docs/source/_static/img/ios_demo_app_choosing_package.png" alt="iOS LLaMA App Choosing package" style="width:600px">
</p>

Click “Run” to build the app and run in on your iPhone. If the app successfully run on your device, you should see something like below:

<p align="center">
<img src="https://raw.githubusercontent.com/pytorch/executorch/refs/heads/main/docs/source/_static/img/ios_demo_app.jpg" alt="iOS LLaMA App" style="width:300px">
</p>

For Llava 1.5 models, you can select and image (via image/camera selector button) before typing prompt and send button.

<p align="center">
<img src="https://raw.githubusercontent.com/pytorch/executorch/refs/heads/main/docs/source/_static/img/ios_demo_app_llava.jpg" alt="iOS LLaMA App" style="width:300px">
</p>

## Reporting Issues
If you encountered any bugs or issues following this tutorial please file a bug/issue here on [Github](https://github.com/pytorch/executorch/issues/new).
