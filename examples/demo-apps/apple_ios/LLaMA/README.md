# Building ExecuTorch Llama and Llava iOS Demo App

This app demonstrates the use of the LLM chat app demonstrating local inference use case with ExecuTorch, using [Llama 3.1](https://github.com/meta-llama/llama-models) for text only chat and [Llava](https://github.com/haotian-liu/LLaVA) for image and text chat.

## Prerequisites
* [Xcode 15](https://developer.apple.com/xcode)
* [iOS 17 SDK](https://developer.apple.com/ios)
* Set up your ExecuTorch repo and environment if you haven’t done so by following the [Setting up ExecuTorch](https://pytorch.org/executorch/stable/getting-started-setup) to set up the repo and dev environment:

```bash
git clone https://github.com/pytorch/executorch.git --recursive && cd executorch
```

Then create a virtual or conda environment using either
```bash
python3 -m venv .venv && source .venv/bin/activate
```
or
```bash
conda create -n executorch python=3.10
conda activate executorch
```

After that, run:
```bash
./install_requirements.sh --pybind coreml mps xnnpack
./backends/apple/coreml/scripts/install_requirements.sh
./backends/apple/mps/install_requirements.sh
```

## Exporting models
Please refer to the [ExecuTorch Llama2 docs](https://github.com/pytorch/executorch/blob/main/examples/models/llama2/README.md) to export the Llama 3.1 model.

## Run the App

1. Open the [project](https://github.com/pytorch/executorch/blob/main/examples/demo-apps/apple_ios/LLaMA/LLaMA.xcodeproj) in Xcode.
2. Run the app (cmd+R).
3. In app UI pick a model and tokenizer to use, type a prompt and tap the arrow buton.

```{note}
ExecuTorch runtime is distributed as a Swift package providing some .xcframework as prebuilt binary targets.
Xcode will dowload and cache the package on the first run, which will take some time.
```

## Copy the model to Simulator

1. Drag and drop the Llama 3.1 and Llava models and tokenizer files onto the Simulator window and save them somewhere inside the iLLaMA folder.
2. Pick the files in the app dialog, type a prompt and click the arrow-up button.

## Copy the model to Device

1. Wire-connect the device and open the contents in Finder.
2. Navigate to the Files tab and drag and drop the models and tokenizer files onto the iLLaMA folder.
3. Wait until the files are copied.

Click the image below to see a demo video of the app running Llama 3.1 and Llava on an iPhone 15 Pro device:

<a href="https://drive.google.com/file/d/1yQ7UoB79vMEBuBaoYvO53dosYTjpOZhd/view?usp=sharing">
  <img src="llama31.png" width="350" alt="iOS app running Llama 3.1">
</a> <a href="https://drive.google.com/file/d/1yQ7UoB79vMEBuBaoYvO53dosYTjpOZhd/view?usp=sharing">
  <img src="llava.png" width="350" alt="iOS app running Llava">
</a>

## Reporting Issues
If you encountered any bugs or issues following this tutorial please file a bug/issue here on [Github](https://github.com/pytorch/executorch/issues/new).
