# Building ExecuTorch LLaMA iOS Demo App

This app demonstrates the use of the LLaMA chat app demonstrating local inference use case with ExecuTorch.

## Prerequisites
* [Xcode 15](https://developer.apple.com/xcode)
* [iOS 17 SDK](https://developer.apple.com/ios)
* Set up your ExecuTorch repo and environment if you haven’t done so by following the [Setting up ExecuTorch](https://pytorch.org/executorch/stable/getting-started-setup) to set up the repo and dev environment:

```bash
git clone -b release/0.2 https://github.com/pytorch/executorch.git
cd executorch
git submodule update --init

python3 -m venv .venv && source .venv/bin/activate

./install_requirements.sh
```

## Exporting models
Please refer to the [ExecuTorch Llama2 docs](https://github.com/pytorch/executorch/blob/main/examples/models/llama2/README.md) to export the model.

## Run the App

1. Open the [project](https://github.com/pytorch/executorch/blob/main/examples/demo-apps/apple_ios/LLaMA/LLaMA.xcodeproj) in Xcode.
2. Run the app (cmd+R).
3. In app UI pick a model and tokenizer to use, type a prompt and tap the arrow buton

```{note}
ExecuTorch runtime is distributed as a Swift package providing some .xcframework as prebuilt binary targets.
Xcode will dowload and cache the package on the first run, which will take some time.
```

## Copy the model to Simulator

1. Drag&drop the model and tokenizer files onto the Simulator window and save them somewhere inside the iLLaMA folder.
2. Pick the files in the app dialog, type a prompt and click the arrow-up button.

## Copy the model to Device

1. Wire-connect the device and open the contents in Finder.
2. Navigate to the Files tab and drag&drop the model and tokenizer files onto the iLLaMA folder.
3. Wait until the files are copied.

Click the image below to see it in action!

<a href="https://pytorch.org/executorch/main/_static/img/llama_ios_app.mp4">
  <img src="https://pytorch.org/executorch/main/_static/img/llama_ios_app.png" width="600" alt="iOS app running a LlaMA model">
</a>

## Reporting Issues
If you encountered any bugs or issues following this tutorial please file a bug/issue here on [Github](https://github.com/pytorch/executorch/issues/new).
