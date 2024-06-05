# Summary
This example demonstrates how to run a [Phi-3-mini](https://huggingface.co/microsoft/Phi-3-mini-128k-instruct) 3.8B model via ExecuTorch. We use XNNPACK to accelarate the performance and XNNPACK symmetric per channel quantization.

# Instructions
## Step 1: Setup
1. Follow the [tutorial](https://pytorch.org/executorch/main/getting-started-setup) to set up ExecuTorch. For installation run `./install_requirements.sh --pybind xnnpack`
2. Phi-3 Mini-128K-Instruct has been integrated in the development version (4.41.0.dev0) of transformers. Make sure that you install transformers with version at least 4.41.0: `pip uninstall -y transformers && pip install git+https://github.com/huggingface/transformers`


## Step 2: Prepare and run the model
1. Download the `tokenizer.model` from HuggingFace.
```
cd examples/models/phi-3-mini
wget -O tokenizer.model https://huggingface.co/microsoft/Phi-3-mini-128k-instruct/resolve/main/tokenizer.model?download=true
```
2. Export the model. This step will take a few minutes to finish.
```
python export_model.py
```
3. Build and run the runner.
```
mkdir cmake-out
cd cmake-out
cmake ..
cd ..
cmake --build cmake-out -j10
./cmake-out/phi_3_mini_runner
```
