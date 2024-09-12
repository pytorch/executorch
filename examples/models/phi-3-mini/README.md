# Summary
This example demonstrates how to run a [Phi-3-mini](https://huggingface.co/microsoft/Phi-3-mini-128k-instruct) 3.8B model via ExecuTorch. We use XNNPACK to accelarate the performance and XNNPACK symmetric per channel quantization.

# Instructions
## Step 1: Setup
1. Follow the [tutorial](https://pytorch.org/executorch/main/getting-started-setup) to set up ExecuTorch. For installation run `./install_requirements.sh --pybind xnnpack`
2. Currently, we support transformers v4.44.2. Install transformers with the following command:
```
pip uninstall -y transformers ; pip install transformers==4.44.2
```
## Step 2: Prepare and run the model
1. Download the `tokenizer.model` from HuggingFace and create `tokenizer.bin`.
```
cd executorch
wget -O tokenizer.model "https://huggingface.co/microsoft/Phi-3-mini-128k-instruct/resolve/main/tokenizer.model?download=true"
python -m extension.llm.tokenizer.tokenizer -t tokenizer.model -o tokenizer.bin
```
2. Export the model. This step will take a few minutes to finish.
```
python -m examples.models.phi-3-mini.export_phi-3-mini -c "4k" -s 128 -o phi-3-mini.pte
```
3. Build and run the model.
- Build executorch with optimized CPU performance as follows. Build options available [here](https://github.com/pytorch/executorch/blob/main/CMakeLists.txt#L59).
 ```
 cmake -DPYTHON_EXECUTABLE=python \
     -DCMAKE_INSTALL_PREFIX=cmake-out \
     -DEXECUTORCH_ENABLE_LOGGING=1 \
     -DCMAKE_BUILD_TYPE=Release \
     -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
     -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
     -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
     -DEXECUTORCH_BUILD_XNNPACK=ON \
     -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON \
     -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON \
     -DEXECUTORCH_BUILD_KERNELS_CUSTOM=ON \
     -Bcmake-out .

 cmake --build cmake-out -j16 --target install --config Release
 ```
- Build Phi-3-mini runner.
```
cmake -DPYTHON_EXECUTABLE=python \
    -DCMAKE_INSTALL_PREFIX=cmake-out \
    -DCMAKE_BUILD_TYPE=Release \
    -DEXECUTORCH_BUILD_KERNELS_CUSTOM=ON \
    -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON \
    -DEXECUTORCH_BUILD_XNNPACK=ON \
    -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON \
    -Bcmake-out/examples/models/phi-3-mini \
    examples/models/phi-3-mini

cmake --build cmake-out/examples/models/phi-3-mini -j16 --config Release
```
- Run model. Options available [here](https://github.com/pytorch/executorch/blob/main/examples/models/phi-3-mini/main.cpp#L13-L30)
```
cmake-out/examples/models/phi-3-mini/phi_3_mini_runner \
    --model_path=phi-3-mini.pte \
    --tokenizer_path=tokenizer.bin \
    --seq_len=128 \
    --temperature=0 \
    --prompt="<|system|>
You are a helpful assistant.<|end|>
<|user|>
What is the capital of France?<|end|>
<|assistant|>"
```
