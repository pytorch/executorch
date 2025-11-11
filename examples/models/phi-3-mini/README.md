# Summary
This example demonstrates how to run a [Phi-3-mini](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct) 3.8B model via ExecuTorch. We use XNNPACK to accelarate the performance and XNNPACK symmetric per channel quantization.

# Instructions
## Step 1: Setup
1. Follow the [tutorial](https://pytorch.org/executorch/main/getting-started-setup) to set up ExecuTorch. For installation run `./install_executorch.sh`
2. Currently, we support transformers v4.56.1. Install transformers with the following command:
```
pip uninstall -y transformers ; pip install transformers==4.56.1
```
3. Install `optimum-executorch`:

```
OPTIMUM_ET_VERSION=$(cat .ci/docker/ci_commit_pins/optimum-executorch.txt)
pip install git+https://github.com/huggingface/optimum-executorch.git@${OPTIMUM_ET_VERSION}
```

## Step 2: Prepare and run the model
1. Download the `tokenizer.model` from HuggingFace.
```
cd executorch
wget -O tokenizer.model "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct/resolve/main/tokenizer.model?download=true"
```
2. Export the model. This step will take a few minutes to finish.
```
optimum-cli export executorch --model microsoft/Phi-3-mini-4k-instruct --task text-generation --recipe xnnpack --qlinear 8da4w --qembedding 8w --output_dir ./
```
The model artifact `model.pte` size is about 2.0GB.

3. Build and run the model.
- Build executorch with LLM preset:
```
cmake --preset llm -DCMAKE_INSTALL_PREFIX=cmake-out

cmake --build cmake-out -j16 --target install --config Release
```
- Build Phi-3-mini runner.
```
cmake -DCMAKE_PREFIX_PATH=cmake-out \
      -DCMAKE_BUILD_TYPE=Release \
      -Bcmake-out/examples/models/phi-3-mini \
      examples/models/phi-3-mini

cmake --build cmake-out/examples/models/phi-3-mini -j16 --config Release
```
- Run model. Options available [here](https://github.com/pytorch/executorch/blob/main/examples/models/phi-3-mini/main.cpp#L16-L33)
```
cmake-out/examples/models/phi-3-mini/phi_3_mini_runner \
    --model_path=model.pte \
    --tokenizer_path=tokenizer.model \
    --seq_len=60 \
    --temperature=0 \
    --prompt="<|system|>
You are a helpful assistant.<|end|>
<|user|>
What is the capital of France?<|end|>
<|assistant|>"
```
