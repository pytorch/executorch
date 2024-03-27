# Experimental Features

This subdirectory is under heavy development so proceed with caution.

We are demonstrating how we can import a llama model in gguf format back into PyTorch/ExecuTorch world, run it and perform different optimizations using PyTorch/ExecuTorch APIs.

The first and most important step would be loading a gguf model into PyTorch.

## Load GGUF Q4_0 Quantized Model

Let's say we've went through the process of building and running llama.cpp and was able to quantize a Llama model using the following command:

```bash
# checkpoint download to models/llama7b
<omitted>
# build
mkdir build
cd build
cmake ..
cmake --build . --config Release

# prepare model
python3 convert.py models/llama7b

# quantize. Notice we use --pure to avoid Q6_K from showing up.
build/bin/quantize --pure models/llama7b/ggml-model-f16.gguf models/llama7b/ggml-model-Q4_0.gguf Q4_0

```

We want to load it back into a `torch.nn.Module` and run in eager mode. The way it works is through a Tensor subclass.


## Generate Tokens in PyTorch Eager
```bash
python3 generate.py --prompt "Once upon a time" --gguf_file models/llama7b/ggml-model-Q4_0.gguf --tokenizer_path models/llama7b/tokenizer.model
```
