./install_executorch.sh --pybind xnnpack

examples/models/llama/install_requirements.sh

cmake -DPYTHON_EXECUTABLE=python \
    -DCMAKE_INSTALL_PREFIX=cmake-out \
    -DEXECUTORCH_ENABLE_LOGGING=1 \
    -DCMAKE_BUILD_TYPE=Debug \
    -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
    -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
    -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
    -DEXECUTORCH_BUILD_XNNPACK=ON \
    -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON \
    -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON \
    -DEXECUTORCH_BUILD_KERNELS_CUSTOM=ON \
    -Bcmake-out .

cmake --build cmake-out -j16 --target install --config Debug

cmake -DPYTHON_EXECUTABLE=python \
    -DCMAKE_INSTALL_PREFIX=cmake-out \
    -DCMAKE_BUILD_TYPE=Debug \
    -DEXECUTORCH_BUILD_KERNELS_CUSTOM=ON \
    -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON \
    -DEXECUTORCH_BUILD_XNNPACK=ON \
    -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON \
    -Bcmake-out/examples/models/llama \
    examples/models/llama

cmake --build cmake-out/examples/models/llama -j16 --config Debug

cmake-out/examples/models/llama/llama_main --model_path=llama3_2_plain.pte --tokenizer_path=../llama3b/tokenizer.model --prompt="Hi"
cmake-out/examples/models/llama/llama_main --model_path=llama3_2_lora.pte --tokenizer_path=../llama3b/tokenizer.model --prompt="Hi"

# Other prompts
# What happens if you eat watermelon seeds?

-- export
# No quantization
# Set these paths to point to the downloaded files
LLAMA_CHECKPOINT=../llama3b/consolidated.00.pth
LLAMA_PARAMS=../llama3b/params.json

python -m examples.models.llama.export_llama \
  --model "llama3_2" \
  --checkpoint "${LLAMA_CHECKPOINT:?}" \
  --params "${LLAMA_PARAMS:?}" \
  -kv \
  --use_sdpa_with_kv_cache \
  -d bf16 \
  --metadata '{"get_bos_id":128000, "get_eos_ids":[128009, 128001]}' \
  --output_name="llama3_2.pte"


Notes:
- We can't use llama_transformer, unless we update the model definition with additional layers.
- When using custom export script, we need to take in the checkpoint+adapter files.
- Check that eager works with consolidated.00.pth and adapter.pt?
