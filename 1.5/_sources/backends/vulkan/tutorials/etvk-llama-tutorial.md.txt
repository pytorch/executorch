# Exporting Llama 3.2 1B/3B Instruct to ExecuTorch Vulkan and running on device

This tutorial assumes that you have a working local copy of the ExecuTorch repo,
and have gone through the steps to install the executorch pip package or have
installed it by building from source.

This tutorial also assumes that you have the Android SDK tools installed and
that you are able to connect to an Android device via `adb`.

Finally, the Android NDK should also be installed, and your environment should
have a variable `ANDROID_NDK` that points to the root directory of the NDK.

```shell
export ANDROID_NDK=<path_to_ndk>
```

## Download the Llama 3.2 1B/3B Instruct model checkpoint and tokenizer

The model checkpoint and tokenizer can be downloaded from the
[Meta Llama website](https://www.llama.com/llama-downloads/).

The model files should be downloaded to `~/.llama/checkpoints/Llama3.2-1B-Instruct`.

## Export the Llama 3.2 1B/3B model

First, navigate to the root of the ExecuTorch repo.

```shell
# Navigate to executorch root
cd ~/executorch
```

Then, set some environment variables to describe how the model should be
exported. Feel free to tune the values to your preferences.

```shell
export LLM_NAME=Llama3.2 && \
export LLM_SIZE=1B && \
export LLM_SUFFIX="-Instruct" && \
export QUANT=8da4w && \
export BACKEND=vulkan && \
export GROUP_SIZE=64 && \
export CONTEXT_LENGTH=2048
```

Then, export the Llama 3.2 1B/3B Instruct model to ExecuTorch Vulkan. Note that
that `--vulkan-force-fp16` flag is set, which will improve model inference
latency at the cost of model accuracy. Feel free to remove this flag.

```shell
python -m examples.models.llama.export_llama \
    -c $HOME/.llama/checkpoints/${LLM_NAME}-${LLM_SIZE}${LLM_SUFFIX}/consolidated.00.pth \
    -p $HOME/.llama/checkpoints/${LLM_NAME}-${LLM_SIZE}${LLM_SUFFIX}/params.json \
    -d fp32 --${BACKEND} \
    -qmode ${QUANT} -G ${GROUP_SIZE} \
    --max_seq_length ${CONTEXT_LENGTH} \
    --max_context_length ${CONTEXT_LENGTH} \
    -kv --use_sdpa_with_kv_cache \
    --metadata '{"append_eos_to_prompt": 0, "get_bos_id":128000, "get_eos_ids":[128009, 128001]}' \
    --model "llama3_2" \
    --output_name $HOME/.llama/checkpoints/${LLM_NAME}-${LLM_SIZE}${LLM_SUFFIX}/${LLM_NAME}-${LLM_SIZE}${LLM_SUFFIX}_${BACKEND}_${QUANT}_g${GROUP_SIZE}_c${CONTEXT_LENGTH}.pte

```

After exporting the model, push the exported `.pte` file and the tokenizer to
your device.

```shell
adb shell mkdir -p /data/local/tmp/llama && \
adb push ~/.llama/checkpoints/${LLM_NAME}-${LLM_SIZE}${LLM_SUFFIX}/tokenizer.model \
  /data/local/tmp/llama/${LLM_NAME}-${LLM_SIZE}${LLM_SUFFIX}_tokenizer.model && \
adb push ~/.llama/checkpoints/${LLM_NAME}-${LLM_SIZE}${LLM_SUFFIX}/${LLM_NAME}-${LLM_SIZE}${LLM_SUFFIX}_${BACKEND}_${QUANT}_g${GROUP_SIZE}_c${CONTEXT_LENGTH}.pte \
  /data/local/tmp/llama/${LLM_NAME}-${LLM_SIZE}${LLM_SUFFIX}_${BACKEND}_${QUANT}_g${GROUP_SIZE}_c${CONTEXT_LENGTH}.pte
```

## Build Core Executorch Components

To be able to run the `.pte` file on device, first the core libraries,
including the Vulkan backend, must be compiled for Android.

```shell
cmake . \
    -DCMAKE_INSTALL_PREFIX=cmake-out-android-so \
    -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
    -DANDROID_SUPPORT_FLEXIBLE_PAGE_SIZES=ON \
    --preset "android-arm64-v8a" \
    -DANDROID_PLATFORM=android-28 \
    -DPYTHON_EXECUTABLE=python \
    -DCMAKE_BUILD_TYPE=Release \
    -DEXECUTORCH_PAL_DEFAULT=posix \
    -DEXECUTORCH_BUILD_LLAMA_JNI=ON \
    -DEXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP=ON \
    -DEXECUTORCH_BUILD_VULKAN=ON \
    -DEXECUTORCH_BUILD_TESTS=OFF \
    -Bcmake-out-android-so && \
cmake --build cmake-out-android-so -j16 --target install --config Release
```

## Build and push the llama runner binary to Android

Then, build a binary that can be used to run the `.pte` file.

```shell
cmake examples/models/llama \
    -DCMAKE_INSTALL_PREFIX=cmake-out-android-so \
    -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake  \
    -DANDROID_SUPPORT_FLEXIBLE_PAGE_SIZES=ON \
    -DEXECUTORCH_ENABLE_LOGGING=ON \
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_PLATFORM=android-28 \
    -DCMAKE_BUILD_TYPE=Release \
    -DPYTHON_EXECUTABLE=python \
    -Bcmake-out-android-so/examples/models/llama && \
cmake --build cmake-out-android-so/examples/models/llama -j16 --config Release
```

Once the binary is built, it can be pushed to your Android device.

```shell
adb shell mkdir /data/local/tmp/etvk/ && \
adb push cmake-out-android-so/examples/models/llama/llama_main /data/local/tmp/etvk/
```

## Execute the llama runner binary

Finally, we can execute the lowered `.pte` file on your device.

```shell
adb shell /data/local/tmp/etvk/llama_main \
  --model_path=/data/local/tmp/llama/${LLM_NAME}-${LLM_SIZE}${LLM_SUFFIX}_${BACKEND}_${QUANT}_g${GROUP_SIZE}_c${CONTEXT_LENGTH}.pte \
  --tokenizer_path=/data/local/tmp/llama/${LLM_NAME}-${LLM_SIZE}${LLM_SUFFIX}_tokenizer.model \
  --temperature=0 --seq_len=400 --warmup \
  --prompt=\"\<\|begin_of_text\|\>\<\|start_header_id\|\>system\<\|end_header_id\|\>Write me a short poem.\<\|eot_id\|\>\<\|start_header_id\|\>assistant\<\|end_header_id\|\>\"
```

Here is some sample output captured from a Galaxy S24:

```shell
E tokenizers:hf_tokenizer.cpp:60] Error parsing json file: [json.exception.parse_error.101] parse error at line 1, column 1: syntax error while parsing value - invalid literal; last read: 'I'
<|begin_of_text|><|start_header_id|>system<|end_header_id|>Write me a short poem.<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Here is a short poem I came up with:

"Moonlight whispers secrets to the night
A gentle breeze that rustles the light
The stars up high, a twinkling show
A peaceful world, where dreams grow slow"

I hope you enjoy it!<|eot_id|>

PyTorchObserver {"prompt_tokens":14,"generated_tokens":54,"model_load_start_ms":1760077800721,"model_load_end_ms":1760077802998,"inference_start_ms":1760077802998,"inference_end_ms":1760077804187,"prompt_eval_end_ms":1760077803162,"first_token_ms":1760077803162,"aggregate_sampling_time_ms":19,"SCALING_FACTOR_UNITS_PER_SECOND":1000}
        Prompt Tokens: 14    Generated Tokens: 54
        Model Load Time:                2.277000 (seconds)
        Total inference time:           1.189000 (seconds)               Rate:  45.416316 (tokens/second)
                Prompt evaluation:      0.164000 (seconds)               Rate:  85.365854 (tokens/second)
                Generated 54 tokens:    1.025000 (seconds)               Rate:  52.682927 (tokens/second)
        Time to first generated token:  0.164000 (seconds)
        Sampling time over 68 tokens:   0.019000 (seconds)
```
