# Building and Running Llama 3 8B Instruct with Qualcomm AI Engine Direct Backend

This tutorial demonstrates how to export Llama 3 8B Instruct for Qualcomm AI Engine Direct Backend and running the model on a Qualcomm device.

## Prerequisites

- Set up your ExecuTorch repo and environment if you haven’t done so by following [the Setting up ExecuTorch](../getting-started-setup.md) to set up the repo and dev environment.
- Read [the Building and Running ExecuTorch with Qualcomm AI Engine Direct Backend page](../build-run-qualcomm-ai-engine-direct-backend.md) to understand how to export and run a model with Qualcomm AI Engine Direct Backend on Qualcomm device.
- Follow [the README for executorch llama](https://github.com/pytorch/executorch/tree/main/examples/models/llama2) to know how to run a llama model on mobile via ExecuTorch.
- A Qualcomm device with 16GB RAM
  - We are continuing to optimize our memory usage to ensure compatibility with lower memory devices.
- The version of [Qualcomm AI Engine Direct SDK](https://developer.qualcomm.com/software/qualcomm-ai-engine-direct-sdk) is 2.26.0 or above.

## Instructions

### Step1: Prepare the checkpoint of the model and optimized matrix from [Spin Quant](https://github.com/facebookresearch/SpinQuant)

1. For Llama 3 tokenizer and checkpoint, please refer to https://github.com/meta-llama/llama-models/blob/main/README.md for further instructions on how to download `tokenizer.model`, `consolidated.00.pth` and `params.json`.
2. To get the optimized matrix, please refer to [SpinQuant on GitHub](https://github.com/facebookresearch/SpinQuant). You can download the optimized rotation matrices in the Quantized Models section. Please choose **LLaMA-3-8B/8B_W4A16KV16_lr_1.5_seed_0**.

### Step2: Export to ExecuTorch with Qualcomm AI Engine Direct Backend
Deploying large language models like Llama 3 on-device presents the following challenges:

1. The model size is too large to fit in device memory for inference.
2. High model loading and inference time.
3. Difficulty in quantization.

To address these challenges, we have implemented the following solutions:
1. Using `--pt2e_quantize qnn_16a4w` to quantize activations and weights, thereby reducing the on-disk model size and alleviating memory pressure during inference.
2. Using `--num_sharding 8` to shard the model into sub-parts.
3. Performing graph transformations to convert or decompose operations into more accelerator-friendly operations.
4. Using `--optimized_rotation_path <path_to_optimized_matrix>` to apply R1 and R2 of [Spin Quant](https://github.com/facebookresearch/SpinQuant) to improve accuracy.
5. Using `--calibration_data "<|start_header_id|>system<|end_header_id|..."` to ensure that during the quantization of Llama 3 8B instruct, the calibration includes special tokens in the prompt template. For more details on the prompt template, refer to [the model card of meta llama3 instruct](https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3/).

To export Llama 3 8B instruct with the Qualcomm AI Engine Direct Backend, ensure the following:

1. The host machine has more than 100GB of memory (RAM + swap space).
2. The entire process takes a few hours.

```bash
# Please note that calibration_data must include the prompt template for special tokens.
python -m examples.models.llama2.export_llama  -t <path_to_tokenizer.model>
llama3/Meta-Llama-3-8B-Instruct/tokenizer.model -p <path_to_params.json> -c <path_to_checkpoint_for_Meta-Llama-3-8B-Instruct>  --use_kv_cache  --qnn --pt2e_quantize qnn_16a4w --disable_dynamic_shape --num_sharding 8 --calibration_tasks wikitext --calibration_limit 1 --calibration_seq_length 128 --optimized_rotation_path <path_to_optimized_matrix> --calibration_data "<|start_header_id|>system<|end_header_id|>\n\nYou are a funny chatbot.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nCould you tell me about Facebook?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
```

### Step3: Invoke the Runtime on an Android smartphone with Qualcomm SoCs
1. Build executorch with Qualcomm AI Engine Direct Backend for android
    ```bash
    cmake \
        -DCMAKE_TOOLCHAIN_FILE="${ANDROID_NDK_ROOT}/build/cmake/android.toolchain.cmake" \
        -DANDROID_ABI=arm64-v8a \
        -DANDROID_PLATFORM=android-23 \
        -DCMAKE_INSTALL_PREFIX=cmake-android-out \
        -DCMAKE_BUILD_TYPE=Release \
        -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
        -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
        -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
        -DEXECUTORCH_BUILD_QNN=ON \
        -DQNN_SDK_ROOT=${QNN_SDK_ROOT} \
        -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON \
        -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON \
        -DEXECUTORCH_BUILD_KERNELS_CUSTOM=ON \
        -Bcmake-android-out .

    cmake --build cmake-android-out -j16 --target install --config Release
    ```
2. Build llama runner for android
```bash
    cmake \
        -DCMAKE_TOOLCHAIN_FILE="${ANDROID_NDK_ROOT}"/build/cmake/android.toolchain.cmake  \
        -DANDROID_ABI=arm64-v8a \
        -DANDROID_PLATFORM=android-23 \
        -DCMAKE_INSTALL_PREFIX=cmake-android-out \
        -DCMAKE_BUILD_TYPE=Release -DPYTHON_EXECUTABLE=python \
        -DEXECUTORCH_BUILD_QNN=ON \
        -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON \
        -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON \
        -DEXECUTORCH_BUILD_KERNELS_CUSTOM=ON \
        -Bcmake-android-out/examples/models/llama2 examples/models/llama2

    cmake --build cmake-android-out/examples/models/llama2 -j16 --config Release
```
3. Run on Android via adb shell
*Pre-requisite*: Make sure you enable USB debugging via developer options on your phone

**3.1 Connect your android phone**

**3.2 We need to push required QNN libraries to the device.**
```bash
# make sure you have write-permission on below path.
DEVICE_DIR=/data/local/tmp/llama
adb shell mkdir -p ${DEVICE_DIR}
adb push ${QNN_SDK_ROOT}/lib/aarch64-android/libQnnHtp.so ${DEVICE_DIR}
adb push ${QNN_SDK_ROOT}/lib/aarch64-android/libQnnSystem.so ${DEVICE_DIR}
adb push ${QNN_SDK_ROOT}/lib/aarch64-android/libQnnHtpV69Stub.so ${DEVICE_DIR}
adb push ${QNN_SDK_ROOT}/lib/aarch64-android/libQnnHtpV73Stub.so ${DEVICE_DIR}
adb push ${QNN_SDK_ROOT}/lib/aarch64-android/libQnnHtpV75Stub.so ${DEVICE_DIR}
adb push ${QNN_SDK_ROOT}/lib/hexagon-v69/unsigned/libQnnHtpV69Skel.so ${DEVICE_DIR}
adb push ${QNN_SDK_ROOT}/lib/hexagon-v73/unsigned/libQnnHtpV73Skel.so ${DEVICE_DIR}
adb push ${QNN_SDK_ROOT}/lib/hexagon-v75/unsigned/libQnnHtpV75Skel.so ${DEVICE_DIR}
```

**3.3 Upload model, tokenizer and llama runner binary to phone**
```bash
adb push <model.pte> ${DEVICE_DIR}
adb push <tokenizer.model> ${DEVICE_DIR}
adb push cmake-android-out/lib/libqnn_executorch_backend.so ${DEVICE_DIR}
adb push cmake-out-android/examples/models/llama2/llama_main ${DEVICE_DIR}
```

**3.4 Run model**
```bash
adb shell "cd ${DEVICE_DIR} && ./llama_main --model_path <model.pte> --tokenizer_path <tokenizer.model> --prompt \"<|start_header_id|>system<|end_header_id|>\n\nYou are a funny chatbot.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nCould you tell me about Facebook?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n\" --seq_len 128"
```
You should see the message:
```
<|start_header_id|>system<|end_header_id|>\n\nYou are a funny chatbot.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nCould you tell me about Facebook?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nHello! I'd be delighted to chat with you about Facebook. Facebook is a social media platform that was created in 2004 by Mark Zuckerberg and his colleagues while he was a student at Harvard University. It was initially called "Facemaker" but later changed to Facebook, which is a combination of the words "face" and "book". The platform was initially intended for people to share their thoughts and share information with their friends, but it quickly grew to become one of the
```

## What is coming?
- Improve the performance for Llama 3 Instruct
- Reduce the memory pressure during inference to support 12GB Qualcomm devices
- Support more LLMs

## FAQ

If you encounter any issues while reproducing the tutorial, please file a github
issue on ExecuTorch repo and tag use `#qcom_aisw` tag
