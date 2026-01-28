set ex

cmake  -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_PLATFORM=android-29 \
    -DCMAKE_INSTALL_PREFIX=cmake-out-android-arm64-v8a \
    -DCMAKE_BUILD_TYPE=Release \
    -DPYTHON_EXECUTABLE=python \
    -DEXECUTORCH_BUILD_XNNPACK=ON \
    -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON \
    -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON \
    -DEXECUTORCH_BUILD_KERNELS_CUSTOM=ON \
    -Bcmake-out-android-arm64-v8a/examples/models/llama \
    examples/models/llama

cmake --build cmake-out-android-arm64-v8a/examples/models/llama -j8 --config Release

adb push cmake-out-android-arm64-v8a/examples/models/llama/llama_main /data/local/tmp/llama

adb shell 'cd /data/local/tmp/llama; ls; ./llama_main --model_path stories15m_h.pte --tokenizer_path tokenizer.bin --prompt "Once upon a time" --temperature 0'
