# Welcome to the ExecuTorch LLaMA Android Demo App

This app demonstrates the use of the LLaMA local inference use case with ExecuTorch.


## Getting models

Please refer to the LLaMA tutorial to generate the model.

Once you have the model, you can push it to your device:
```
adb shell mkdir -p /data/local/tmp/llama
adb push model.pte /data/local/tmp/llama
adb push tokenizer.bin /data/local/tmp/llama
```

The demo app searches in `/data/local/tmp/llama` for .pte and .bin files as LLAMA model and tokenizer.

## Build JNI library
1. Open a terminal window and navigate to the root directory of the `executorch`.
2. Set the following environment variables:
```
export BUCK=<path_to_buck>
export ANDROID_NDK=<path_to_android_ndk>
export PYTHON_EXECUTABLE=<path_to_python_executable>
export ANDROID_ABI=<abi_type> # ex: arm64-v8a
export FLATC=<path_to_flatc_executable>
```
3. Create a new directory for the CMake build output:
```
mkdir cmake-out
pushd cmake-out
```
4. Run the following command to configure the CMake build:
```
cmake .. -DCMAKE_INSTALL_PREFIX=cmake-out \
  -DCMAKE_TOOLCHAIN_FILE="${ANDROID_NDK}/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI="${ANDROID_ABI}" \
  -DBUCK2="${BUCK2}" \
  -DEXECUTORCH_BUILD_XNNPACK=ON \
  -DEXECUTORCH_BUILD_FLATC=OFF \
  -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
  -DFLATC_EXECUTABLE="${FLATC}" \
  -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON

cmake --build . -j16 --target install

cmake ../examples/models/llama2 -DBUCK2="$BUCK" \
         -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
         -DANDROID_ABI="$ANDROID_ABI" \
         -DCMAKE_INSTALL_PREFIX=cmake-out \
         -Bexamples/models/llama2

cmake --build examples/models/llama2 -j16

cmake ../extension/android -DBUCK2="${BUCK2}" \
  -DCMAKE_TOOLCHAIN_FILE=${ANDROID_NDK}/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI="${ANDROID_ABI}" \
  -DCMAKE_INSTALL_PREFIX=cmake-out \
  -DEXECUTORCH_BUILD_LLAMA_JNI=ON \
  -Bextension/android

cmake --build extension/android -j16

popd
```

5. Copy the built library to your app:
```
JNI_LIBS_PATH="examples/demo-apps/android/LlamaDemo/app/src/main/jniLibs"
mkdir -p "${JNI_LIBS_PATH}/${ANDROID_ABI}"
cp cmake-out/extension/android/libexecutorch_llama_jni.so "${JNI_LIBS_PATH}/${ANDROID_ABI}/"
```

## Build Java app
1. Open Android Studio and select "Open an existing Android Studio project" to open examples/demo-apps/android/LlamaDemo.
2. Run the app (^R).
