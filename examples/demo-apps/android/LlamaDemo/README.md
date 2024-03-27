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
cmake .. -DBUCK2="$BUCK" \
         -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
         -DANDROID_ABI="$ANDROID_ABI" \
         -DCMAKE_INSTALL_PREFIX=cmake-out \
         -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
         -DEXECUTORCH_BUILD_FLATC=OFF \
         -DFLATC_EXECUTABLE="${FLATC}" \
         -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
         -DPYTHON_EXECUTABLE="$PYTHON_EXECUTABLE" \
         -DEXECUTORCH_BUILD_ANDROID_JNI=ON \
         -DEXECUTORCH_BUILD_XNNPACK=ON
```
5. Run the following command to build the JNI library:
```
cmake --build . -j50
popd
```
6.
Copy the built library to your app:
```
JNI_LIBS_PATH="examples/demo-apps/android/LlamaDemo/app/src/main/jniLibs"
mkdir -p "${JNI_LIBS_PATH}/${ANDROID_ABI}"
cp cmake-out/extension/android/libexecutorch_llama_jni.so "${JNI_LIBS_PATH}/${ANDROID_ABI}/"
```

## Build Java library
The Java part of the ExecuTorch library can be built with gradlew:
```
pushd extension/android
./gradlew build
popd
```
In the android app, we set up the relative path to the built aar, so no further action is needed.

## Build Java app
1. Open Android Studio and select "Open an existing Android Studio project" to open examples/demo-apps/android/LlamaDemo.
2. Run the app (^R).
