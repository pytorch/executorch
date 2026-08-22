# Executing and profiling an ExecuTorch Vulkan model on device

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

## Lower a model to ExecuTorch Vulkan and obtain the `.pte` file


The commands in this tutorial are assumed to be executed from ExecuTorch's root
directory.

```shell
cd ~/executorch
```

For this tutorial, we will use the export script in
[`executorch/examples/vulkan/export.py`](https://github.com/pytorch/executorch/tree/main/examples/vulkan),
however any method of generating a `.pte` file will suffice. In this tutorial,
the InceptionV3 model is exported.

```shell
python -m examples.vulkan.export --model_name=ic3 -o . -fp16
```

After exporting, there should be a file called `ic3_vulkan.pte` in the root
directory of ExecuTorch. Feel free to modify the `-o` argument of the script to
control where the `.pte` file will be stored.

Then, push the `.pte` file to device.

```shell
adb shell mkdir -p /data/local/tmp/etvk/models/ && \
adb push ic3_vulkan.pte /data/local/tmp/etvk/models/ic3_vulkan.pte
```

## Build the `executor_runner` binary and push to device

To be able to run the `.pte` file on device, first the core libraries,
including the Vulkan backend, must be compiled for Android. Note that
`-DEXECUTORCH_ENABLE_EVENT_TRACER=ON` is used to turn on profiling, and
`-DEXECUTORCH_BUILD_EXECUTOR_RUNNER=ON` is used to build the runner binary that
will be used to execute and profile the `.pte` file.


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
    -DEXECUTORCH_BUILD_EXTENSION_EVALUE_UTIL=ON \
    -DEXECUTORCH_BUILD_EXECUTOR_RUNNER=ON \
    -DEXECUTORCH_ENABLE_EVENT_TRACER=ON \
    -Bcmake-out-android-so && \
cmake --build cmake-out-android-so -j16 --target install --config Release
```

Once the build completes, we can push the runner binary to device.

```shell
adb push cmake-out-android-so/executor_runner /data/local/tmp/etvk/executor_runner
```

## Execute the `.pte` file

Finally, we can execute the lowered `.pte` file on your device. To test run the
model file without profiling:

```shell
adb shell /data/local/tmp/etvk/executor_runner \
  --model_path /data/local/tmp/etvk/models/ic3_vulkan.pte
```

Now, with profiling:

```shell
MODEL_NAME=ic3 && \
BACKEND=vulkan && \
NUM_ITERS=3 && \
adb shell mkdir -p /data/local/tmp/etvk/etdumps/ && \
adb shell /data/local/tmp/etvk/executor_runner \
  --model_path /data/local/tmp/etvk/models/${MODEL_NAME}_${BACKEND}.pte \
  --num_executions=${NUM_ITERS} \
  --etdump_path /data/local/tmp/etvk/etdumps/${MODEL_NAME}_${BACKEND}.etdp && \
adb pull /data/local/tmp/etvk/etdumps/${MODEL_NAME}_${BACKEND}.etdp ${MODEL_NAME}_${BACKEND}.etdp && \
adb shell rm /data/local/tmp/etvk/etdumps/${MODEL_NAME}_${BACKEND}.etdp && \
python devtools/inspector/inspector_cli.py \
  --etdump_path ${MODEL_NAME}_${BACKEND}.etdp
```

Here is some sample (tailed) output from a Samsung Galaxy S24:

```shell
├─────┼────────────────────┼────────────────────────────────────────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────────┼────────────┼───────────────────┼─────────────────────────┼────────────────────────────────────────────────────────┤
│ 165 │ Execute            │ conv2d_clamp_half_163                  │   0.345082   │   0.346164   │   0.346247   │   0.345748   │   0.344812   │   0.346268   │ []         │ True              │                         │ [2081488974948084, 2081488995911052, 2081489016763676] │
├─────┼────────────────────┼────────────────────────────────────────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────────┼────────────┼───────────────────┼─────────────────────────┼────────────────────────────────────────────────────────┤
│ 166 │ Execute            │ conv2d_clamp_half_164                  │   0.306124   │   0.30654    │   0.306998   │   0.306557   │   0.30602    │   0.307112   │ []         │ True              │                         │ [2081488975294716, 2081488996256228, 2081489017110204] │
├─────┼────────────────────┼────────────────────────────────────────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────────┼────────────┼───────────────────┼─────────────────────────┼────────────────────────────────────────────────────────┤
│ 167 │ Execute            │ set_zero_int32_165                     │   0.00240245 │   0.00244403 │   0.00248561 │   0.00244403 │   0.00239205 │   0.002496   │ []         │ True              │                         │ [2081488975601100, 2081488996563132, 2081489017417680] │
├─────┼────────────────────┼────────────────────────────────────────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────────┼────────────┼───────────────────┼─────────────────────────┼────────────────────────────────────────────────────────┤
│ 168 │ Execute            │ concat_2_texture3d_half_166            │   0.0122305  │   0.01248    │   0.0125634  │   0.0124108  │   0.0121682  │   0.0125842  │ []         │ True              │                         │ [2081488975603960, 2081488996565940, 2081489017420436] │
├─────┼────────────────────┼────────────────────────────────────────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────────┼────────────┼───────────────────┼─────────────────────────┼────────────────────────────────────────────────────────┤
│ 169 │ Execute            │ set_zero_int32_167                     │   0.00157056 │   0.00161195 │   0.00161214 │   0.00159478 │   0.00156021 │   0.00161219 │ []         │ True              │                         │ [2081488975616804, 2081488996578888, 2081489017432968] │
├─────┼────────────────────┼────────────────────────────────────────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────────┼────────────┼───────────────────┼─────────────────────────┼────────────────────────────────────────────────────────┤
│ 170 │ Execute            │ concat_3_texture3d_half_168            │   0.0420369  │   0.0423281  │   0.0427857  │   0.0423974  │   0.0419641  │   0.0429001  │ []         │ True              │                         │ [2081488975618728, 2081488996580864, 2081489017434944] │
├─────┼────────────────────┼────────────────────────────────────────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────────┼────────────┼───────────────────┼─────────────────────────┼────────────────────────────────────────────────────────┤
│ 171 │ Execute            │ update_concat_offset_3_int32_169       │   0.00261035 │   0.00265193 │   0.00265212 │   0.00263468 │   0.00259995 │   0.00265217 │ []         │ True              │                         │ [2081488975661992, 2081488996623556, 2081489017477272] │
├─────┼────────────────────┼────────────────────────────────────────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────────┼────────────┼───────────────────┼─────────────────────────┼────────────────────────────────────────────────────────┤
│ 172 │ Execute            │ concat_1_texture3d_half_170            │   0.00758157 │   0.00774789 │   0.00803914 │   0.00779994 │   0.00753999 │   0.00811195 │ []         │ True              │                         │ [2081488975664956, 2081488996626572, 2081489017480288] │
├─────┼────────────────────┼────────────────────────────────────────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────────┼────────────┼───────────────────┼─────────────────────────┼────────────────────────────────────────────────────────┤
│ 173 │ Execute            │ mean2d_half_171                        │   0.0147889  │   0.0148721  │   0.0150384  │   0.0149067  │   0.0147681  │   0.01508    │ []         │ True              │                         │ [2081488975673432, 2081488996634476, 2081489017488400] │
├─────┼────────────────────┼────────────────────────────────────────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────────┼────────────┼───────────────────┼─────────────────────────┼────────────────────────────────────────────────────────┤
│ 174 │ Execute            │ view_half_172                          │   0.00644803 │   0.00644803 │   0.00653119 │   0.00648268 │   0.00644803 │   0.00655198 │ []         │ True              │                         │ [2081488975688876, 2081488996649712, 2081489017503532] │
├─────┼────────────────────┼────────────────────────────────────────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────────┼────────────┼───────────────────┼─────────────────────────┼────────────────────────────────────────────────────────┤
│ 175 │ Execute            │ view_half_173                          │   0.00488806 │   0.00488806 │   0.00488806 │   0.00488806 │   0.00488806 │   0.00488806 │ []         │ True              │                         │ [2081488975695688, 2081488996656524, 2081489017510448] │
├─────┼────────────────────┼────────────────────────────────────────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────────┼────────────┼───────────────────┼─────────────────────────┼────────────────────────────────────────────────────────┤
│ 176 │ Execute            │ linear_naive_texture3d_half_174        │   0.586726   │   0.590096   │   0.595338   │   0.590876   │   0.585884   │   0.596648   │ []         │ True              │                         │ [2081488975700940, 2081488996661776, 2081489017515700] │
├─────┼────────────────────┼────────────────────────────────────────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────────┼────────────┼───────────────────┼─────────────────────────┼────────────────────────────────────────────────────────┤
│ 177 │ Execute            │ image_to_nchw_texture3d_half_float_175 │   0.00270395 │   0.00270414 │   0.00274572 │   0.00272139 │   0.00270391 │   0.00275612 │ []         │ True              │                         │ [2081488976297952, 2081488997248024, 2081489018106160] │
├─────┼────────────────────┼────────────────────────────────────────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────────┼────────────┼───────────────────┼─────────────────────────┼────────────────────────────────────────────────────────┤
│ 178 │ Execute            │ DELEGATE_CALL                          │  20.8864     │  20.9461     │  21.5925     │  21.1906     │  20.8715     │  21.7541     │ []         │ False             │                         │ [358395625, 380178646, 401147657]                      │
├─────┼────────────────────┼────────────────────────────────────────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────────┼────────────┼───────────────────┼─────────────────────────┼────────────────────────────────────────────────────────┤
│ 179 │ Execute            │ Method::execute                        │  20.8867     │  20.9464     │  21.593      │  21.191      │  20.8718     │  21.7547     │ []         │ False             │                         │ [358395521, 380178542, 401147552]                      │
╘═════╧════════════════════╧════════════════════════════════════════╧══════════════╧══════════════╧══════════════╧══════════════╧══════════════╧══════════════╧════════════╧═══════════════════╧═════════════════════════╧════════════════════════════════════════════════════════╛
```
