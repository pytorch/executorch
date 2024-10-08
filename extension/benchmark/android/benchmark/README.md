Minibench: ExecuTorch Android Benchmark App
===

Minibench is a benchmarking app for testing the performance of the ExecuTorch runtime on Android devices.

It supports both generic (vision, audio, etc) models and LLM.

- For generic model, it reports metrics such as model load time, and average inference time.
- For LLM, it reports metrics such as model load time, and tokens per second.
- We are working on providing more metrics in the future.

Minibench is usedful for giving reference performance data when developers integrate ExecuTorch with their own Android app.

## Build
You will need executorch AAR for Java and JNI dependencies.
```
export ANDROID_NDK=<path_to_android_ndk>
sh build/build_android_llm_demo.sh
```
and copy the AAR to `app/libs`.
```
mkdir -p app/libs
cp $BUILD_AAR_DIR/executorch.aar app/libs
```

You can also refer to [this script](https://github.com/pytorch/executorch/blob/62024d8/.github/workflows/android-perf.yml#L226-L235) to see how it is built.

Then you can build and install the app on Android Studio, or simply run
```
./gradlew installDebug
```

## Usage
This apk does not come with a launcher icon. Instead, trigger it from command line

### Push model to a directory
```
adb shell mkdir /data/local/tmp/minibench
adb push my_model.pte /data/local/tmp/minibench
# optionally, push tokenizer for LLM
adb push tokenizer.bin /data/local/tmp/minibench
```

### Generic model
```
adb shell am start -W -S -n org.pytorch.minibench/org.pytorch.minibench.LlmBenchmarkActivity \
 --es model_dir /data/local/tmp/minibench
```

### LLM
```
adb shell am start -W -S -n org.pytorch.minibench/org.pytorch.minibench.LlmBenchmarkActivity \
 --es model_dir /data/local/tmp/minibench --es tokenizer_path /data/local/tmp/minibench/tokenizer.bin
```

### Fetch results
```
adb shell run-as org.pytorch.minibench cat files/benchmark_results.json
```
If the ExecuTorch runner is initialized and loads your model, but there is a load error or run error, you will see error code from that JSON.
