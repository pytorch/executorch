# ExecuTorch Neuron Backend examples

## Directory structure
```bash
examples/mediatek
├── mtk_executor_runner               # Contains an example C++ wrapper around the ExecuTorch runtime
├── CMakeLists.txt                    # The build file
└── README.md                         # This file
```

## Environment

Please follow the build step in backends/mediatek/scripts.
The mtk_executor_runner will be automatically generated.

## Execute

Push the .pte files and the mtk_executor_runner into the Android device.

```bash
adb push mtk_executor_runner <PHONE_PATH, e.g. /data/local/tmp>
adb push <MODEL_NAME>.pte <PHONE_PATH, e.g. /data/local/tmp>
```

Run the model

```bash
adb shell "<PHONE_PATH>/mtk_executor_runner  --model_path <MODEL_PATH> --iteration <ITER_TIMES>"
```
