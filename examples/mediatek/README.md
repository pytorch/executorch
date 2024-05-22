## Directory Structure

Below is the layout of the `examples/mediatek` directory, which includes the necessary files for the example applications:

```plaintext
examples/mediatek
├── mtk_executor_runner               # Example C++ wrapper for the ExecuTorch runtime
├── CMakeLists.txt                    # CMake build configuration file for compiling examples
└── README.md                         # Documentation for the examples (this file)
```

## Supported Chips

The examples provided in this repository are tested and supported on the following MediaTek chip:

- MediaTek Dimensity 9300 (D9300)

## Environment Setup

To set up the build environment for the `mtk_executor_runner`:

1. Navigate to the `backends/mediatek/scripts` directory within the repository.
2. Follow the detailed build steps provided in that location.
3. Upon successful completion of the build steps, the `mtk_executor_runner` binary will be generated.

## Deploying and Running on the Device

### Pushing Files to the Device

Transfer the `.pte` model files and the `mtk_executor_runner` binary to your Android device using the following commands:

```bash
adb push mtk_executor_runner <PHONE_PATH, e.g. /data/local/tmp>
adb push <MODEL_NAME>.pte <PHONE_PATH, e.g. /data/local/tmp>
```

Make sure to replace `<MODEL_NAME>` with the actual name of your model file. And, replace the `<PHONE_PATH>` with the desired detination on the device.

### Executing the Model

Execute the model on your Android device by running:

```bash
adb shell "/data/local/tmp/mtk_executor_runner --model_path /data/local/tmp/<MODEL_NAME>.pte --iteration <ITER_TIMES>"
```

In the command above, replace `<MODEL_NAME>` with the name of your model file and `<ITER_TIMES>` with the desired number of iterations to run the model.
