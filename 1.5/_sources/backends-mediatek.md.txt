# MediaTek Backend

The MediaTek backend enables acceleration of PyTorch models on edge devices with MediaTek Neuron Processing Units (NPUs). This backend provides tools for exporting, building, and deploying models to leverage MediaTek hardware.

## Features

- Acceleration of PyTorch models on MediaTek NPUs
- Tools for model export and lowering
- Example scripts for model deployment and execution

## Target Requirements

- **Hardware:** MediaTek Dimensity 9300 (D9300), Dimensity 9400 (D9400)
- **Host OS:** Linux
- **SDK:** [NeuroPilot Express SDK](https://neuropilot.mediatek.com/resources/public/npexpress/en/docs/npexpress)

## Development Requirements

- Linux operating system
- Python dependencies:
  ```bash
  pip3 install -r requirements.txt
  ```
- NeuroPilot SDK Python wheels (download from [NeuroPilot Express SDK](https://neuropilot.mediatek.com/resources/public/npexpress/en/docs/npexpress)):
  ```bash
  pip3 install mtk_neuron-8.2.23-py3-none-linux_x86_64.whl
  pip3 install mtk_converter-8.13.0+public-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
  ```

## Using the MediaTek Backend

### Exporting and Lowering a Model

To export and lower a model for the MediaTek backend, use the provided shell script:
```bash
cd executorch
./examples/mediatek/shell_scripts/export_oss.sh mobilenetv3
```
The exported `.pte` file is saved in a directory named after the model.

### Quantizer API

Quantizer can be configured with different precision. We currently support A16W16, A16W8, A16W4, A8W8 and A8W4

The example code will be
```python
precision = "A16W16"
quantizer = NeuropilotQuantizer()
quantizer.setup_precision(getattr(Precision, precision))
```

### Partitioner API

A list of CompileSpec is suppported by MediaTek backend:
- `platform-config`: Specifies the targeted MediaTek platform name to compile for.

## Runtime Integration

This section presents an example of exporting and deploying a model. Please refer to `executorch/examples/mediatek/` for export and execution examples of various of models.

### Building Example Runners

Build example runners:
```bash
./mtk_build_examples.sh
```
Runners are located in `cmake-android-out/examples/mediatek/`.

### Deploying to Device

1. Push `libneuron_backend.so`, `libneuronusdk_adapter.mtk.so` and `libneuron_buffer_allocator.so` to the device.
2. Set the library path before running ExecuTorch:
   ```bash
   export LD_LIBRARY_PATH=<path_to_neuron_backend>:<path_to_usdk>:<path_to_buffer_allocator>:$LD_LIBRARY_PATH
   ```

### Building the Backend from Source
1. Copy `NeuronAdapter.h` to `backends/mediatek/runtime/include/api/`

2. Set NDK Path: Ensure that the `$ANDROID_NDK` environment variable is set to the path where the NDK is located.
   ```bash
   export ANDROID_NDK=<path_to_android_ndk>
   ```

3. Build the backend library `libneuron_backend.so`:
    ```bash
    cd backends/mediatek/scripts/
    ./mtk_build.sh
    ```
The output is `libneuron_backend.so` in `cmake-android-out/backends/mediatek/`.
