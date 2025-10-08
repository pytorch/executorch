# YOLO12 Detection C++ Inference with ExecuTorch

This example demonstrates how to perform inference of [YOLO12 family](https://docs.ultralytics.com/models/yolo12/) detection models in C++ leveraging the Executorch backends:

- [OpenVINO](../../../backends/openvino/README.md)
- [XNNPACK](../../../backends/xnnpack/README.md)

## Performance Evaluation

| CPU                            | Model   | Backend  | Device | Precision | Average Latency, ms |
|--------------------------------|---------|----------|--------|-----------|---------------------|
| Intel(R) Core(TM) Ultra 7 155H | yolo12s | openvino | CPU    | FP32      | 88.3549             |
| Intel(R) Core(TM) Ultra 7 155H | yolo12s | openvino | CPU    | INT8      | 53.066              |
| Intel(R) Core(TM) Ultra 7 155H | yolo12l | openvino | CPU    | FP32      | 317.953             |
| Intel(R) Core(TM) Ultra 7 155H | yolo12l | openvino | CPU    | INT8      | 150.846             |
| Intel(R) Core(TM) Ultra 7 155H | yolo12s | openvino | GPU    | FP32      | 32.71               |
| Intel(R) Core(TM) Ultra 7 155H | yolo12l | openvino | GPU    | FP32      | 70.885              |
| Intel(R) Core(TM) Ultra 7 155H | yolo12s | xnnpack  | CPU    | FP32      | 169.36              |
| Intel(R) Core(TM) Ultra 7 155H | yolo12l | xnnpack  | CPU    | FP32      | 436.876             |

## Instructions

### Step 1: Install ExecuTorch

To install ExecuTorch, follow this [guide](https://pytorch.org/executorch/stable/getting-started-setup.html).

### Step 2: Install the backend of your choice

- [OpenVINO backend installation guide](../../../backends/openvino/README.md#build-instructions)
- [XNNPACK backend installation guilde](https://pytorch.org/executorch/stable/tutorial-xnnpack-delegate-lowering.html#running-the-xnnpack-model-with-cmake)

### Step 3: Install the demo requirements

Python demo requirements:

```bash
python -m pip install -r examples/models/yolo12/requirements.txt
```

Demo infenrece dependency - OpenCV library:
<https://opencv.org/get-started/>

### Step 4: Export the YOLO12 model to the ExecuTorch

OpenVINO:

```bash
python export_and_validate.py --model_name yolo12s --input_dims=[1920,1080]  --backend openvino --device CPU
```

OpenVINO quantized model:

```bash
python export_and_validate.py --model_name yolo12s --input_dims=[1920,1080]  --backend openvino --quantize --video_input /path/to/calibration/video --device CPU
```

XNNPACK:

```bash
python export_and_validate.py --model_name yolo12s --input_dims=[1920,1080] --backend xnnpack
```

> **_NOTE:_**  Quantization for XNNPACK backend is WIP. Please refere to <https://github.com/pytorch/executorch/issues/11523> for more details.

Exported model could be validated using the `--validate` key:

```bash
python export_and_validate.py --model_name yolo12s --backend ... --validate dataset_name.yaml
```

A list of available datasets and instructions on how to use a custom dataset can be found [here](https://docs.ultralytics.com/datasets/detect/).
Validation only supports the default `--input_dims`; please do not specify this parameter when using the `--validate` flag.

To get a full parameters description please use the following command:

```bash
python export_and_validate.py --help
```

### Step 5: Build the demo project

OpenVINO:

```bash
cd examples/models/yolo12
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DUSE_OPENVINO_BACKEND=ON ..
make -j$(nproc)
```

XNNPACK:

```bash
cd examples/models/yolo12
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DUSE_XNNPACK_BACKEND=ON ..
make -j$(nproc)
```

### Step 6: Run the demo

```bash
./build/Yolo12DetectionDemo -model_path /path/to/exported/model -input_path /path/to/video/file -output_path /path/to/output/annotated/video
```

To get a full parameters description please use the following command:

```bash
./build/Yolo12DetectionDemo --help
```

## Credits

Ultralytics examples: <https://github.com/ultralytics/ultralytics/tree/main/examples>
