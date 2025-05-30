# YOLO12 Detection C++ Inference with ExecuTorch

<p align="center">
      <br>
      <img src="./yolo12s_demo.gif">
      <br>
</p>

This example demonstrates how to perform inference of [Ultralytics YOLO12 family](https://docs.ultralytics.com/models/yolo12/) detection models in C++ leveraging the Executorch backends:
- [OpenVINO](../../../backends/openvino/README.md)
- [XNNPACK](../../../backends/xnnpack/README.md)


# Instructions

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
https://opencv.org/get-started/


### Step 4: Export the Yolo12 model to the ExecuTorch


OpenVINO:
```bash
python export_and_validate.py --model_name yolo12s --input_dims=[1920,1080]  --backend openvino --device CPU
```

XNNPACK:
```bash
python export_and_validate.py --model_name yolo12s --input_dims=[1920,1080] --backend xnnpack
```

> **_NOTE:_**  Quantization is comming soon!

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
```
./build/Yolo12DetectionDemo --help
```


# Credits:

Ultralytics examples: https://github.com/ultralytics/ultralytics/tree/main/examples

Sample video: https://www.pexels.com/@shanu-1040189/
