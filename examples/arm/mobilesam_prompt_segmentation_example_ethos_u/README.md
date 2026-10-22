# MobileSAM Prompt Segmentation on Ethos-U

This example turns a point on an image into an object mask. It shows the full
ExecuTorch flow: export MobileSAM, quantize it, delegate it to Ethos-U85, run it
on the Corstone-320 FVP, and compare the target result with the host result.

There is one tested configuration: MobileSAM `vit_t`, a `448x448` input, and
Ethos-U85-256. The image can change at runtime, but the point prompt is embedded
in the exported model. Changing the prompt requires re-exporting the model.

## Run It

From the ExecuTorch repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
./install_executorch.sh --optional-dependency ethos_u
./examples/arm/setup.sh --i-agree-to-the-contained-eula
./examples/arm/mobilesam_prompt_segmentation_example_ethos_u/run.sh
```

The final command performs the complete flow and prints
`MobileSAM example: PASS`. Its main result is:

`arm_test/mobilesam/result/fvp_comparison.png`

## What It Does

1. Fetches the pinned official MobileSAM source and checkpoint outside the
   repository.
2. Runs `torch.export`, PT2E quantization, and `EthosUPartitioner` to create a
   `.pte` containing one Ethos-U delegate.
3. Builds the standard Arm ExecuTorch runner and runs one inference on FVP.
4. Compares the FVP mask with the host quantized mask and requires `0.9` IoU.

Successful completion creates:

- Program: `arm_test/mobilesam/export/mobilesam.pte`
- Host masks: `arm_test/mobilesam/export/fp32_mask.png` and
  `arm_test/mobilesam/export/quantized_mask.png`
- FVP log: `arm_test/mobilesam/fvp.log`
- Comparison: `arm_test/mobilesam/result/fvp_comparison.png`
- FVP validation: `arm_test/mobilesam/result/metrics.json`
- TOSA and Vela artifacts: `arm_test/mobilesam/export/artifacts`

The Python installer uses this source checkout and installs the dependencies
needed for ahead-of-time Ethos-U export. The Arm setup script installs the
cross compiler and FVP. Do not install a separate PyPI `executorch` wheel for
this source example.

On macOS, Docker must be running and the
[FVPs-on-Mac](https://github.com/Arm-Examples/FVPs-on-Mac) wrapper must be on
`PATH`.

## Code Map

- [`prepare_mobilesam.py`](model_export/prepare_mobilesam.py) fetches and
  verifies the external model.
- [`export_mobilesam.py`](model_export/export_mobilesam.py) contains the model,
  quantization, validation, and lowering flow.
- [`run.sh`](run.sh) uses ExecuTorch's standard Arm runner for target execution.
- [`visualize_fvp_output.py`](runtime/visualize_fvp_output.py) checks and plots
  the raw output tensor.

There is no MobileSAM-specific C++ runtime or CMake project.

## Limitations

- The exported model accepts one image tensor and uses one fixed positive point.
- It returns a low-resolution mask. Upsampling and thresholding are host-side
  post-processing.
- The demo image is also the calibration image. Product use requires a
  representative calibration set.
- The default fast FVP mode validates correctness. Its counters are not a
  performance benchmark or a measurement of real-device latency.

See [model export](model_export/README.md) and
[runtime](runtime/README.md) for details of each stage.
