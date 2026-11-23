# Lower and Run a Model on the ESP32-S3

This tutorial quantizes a small convolutional network, lowers it to the Espressif ESP-NN backend, and runs it through the ExecuTorch runtime: first in the `esp-emu` emulator, then on an ESP32-S3 board. The program carries its own reference outputs (BundleIO), so the runner checks the results on the device.

## Prerequisites

- A Linux x86_64 machine with Python 3.11 or 3.12, and an ExecuTorch that includes the Espressif backend: a nightly build or a source install, as described under *Development Requirements* in {doc}`../espressif-overview`.
- CMake 3.25 or newer, and Ninja.
- An ExecuTorch source checkout with its submodules. The commands below run from its root.
- Optional: an ESP32-S3 board with PSRAM, for the last step.

Install ESP-IDF and the emulator with your ExecuTorch environment active, then load them into the shell:

```bash
examples/espressif/setup.sh --minimal
source ~/.cache/executorch-espressif/setup_path.sh
```

## Step 1: Quantize, Lower and Bundle the Model

Save the following as `export_tutorial_model.py`. It quantizes the model with `EspNNQuantizer` and lowers it with `to_espnn_edge`. It then bundles one input with the expected outputs, computed by `ProgramEmulator`, the backend's bit-exact host model of the delegate.

```python
import torch
import torch.nn as nn
from executorch.backends.espressif import EspNNQuantizer, to_espnn_edge
from executorch.backends.espressif.reference import ProgramEmulator
from executorch.devtools.bundled_program.config import MethodTestCase, MethodTestSuite
from executorch.devtools.bundled_program.core import BundledProgram
from executorch.devtools.bundled_program.serialize import (
    serialize_from_bundled_program_to_flatbuffer,
)
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1, groups=16), nn.ReLU6(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.classifier = nn.Linear(16, 10)

    def forward(self, x):
        return self.classifier(self.features(x))


torch.manual_seed(0)
model = SmallCNN().eval()
example_inputs = (torch.randn(1, 3, 32, 32),)

prepared = prepare_pt2e(
    torch.export.export(model, example_inputs).module(), EspNNQuantizer()
)
with torch.no_grad():
    for _ in range(32):  # use real calibration data for a real model
        prepared(torch.randn(1, 3, 32, 32))
quantized = convert_pt2e(prepared)

edge, report = to_espnn_edge(torch.export.export(quantized, example_inputs))
print(f"{report['partitions']} delegate(s), {len(report['cpu_ops'])} CPU operator(s)")

expected = ProgramEmulator(edge.exported_program())(*example_inputs).to(torch.float32)
program = edge.to_executorch()
bundle = BundledProgram(
    program,
    [MethodTestSuite("forward", [MethodTestCase(list(example_inputs), [expected])])],
)
with open("small_cnn.bpte", "wb") as file:
    file.write(serialize_from_bundled_program_to_flatbuffer(bundle))

plan = program.executorch_program.execution_plan[0]
print("operators:", sorted(f"{op.name}.{op.overload}" for op in plan.operators))
```

```bash
python export_tutorial_model.py
```

Every layer of this model is supported, so the report shows one delegate and no CPU operators:

```
1 delegate(s), 0 CPU operator(s)
operators: []
```

If your model has CPU operators, the operator list shows them; select the ones outside `quantized_decomposed::` when building the runtime.

## Step 2: Cross-Compile the ExecuTorch Runtime

```bash
cmake --preset esp-baremetal -G Ninja -B cmake-out-esp \
    -DCMAKE_TOOLCHAIN_FILE="${IDF_PATH}/tools/cmake/toolchain-esp32s3.cmake" \
    -DCMAKE_BUILD_TYPE=Release \
    -DEXECUTORCH_BUILD_DEVTOOLS=ON \
    -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON \
    -DEXECUTORCH_SELECT_OPS_LIST=aten::add.out \
    -DMAX_KERNEL_NUM=256 \
    -DEXECUTORCH_BUILD_PTHREADPOOL=OFF -DEXECUTORCH_BUILD_CPUINFO=OFF
cmake --build cmake-out-esp --target install
```

`aten::add.out` is a placeholder: the runner links the selected-kernels library, which needs at least one operator. `EXECUTORCH_BUILD_DEVTOOLS` provides the BundleIO verification.

## Step 3: Run in the Emulator

The emulator runs the same firmware as the board, including ESP-NN's optimized ESP32-S3 kernels. These need `esp-emu` 0.51.0 or newer, which `setup.sh` installs.

```bash
idf.py -C examples/espressif/project -B build-emu \
    -DIDF_TARGET=esp32s3 \
    -DSDKCONFIG=$PWD/build-emu.sdkconfig \
    -DET_ESPNN_BACKEND=ON \
    -DET_BUILD_DIR_PATH=$PWD/cmake-out-esp \
    -DET_PTE_FILE_PATH=$PWD/small_cnn.bpte \
    -DET_BUNDLE_IO=ON -DET_NUM_INFERENCES=1 \
    -DET_ATOL=1e-5 -DET_RTOL=1e-5 \
    build merge-bin

esp-emu --chip esp32s3 \
    --firmware build-emu/merged-binary.bin \
    --elf build-emu/executorch_esp_runner.elf \
    --net user --psram-size 8M --timeout 60s --exit-on 'Program complete.' | tee emulator.log
python examples/espressif/test_emulator.py emulator.log
```

The runner logs the error statistics and the result of the comparison:

```
max_absolute_error:  0.000000
TEST: BundleIO index[0] Test_result: PASS
```

A fully delegated program matches the host reference exactly, so the tolerance can be 1e-5. For a program with CPU operators, allow for the CPU kernels' float rounding, for example 1% of the largest output.

To test with ESP-NN's portable C reference kernels instead, add `CONFIG_NN_ANSI_C=y` through an extra sdkconfig defaults file:

```bash
printf 'CONFIG_NN_ANSI_C=y\n' > ansi.defaults
# add to the idf.py command above:
#   "-DSDKCONFIG_DEFAULTS=sdkconfig.defaults;$PWD/ansi.defaults"
```

Both kernel sets give the same results.

## Step 4: Run on an ESP32-S3

Build the same program with the default ESP-NN configuration, which uses the S3's SIMD kernels, and flash it:

```bash
idf.py -C examples/espressif/project -B build-board \
    -DIDF_TARGET=esp32s3 \
    -DET_ESPNN_BACKEND=ON \
    -DET_BUILD_DIR_PATH=$PWD/cmake-out-esp \
    -DET_PTE_FILE_PATH=$PWD/small_cnn.bpte \
    -DET_BUNDLE_IO=ON -DET_NUM_INFERENCES=10 \
    -DET_ATOL=1e-5 -DET_RTOL=1e-5 \
    -p /dev/ttyACM0 build flash monitor
```

The board prints the average inference time, then the same `Test_result: PASS` line. On an ESP32-S3 at 240 MHz this model takes about 14 ms per inference:

```
Average inference time: 13.563 ms
 max_absolute_error:  0.000000
TEST: BundleIO index[0] Test_result: PASS
```

## Next Steps

- Lower your own model. For networks such as MobileNetV3, use quantization-aware training; see {doc}`../espressif-quantization`.
- See {doc}`../espressif-overview` to integrate the backend into your own ESP-IDF application.
- [`examples/espressif/camera_stream`](https://github.com/pytorch/executorch/tree/main/examples/espressif/camera_stream) runs live image classification from an ESP32-S3 camera board on the ExecuTorch runtime.
