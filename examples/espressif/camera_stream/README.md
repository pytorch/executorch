# Camera inference stream (ESP32-S3)

Live ImageNet classification on an ESP32-S3 camera board. The model is an
ExecuTorch program (`.pte`) run by the ExecuTorch runtime, with its int8
graph in the ESP-NN delegate (`backends/espressif`). Results and a preview
stream over USB to a terminal or a browser.

Tested with an OV3660 on the ESP32-S3-EYE / Freenove ESP32-S3-WROOM CAM
pinout (selectable in menuconfig, as is the XIAO ESP32S3 Sense).

```bash
# 1. Quantize and lower a model. MobileNetV3-Small needs QAT; plain PTQ loses
#    most of its accuracy.
python export_model.py --calib-dir imagenette2-160/train \
    --eval-dir imagenette2-160/val --qat-epochs 5 --qat-lr 1e-3 \
    --output models/mobilenet_v3_small_224.pte

# 2. Cross-compile ExecuTorch (see backends/espressif/README.md) into
#    cmake-out-esp, then build and flash from an activated ESP-IDF shell
ET_BUILD_DIR=$PWD/../../../cmake-out-esp ./flash.sh /dev/ttyACM0

# 3. Watch: terminal dashboard, or Chrome at http://127.0.0.1:8765/
python viewer.py --port /dev/ttyACM0
python viewer.py --port /dev/ttyACM0 --web
```

`export_model.py` prints the CPU operators the program needs and writes them
to `models/mobilenet_v3_small_224.ops.txt` for `EXECUTORCH_SELECT_OPS_LIST`.
MobileNetV3-Small lowers to one delegate with no CPU operators, so any
selection works.

The exporter leaves input allocation to the camera application. The runtime
shares the normalized image buffer, avoiding a second float input buffer and
a copy on each inference. Re-export existing models to use this memory plan.

The board captures JPEG frames from the sensor, decodes and center-crops
them, runs the model on core 1, and streams results, the model's input image
and a preview of about 12 fps on core 0.

Measured on an ESP32-S3 N16R8 at 240 MHz with MobileNetV3-Small at 224×224:

| | |
|---|---|
| Inference | 646 ms, of which 626 ms in the ESP-NN delegate (593 ms in kernels) |
| Results | 1.3 per second, about 0.8 s from capture to result |
| Accuracy (200 Imagenette val images, 1000-way top-1) | float 134, int8 program 152 |

The int8 program beats the float model there because QAT fine-tunes on
Imagenette's 10 classes. It is not a general ImageNet result.

The delegate copies the weights into PSRAM and can split kernels across both
cores (`CONFIG_ESPNN_DELEGATE_DUAL_CORE`, which slows the preview).

Notes from bring-up: capture JPEG rather than RGB565 (continuous RGB565
capture competes with inference for PSRAM bandwidth), keep the JPEG frame
buffers in internal RAM, and do not use the camera driver's PSRAM DMA mode.
Keep XCLK at 10 MHz (the default): at 20 MHz the OV3660 started dropping
and corrupting frames after a few minutes, which showed up as a black or
frozen preview and `dec_err` counts in the results.
