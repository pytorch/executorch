# Camera inference stream (ESP32-S3)

Live ImageNet classification on an ESP32-S3 camera board with the standalone
`espnn_runtime`, streamed over USB to a terminal or a browser.

Tested with an OV3660 on the ESP32-S3-EYE / Freenove ESP32-S3-WROOM CAM
pinout (selectable in menuconfig, as is the XIAO ESP32S3 Sense).

```bash
# 1. Compile a model (QAT recovers MobileNetV3-Small accuracy; plain PTQ loses most of it)
python ../aot_espnn_compiler.py --model mobilenet_v3_small --mode standalone \
    --calib-dir imagenette2-160/train --eval-dir imagenette2-160/val \
    --calib-samples 1024 --qat-epochs 3 --qat-lr 1e-4 --output models/mobilenet_v3_small_224.xtb

# 2. Build and flash (activated ESP-IDF shell)
./flash.sh /dev/ttyACM0

# 3. Watch: terminal dashboard, or Chrome at http://127.0.0.1:8765/
python viewer.py
python viewer.py --web
```

The board captures JPEG frames from the sensor, decodes and center-crops them,
runs the model on core 1 and streams results, the model's input image and a
15 fps preview on core 0. On the ESP32-S3 with MobileNetV3-Small this gives
about 1.4 results per second with roughly 0.75 s from capture to result.

Notes from bring-up: capture JPEG rather than RGB565 (continuous RGB565
capture competes with inference for PSRAM bandwidth), keep the JPEG frame
buffers in internal RAM, and do not use the camera driver's PSRAM DMA mode.
