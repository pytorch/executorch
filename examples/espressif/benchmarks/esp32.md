# ESP-NN backend benchmarks: esp32

Board: esp32, 2 cores at 240 MHz, 4 MB flash, 0 KB PSRAM, ESP-IDF v5.5.5.

| Model | Input | Path | Operators | Model size | Arena | Latency mean (min) | FPS | On-board check | Host accuracy |
|---|---|---|---|---|---|---|---|---|---|
| Keyword spotting DS-CNN, 49x10 MFCC | 1,1,49,10 | standalone XTS3 | 12 int8 | 0.03 MB | 16 KB | 203.4 ms (203.4) | 4.92 | bit-exact 2/2, ANSI too | top-1 agrees with float 32/32 |
| Small residual CNN, 64x64 | 1,3,64,64 | standalone XTS3 | 16 int8 | 0.24 MB | 36 KB | 705.0 ms (705.0) | 1.42 | bit-exact 2/2, ANSI too | top-1 agrees with float 32/32 |
| MobileNetV3-Small, ImageNet, 96x96 | 1,3,96,96 | standalone XTS3 | 108 int8 | 2.68 MB | 72 KB | 1440.7 ms (1440.7) | 0.69 | bit-exact 2/2, ANSI too | top-1 0/32 (float 6), agrees with float 0/32 |
| MobileNetV3-Small, ImageNet, 224x224 | 1,3,224,224 | standalone XTS3 | 108 int8 | 2.68 MB | 392 KB | - | - | does not fit RAM (280 KB free, no PSRAM) | top-1 6/32 (float 21), agrees with float 4/32 |
| MobileNetV2, ImageNet, 224x224 | 1,3,224,224 | standalone XTS3 | 65 int8 | 3.70 MB | - | - | - | does not fit 4 MB flash | top-1 20/32 (float 20), agrees with float 29/32 |
| MobileNetV3-Large, ImageNet, 224x224 | 1,3,224,224 | standalone XTS3 | 121 int8 | 5.71 MB | - | - | - | does not fit 4 MB flash | top-1 24/32 (float 23), agrees with float 27/32 |
| EfficientNet-B0, ImageNet, 224x224 | 1,3,224,224 | standalone XTS3 | 190 int8 | 5.65 MB | - | - | - | does not fit 4 MB flash | top-1 19/32 (float 23), agrees with float 23/32 |

Latency is the model invocation only (no input preprocessing), timed over repeated runs after a bit-exact (or SQNR) check against the host reference.

Host accuracy is the int8 model on the host (the board's output is bit-identical) after post-training quantization with 64 calibration images and no QAT, on 32 Imagenette validation images mapped to ImageNet classes. Models with random weights report top-1 agreement or SQNR against the float model instead.
