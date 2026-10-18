# ESP-NN backend benchmarks: esp32s3

Board: esp32s3, 2 cores at 240 MHz, 16 MB flash, 8192 KB PSRAM, ESP-IDF v5.5.5.

| Model | Input | Path | Operators | Model size | Arena | Latency mean (min) | FPS | On-board check | Host accuracy |
|---|---|---|---|---|---|---|---|---|---|
| Keyword spotting DS-CNN, 49x10 MFCC | 1,1,49,10 | standalone XTS3 | 12 int8 | 0.03 MB | 16 KB | 16.4 ms (16.4) | 60.80 | bit-exact 2/2, ANSI too | top-1 agrees with float 32/32 |
| Small residual CNN, 64x64 | 1,3,64,64 | standalone XTS3 | 16 int8 | 0.24 MB | 36 KB | 40.3 ms (40.3) | 24.80 | bit-exact 2/2, ANSI too | top-1 agrees with float 32/32 |
| MobileNetV3-Small, ImageNet, 96x96 | 1,3,96,96 | standalone XTS3 | 108 int8 | 2.68 MB | 72 KB | 116.2 ms (116.2) | 8.61 | bit-exact 2/2, ANSI too | top-1 0/32 (float 6), agrees with float 0/32 |
| MobileNetV3-Small, ImageNet, 224x224 | 1,3,224,224 | standalone XTS3 | 108 int8 | 2.68 MB | 392 KB | 581.7 ms (581.7) | 1.72 | bit-exact 2/2, ANSI too | top-1 6/32 (float 21), agrees with float 4/32 |
| MobileNetV2, ImageNet, 224x224 | 1,3,224,224 | standalone XTS3 | 65 int8 | 3.70 MB | 1470 KB | 1609.3 ms (1609.3) | 0.62 | bit-exact 2/2, ANSI too | top-1 20/32 (float 20), agrees with float 29/32 |
| MobileNetV3-Large, ImageNet, 224x224 | 1,3,224,224 | standalone XTS3 | 121 int8 | 5.71 MB | 980 KB | 1581.6 ms (1581.6) | 0.63 | bit-exact 2/2, ANSI too | top-1 24/32 (float 23), agrees with float 27/32 |
| EfficientNet-B0, ImageNet, 224x224 | 1,3,224,224 | standalone XTS3 | 190 int8 | 5.65 MB | 2352 KB | 2498.3 ms (2498.3) | 0.40 | bit-exact 2/2, ANSI too | top-1 19/32 (float 23), agrees with float 23/32 |
| LR-ASPP MobileNetV3 segmentation, 21 classes, 224x224 | 1,3,224,224 | ExecuTorch .pte | 146 delegated, 0 CPU | 3.43 MB | 1045 KB | 3580.4 ms (3579.1); delegate 3580.5, CPU 0.0 | 0.28 | bit-exact | 6.1 dB vs float |
| SSDLite320 MobileNetV3 detection heads, COCO, 320x320 | 1,3,320,320 | ExecuTorch .pte | 243 delegated, 0 CPU | 3.72 MB | 2000 KB | 3830.3 ms (3830.0); delegate 3830.4, CPU 0.0 | 0.26 | bit-exact | 7.4 dB vs float |
| Small vision transformer: LayerNorm and attention on the CPU fallback | 1,3,64,64 | ExecuTorch .pte | 11 delegated, 30 CPU | 0.04 MB | 38 KB | 85.0 ms (84.9); delegate 22.1, CPU 62.8 | 11.77 | 39.3 dB vs host | top-1 agrees with float 31/32 |

Latency is the model invocation only (no input preprocessing), timed over repeated runs after a bit-exact (or SQNR) check against the host reference.

Host accuracy is the int8 model on the host (the board's output is bit-identical) after post-training quantization with 64 calibration images and no QAT, on 32 Imagenette validation images mapped to ImageNet classes. Models with random weights report top-1 agreement or SQNR against the float model instead.
