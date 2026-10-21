# Bundled MobileNetV2 demo

`mv2.ptn` contains torchvision's pretrained MobileNetV2 with
`MobileNet_V2_Weights.IMAGENET1K_V2`. `mv2_input.bin` is the bundled cat after the
weights' standard resize, crop, and normalization, in little-endian FP32 NCHW
`[1, 3, 224, 224]` order. `mv2_expected.bin` contains PyTorch's FP32 `[1, 1000]`
reference logits. `categories.txt` lists the corresponding ImageNet-1K labels.
The top reference prediction is `tiger cat` (index 282).

## Sources and attribution

The model implementation, transforms, and labels come from
[PyTorch torchvision](https://github.com/pytorch/vision), whose code uses the BSD
3-Clause license. The pretrained weights were trained on ImageNet-1K:

```text
https://download.pytorch.org/models/mobilenet_v2-7ebf99e0.pth
SHA-256: 7ebf99e03e254b273379b23edca7ec0da9f48273b23a332b93c1c99d49e86e8f
```

The source image, `examples/native/vision/cat.jpg`, is PyTorch/Caffe2's
[`caffe2/python/trt/data/tabby_tiger_cat.jpg`](https://github.com/pytorch/pytorch/blob/v1.13.1/caffe2/python/trt/data/tabby_tiger_cat.jpg),
distributed in the BSD-licensed PyTorch repository.
Its SHA-256 is `d5f516d7cde858db080760927ea73b1d5bab21a38ca0d4a1aea5b0e6f884969c`.

## Reproduction

Generated with Python 3.12, `torch==2.15.0.dev20261004`,
`torchvision==0.30.0.dev20261009`, `numpy==2.1.3`, and `Pillow==12.3.0`.
The exporter used the fbsource ExecuTorch sources from this stack, including
D126696057's shared `FuseBatchNormWithConvProgramPass`; the installed supporting
ExecuTorch distribution was `1.5.0+a134f8f`. The source tree took precedence over
that distribution, with `program.fbs` and `scalar_type.fbs` staged in
`exir/_serialize` for the source-based exporter.

To regenerate the same model and input with the existing Native vision exporter,
run from the ExecuTorch source root with those exporter sources importable:

```sh
python -m executorch.examples.native.vision.export \
  --model mv2 --image examples/native/vision/cat.jpg --out-dir native-demo
```

The exporter writes `mv2.ptn`, `mv2_input.bin`, `mv2_expected.bin`, and
`categories.txt` under `native-demo/mv2/`, along with a JSON reference and summary.
These four runtime artifacts are bundled here and installed with both vision
examples; exporting requires no Native wheel.

BatchNorm folding uses Native's default AOT pipeline. Inference checks all logits
with `rtol=1e-3, atol=1e-3`; regenerating on other platforms or package versions
may change floating-point results and archive checksums.

## Artifact SHA-256

```text
047961ed03e16e9e3ec4b774a0e50543376dc23b13a2a219adf028facafdf07a  mv2.ptn
33ad149d88714892e4e269453eec10877b11370f30e9625b53acfe9e88fd4028  mv2_input.bin
bfbef51c4f9203093c5cfb2a8a7fb452a488aa03cf4babab74dc5b098fcbc7a4  mv2_expected.bin
ee4fa67b1dd46919ef87529b70e25bdd1ec9ebd350de3ad426e6640eb70320a1  categories.txt
```
