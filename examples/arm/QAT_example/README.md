# Practical-RIFE Accuracy Flow

`qat_loop.py` measures Practical-RIFE eager, PTQ, and QAT accuracy on real
frame triples. It is intended as the accuracy companion to VGF performance
measurements, so each reported artifact can be tied back to a reproducible
model, ExecuTorch revision, and export command.

## Inputs

Use a standard Practical-RIFE checkout with this layout:

```text
<model-root>/train_log/IFNet_HDv3.py
```

Pass `--checkpoint /path/to/flownet.pkl` to load trained weights. The
checkpoint is optional; omitting it uses the model's initial weights.

The triples file must identify frame interpolation samples in this order:

```text
name,input0,input1,target
sample_00000,/path/frame_0.png,/path/frame_2.png,/path/frame_1.png
```

The script treats `input0` and `input1` as the two model inputs and `target`
as the expected middle frame.

## Real Accuracy Run

Use real frame triples for accuracy reporting. This example compares eager,
PTQ, and QAT at the default 768x384 shape while preserving uint8 image IO:

```bash
python examples/arm/QAT_example/qat_loop.py \
  --triples-list <path-to-real-triples.csv> \
  --model-root <path-to-Practical-RIFE> \
  --checkpoint <path-to-Practical-RIFE>/train_log/flownet.pkl \
  --height 768 \
  --width 384 \
  --max-triples 10 \
  --calibration-samples 8 \
  --mode all \
  --io-quantization uint8 \
  --qat-samples 8 \
  --qat-steps 50 \
  --qat-lr 1e-5 \
  --output-dir out/rife_accuracy_10_real_triples_qat50
```

The output directory contains:

- `report.md`: human-readable aggregate metrics.
- `metrics.json`: aggregate and per-sample metrics.
- `metrics.csv`: tabular per-sample metrics.
- `quantization_coverage/`: PTQ and QAT graph coverage details.

The flow uses the VGF SNORM-safe activation range `[-127, 127]`. Keeping this
range consistent across the graph avoids requantization boundaries around the
fused flow-offset samplers while ensuring sampled images never use the
ambiguous SNORM value `-128`.

## Smoke Runs

Small random or short QAT runs are useful only for checking that the flow runs.
Do not use them as model-quality evidence. For example, two samples and three
QAT steps can produce very low PSNR while still proving that eager, PTQ, and
QAT execution is wired correctly.

## Model Selection

Use `--model-class` to select a compatible model class from
`train_log/IFNet_HDv3.py` in your Practical-RIFE checkout. For example,
add `--model-class IFNetTwoBlock` to the command above.

### Packed Intermediate Warps

The RIFE VGF extension runs `PackRifeWarpsAndConcatPass` after TOSA lowering.
For the supported integer pattern, one shader samples both images and both
feature tensors and writes the second block's CONCAT input directly as C32.
It copies the existing full-resolution mask, features and flow, inserts the
constant timestep, and adds four dummy channels. Four zero weight columns
make the following convolution ignore those channels.

The pass checks shader identities, descriptors, matching sampling qparams,
private warp consumers, and symmetric convolution weights before rewriting.
Unsupported graphs are left unchanged. When the original block0 tail matches,
PixelShuffle, slicing, state resizing and rescaling run inside the packed shader.
The shader writes only C32, including the original INT8 previous-flow codes in
channels 24–27. The final warp shader reads those channels directly for block1's
flow update. No separate flow output is written.
The final block0 transpose-convolution weights, bias and per-channel rescale
parameters are reordered during export from channel-major to phase-major order,
so each PixelShuffle lookup reads 13 contiguous channels from C52. No runtime weight
reorder is added. Resize bounds come from the tensor dimensions; C32 stores
remain loop-based.

`FuseRifeFinalWarpsPass` fuses block1's PixelShuffle, flow update, sigmoid mask
lookup and both final image warps into one shader. It reorders block1's final
transpose-convolution weights, bias and rescale parameters at export time so
each PixelShuffle phase reads five contiguous channels from C20. Integer flow
rescales preserve TOSA's single-round behavior, and each warp retains its own
sampling quantization parameters. The shader outputs two warped RGBA images
and the quantized NCHW mask; final image blending remains in TOSA. Unsupported
patterns and tails with additional consumers are left unchanged.

`PadRifeBlock0ConcatPass` pads the two RGB image branches to C4 before their
half-resolution RESIZE operations. Image0 carries the constant timestep in
its fourth channel; image1 carries a dummy channel. The CONCAT joins four C4
branches without separate constant inputs. The pass reorders the convolution's
input weight columns and adds a zero column for the dummy channel, preserving
the convolution result. Existing image PADs are shared with the warp shaders
when their fourth channel is unused. Unsupported resize or quantization patterns
are skipped.

`glslc` must be available when exporting a matching graph.

## Reference Samples

Keep reference images outside this repository and pass them through
`--triples-list`. For internal comparisons, use a fixed triples list and keep
the image bundle unchanged across graph and quantization experiments.

The triples list is the reproducibility contract. It should use repository- or
bundle-relative paths where possible, so another user can unpack the same image
bundle and run the same command without editing absolute paths.

## Reporting Checklist

When sharing accuracy or performance data, include:

- Practical-RIFE commit hash.
- ExecuTorch commit hash.
- `qat_loop.py` command line and flags.
- Dataset or triples-list identity.
- Input shape and preprocessing mode.
- IO quantization mode.
- PTQ calibration sample count.
- QAT sample count, step count, and learning rate.
- Reference image bundle identity.
- Whether a VGF or PTE was exported from the same run.

This keeps accuracy numbers comparable with board performance results and
avoids mixing smoke-test numbers with real-data validation.
